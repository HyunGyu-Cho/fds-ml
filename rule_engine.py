from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

# 팀 저장소 backend FdsEvaluationTransactionService.java의 Rule(R01~R08)을 그대로 옮긴 오프라인 재현
# R05(고위험 국가 IP) · R07(차단 목록)은 합성 데이터에 IP · 차단 목록이 없어 항상 꺼진다

HIGH_AMOUNT_THRESHOLD = 500_000
HIGH_AMOUNT_SCORE = 30
REPEAT_WINDOW = timedelta(minutes=10)
REPEAT_COUNT_THRESHOLD = 3
REPEAT_SCORE = 30
NEW_DEVICE_SCORE = 25
NEW_CARD_SCORE = 25
SMALL_AMOUNT_MAX = 10_000
CARD_TESTING_SMALL_COUNT = 5
CARD_TESTING_CARD_COUNT = 3
CARD_TESTING_DECLINED_COUNT = 3
CARD_TESTING_SCORE = 45
DECLINED_RETRY_COUNT_THRESHOLD = 2
DECLINED_RETRY_SCORE = 35
MAX_SCORE = 100
REVIEW_MIN_SCORE = 50
DENY_MIN_SCORE = 80

ALWAYS_OFF_RULES = ("R05", "R07")


@dataclass(frozen=True)
class Payment:
    payment_id: str
    requested_at: datetime
    amount: int
    device_id: str | None
    card_token: str | None
    authorization_result: str


@dataclass(frozen=True)
class RuleResult:
    score: int
    decision: str
    hits: tuple[str, ...]


def parse_time(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("거래 시각에 시간대 정보가 없습니다.")
    return timestamp.astimezone(timezone.utc)


def decide(score: int) -> str:
    # Java resultFor와 같은 경계: 80 이상 거절, 50 이상 보류
    if score >= DENY_MIN_SCORE:
        return "DENY"
    if score >= REVIEW_MIN_SCORE:
        return "REVIEW"
    return "ALLOW"


def _optional(value: str | None) -> str | None:
    # CSV의 빈 칸은 Java의 null과 같게 본다
    return value or None


def to_payment(row: dict) -> Payment:
    return Payment(
        payment_id=str(row["paymentId"]),
        requested_at=parse_time(row["paymentRequestedAt"]),
        amount=int(row["amount"]),
        device_id=_optional(row.get("deviceId")),
        card_token=_optional(row.get("cardToken")),
        authorization_result=row.get("authorizationResult", ""),
    )


@dataclass
class CustomerHistory:
    # 같은 가맹점 · 같은 손님의 앞선 거래 요약 — 매번 전체 이력을 훑지 않게 (TabFormer처럼 손님당 수천 건일 때)
    recent: deque[Payment] = field(default_factory=deque)
    has_approved: bool = False
    approved_devices: set[str | None] = field(default_factory=set)
    approved_cards: set[str | None] = field(default_factory=set)

    def add(self, payment: Payment) -> None:
        self.recent.append(payment)
        if payment.authorization_result == "APPROVED":
            self.has_approved = True
            self.approved_devices.add(payment.device_id)
            self.approved_cards.add(payment.card_token)


def evaluate_payment(current: Payment, history: CustomerHistory) -> RuleResult:
    # history는 같은 가맹점 · 같은 손님의 앞선 거래만 (현재 결제의 승인 결과는 보지 않는다)
    hits = []
    score = 0
    window_start = current.requested_at - REPEAT_WINDOW

    if current.amount >= HIGH_AMOUNT_THRESHOLD:
        hits.append("R01")
        score += HIGH_AMOUNT_SCORE

    # 10분 경계 시각보다 오래된 거래는 다시 볼 일이 없다 (거래를 시각 순서로 처리하므로)
    while history.recent and history.recent[0].requested_at < window_start:
        history.recent.popleft()

    # R04는 JPA Between이라 10분 경계 시각도 포함한다
    attempt_ids = {p.payment_id for p in history.recent}
    attempt_ids.add(current.payment_id)
    if len(attempt_ids) >= REPEAT_COUNT_THRESHOLD:
        hits.append("R04")
        score += REPEAT_SCORE

    # 앞선 승인 거래가 있고, 그중 같은 기기 · 카드가 하나도 없을 때
    if current.device_id and history.has_approved and current.device_id not in history.approved_devices:
        hits.append("R02")
        score += NEW_DEVICE_SCORE
    if current.card_token and history.has_approved and current.card_token not in history.approved_cards:
        hits.append("R03")
        score += NEW_CARD_SCORE

    # R06 · R08은 isWithinWindow라 10분 경계 시각은 빼고 현재 시각까지 포함한다
    window = [p for p in history.recent if window_start < p.requested_at]
    small_ids = {p.payment_id for p in window if p.amount <= SMALL_AMOUNT_MAX}
    if current.amount <= SMALL_AMOUNT_MAX:
        small_ids.add(current.payment_id)
    cards = {p.card_token for p in window if p.card_token}
    if current.card_token:
        cards.add(current.card_token)
    declined = len({p.payment_id for p in window if p.authorization_result == "DECLINED"})

    if (
        current.card_token
        and len(small_ids) >= CARD_TESTING_SMALL_COUNT
        and len(cards) >= CARD_TESTING_CARD_COUNT
        and declined >= CARD_TESTING_DECLINED_COUNT
    ):
        hits.append("R06")
        score += CARD_TESTING_SCORE
    if declined >= DECLINED_RETRY_COUNT_THRESHOLD:
        hits.append("R08")
        score += DECLINED_RETRY_SCORE

    score = min(MAX_SCORE, score)
    return RuleResult(score=score, decision=decide(score), hits=tuple(hits))


def evaluate_rules(transactions: list[dict]) -> list[RuleResult]:
    # 시각 순서대로 돌며 결제마다 그 시점까지의 이력으로 점수를 매긴다 (반환 순서 = 입력 순서)
    payments = [to_payment(row) for row in transactions]
    order = sorted(range(len(payments)), key=lambda i: (payments[i].requested_at, int(payments[i].payment_id)))
    history: dict[tuple[str, str], CustomerHistory] = defaultdict(CustomerHistory)
    results: list[RuleResult | None] = [None] * len(payments)

    for index in order:
        row = transactions[index]
        key = (str(row["merchantId"]), str(row["customerRef"]))
        results[index] = evaluate_payment(payments[index], history[key])
        history[key].add(payments[index])

    return results
