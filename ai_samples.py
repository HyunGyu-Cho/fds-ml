from __future__ import annotations

import argparse
import csv
import json
import random
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

from rule_engine import evaluate_rules, parse_time

# v3 합성 데이터의 REVIEW 결제로 ai-server 측정 샘플(measure_*.json 형식)을 만든다
# 정답(사기 여부 · 상황)은 labels.csv에만 두고 샘플 JSON에는 넣지 않는다

ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "data" / "v3"
OUTPUT_DIR = ML_DIR / "exports" / "ai_samples"
SEED = 42
HIGH_RISK_COUNTRIES = ("NG",)
LOOKBACK = timedelta(days=30)
RULE_WINDOW = timedelta(minutes=10)
HISTORY_MAX = 20
FIRST_CASE_ID = 1001

# 상황별 뽑을 개수 — 있는 수보다 많으면 있는 만큼 전부 (정상 50 · 사기 50)
QUOTAS = {
    "NORMAL": 18,
    "ODD_NEW_CARD_DEVICE": 18,
    "ABROAD_TRAVEL": 7,
    "ABROAD_RESIDENT": 2,
    "ODD_BURST": 3,
    "ODD_BIG_FIRST": 2,
    "FRAUD_SAME_COUNTRY": 17,
    "FRAUD_FOREIGN": 13,
    "FRAUD_HIGH_RISK": 20,
}

# Spring FdsEvaluationTransactionService의 규칙 이름 · 점수와 같다
RULES = {
    "R01": ("고액 결제", 30),
    "R02": ("신규 단말기", 25),
    "R03": ("신규 카드", 25),
    "R04": ("단시간 반복 결제", 30),
    "R05": ("고위험 국가 IP", 40),
    "R06": ("카드 테스팅", 45),
    "R08": ("거절 후 재시도", 35),
}

LABEL_COLUMNS = (
    "caseId",
    "paymentId",
    "sampleFile",
    "fraudLabel",
    "condition",
    "scenarioType",
    "ruleHits",
    "rareSignal",
)


def _iso(value) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


# 만 원 단위면 "50만 원", 아니면 "12,345원" (Spring RuleCriterionFormatter.won과 같다)
def won(amount: int) -> str:
    if amount != 0 and amount % 10_000 == 0:
        return f"{amount // 10_000:,}만 원"
    return f"{amount:,}원"


# 지난 결제 상태 — 보류(REVIEW) 결제는 24시간 뒤 자동 취소되므로 사기 · 정상 모두 만료로 둔다 (정답이 새지 않게)
def history_status(row: dict) -> str:
    if row["authorizationResult"] == "DECLINED":
        return "DECLINED"
    return {"ALLOW": "CAPTURED", "REVIEW": "EXPIRED", "DENY": "DENIED"}[row["_decision"]]


def load_v3(data_dir: Path = DATA_DIR) -> list[dict]:
    with (data_dir / "transactions.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    with (data_dir / "conditions.csv").open(encoding="utf-8", newline="") as file:
        conditions = {r["paymentId"]: r["condition"] for r in csv.DictReader(file)}
    for row in rows:
        row["_condition"] = conditions[row["paymentId"]]
    return rows


# 모든 결제에 Rule 판정과 시각을 붙이고, 고객 · 카드 · 기기별 시간순 목록을 만든다
class Index:
    def __init__(self, rows: list[dict]) -> None:
        for row, result in zip(rows, evaluate_rules(rows, HIGH_RISK_COUNTRIES), strict=True):
            row["_decision"] = result.decision
            row["_score"] = result.score
            row["_hits"] = result.hits
            row["_at"] = parse_time(row["paymentRequestedAt"])
        ordered = sorted(rows, key=lambda r: (r["_at"], int(r["paymentId"])))
        self.rows = rows
        self.by_customer = defaultdict(list)
        self.by_card = defaultdict(list)
        self.by_device = defaultdict(list)
        for row in ordered:
            self.by_customer[(row["merchantId"], row["customerRef"])].append(row)
            if row["cardToken"]:
                self.by_card[(row["merchantId"], row["cardToken"])].append(row)
            if row["deviceId"]:
                self.by_device[(row["merchantId"], row["deviceId"])].append(row)

    # 이 결제보다 앞선 결제만 (since 이후) — 같은 시각 결제는 결제 ID 순서로 앞선 것만
    @staticmethod
    def before(group: list[dict], row: dict, since) -> list[dict]:
        key = (row["_at"], int(row["paymentId"]))
        times = [(r["_at"], int(r["paymentId"])) for r in group]
        end = bisect_left(times, key)
        return [r for r in group[:end] if r["_at"] >= since]


def select(rows: list[dict], quotas: dict[str, int] = QUOTAS, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    review = defaultdict(list)
    for row in rows:
        if row["_decision"] == "REVIEW":
            review[row["_condition"]].append(row)
    chosen = []
    for condition, quota in quotas.items():
        pool = sorted(review[condition], key=lambda r: int(r["paymentId"]))
        chosen.extend(rng.sample(pool, min(quota, len(pool))))
    # 상황을 섞어 둔다 — --limit 5로 앞 몇 건만 돌려도 정상 · 사기가 섞이게
    rng.shuffle(chosen)
    return chosen


def _criterion(code: str, row: dict, approved_count: int, window: list[dict]) -> tuple:
    attempts = len({r["paymentId"] for r in window if r["_at"] >= row["_at"] - RULE_WINDOW}) + 1
    declined = sum(1 for r in window if r["_at"] > row["_at"] - RULE_WINDOW and r["authorizationResult"] == "DECLINED")
    if code == "R01":
        return "50만 원 이상", won(int(row["amount"]))
    if code in ("R02", "R03"):
        target = "단말기" if code == "R02" else "카드"
        return f"이전 승인 결제에 없던 {target}", f"이전 승인 {approved_count}건 중 같은 {target} 없음"
    if code == "R04":
        return "10분 안 3회 이상", f"{attempts}회"
    if code == "R08":
        return "10분 안 거절 2회 이상", f"거절 {declined}회"
    # R05는 식별자(국가)가 들어 Spring도 비운다, R06은 이 샘플에 거의 없어 비운다
    return None, None


def _linked(index: Index, row: dict, by: dict, key_column: str) -> dict:
    since = row["_at"] - LOOKBACK
    others = [
        r
        for r in index.before(by[(row["merchantId"], row[key_column])], row, since)
        if r["customerRef"] != row["customerRef"]
    ]
    results = Counter(r["_decision"] for r in others)
    return {
        "provided": True,
        "otherCustomerCount": len({r["customerRef"] for r in others}),
        "paymentCount": len(others),
        "resultCounts": [{"result": k, "count": results[k]} for k in ("ALLOW", "REVIEW", "DENY") if results[k]],
        "declinedCount": sum(1 for r in others if r["authorizationResult"] == "DECLINED"),
    }


# 정상 고객에게 드문 정황 (#490 기준) — 거절 후 재시도 · 카드 테스팅 · 같은 카드나 기기를 다른 고객이 씀
def has_rare_signal(sample: dict) -> bool:
    hits = {h["code"] for p in sample["fdsEvidence"]["payments"] for h in p["ruleHits"]}
    linked = sample["linked"]
    return bool(hits & {"R06", "R08"}) or any(linked[k]["otherCustomerCount"] > 0 for k in ("card", "device"))


def build_sample(index: Index, row: dict, case_id: str) -> dict:
    at = row["_at"]
    retrieved = _iso(at + timedelta(minutes=5))
    customer = index.by_customer[(row["merchantId"], row["customerRef"])]
    earlier = index.before(customer, row, at - LOOKBACK)
    all_earlier = index.before(customer, row, customer[0]["_at"])
    approved_count = sum(1 for r in all_earlier if r["authorizationResult"] == "APPROVED")
    head = {"caseId": case_id, "evidenceVersion": 1}

    hits = []
    for code in row["_hits"]:
        name, score = RULES[code]
        criterion, observed = _criterion(code, row, approved_count, earlier)
        hits.append({"code": code, "name": name, "appliedScore": score, "criterion": criterion, "observed": observed})

    recent = list(reversed(earlier))
    shown = recent[:HISTORY_MAX]
    statuses = Counter(history_status(r) for r in shown)
    order = ("CAPTURED", "HELD", "PENDING", "VOIDED", "DECLINED", "DENIED", "EXPIRED")

    return {
        "_measure": "fds-ml v3 합성 데이터 REVIEW 결제 — 정답은 labels.csv에만 있다",
        "casePayments": {
            **head,
            "source": "P3·P5",
            "retrievedAt": retrieved,
            "payments": [
                {"paymentId": row["paymentId"], "amount": int(row["amount"]), "paymentStatus": "HELD", "requestedAt": _iso(at)}
            ],
            "missingPaymentIds": [],
        },
        "fdsEvidence": {
            **head,
            "source": "P2",
            "retrievedAt": retrieved,
            "payments": [
                {
                    "paymentId": row["paymentId"],
                    "riskScore": row["_score"],
                    "fdsResult": row["_decision"],
                    "ruleHits": hits,
                    "signals": {
                        "cardDisplayLabel": row["cardDisplayLabel"] or None,
                        "cardProvided": bool(row["cardToken"]),
                        "deviceProvided": bool(row["deviceId"]),
                        "ipCountryCode": row["ipCountryCode"] or None,
                        "ipProvided": bool(row["ipCountryCode"]),
                    },
                }
            ],
            "missingPaymentIds": [],
        },
        "history": {
            **head,
            "source": "P5",
            "retrievedAt": retrieved,
            "lookbackDays": LOOKBACK.days,
            "payments": [
                {"paymentId": r["paymentId"], "amount": int(r["amount"]), "paymentStatus": history_status(r), "requestedAt": _iso(r["_at"])}
                for r in shown
            ],
            "statusCounts": [{"paymentStatus": s, "count": statuses[s]} for s in order if statuses[s]],
            "truncated": len(recent) > HISTORY_MAX,
        },
        "linked": {
            **head,
            "source": "P2",
            "retrievedAt": retrieved,
            "lookbackDays": LOOKBACK.days,
            "card": _linked(index, row, index.by_card, "cardToken"),
            "device": _linked(index, row, index.by_device, "deviceId"),
            # v3 데이터에는 IP 원문이 없어 비교할 수 없다 → "정보 없음"
            "ip": {"provided": False, "otherCustomerCount": 0, "paymentCount": 0, "resultCounts": [], "declinedCount": 0},
        },
    }


def export(rows: list[dict], output_dir: Path = OUTPUT_DIR, quotas: dict[str, int] = QUOTAS, seed: int = SEED) -> list[dict]:
    index = Index(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for offset, row in enumerate(select(rows, quotas, seed)):
        case_id = str(FIRST_CASE_ID + offset)
        sample = build_sample(index, row, case_id)
        name = f"case-{case_id}.json"
        (output_dir / name).write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
        labels.append(
            {
                "caseId": case_id,
                "paymentId": row["paymentId"],
                "sampleFile": name,
                "fraudLabel": row["fraudLabel"],
                "condition": row["_condition"],
                "scenarioType": row["scenarioType"],
                "ruleHits": "+".join(row["_hits"]),
                "rareSignal": int(has_rare_signal(sample)),
            }
        )
    with (output_dir / "labels.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LABEL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(labels)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 REVIEW 결제 → ai-server 측정 샘플")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    labels = export(load_v3(args.data), args.out)
    print(f"샘플 {len(labels)}건 → {args.out}")
    for condition, count in Counter(r["condition"] for r in labels).items():
        rare = sum(int(r["rareSignal"]) for r in labels if r["condition"] == condition)
        print(f"  {condition:<22} {count:>3}건 (드문 정황 {rare})")


if __name__ == "__main__":
    main()
