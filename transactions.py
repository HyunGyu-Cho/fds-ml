
import random
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone


# 한국 표준시와 거래 생성 기간
KST = timezone(timedelta(hours=9))
START_DATE = datetime(2026, 4, 1, tzinfo=KST)
END_DATE = datetime(2026, 9, 1, tzinfo=KST)


# 정상 소액 구매 발생 확률
# 실제 금융 통계가 아닌 합성 데이터 실험용 가정
NORMAL_SMALL_PAYMENT_RATE = 0.04


# 정상 연속 결제 후보 쌍의 선택 확률
# 전체 정상 거래의 8%라는 뜻은 아니다.
NORMAL_REPEAT_PAIR_RATE = 0.08


def to_utc_string(value: datetime) -> str:
    """시간을 UTC ISO 8601 문자열로 변환한다."""

    utc_time = value.astimezone(timezone.utc)
    return utc_time.isoformat().replace("+00:00", "Z")


def generate_payment_time(
    rng: random.Random,
    customer: dict,
) -> datetime:
    """고객의 선호 시간대를 고려해 결제 시각을 생성한다."""

    # 거래 기간 안에서 날짜를 선택
    total_days = (END_DATE - START_DATE).days
    day_offset = rng.randrange(total_days)

    # 대부분 평소 선호 시간 주변에서 결제
    if rng.random() < 0.10:
        hour = rng.randrange(24)
    else:
        hour = round(
            rng.gauss(customer["preferredHour"], 2.0)
        ) % 24

    # 분과 초를 생성해 거래 시각 완성
    minute = rng.randrange(60)
    second = rng.randrange(60)

    return START_DATE + timedelta(
        days=day_offset,
        hours=hour,
        minutes=minute,
        seconds=second,
    )


def generate_payment_amount(
    rng: random.Random,
    customer: dict,
) -> int:
    """고객의 소비 성향을 반영해 정상 거래 금액을 생성한다."""

    # 일부 정상 구매는 평소 소비 금액보다 작은 소액 거래
    if rng.random() < NORMAL_SMALL_PAYMENT_RATE:
        return rng.randrange(1000, 9001, 100)

    # 로그정규분포로 평소 소비 금액 주변의 변동성 생성
    multiplier = rng.lognormvariate(0, 0.35)
    amount = customer["normalAmount"] * multiplier

    # 정상 거래에도 드물게 큰 금액의 구매 발생
    if rng.random() < 0.05:
        amount *= rng.uniform(1.7, 3.5)

    # 최소 1,000원, 금액은 100원 단위로 정리
    amount = max(1000, round(amount / 100) * 100)

    return int(amount)


def generate_payment_device(
    rng: random.Random,
    customer: dict,
) -> str:
    """평소 기기 또는 정상적인 추가 기기 사용을 생성한다."""

    # 소수의 정상 거래에서 추가 기기 사용
    if rng.random() < 0.04:
        return f"SIM-DEVICE-{customer['customerRef']}-EXTRA"

    # 나머지는 기존 고객 프로필의 기기 사용
    return rng.choice(customer["deviceIds"])


def generate_payment_card(
    rng: random.Random,
    customer: dict,
) -> str:
    """평소 카드 또는 정상적인 추가 카드 사용을 생성한다."""

    # 소수의 정상 거래에서 추가 카드 사용
    if rng.random() < 0.02:
        return f"SIM-CARD-{customer['customerRef']}-EXTRA"

    # 나머지는 기존 고객 프로필의 카드 사용
    return rng.choice(customer["cardTokens"])


def generate_authorization_result(
    rng: random.Random,
) -> str:
    """가상의 카드 승인 결과를 생성한다."""

    # 정상 거래라도 잔액 부족 등으로 승인 실패 가능
    if rng.random() < 0.02:
        return "DECLINED"

    return "APPROVED"


def inject_normal_repeat_payments(
    rng: random.Random,
    events: list[dict],
) -> None:
    """일부 정상 거래를 같은 고객의 연속 구매로 배치한다.

    새로운 거래를 추가하지 않고 기존 거래의 시각을 조정한다.
    정상 거래 총수와 고객·가맹점 관계는 유지한다.
    """

    # 동일 가맹점·고객의 거래를 모은다.
    groups = defaultdict(list)

    for event in events:
        key = (
            event["merchantId"],
            event["customerRef"],
        )
        groups[key].append(event)

    # 고객별로 연속 결제 후보를 선택한다.
    for group in groups.values():
        if len(group) < 2:
            continue

        # 같은 고객의 거래를 무작위로 짝짓는다.
        rng.shuffle(group)

        # 각 거래가 한 번만 짝지어지도록 2칸씩 이동
        for i in range(0, len(group) - 1, 2):

            # 후보 쌍 중 일부만 연속 결제로 변경
            if rng.random() >= NORMAL_REPEAT_PAIR_RATE:
                continue

            first = group[i]
            second = group[i + 1]

            base_time = first["_paymentTime"]

            # 종료일에 가까운 거래는 변경하지 않는다.
            if base_time >= END_DATE - timedelta(minutes=30):
                continue

            # 약 55%는 45초~5분 간격
            # 나머지는 6~30분 간격
            if rng.random() < 0.55:
                gap_seconds = rng.randint(45, 300)
            else:
                gap_seconds = rng.randint(360, 1800)

            # 첫 번째 거래 이후에 두 번째 거래 배치
            second_time = base_time + timedelta(
                seconds=gap_seconds
            )

            if second_time >= END_DATE:
                continue

            # 두 번째 거래의 결제 요청 시각 변경
            second["_paymentTime"] = second_time

            # 주문 생성 시각도 변경된 결제 시각에 맞춘다.
            lead_minutes = rng.randint(
                0,
                min(15, gap_seconds // 60),
            )

            second["_orderTime"] = second_time - timedelta(
                minutes=lead_minutes
            )

            # 정상 연속 구매에서는 같은 기기를 재사용할 수 있다.
            if rng.random() < 0.90:
                second["deviceId"] = first["deviceId"]

            # 정상 연속 구매에서는 같은 카드를 재사용할 수 있다.
            if rng.random() < 0.90:
                second["cardToken"] = first["cardToken"]


def generate_normal_transactions(
    rng: random.Random,
    customers: list[dict],
    count: int = 18000,
) -> list[dict]:
    """고객 프로필 기반 정상 거래 이력을 생성한다."""

    if not customers:
        raise ValueError("고객 프로필이 비어 있습니다.")

    if count <= 0:
        raise ValueError("거래 건수는 1 이상이어야 합니다.")

    # 월 구매 빈도가 높은 고객에게 더 큰 선택 가중치 부여
    weights = [
        c["monthlyFrequency"]
        for c in customers
    ]

    selected_customers = rng.choices(
        customers,
        weights=weights,
        k=count,
    )

    # 최종 거래 ID를 부여하기 전 임시 이벤트 생성
    events = []

    for customer in selected_customers:

        # 고객 행동 패턴에 따라 금액과 시각 생성
        payment_time = generate_payment_time(rng, customer)
        amount = generate_payment_amount(rng, customer)

        # 고객별 카드·기기 사용 패턴 생성
        device_id = generate_payment_device(rng, customer)
        card_token = generate_payment_card(rng, customer)

        # 주문 생성은 결제 요청보다 최대 15분 빠름
        order_time = payment_time - timedelta(
            minutes=rng.randint(0, 15)
        )

        # 카드 승인 결과는 사기 여부와 별도로 생성
        authorization = generate_authorization_result(rng)

        event = {
            "_paymentTime": payment_time,
            "_orderTime": order_time,
            "merchantId": customer["merchantId"],
            "customerRef": customer["customerRef"],
            "amount": amount,
            "deviceId": device_id,
            "cardToken": card_token,
            "cardDisplayLabel": "SIM **** 0000",
            "buyerEmail": customer["buyerEmail"],
            "authorizationResult": authorization,
            "fraudLabel": 0,
            "scenarioType": "NORMAL",
        }

        events.append(event)

    # 일부 정상 거래를 연속 결제로 변경
    # 정렬 이전에 실행해야 변경된 시각이 반영된다.
    inject_normal_repeat_payments(
        rng=rng,
        events=events,
    )

    # 전체 거래를 실제 발생 시각 기준으로 정렬
    events.sort(
        key=lambda e: (
            e["_paymentTime"],
            e["merchantId"],
            e["customerRef"],
        )
    )

    # 정렬된 거래에 고유 ID와 API 대응 필드 부여
    transactions = []

    for index, event in enumerate(events, start=1):

        payment_time = to_utc_string(
            event["_paymentTime"]
        )

        order_time = to_utc_string(
            event["_orderTime"]
        )

        # UUID5를 사용해 재현 가능한 평가 요청 ID 생성
        evaluation_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"donworry-normal-{index}-{payment_time}",
        )

        transaction = {
            "paymentId": index,
            "evaluationRequestId": str(evaluation_id),
            "orderRef": f"SIM-ORDER-N-{index:08d}",
            "merchantId": event["merchantId"],
            "customerRef": event["customerRef"],
            "amount": event["amount"],
            "orderCreatedAt": order_time,
            "paymentRequestedAt": payment_time,
            "deviceId": event["deviceId"],
            "cardToken": event["cardToken"],
            "cardDisplayLabel": event["cardDisplayLabel"],
            "buyerEmail": event["buyerEmail"],
            "authorizationResult": event["authorizationResult"],
            "fraudLabel": event["fraudLabel"],
            "scenarioType": event["scenarioType"],
        }

        transactions.append(transaction)

    return transactions