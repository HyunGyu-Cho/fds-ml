
import random
from collections import defaultdict
from datetime import datetime, timedelta

from transactions import KST, START_DATE, END_DATE, to_utc_string


# 사기 시나리오별 최소 사건 구성 거래 수
SCENARIOS = {
    "CARD_TESTING": 5,
    "ACCOUNT_TAKEOVER": 2,
    "STOLEN_CARD": 2,
    "LOW_AND_SLOW": 4,
    "SPLIT_PAYMENT": 3,
    "BEHAVIORAL_ANOMALY": 1,
    "CARD_DEVICE_CHANGE": 2,
}


# 정상 거래 이력이 적은 고객도 공격 대상에 포함할 시나리오
COLD_START_SCENARIOS = {
    "CARD_TESTING",
    "LOW_AND_SLOW",
    "SPLIT_PAYMENT",
}

# 해당 시나리오의 Episode 중 저이력 대상으로 생성할 확률
# 실제 금융 통계가 아닌 합성 데이터 실험용 가정
COLD_START_PROBABILITY = 0.30


def parse_kst(value: str) -> datetime:
    """UTC 문자열을 KST datetime으로 변환한다."""

    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    ).astimezone(KST)


def partition_count(total: int, minimum: int) -> list[int]:
    """총 거래 수를 최소 사건 크기 단위로 나눈다."""

    groups = [minimum] * (total // minimum)

    # 나머지 거래를 기존 사건에 분배
    for i in range(total % minimum):
        groups[i] += 1

    return groups


def split_target_amount(
    rng: random.Random,
    target_amount: int,
    payment_count: int,
    min_amount: int = 100_000,
    max_amount: int = 450_000,
) -> list[int]:
    """목표 총액을 100원 단위의 여러 결제 금액으로 정확히 분할한다.

    개별 금액 범위를 지키면서 합계가 target_amount와 같도록 보장한다.
    금액 범위 및 목표 총액은 실제 금융 통계가 아닌 합성 데이터 가정이다.
    """

    # 최소 2건이며, 모든 금액은 100원 단위의 정수여야 한다.
    if payment_count < 2:
        raise ValueError("분할 결제 건수는 2건 이상이어야 합니다.")

    values = (target_amount, min_amount, max_amount)

    if any(
        not isinstance(value, int) or value % 100 != 0
        for value in values
    ):
        raise ValueError("금액은 100원 단위의 정수여야 합니다.")

    if (
        min_amount < 1_000
        or max_amount >= 500_000
        or min_amount > max_amount
    ):
        raise ValueError("개별 분할 금액의 범위가 유효하지 않습니다.")

    if not (
        min_amount * payment_count
        <= target_amount
        <= max_amount * payment_count
    ):
        raise ValueError(
            "해당 목표 총액은 지정한 건수와 금액 범위로 나눌 수 없습니다."
        )

    remaining = target_amount
    amounts = []

    # 남은 거래의 최소·최대 금액을 고려해 분할
    for remaining_count in range(payment_count, 1, -1):
        lower = max(
            min_amount,
            remaining - (remaining_count - 1) * max_amount,
        )

        upper = min(
            max_amount,
            remaining - (remaining_count - 1) * min_amount,
        )

        amount = rng.randrange(lower, upper + 100, 100)
        amounts.append(amount)
        remaining -= amount

    # 마지막 결제는 남은 금액으로 결정
    amounts.append(remaining)
    rng.shuffle(amounts)

    assert sum(amounts) == target_amount
    return amounts


def make_fraud_transaction(
    rng: random.Random,
    customer: dict,
    when: datetime,
    amount: float,
    device: str,
    card: str,
    status: str,
    scenario: str,
) -> dict:
    """P5·P2 대응 필드와 사기 Label을 가진 거래를 만든다."""

    # 정상 거래와 동일하게 금액을 100원 단위로 처리
    amount = max(1000, int(round(amount / 100) * 100))

    # 주문 생성은 결제 요청보다 최대 15분 이전
    order_time = when - timedelta(minutes=rng.randint(0, 15))

    return {
        "paymentId": None,
        "evaluationRequestId": None,
        "orderRef": None,
        "merchantId": customer["merchantId"],
        "customerRef": customer["customerRef"],
        "amount": amount,
        "orderCreatedAt": to_utc_string(order_time),
        "paymentRequestedAt": to_utc_string(when),
        "deviceId": device,
        "cardToken": card,
        "cardDisplayLabel": "SIM **** 0000",
        "buyerEmail": customer["buyerEmail"],
        "authorizationResult": status,
        "fraudLabel": 1,
        "scenarioType": scenario,
    }


def generate_fraud_transactions(
    rng: random.Random,
    customers: list[dict],
    normal_transactions: list[dict],
    count: int = 2000,
) -> list[dict]:
    """정상 고객 이력을 바탕으로 7종 사기 거래를 생성한다.

    일부 시나리오는 정상 거래 이력이 적은 고객도 공격 대상으로 선택한다.
    """

    if count < 35:
        raise ValueError("사기 거래는 최소 35건 이상이어야 합니다.")

    # 고객별 정상 거래 이력 구성
    histories = defaultdict(list)

    for tx in normal_transactions:
        histories[tx["customerRef"]].append(tx)

    # 공격 발생 이후 거래를 생성할 수 있도록 종료일 여유 확보
    upper = END_DATE - timedelta(days=2)

    # 기존 방식에 사용할 정상 거래 이력이 충분한 고객
    eligible = [
        customer
        for customer in customers
        if len(histories[customer["customerRef"]]) >= 3
        and parse_kst(
            histories[customer["customerRef"]][2]["paymentRequestedAt"]
        ) < upper - timedelta(days=1)
    ]

    if not eligible:
        raise ValueError("정상 거래 이력이 충분한 고객이 없습니다.")

    # 고객별 세 번째 정상 거래 시점을 저장
    # 이 시점 이전에는 정상 거래 이력이 최대 2건이다.
    third_normal_time = {}

    for customer in customers:
        history = histories[customer["customerRef"]]

        if len(history) >= 3:
            third_normal_time[customer["customerRef"]] = parse_kst(
                history[2]["paymentRequestedAt"]
            )
        else:
            third_normal_time[customer["customerRef"]] = None

    # 도난 카드 시나리오에서 사용할 가맹점별 고객 목록
    by_merchant = defaultdict(list)

    for customer in customers:
        by_merchant[customer["merchantId"]].append(customer)

    frauds = []
    scenario_names = list(SCENARIOS)
    episode_serial = 0

    # 사기 건수를 7종 시나리오에 균등하게 배분
    for scenario_idx, (name, minimum) in enumerate(SCENARIOS.items()):
        target = count // len(scenario_names)

        if scenario_idx < count % len(scenario_names):
            target += 1

        # 시나리오별 여러 거래를 하나의 공격 사건으로 구성
        for episode_size in partition_count(target, minimum):
            episode_serial += 1
            attack_event_id = f"ATTACK-{episode_serial:06d}"

            # 일부 시나리오에서 저이력 고객 공격을 생성
            use_cold_start = (
                name in COLD_START_SCENARIOS
                and rng.random() < COLD_START_PROBABILITY
            )

            if use_cold_start:
                # 생성 기간 내에서 공격 시작 시각 선택
                duration_seconds = int(
                    (upper - START_DATE).total_seconds()
                )

                anchor = START_DATE + timedelta(
                    seconds=rng.randrange(duration_seconds)
                )

                # 해당 시점에 정상 거래가 0~2건인 고객을 선택
                candidates = [
                    c
                    for c in customers
                    if (
                        third_normal_time[c["customerRef"]] is None
                        or third_normal_time[c["customerRef"]] > anchor
                    )
                ]

                if not candidates:
                    raise ValueError(
                        "정상 거래 이력이 적은 공격 대상 고객이 없습니다."
                    )

                customer = rng.choice(candidates)

            else:
                # 기존 방식: 정상 거래 3건 이상을 가진 고객 선택
                customer = rng.choice(eligible)
                history = histories[customer["customerRef"]]

                # 세 번째 정상 거래 이후의 거래를 기준으로 선택
                anchors = [
                    tx
                    for tx in history[2:]
                    if parse_kst(tx["paymentRequestedAt"])
                    < upper - timedelta(days=1)
                ]

                base = parse_kst(
                    rng.choice(anchors)["paymentRequestedAt"]
                )

                # 기준 거래 이후 1~3일 사이 공격 시작
                anchor = min(
                    base + timedelta(
                        days=rng.randint(1, 3),
                        minutes=rng.randint(0, 60),
                    ),
                    upper,
                )

            # 고객이 기존에 사용하던 카드와 기기 선택
            card = rng.choice(customer["cardTokens"])
            device = rng.choice(customer["deviceIds"])

            # 도난 카드: 같은 가맹점의 다른 고객이 사용한 카드 선택
            if name == "STOLEN_CARD":
                donors = [
                    other
                    for other in by_merchant[customer["merchantId"]]
                    if other["customerRef"] != customer["customerRef"]
                    and any(
                        parse_kst(tx["paymentRequestedAt"]) < anchor
                        for tx in histories[other["customerRef"]]
                    )
                ]

                if not donors:
                    raise ValueError("도난 카드의 과거 이력이 부족합니다.")

                donor = rng.choice(donors)

                prior_cards = [
                    tx["cardToken"]
                    for tx in histories[donor["customerRef"]]
                    if parse_kst(tx["paymentRequestedAt"]) < anchor
                ]

                card = rng.choice(prior_cards)

            when = anchor

            # 분할 결제 사건의 목표 총액 설정
            split_amounts = None

            if name == "SPLIT_PAYMENT":
                target_amount = rng.randrange(
                    600_000,
                    900_001,
                    100,
                )

                split_amounts = split_target_amount(
                    rng=rng,
                    target_amount=target_amount,
                    payment_count=episode_size,
                )

            # 하나의 공격 사건에 속한 거래를 시간순으로 생성
            for j in range(episode_size):

                if j:
                    if name == "CARD_TESTING":
                        when += timedelta(
                            seconds=rng.randint(25, 80)
                        )

                    elif name == "LOW_AND_SLOW":
                        when += timedelta(
                            minutes=rng.randint(20, 50)
                        )

                    elif name == "SPLIT_PAYMENT":
                        when += timedelta(
                            minutes=rng.randint(12, 25)
                        )

                    else:
                        when += timedelta(
                            minutes=rng.randint(4, 20)
                        )

                # 1. CARD_TESTING
                # 여러 카드로 짧은 시간에 소액 결제 시도
                if name == "CARD_TESTING":
                    amount = rng.randint(1000, 9000)

                    card_id = (
                        f"SIM-CARD-{customer['customerRef']}-{100 + j}"
                    )

                    device_id = device

                    status = (
                        "DECLINED"
                        if j < 3
                        else rng.choice(["APPROVED", "DECLINED"])
                    )

                # 2. ACCOUNT_TAKEOVER
                # 계정 탈취로 인한 기기·카드·금액 변화
                elif name == "ACCOUNT_TAKEOVER":
                    amount = (
                        customer["normalAmount"]
                        * rng.uniform(1.2, 3.1)
                    )

                    card_id = (
                        card
                        if rng.random() < 0.5
                        else f"SIM-CARD-{customer['customerRef']}-EXTRA"
                    )

                    device_id = (
                        f"SIM-DEVICE-{customer['customerRef']}-EXTRA"
                    )

                    status = (
                        "APPROVED"
                        if rng.random() < 0.87
                        else "DECLINED"
                    )

                # 3. STOLEN_CARD
                # 다른 고객이 사용한 카드를 무단 사용
                elif name == "STOLEN_CARD":
                    amount = (
                        customer["normalAmount"]
                        * rng.uniform(0.7, 2.1)
                    )

                    card_id = card

                    device_id = (
                        device
                        if rng.random() < 0.5
                        else f"SIM-DEVICE-{customer['customerRef']}-EXTRA"
                    )

                    status = (
                        "APPROVED"
                        if rng.random() < 0.60
                        else "DECLINED"
                    )

                # 4. LOW_AND_SLOW
                # 짧은 탐지 시간창을 회피하는 저속 반복 거래
                elif name == "LOW_AND_SLOW":
                    amount = (
                        customer["normalAmount"]
                        * rng.uniform(0.8, 1.6)
                    )

                    card_id = card
                    device_id = device

                    status = (
                        "APPROVED"
                        if rng.random() < 0.94
                        else "DECLINED"
                    )

                # 5. SPLIT_PAYMENT
                # 목표 총액을 여러 결제로 분할
                elif name == "SPLIT_PAYMENT":
                    amount = split_amounts[j]

                    card_id = card
                    device_id = device

                    status = (
                        "APPROVED"
                        if rng.random() < 0.96
                        else "DECLINED"
                    )

                # 6. BEHAVIORAL_ANOMALY
                # 고객의 평소 금액과 시간대에서 벗어난 거래
                elif name == "BEHAVIORAL_ANOMALY":
                    mode = rng.choices(
                        ["AMOUNT", "TIME", "BOTH", "SUBTLE"],
                        weights=[30, 30, 20, 20],
                        k=1,
                    )[0]

                    # 금액 중심 또는 복합 이상
                    if mode in {"AMOUNT", "BOTH"}:
                        multiplier = (
                            rng.uniform(0.25, 0.65)
                            if rng.random() < 0.5
                            else rng.uniform(1.6, 2.7)
                        )
                    else:
                        multiplier = rng.uniform(0.75, 1.45)

                    amount = customer["normalAmount"] * multiplier
                    card_id = card

                    # 시간 중심 또는 복합 이상
                    if mode in {"TIME", "BOTH"}:
                        direction = rng.choice([-1, 1])
                        shift_hours = direction * rng.randint(5, 9)

                    elif mode == "AMOUNT":
                        shift_hours = rng.randint(-2, 2)

                    else:
                        direction = rng.choice([-1, 1])
                        shift_hours = direction * rng.randint(1, 5)

                    when += timedelta(hours=shift_hours)

                    # 정상 거래에서도 나타나는 추가 기기 사용 허용
                    device_id = (
                        device
                        if rng.random() < 0.5
                        else f"SIM-DEVICE-{customer['customerRef']}-EXTRA"
                    )

                    status = (
                        "APPROVED"
                        if rng.random() < 0.96
                        else "DECLINED"
                    )

                # 7. CARD_DEVICE_CHANGE
                # 평소와 다른 카드·기기 조합을 교대로 사용
                else:
                    amount = (
                        customer["normalAmount"]
                        * rng.uniform(0.6, 1.8)
                    )

                    card_id = (
                        card
                        if j % 2 == 0
                        else f"SIM-CARD-{customer['customerRef']}-EXTRA"
                    )

                    device_id = (
                        f"SIM-DEVICE-{customer['customerRef']}-EXTRA"
                        if j % 2 == 0
                        else device
                    )

                    status = (
                        "APPROVED"
                        if rng.random() < 0.94
                        else "DECLINED"
                    )

                # 사기 거래 생성
                fraud_tx = make_fraud_transaction(
                    rng=rng,
                    customer=customer,
                    when=when,
                    amount=amount,
                    device=device_id,
                    card=card_id,
                    status=status,
                    scenario=name,
                )

                # 동일 Episode의 거래들을 내부 사건 ID로 연결
                fraud_tx["_attackEventId"] = attack_event_id
                frauds.append(fraud_tx)

    # 전체 사기 거래를 시간순으로 정렬
    frauds.sort(
        key=lambda tx: (
            tx["paymentRequestedAt"],
            tx["merchantId"],
            tx["customerRef"],
            tx["amount"],
        )
    )

    return frauds