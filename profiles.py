import random


# 합성 데이터에서 사용할 가상 가맹점
# 실제 DonWorry DB의 merchantId와는 무관하다.
MERCHANTS = [
    {"merchantId": 1001, "name": "SIM_CAKESHOP", "priceScale": 1.0},
    {"merchantId": 1002, "name": "SIM_SHOP_B", "priceScale": 0.7},
    {"merchantId": 1003, "name": "SIM_SHOP_C", "priceScale": 1.4},
]


# 고객의 소비 유형별 설정
# 비율과 금액 범위는 실제 금융 통계가 아닌 실험용 가정이다.
SEGMENTS = [
    {
        "name": "light",
        "weight": 0.55,
        "amountRange": (15000, 40000),
        "monthlyFrequency": (0.5, 2.0),
    },
    {
        "name": "regular",
        "weight": 0.35,
        "amountRange": (30000, 100000),
        "monthlyFrequency": (2.0, 5.0),
    },
    {
        "name": "heavy",
        "weight": 0.10,
        "amountRange": (70000, 200000),
        "monthlyFrequency": (5.0, 10.0),
    },
]


def generate_customer_profiles(rng: random.Random, count: int = 1500):
    """고객별 정상 소비 성향을 생성한다."""

    if count <= 0:
        raise ValueError("고객 수는 1 이상이어야 합니다.")

    customers = []

    for i in range(count):
        # P5의 customerRef 검증 규칙에 맞는 숫자 문자열
        customer_ref = str(100000000001 + i)

        # 고객의 소비 유형을 가중 확률로 선택
        segment = rng.choices(
            SEGMENTS,
            weights=[s["weight"] for s in SEGMENTS],
            k=1,
        )[0]

        # 가상의 가맹점을 선택
        merchant = rng.choice(MERCHANTS)

        # 소비 유형과 가맹점 특성을 반영한 기준 금액
        base_amount = rng.uniform(*segment["amountRange"])
        normal_amount = round(base_amount * merchant["priceScale"])

        # 고객별 평균 구매 빈도와 선호 시간
        monthly_frequency = rng.uniform(*segment["monthlyFrequency"])
        preferred_hour = rng.randint(9, 22)

        # 대부분의 고객은 소수의 카드·기기를 사용
        card_count = rng.choices(
            [1, 2, 3],
            weights=[0.75, 0.22, 0.03],
            k=1,
        )[0]

        device_count = rng.choices(
            [1, 2],
            weights=[0.80, 0.20],
            k=1,
        )[0]

        # 실제 카드·기기와 무관한 가상 식별자 생성
        card_tokens = [
            f"SIM-CARD-{customer_ref}-{j}"
            for j in range(1, card_count + 1)
        ]

        device_ids = [
            f"SIM-DEVICE-{customer_ref}-{j}"
            for j in range(1, device_count + 1)
        ]

        # 고객 프로필 구성
        customer = {
            "customerRef": customer_ref,
            "merchantId": merchant["merchantId"],
            "segment": segment["name"],
            "normalAmount": normal_amount,
            "monthlyFrequency": round(monthly_frequency, 2),
            "preferredHour": preferred_hour,
            "cardTokens": card_tokens,
            "deviceIds": device_ids,
            "buyerEmail": f"user{customer_ref}@example.invalid",
        }

        customers.append(customer)

    return customers
