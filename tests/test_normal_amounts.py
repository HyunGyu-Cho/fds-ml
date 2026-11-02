
"""정상 거래 금액 및 연속 결제 생성 규칙 검증."""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

FDS_ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FDS_ML_DIR))

import transactions
from transactions import generate_payment_amount


def test_normal_small_payments():
    """정상 거래에서도 4,000원 이하 구매가 발생해야 한다."""

    rng = random.Random(42)
    customer = {"normalAmount": 60000}

    amounts = [
        generate_payment_amount(rng, customer)
        for _ in range(2000)
    ]

    assert any(
        1000 <= amount <= 4000
        for amount in amounts
    )

    assert any(amount > 9000 for amount in amounts)

    assert all(
        amount >= 1000 and amount % 100 == 0
        for amount in amounts
    )


def test_normal_repeat_payments(monkeypatch):
    """정상 연속 결제가 45초~30분 간격으로 생성되는지 검증."""

    # 테스트 중에만 연속 결제 선택 확률을 100%로 변경
    monkeypatch.setattr(
        transactions,
        "NORMAL_REPEAT_PAIR_RATE",
        1.0,
    )

    first_time = datetime(
        2026, 5, 5, 10, 0, tzinfo=transactions.KST
    )
    second_time = datetime(
        2026, 6, 5, 11, 0, tzinfo=transactions.KST
    )

    # 같은 고객의 정상 거래 두 건을 준비
    events = [
        {
            "merchantId": 1001,
            "customerRef": "TEST-CUSTOMER",
            "_paymentTime": first_time,
            "_orderTime": first_time - timedelta(minutes=5),
            "deviceId": "DEVICE-A",
            "cardToken": "CARD-A",
        },
        {
            "merchantId": 1001,
            "customerRef": "TEST-CUSTOMER",
            "_paymentTime": second_time,
            "_orderTime": second_time - timedelta(minutes=5),
            "deviceId": "DEVICE-B",
            "cardToken": "CARD-B",
        },
    ]

    # 연속 결제 시각 생성
    transactions.inject_normal_repeat_payments(
        rng=random.Random(42),
        events=events,
    )

    # 거래 건수는 그대로 유지
    assert len(events) == 2

    # 생성된 두 거래의 시간 차이를 계산
    times = sorted(event["_paymentTime"] for event in events)
    gap_seconds = (times[1] - times[0]).total_seconds()

    assert 45 <= gap_seconds <= 1800

    # 주문 생성 시각이 결제 요청 시각보다 늦으면 안 된다.
    assert all(
        event["_orderTime"] <= event["_paymentTime"]
        for event in events
    )

    # 거래가 데이터 생성 기간을 벗어나지 않아야 한다.
    assert all(
        transactions.START_DATE
        <= event["_paymentTime"]
        < transactions.END_DATE
        for event in events
    )