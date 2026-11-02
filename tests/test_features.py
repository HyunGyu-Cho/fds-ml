
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# fds-ml 디렉터리의 features.py를 import한다.
ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from features import FEATURE_COLUMNS, build_features, load_feature_rows


BASE_TIME = datetime(
    2026, 4, 1, 0, 0, tzinfo=timezone.utc
)


def make_tx(
    minutes: int,
    customer: str = "customer-a",
    amount: int = 10000,
    card: str = "card-a",
    device: str = "device-a",
) -> dict:
    """테스트용 거래 한 건을 생성한다."""

    timestamp = BASE_TIME + timedelta(minutes=minutes)

    return {
        "paymentRequestedAt": timestamp.isoformat(),
        "customerRef": customer,
        "merchantId": "1001",
        "amount": str(amount),
        "cardToken": card,
        "deviceId": device,
        "fraudLabel": "0",
        "scenarioType": "NORMAL",
        "authorizationResult": "APPROVED",
    }


def test_first_transaction_has_no_history():
    rows = [make_tx(0)]
    features = build_features(rows)

    assert len(features) == 1
    assert tuple(features[0]) == FEATURE_COLUMNS

    first = features[0]

    assert first["customerPastTxCount"] == 0
    assert first["customerAvgAmount"] == 0
    assert first["amountRatio"] == 0
    assert first["txCount5m"] == 0
    assert first["txCount30m"] == 0
    assert first["txCount24h"] == 0
    assert first["minutesSinceLastTx"] == -1
    assert first["cardSeenBefore"] == 0
    assert first["deviceSeenBefore"] == 0


def test_past_count_average_and_ratio():
    rows = [
        make_tx(0, amount=10000),
        make_tx(3, amount=30000),
        make_tx(8, amount=20000),
    ]

    features = build_features(rows)

    second = features[1]
    third = features[2]

    assert second["customerPastTxCount"] == 1
    assert second["customerAvgAmount"] == 10000
    assert second["amountRatio"] == 3
    assert second["minutesSinceLastTx"] == 3

    assert third["customerPastTxCount"] == 2
    assert third["customerAvgAmount"] == 20000
    assert third["amountRatio"] == 1
    assert third["minutesSinceLastTx"] == 5


def test_rolling_transaction_windows():
    rows = [
        make_tx(0),
        make_tx(2),
        make_tx(9),
        make_tx(40),
        make_tx(1480),
    ]

    features = build_features(rows)

    # 9분 시점에는 5분 내 거래가 없지만
    # 30분 및 24시간 이력에는 두 건이 존재한다.
    assert features[2]["txCount5m"] == 0
    assert features[2]["txCount30m"] == 2
    assert features[2]["txCount24h"] == 2

    # 40분 시점에는 30분 내 거래가 없다.
    assert features[3]["txCount5m"] == 0
    assert features[3]["txCount30m"] == 0
    assert features[3]["txCount24h"] == 3

    # 정확히 24시간 전의 거래는 포함한다.
    assert features[4]["txCount24h"] == 1


def test_card_and_device_history():
    rows = [
        make_tx(
            0, customer="A",
            card="card-1", device="device-1",
        ),
        make_tx(
            1, customer="A",
            card="card-1", device="device-1",
        ),
        make_tx(
            2, customer="B",
            card="card-1", device="device-2",
        ),
        make_tx(
            3, customer="A",
            card="card-2", device="device-2",
        ),
    ]

    features = build_features(rows)

    assert features[1]["cardSeenBefore"] == 1
    assert features[1]["deviceSeenBefore"] == 1

    assert features[2]["cardSeenBefore"] == 0
    assert features[2]["cardUsedByOtherCustomer"] == 1
    assert features[2]["deviceSeenBefore"] == 0

    # device-2는 다른 고객이 사용했지만
    # A 본인의 과거 기기는 아니다.
    assert features[3]["deviceSeenBefore"] == 0
    assert features[3]["cardSeenBefore"] == 0


def test_same_timestamp_does_not_leak_history():
    rows = [
        make_tx(
            0, customer="A",
            card="shared", device="device-a",
        ),
        make_tx(
            0, customer="B",
            card="shared", device="device-b",
        ),
        make_tx(
            1, customer="C",
            card="shared", device="device-c",
        ),
    ]

    features = build_features(rows)

    # 동일 시각 거래끼리는 서로 과거 이력이 아니다.
    assert features[0]["cardUsedByOtherCustomer"] == 0
    assert features[1]["cardUsedByOtherCustomer"] == 0

    # 1분 뒤의 거래는 이전 두 고객의 사용 이력을 참조한다.
    assert features[2]["cardUsedByOtherCustomer"] == 1


def test_unsorted_input_returns_original_order():
    rows = [
        make_tx(8),
        make_tx(0),
        make_tx(3),
    ]

    features = build_features(rows)

    assert features[0]["customerPastTxCount"] == 2
    assert features[1]["customerPastTxCount"] == 0
    assert features[2]["customerPastTxCount"] == 1


def test_future_transactions_do_not_change_past_features():
    original = [
        make_tx(0),
        make_tx(3),
    ]

    before = build_features(original)

    extended = original + [
        make_tx(100),
        make_tx(200),
    ]

    after = build_features(extended)

    assert before == after[:2]


def test_labels_and_authorization_do_not_affect_features():
    rows = [
        make_tx(0),
        make_tx(3),
        make_tx(8),
    ]

    before = build_features(rows)

    changed = [row.copy() for row in rows]

    for row in changed:
        row["fraudLabel"] = "1"
        row["scenarioType"] = "STOLEN_CARD"
        row["authorizationResult"] = "DECLINED"

    after = build_features(changed)

    assert before == after


def test_kst_hour_conversion():
    row = make_tx(0)

    # UTC 00:00은 KST 09:00이다.
    features = build_features([row])

    assert features[0]["hourOfDay"] == 9


def test_naive_timestamp_is_rejected():
    row = make_tx(0)
    row["paymentRequestedAt"] = "2026-04-01T00:00:00"

    with pytest.raises(ValueError, match="시간대"):
        build_features([row])


def test_load_feature_rows_from_csv(tmp_path):
    csv_path = tmp_path / "transactions.csv"

    rows = [
        make_tx(0),
        make_tx(3),
    ]

    with open(
        csv_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    features = load_feature_rows(csv_path)

    assert len(features) == 2
    assert features[0]["customerPastTxCount"] == 0
    assert features[1]["customerPastTxCount"] == 1
