
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from sequences import (
    SEQUENCE_FEATURES,
    build_sequences,
)


BASE_TIME = datetime(
    2026, 4, 1, tzinfo=timezone.utc
)


def make_tx(
    minutes,
    customer="A",
    amount=10000,
    merchant="1001",
    device="device-1",
    card="card-1",
):
    return {
        "paymentRequestedAt": (
            BASE_TIME + timedelta(minutes=minutes)
        ).isoformat(),
        "customerRef": customer,
        "amount": str(amount),
        "merchantId": merchant,
        "deviceId": device,
        "cardToken": card,
        "fraudLabel": "0",
        "scenarioType": "NORMAL",
        "authorizationResult": "APPROVED",
    }


def test_first_transaction_is_all_padding():
    windows = build_sequences([make_tx(0)])

    assert len(windows) == 1
    assert windows[0].mask == (0, 0, 0, 0, 0)
    assert len(windows[0].tokens) == 5

    for token in windows[0].tokens:
        assert tuple(token) == SEQUENCE_FEATURES
        assert token["merchantId"] == "__PAD__"


def test_sequence_contains_only_past_transactions():
    rows = [
        make_tx(0, amount=10000),
        make_tx(3, amount=20000),
        make_tx(8, amount=30000),
    ]

    windows = build_sequences(rows)

    assert windows[0].mask == (0, 0, 0, 0, 0)
    assert windows[1].mask == (0, 0, 0, 0, 1)
    assert windows[2].mask == (0, 0, 0, 1, 1)

    tokens = windows[2].tokens

    assert tokens[3]["logAmount"] == pytest.approx(
        math.log1p(10000)
    )
    assert tokens[4]["logAmount"] == pytest.approx(
        math.log1p(20000)
    )

    assert tokens[3]["logMinutesBeforeCurrent"] == (
        pytest.approx(math.log1p(8))
    )

    assert tokens[4]["logMinutesBeforeCurrent"] == (
        pytest.approx(math.log1p(5))
    )


def test_sequence_keeps_only_recent_five():
    rows = [
        make_tx(i, amount=10000 + i)
        for i in range(7)
    ]

    windows = build_sequences(rows)

    last = windows[-1]

    assert last.mask == (1, 1, 1, 1, 1)

    amounts = [
        math.expm1(token["logAmount"])
        for token in last.tokens
    ]

    assert amounts == pytest.approx([
        10001, 10002, 10003, 10004, 10005
    ])


def test_same_timestamp_transactions_are_not_history():
    rows = [
        make_tx(0, amount=10000),
        make_tx(0, amount=20000),
        make_tx(1, amount=30000),
    ]

    windows = build_sequences(rows)

    assert windows[0].mask == (0, 0, 0, 0, 0)
    assert windows[1].mask == (0, 0, 0, 0, 0)
    assert windows[2].mask == (0, 0, 0, 1, 1)


def test_different_customers_have_separate_history():
    rows = [
        make_tx(0, customer="A"),
        make_tx(1, customer="B"),
        make_tx(2, customer="A"),
    ]

    windows = build_sequences(rows)

    assert sum(windows[0].mask) == 0
    assert sum(windows[1].mask) == 0
    assert sum(windows[2].mask) == 1


def test_future_transactions_do_not_change_past():
    original = [
        make_tx(0),
        make_tx(3),
    ]

    before = build_sequences(original)

    extended = original + [
        make_tx(100),
        make_tx(200),
    ]

    after = build_sequences(extended)

    assert before == after[:2]


def test_labels_do_not_affect_sequences():
    rows = [
        make_tx(0),
        make_tx(3),
        make_tx(8),
    ]

    before = build_sequences(rows)

    changed = [row.copy() for row in rows]

    for row in changed:
        row["fraudLabel"] = "1"
        row["scenarioType"] = "STOLEN_CARD"
        row["authorizationResult"] = "DECLINED"

    after = build_sequences(changed)

    assert before == after


def test_unsorted_input_preserves_original_order():
    rows = [
        make_tx(8),
        make_tx(0),
        make_tx(3),
    ]

    windows = build_sequences(rows)

    assert sum(windows[0].mask) == 2
    assert sum(windows[1].mask) == 0
    assert sum(windows[2].mask) == 1


def test_new_device_and_card_match_flags():
    rows = [
        make_tx(0, device="D1", card="C1"),
        make_tx(2, device="D2", card="C1"),
        make_tx(4, device="D2", card="C2"),
    ]

    windows = build_sequences(rows)

    # 세 번째 거래를 기준으로:
    # 첫 거래의 기기/카드는 모두 다르다.
    # 두 번째 거래의 기기만 동일하다.
    first_past = windows[2].tokens[-2]
    second_past = windows[2].tokens[-1]

    assert first_past["sameDeviceAsCurrent"] == 0
    assert first_past["sameCardAsCurrent"] == 0

    assert second_past["sameDeviceAsCurrent"] == 1
    assert second_past["sameCardAsCurrent"] == 0
