
"""FDS 합성 거래 데이터 및 CSV 출력 검증 테스트."""

import csv
import hashlib
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest


# 하이픈이 포함된 fds-ml 폴더를 import 경로에 추가
FDS_ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FDS_ML_DIR))

from generate_data import (
    SEED,
    CUSTOMER_COUNT,
    NORMAL_COUNT,
    FRAUD_COUNT,
    TOTAL_COUNT,
    build_dataset,
)
from scenarios import SCENARIOS
from transactions import KST, START_DATE, END_DATE
from csv_export import CSV_COLUMNS, save_transactions_csv


@pytest.fixture(scope="module")
def dataset():
    """동일 테스트 모듈에서 합성 데이터를 한 번만 생성한다."""
    return build_dataset(SEED)


def parse_utc(value):
    """UTC 문자열을 timezone-aware datetime으로 변환한다."""
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def csv_value(value):
    """CSV 저장 규칙에 따라 Python 값을 문자열로 변환한다."""
    return "" if value is None else str(value)


def test_dataset_counts(dataset):
    """고객 및 정상·사기 거래 수 검증."""
    customers, transactions = dataset

    labels = Counter(
        tx["fraudLabel"] for tx in transactions
    )

    assert len(customers) == CUSTOMER_COUNT == 1500
    assert len(transactions) == TOTAL_COUNT == 20000
    assert labels == {
        0: NORMAL_COUNT,
        1: FRAUD_COUNT,
    }


def test_fraud_scenarios(dataset):
    """7종 사기 시나리오 및 거래 수 검증."""
    _, transactions = dataset

    counts = Counter(
        tx["scenarioType"]
        for tx in transactions
        if tx["fraudLabel"] == 1
    )

    assert len(counts) == 7
    assert set(counts) == set(SCENARIOS)
    assert sum(counts.values()) == FRAUD_COUNT

    # 2000건을 7종에 거의 균등하게 배분한 초기 설계
    assert all(
        count in (285, 286)
        for count in counts.values()
    )


def test_unique_identifiers(dataset):
    """결제·주문·평가 요청 식별자 중복 검증."""
    _, transactions = dataset

    for field in (
        "paymentId",
        "orderRef",
        "evaluationRequestId",
    ):
        values = [tx[field] for tx in transactions]

        assert len(values) == len(set(values)), (
            f"{field}: 중복 식별자가 존재합니다."
        )

    assert [
        tx["paymentId"] for tx in transactions
    ] == list(range(1, TOTAL_COUNT + 1))


def test_transaction_timestamps(dataset):
    """거래 발생 순서 및 KST 기간 검증."""
    _, transactions = dataset

    payment_times = [
        parse_utc(tx["paymentRequestedAt"])
        for tx in transactions
    ]

    assert payment_times == sorted(payment_times)

    for tx, payment_time in zip(
        transactions, payment_times
    ):
        order_time = parse_utc(tx["orderCreatedAt"])

        assert order_time <= payment_time
        assert START_DATE <= payment_time.astimezone(KST) < END_DATE


def test_transaction_fields(dataset):
    """필수 컬럼, 금액, 승인 결과 및 가상 데이터 검증."""
    _, transactions = dataset

    expected_columns = set(CSV_COLUMNS)

    for tx in transactions:
        assert set(tx) == expected_columns
        assert tx["amount"] > 0

        assert tx["authorizationResult"] in {
            "APPROVED",
            "DECLINED",
        }

        assert tx["buyerEmail"].endswith(
            "@example.invalid"
        )

        assert "ipAddress" not in tx
        assert "ipCountryCode" not in tx


def test_customer_references(dataset):
    """거래 고객이 가상 고객 프로필에 존재하는지 검증."""
    customers, transactions = dataset

    valid_customers = {
        (
            str(customer["merchantId"]),
            str(customer["customerRef"]),
        )
        for customer in customers
    }

    for tx in transactions:
        key = (
            str(tx["merchantId"]),
            str(tx["customerRef"]),
        )

        assert key in valid_customers, (
            f"존재하지 않는 고객 참조: {key}"
        )


def test_csv_round_trip(dataset, tmp_path):
    """CSV 저장 후 다시 읽어 원본과 일치하는지 검증."""
    _, transactions = dataset

    csv_path = tmp_path / "transactions.csv"

    saved_path, saved_count = save_transactions_csv(
        transactions,
        csv_path,
    )

    assert saved_path == csv_path
    assert saved_count == TOTAL_COUNT
    assert csv_path.is_file()

    with csv_path.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file)

        assert reader.fieldnames == list(CSV_COLUMNS)

        rows = list(reader)

    assert len(rows) == TOTAL_COUNT

    for line_number, (original, saved) in enumerate(
        zip(transactions, rows), start=2
    ):
        expected = {
            column: csv_value(original[column])
            for column in CSV_COLUMNS
        }

        assert saved == expected, (
            f"CSV {line_number}행의 값이 원본과 다릅니다."
        )


def test_seed_and_csv_reproducibility(
    dataset, tmp_path
):
    """동일 Seed의 데이터와 CSV 바이트 재현성 검증."""
    customers, transactions = dataset

    replay_customers, replay_transactions = (
        build_dataset(SEED)
    )

    assert customers == replay_customers
    assert transactions == replay_transactions

    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"

    save_transactions_csv(
        transactions, first_path
    )

    save_transactions_csv(
        replay_transactions, second_path
    )

    first_hash = hashlib.sha256(
        first_path.read_bytes()
    ).hexdigest()

    second_hash = hashlib.sha256(
        second_path.read_bytes()
    ).hexdigest()

    assert first_hash == second_hash


def test_reject_unexpected_ip_columns(
    dataset, tmp_path
):
    """허용되지 않은 IP 필드가 CSV에 기록되지 않도록 검증."""
    _, transactions = dataset

    invalid_transaction = dict(transactions[0])
    invalid_transaction["ipAddress"] = "192.0.2.1"

    output_path = tmp_path / "invalid.csv"

    with pytest.raises(ValueError):
        save_transactions_csv(
            [invalid_transaction],
            output_path,
        )

    assert not output_path.exists()


def test_reject_missing_columns(
    dataset, tmp_path
):
    """필수 컬럼 누락을 감지하는지 검증."""
    _, transactions = dataset

    invalid_transaction = dict(transactions[0])
    del invalid_transaction["merchantId"]

    output_path = tmp_path / "missing.csv"

    with pytest.raises(ValueError):
        save_transactions_csv(
            [invalid_transaction],
            output_path,
        )

    assert not output_path.exists()


def test_fraud_with_short_customer_history(dataset):
    """과거 거래가 3건 미만인 고객에게도 사기가 발생하는지 검증."""
    _, transactions = dataset

    # 고객별 현재 거래 이전의 거래 수를 기록한다.
    history_counts = Counter()
    short_history_frauds = 0

    # 전체 거래는 결제 요청 시각순으로 정렬되어 있다.
    for tx in transactions:
        key = (tx["merchantId"], tx["customerRef"])
        prior_count = history_counts[key]

        # 과거 거래가 0~2건인 사기 거래를 확인한다.
        if tx["fraudLabel"] == 1 and prior_count < 3:
            short_history_frauds += 1

        history_counts[key] += 1

    assert short_history_frauds > 0, (
        "과거 거래가 3건 미만인 사기 거래가 존재하지 않습니다."
    )
