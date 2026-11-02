"""FDS 사기 공격 사건 관계와 CSV 검증 테스트."""

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pytest

FDS_ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FDS_ML_DIR))

from csv_export import CSV_COLUMNS
from episode_export import EPISODE_COLUMNS, save_episode_links_csv
from generate_data import (
    FRAUD_COUNT,
    SEED,
    build_dataset,
    build_dataset_with_episodes,
    validate_episode_links,
)
from scenarios import SCENARIOS


@pytest.fixture(scope="module")
def dataset_with_episodes():
    return build_dataset_with_episodes(SEED)


def test_old_dataset_contract_is_unchanged(dataset_with_episodes):
    """기존 거래 15컬럼과 반환 구조를 보존한다."""
    customers, transactions, episode_links = dataset_with_episodes
    old_customers, old_transactions = build_dataset(SEED)
    assert (customers, transactions) == (old_customers, old_transactions)
    assert len(episode_links) == FRAUD_COUNT
    assert all(set(tx) == set(CSV_COLUMNS) for tx in transactions)
    assert all("_attackEventId" not in tx for tx in transactions)


def test_each_fraud_payment_has_exactly_one_episode(dataset_with_episodes):
    """사기 거래만 누락·중복 없이 사건에 연결한다."""
    _, transactions, episode_links = dataset_with_episodes
    episode_count = validate_episode_links(transactions, episode_links)
    assert episode_count > 0

    fraud_ids = {tx["paymentId"] for tx in transactions if tx["fraudLabel"] == 1}
    ids = [link["paymentId"] for link in episode_links]
    assert len(ids) == len(set(ids)) == FRAUD_COUNT
    assert set(ids) == fraud_ids


def test_episode_minimum_sizes_and_sequence_gaps(dataset_with_episodes):
    """각 사건은 동일 고객·가맹점이며 시나리오의 간격 규칙을 지킨다."""
    _, transactions, episode_links = dataset_with_episodes
    payment_by_id = {tx["paymentId"]: tx for tx in transactions}
    groups = defaultdict(list)
    for link in episode_links:
        groups[link["attackEventId"]].append(payment_by_id[link["paymentId"]])

    scenario_group_counts = Counter()
    for attack_id, group in groups.items():
        group.sort(key=lambda tx: tx["paymentRequestedAt"])
        scenario = group[0]["scenarioType"]
        scenario_group_counts[scenario] += 1
        assert len(group) >= SCENARIOS[scenario], attack_id
        assert len({(tx["merchantId"], tx["customerRef"]) for tx in group}) == 1
        assert len({tx["scenarioType"] for tx in group}) == 1

        for prev, current in zip(group, group[1:]):
            t1 = datetime.fromisoformat(prev["paymentRequestedAt"].replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(current["paymentRequestedAt"].replace("Z", "+00:00"))
            seconds = (t2 - t1).total_seconds()
            if scenario == "CARD_TESTING":
                assert 25 <= seconds <= 80
            elif scenario == "LOW_AND_SLOW":
                assert 20 * 60 <= seconds <= 50 * 60
            elif scenario == "SPLIT_PAYMENT":
                assert 12 * 60 <= seconds <= 25 * 60
            else:
                assert 4 * 60 <= seconds <= 20 * 60

        if scenario == "SPLIT_PAYMENT":
            assert sum(tx["amount"] for tx in group) > 500_000

    assert set(scenario_group_counts) == set(SCENARIOS)


def test_episode_csv_round_trip(dataset_with_episodes, tmp_path):
    """저장한 CSV에서 사건 연결 정보를 손실 없이 복원한다."""
    _, _, episode_links = dataset_with_episodes
    output_path = tmp_path / "fraud_episodes.csv"
    saved_path, count = save_episode_links_csv(episode_links, output_path)
    assert saved_path == output_path
    assert count == FRAUD_COUNT

    with output_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == list(EPISODE_COLUMNS)
        rows = list(reader)

    assert len(rows) == FRAUD_COUNT
    assert rows == [
        {name: str(link[name]) for name in EPISODE_COLUMNS}
        for link in episode_links
    ]


def test_episode_csv_reproducible(dataset_with_episodes, tmp_path):
    """동일 Seed로 동일 사건 CSV 바이트를 재생성한다."""
    _, _, episode_links = dataset_with_episodes
    _, _, replay_links = build_dataset_with_episodes(SEED)
    assert episode_links == replay_links
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    save_episode_links_csv(episode_links, first_path)
    save_episode_links_csv(replay_links, second_path)
    assert hashlib.sha256(first_path.read_bytes()).digest() == (
        hashlib.sha256(second_path.read_bytes()).digest()
    )


def test_episode_csv_rejects_unexpected_fields(dataset_with_episodes, tmp_path):
    """정답 Label 등 추가 컬럼이 혼입되면 저장을 거부한다."""
    _, _, episode_links = dataset_with_episodes
    output_path = tmp_path / "invalid.csv"
    invalid = dict(episode_links[0])
    invalid["fraudLabel"] = 1
    with pytest.raises(ValueError):
        save_episode_links_csv([invalid], output_path)
    assert not output_path.exists()