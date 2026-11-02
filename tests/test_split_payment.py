"""목표 총액 기반 SPLIT_PAYMENT 합성 거래 검증."""

import random
import sys
from collections import defaultdict
from pathlib import Path

import pytest

FDS_ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FDS_ML_DIR))

from generate_data import SEED, build_dataset_with_episodes
from scenarios import split_target_amount


def test_split_amount_preserves_total():
    """다양한 금액·건수·Seed에서도 합계와 건별 범위를 보존한다."""
    for payment_count in (3, 4):
        for target in (600_000, 675_000, 750_000, 900_000):
            for seed in range(10):
                amounts = split_target_amount(
                    rng=random.Random(seed),
                    target_amount=target,
                    payment_count=payment_count,
                )
                assert len(amounts) == payment_count
                assert sum(amounts) == target
                assert all(100_000 <= x <= 450_000 for x in amounts)
                assert all(x % 100 == 0 for x in amounts)


def test_split_amount_rejects_invalid_target():
    """분할 불가능하거나 100원 단위가 아닌 입력을 거부한다."""
    rng = random.Random(42)
    with pytest.raises(ValueError):
        split_target_amount(rng, 600_000, 1)
    with pytest.raises(ValueError):
        split_target_amount(rng, 1_500_000, 3)
    with pytest.raises(ValueError):
        split_target_amount(rng, 600_050, 3)
    with pytest.raises(ValueError):
        split_target_amount(rng, 600_000, 3, max_amount=500_000)


def test_generated_split_episodes_obey_target_range():
    """실제 생성된 사건 95개의 총액·개별 금액을 검증한다."""
    _, transactions, episode_links = build_dataset_with_episodes(SEED)
    by_payment = {tx["paymentId"]: tx for tx in transactions}
    groups = defaultdict(list)

    for link in episode_links:
        if link["scenarioType"] == "SPLIT_PAYMENT":
            groups[link["attackEventId"]].append(
                by_payment[link["paymentId"]]
            )

    assert len(groups) == 95
    assert sum(len(group) for group in groups.values()) == 286

    for attack_id, group in groups.items():
        amounts = [tx["amount"] for tx in group]
        assert len(group) in (3, 4), attack_id
        assert 600_000 <= sum(amounts) <= 900_000, attack_id
        assert all(100_000 <= x <= 450_000 for x in amounts), attack_id
        assert all(x % 100 == 0 for x in amounts), attack_id
