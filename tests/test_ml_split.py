
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from train_random_forest import split_and_purge


BASE_TIME = datetime(
    2026, 4, 1, tzinfo=timezone.utc
)


def make_dataset():
    transactions = []
    episodes = []

    fraud_indices = {1, 13, 14, 16, 18}

    for i in range(20):
        timestamp = BASE_TIME + timedelta(minutes=i)

        payment_id = f"p{i}"
        fraud_label = int(i in fraud_indices)

        transactions.append({
            "paymentId": payment_id,
            "paymentRequestedAt": timestamp.isoformat(),
            "fraudLabel": str(fraud_label),
        })

        if fraud_label:
            episode_id = (
                "CROSSING"
                if i in {13, 14}
                else f"E{i}"
            )

            episodes.append({
                "paymentId": payment_id,
                "attackEventId": episode_id,
            })

    return transactions, episodes


def test_time_split_and_episode_purging():
    transactions, episodes = make_dataset()

    result = split_and_purge(transactions, episodes)

    assert result["crossing_episodes"] == {"CROSSING"}

    assert result["purged_payments"] == {"p13", "p14"}

    indices = result["indices"]

    assert len(indices["Train"]) == 13
    assert len(indices["Validation"]) == 2
    assert len(indices["Test"]) == 3

    assert result["train_cut"] == BASE_TIME + timedelta(
        minutes=14
    )
    assert result["test_cut"] == BASE_TIME + timedelta(
        minutes=17
    )


def test_remaining_episodes_do_not_overlap():
    transactions, episodes = make_dataset()

    result = split_and_purge(transactions, episodes)

    payment_to_episode = {
        row["paymentId"]: row["attackEventId"]
        for row in episodes
    }

    episode_to_split = {}

    for split_name, indices in result["indices"].items():
        for i in indices:
            payment_id = transactions[i]["paymentId"]

            if payment_id not in payment_to_episode:
                continue

            episode_id = payment_to_episode[payment_id]

            assert episode_id not in episode_to_split or (
                episode_to_split[episode_id] == split_name
            )

            episode_to_split[episode_id] = split_name


def test_missing_episode_is_rejected():
    transactions, episodes = make_dataset()

    incomplete_episodes = episodes[:-1]

    with pytest.raises(ValueError, match="Episode 연결"):
        split_and_purge(
            transactions, incomplete_episodes
        )
