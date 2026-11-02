
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

# 실험 설치(requirements-experiments.txt)가 없으면 건너뛴다
torch = pytest.importorskip("torch")
pytest.importorskip("xgboost")

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from train_hybrid_xgboost import (
    split_inner_train,
    extract_sequence_embeddings,
    add_embedding_features,
    fixed_fpr_predictions,
)
from train_transformer import FraudNeuralNetwork


BASE_TIME = datetime(
    2026, 4, 1, tzinfo=timezone.utc
)


def make_inner_dataset():
    transactions = []
    episode_by_payment = {}

    fraud_indices = {
        2, 10, 15, 16, 18, 20, 21, 25
    }

    for index in range(30):
        transactions.append({
            "paymentId": f"p{index}",
            "paymentRequestedAt": (
                BASE_TIME + timedelta(minutes=index)
            ).isoformat(),
            "fraudLabel": str(
                int(index in fraud_indices)
            ),
        })

    for index in fraud_indices:
        if index in (15, 16):
            episode_id = "CROSS_1"
        elif index in (20, 21):
            episode_id = "CROSS_2"
        else:
            episode_id = f"E{index}"

        episode_by_payment[
            f"p{index}"
        ] = episode_id

    return transactions, episode_by_payment


def test_inner_split_purges_crossing_episodes():
    transactions, episode_map = make_inner_dataset()

    result = split_inner_train(
        transactions,
        list(range(len(transactions))),
        episode_map,
    )

    assert result["crossing_episodes"] == {
        "CROSS_1", "CROSS_2"
    }

    assert result["purged_payments"] == {
        "p15", "p16", "p20", "p21"
    }

    groups = result["groups"]

    all_indices = [
        i
        for indices in groups.values()
        for i in indices
    ]

    assert len(all_indices) == 26
    assert len(set(all_indices)) == 26

    remaining_ids = {
        transactions[i]["paymentId"]
        for i in all_indices
    }

    assert "p15" not in remaining_ids
    assert "p21" not in remaining_ids


def test_inner_split_preserves_time_order():
    transactions, episode_map = make_inner_dataset()

    groups = split_inner_train(
        transactions,
        list(range(30)),
        episode_map,
    )["groups"]

    encoder_train = groups["EncoderTrain"]
    early_stop = groups["EncoderEarlyStop"]
    xgb_train = groups["XGBTrain"]

    assert max(encoder_train) < min(early_stop)
    assert max(early_stop) < min(xgb_train)


def test_embedding_shape_and_padding_safety():
    torch.manual_seed(42)

    model = FraudNeuralNetwork(
        tab_dim=15,
        seq_dim=9,
        use_sequence=True,
    )

    sequences = np.zeros(
        (3, 5, 9),
        dtype=np.float32,
    )

    mask = np.zeros(
        (3, 5),
        dtype=bool,
    )

    embeddings = extract_sequence_embeddings(
        model,
        sequences,
        mask,
        [0, 1, 2],
    )

    assert embeddings.shape == (3, 32)
    assert np.isfinite(embeddings).all()


def test_hybrid_feature_columns():
    rows = [
        {"amount": 10000.0, "merchantId": "1001"},
        {"amount": 20000.0, "merchantId": "1002"},
    ]

    embeddings = np.ones(
        (2, 32),
        dtype=np.float32,
    )

    result = add_embedding_features(
        rows, embeddings
    )

    assert len(result) == 2
    assert len(result[0]) == 34
    assert result[0]["seq_emb_00"] == 1.0
    assert result[0]["seq_emb_31"] == 1.0
    assert "seq_emb_00" not in rows[0]


def test_fixed_fpr_with_tied_scores():
    y = np.asarray([0, 0, 0, 0, 1])

    scores = np.asarray([
        0.4, 0.4, 0.2, 0.1, 0.5
    ])

    threshold, predictions = fixed_fpr_predictions(
        y, scores, target_fpr=0.25
    )

    fp = int(
        ((y == 0) & predictions).sum()
    )

    assert fp <= 1
    assert predictions[-1]