
import sys
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from train_isolation_forest import (
    create_pipeline,
    anomaly_scores,
    threshold_at_fixed_fpr,
)


def test_threshold_respects_fp_budget():
    y = np.array([0, 0, 0, 0, 1, 1])

    scores = np.array([
        0.10, 0.20, 0.30, 0.40,
        0.50, 0.60,
    ])

    threshold, allowed_fp = threshold_at_fixed_fpr(
        y, scores, target_fpr=0.25
    )

    predictions = scores >= threshold

    fp = int(((y == 0) & predictions).sum())

    assert allowed_fp == 1
    assert fp == 1
    assert predictions[4]
    assert predictions[5]


def test_tied_scores_do_not_exceed_fp_budget():
    y = np.array([0, 0, 0, 0, 1])

    scores = np.array([
        0.40, 0.40, 0.20, 0.10, 0.50,
    ])

    threshold, allowed_fp = threshold_at_fixed_fpr(
        y, scores, target_fpr=0.25
    )

    predictions = scores >= threshold
    fp = int(((y == 0) & predictions).sum())

    assert allowed_fp == 1
    assert fp <= allowed_fp


def test_anomaly_score_direction():
    normal = [
        {"amount": 100.0 + i, "merchantId": "1001"}
        for i in range(300)
    ]

    pipeline = create_pipeline()
    pipeline.fit(normal)

    sample = [
        {"amount": 150.0, "merchantId": "1001"},
        {"amount": 10000.0, "merchantId": "1001"},
    ]

    scores = anomaly_scores(pipeline, sample)

    assert len(scores) == 2
    assert np.isfinite(scores).all()

    # 현재 정의에서는 높은 점수가 더 이상한 거래를 의미한다.
    assert np.allclose(
        scores,
        -pipeline.score_samples(sample),
    )
