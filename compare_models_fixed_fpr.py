
from pathlib import Path
import math

import joblib
import numpy as np
import torch

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from features import FEATURE_COLUMNS, build_features
from sequences import build_sequences
from train_random_forest import (
    read_csv,
    split_and_purge,
    make_xy,
)
from train_transformer import FraudNeuralNetwork


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

TARGET_FPR = 50 / 2555


def check_boundaries(bundle, split_info):
    if bundle["train_cut"] != split_info["train_cut"].isoformat():
        raise ValueError("Train 분할 경계 불일치")

    if bundle["test_cut"] != split_info["test_cut"].isoformat():
        raise ValueError("Test 분할 경계 불일치")


def tree_scores(
    filename,
    features,
    transactions,
    indices,
    split_info,
):
    bundle = joblib.load(MODEL_DIR / filename)
    check_boundaries(bundle, split_info)

    x, _ = make_xy(
        features,
        transactions,
        indices,
        bundle["feature_columns"],
    )

    return bundle["pipeline"].predict_proba(x)[:, 1]


def transformer_scores(
    features,
    transactions,
    indices,
    split_info,
):
    checkpoint = torch.load(
        MODEL_DIR / "transformer_candidate.pt",
        map_location="cpu",
        weights_only=True,
    )

    check_boundaries(checkpoint, split_info)

    preprocessing = joblib.load(
        MODEL_DIR / "neural_preprocessing.joblib"
    )

    length = checkpoint["sequence_length"]

    windows = build_sequences(
        transactions,
        sequence_length=length,
    )

    tab_rows = [
        {
            name: features[i][name]
            for name in FEATURE_COLUMNS
        }
        for i in indices
    ]

    tab_raw = preprocessing[
        "tab_vectorizer"
    ].transform(tab_rows)

    tab = preprocessing[
        "tab_scaler"
    ].transform(tab_raw).astype(np.float32)

    selected_windows = [
        windows[i] for i in indices
    ]

    all_tokens = [
        token
        for window in selected_windows
        for token in window.tokens
    ]

    seq_raw = preprocessing[
        "seq_vectorizer"
    ].transform(all_tokens)

    seq_raw = seq_raw.reshape(
        len(indices), length, -1
    )

    seq = preprocessing[
        "seq_scaler"
    ].transform(
        seq_raw.reshape(-1, seq_raw.shape[-1])
    )

    seq = seq.reshape(
        len(indices), length, -1
    ).astype(np.float32)

    mask = np.asarray(
        [
            window.mask
            for window in selected_windows
        ],
        dtype=bool,
    )

    seq[~mask] = 0.0

    model = FraudNeuralNetwork(
        tab_dim=checkpoint["tab_dim"],
        seq_dim=checkpoint["seq_dim"],
        use_sequence=True,
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )
    model.eval()

    probabilities = []

    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            end = start + 128

            logits = model(
                torch.from_numpy(tab[start:end]),
                torch.from_numpy(seq[start:end]),
                torch.from_numpy(mask[start:end]),
            )

            probabilities.extend(
                torch.sigmoid(logits).numpy().tolist()
            )

    return np.asarray(
        probabilities,
        dtype=np.float64,
    )


def threshold_with_fp_budget(
    y_true,
    scores,
    target_fpr,
):
    """
    Validation 정상 거래를 이용해 FPR 상한을 적용한다.

    점수는 float64로 통일한다.
    경계 점수보다 엄격히 높은 경우에만 양성 판정한다.

    같은 점수가 여러 건이어도 FP 예산을 초과하지 않는다.
    """

    y_true = np.asarray(y_true)
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr 범위가 잘못되었습니다.")

    if not np.isfinite(scores).all():
        raise ValueError("예측 점수에 NaN 또는 Inf가 있습니다.")

    normal_scores = np.sort(
        scores[y_true == 0]
    )[::-1]

    normal_count = len(normal_scores)

    if normal_count == 0:
        raise ValueError("정상 거래가 없습니다.")

    allowed_fp = math.floor(
        target_fpr * normal_count + 1e-9
    )

    # 허용 개수 바로 다음 정상 거래의 점수를 기준으로 한다.
    threshold = float(
        normal_scores[allowed_fp]
    )

    # 중요: >=가 아니라 >를 사용한다.
    predictions = scores > threshold

    actual_fp = int(
        ((y_true == 0) & predictions).sum()
    )

    if actual_fp > allowed_fp:
        raise RuntimeError(
            f"FPR 예산 초과: "
            f"허용={allowed_fp}, 실제={actual_fp}"
        )

    return threshold, predictions, allowed_fp


def evaluate(y, scores):
    threshold, predictions, allowed_fp = (
        threshold_with_fp_budget(
            y,
            scores,
            TARGET_FPR,
        )
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": threshold,
        "predictions": predictions,
        "allowed_fp": allowed_fp,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "ap": average_precision_score(
            y, scores
        ),
        "fpr": fp / (fp + tn),
    }


def main():
    transactions = read_csv(
        BASE_DIR / "data" / "transactions.csv"
    )

    episodes = read_csv(
        BASE_DIR / "data" / "fraud_episodes.csv"
    )

    features = build_features(transactions)

    split_info = split_and_purge(
        transactions,
        episodes,
    )

    indices = split_info[
        "indices"
    ]["Validation"]

    y = np.asarray([
        int(transactions[i]["fraudLabel"])
        for i in indices
    ])

    print("=== 동일 FPR 모델 비교 ===")
    print(f"Validation: {len(indices):,}건")
    print(f"Target FPR: {TARGET_FPR:.4%}")
    print(f"정상 거래: {int((y == 0).sum()):,}건")
    print(f"사기 거래: {int((y == 1).sum()):,}건")

    score_map = {
        "RF": tree_scores(
            "random_forest_baseline.joblib",
            features,
            transactions,
            indices,
            split_info,
        ),
        "XGBoost": tree_scores(
            "xgboost_candidate.joblib",
            features,
            transactions,
            indices,
            split_info,
        ),
        "Transformer": transformer_scores(
            features,
            transactions,
            indices,
            split_info,
        ),
    }

    results = {}

    print("\n[동일 FPR 기준 Validation 성능]")

    for name, scores in score_map.items():
        result = evaluate(y, scores)
        results[name] = result

        print(
            f"{name:<13} "
            f"TP={result['tp']:>3} "
            f"FP={result['fp']:>3} "
            f"Recall={result['recall']:.4f} "
            f"F1={result['f1']:.4f} "
            f"AP={result['ap']:.4f} "
            f"FPR={result['fpr']:.4f} "
            f"Threshold={result['threshold']:.6f}"
        )

        print(
            f"  허용 FP={result['allowed_fp']}, "
            f"실제 FP={result['fp']}"
        )

    rf = results["RF"]["predictions"]
    xgb = results["XGBoost"]["predictions"]
    transformer = results["Transformer"]["predictions"]

    common_misses = (
        (y == 1) & ~rf & ~xgb
    )

    recovered = (
        common_misses & transformer
    )

    new_fp = (
        (y == 0)
        & ~rf
        & ~xgb
        & transformer
    )

    print("\n[Transformer 추가 탐지 효과]")
    print(
        "RF·XGB 공통 미탐:",
        int(common_misses.sum()),
    )
    print(
        "Transformer 복구:",
        int(recovered.sum()),
    )
    print(
        "추가 정상 오탐:",
        int(new_fp.sum()),
    )

    union = rf | xgb | transformer

    tn, fp, fn, tp = confusion_matrix(
        y, union, labels=[0, 1]
    ).ravel()

    print("\n[세 모델 OR 결합]")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(
        f"Precision="
        f"{precision_score(y, union):.4f}"
    )
    print(
        f"Recall="
        f"{recall_score(y, union):.4f}"
    )
    print(
        f"F1="
        f"{f1_score(y, union):.4f}"
    )
    print(f"FPR={fp / (fp + tn):.4f}")

    print(
        "\n주의: 개별 모델의 FPR을 제한해도 "
        "OR 결합 FPR은 목표값을 초과할 수 있습니다."
    )

    print("\nTest 데이터는 사용하지 않았습니다.")


if __name__ == "__main__":
    main()
