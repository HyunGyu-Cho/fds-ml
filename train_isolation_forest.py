
from collections import Counter, defaultdict
from pathlib import Path
import math

import joblib
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from features import FEATURE_COLUMNS, build_features
from train_random_forest import (
    read_csv,
    split_and_purge,
    make_xy,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

RANDOM_STATE = 42

# RF Validation 결과: 정상 거래 2,555건 중 FP 50건
TARGET_FPR = 50 / 2555


def create_pipeline():
    """정상 거래의 분포를 학습하는 Isolation Forest."""

    return Pipeline([
        (
            "vectorizer",
            DictVectorizer(sparse=False),
        ),
        (
            "isolation_forest",
            IsolationForest(
                n_estimators=200,
                max_samples=256,
                contamination="auto",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


def anomaly_scores(pipeline, x):
    """
    sklearn의 점수 방향을 반전한다.

    반환값이 높을수록 이상한 거래로 해석한다.
    이 값은 사기 확률이 아니다.
    """

    return -pipeline.score_samples(x)


def threshold_at_fixed_fpr(y_true, scores, target_fpr):
    """
    Validation 정상 거래의 허용 오탐 수를 기준으로
    Threshold를 계산한다.

    동점이 있으면 오탐 예산을 초과하지 않도록
    보수적으로 처리한다.
    """

    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr은 0 이상 1 미만이어야 합니다.")

    normal_scores = sorted(
        (
            float(score)
            for label, score in zip(y_true, scores)
            if label == 0
        ),
        reverse=True,
    )

    if not normal_scores:
        raise ValueError("Validation 정상 거래가 없습니다.")

    allowed_fp = math.floor(
        target_fpr * len(normal_scores) + 1e-10
    )

    # 허용 개수 다음에 있는 정상 거래의 점수보다
    # Threshold를 조금 높게 잡는다.
    threshold = float(
        np.nextafter(
            normal_scores[allowed_fp],
            np.inf,
        )
    )

    return threshold, allowed_fp


def calculate_metrics(y_true, predictions, scores=None):
    """분류 결과 및 오탐 비율을 계산한다."""

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    result = {
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if scores is not None:
        result["ap"] = average_precision_score(
            y_true, scores
        )
        result["roc_auc"] = roc_auc_score(
            y_true, scores
        )

    return result


def get_existing_predictions(
    model_name, features, transactions, indices, split_info
):
    """저장된 RF·XGBoost의 Validation 예측을 재사용한다."""

    path = MODEL_DIR / model_name
    bundle = joblib.load(path)

    if bundle["train_cut"] != split_info["train_cut"].isoformat():
        raise ValueError(f"{model_name}: Train 분할 경계 불일치")

    if bundle["test_cut"] != split_info["test_cut"].isoformat():
        raise ValueError(f"{model_name}: Test 분할 경계 불일치")

    x, _ = make_xy(
        features,
        transactions,
        indices,
        bundle["feature_columns"],
    )

    probabilities = bundle["pipeline"].predict_proba(
        x
    )[:, 1]

    return np.asarray(
        probabilities >= bundle["threshold"],
        dtype=bool,
    )


def print_scenario_analysis(
    transactions, features, indices, y,
    if_predictions, rf_predictions, xgb_predictions
):
    """시나리오별 탐지율과 공통 미탐 복구 건수를 출력한다."""

    scenarios = defaultdict(
        lambda: {"total": 0, "detected": 0, "recovered": 0}
    )

    history_groups = defaultdict(
        lambda: {"total": 0, "detected": 0}
    )

    for position, index in enumerate(indices):
        if y[position] != 1:
            continue

        row = transactions[index]
        feature = features[index]

        scenario = row["scenarioType"]
        detected = bool(if_predictions[position])

        missed_by_both = (
            not rf_predictions[position]
            and not xgb_predictions[position]
        )

        scenarios[scenario]["total"] += 1
        scenarios[scenario]["detected"] += int(detected)
        scenarios[scenario]["recovered"] += int(
            missed_by_both and detected
        )

        history_group = (
            "0~2건"
            if feature["customerPastTxCount"] <= 2
            else "3건 이상"
        )

        history_groups[history_group]["total"] += 1
        history_groups[history_group]["detected"] += int(
            detected
        )

    print("\n[Isolation Forest 시나리오별 Recall]")
    print("recovered = RF·XGB 공통 미탐 중 추가 탐지")

    for name, counts in sorted(scenarios.items()):
        total = counts["total"]
        detected = counts["detected"]

        print(
            f"{name:<23} "
            f"TP={detected:>3} "
            f"FN={total - detected:>3} "
            f"Recall={detected / total:.2%} "
            f"recovered={counts['recovered']}"
        )

    print("\n[고객 과거 거래 수에 따른 Recall]")

    for name, counts in sorted(history_groups.items()):
        total = counts["total"]
        detected = counts["detected"]

        print(
            f"{name}: {detected}/{total} "
            f"({detected / total:.2%})"
        )


def main():
    transactions = read_csv(
        DATA_DIR / "transactions.csv"
    )
    episodes = read_csv(
        DATA_DIR / "fraud_episodes.csv"
    )

    features = build_features(transactions)
    split_info = split_and_purge(transactions, episodes)

    train_indices = split_info["indices"]["Train"]
    val_indices = split_info["indices"]["Validation"]

    # 학습에는 Train의 정상 거래만 사용한다.
    normal_train_indices = [
        index
        for index in train_indices
        if transactions[index]["fraudLabel"] == "0"
    ]

    assert len(normal_train_indices) == 12800
    assert len(val_indices) == 2998

    print("=== Isolation Forest Baseline ===")
    print(f"Train 정상 거래: {len(normal_train_indices):,}건")
    print(f"Validation 전체 거래: {len(val_indices):,}건")
    print(f"Target FPR: {TARGET_FPR:.2%}")
    print(f"Purged transactions: {len(split_info['purged_payments'])}")

    variants = {
        "A_full": list(FEATURE_COLUMNS),
        "B_no_shared_card": [
            name
            for name in FEATURE_COLUMNS
            if name != "cardUsedByOtherCustomer"
        ],
    }

    candidates = []

    print("\n=== Validation 후보 비교 ===")

    for variant_name, columns in variants.items():
        train_x, _ = make_xy(
            features,
            transactions,
            normal_train_indices,
            columns,
        )

        val_x, val_y = make_xy(
            features,
            transactions,
            val_indices,
            columns,
        )

        pipeline = create_pipeline()
        pipeline.fit(train_x)

        scores = anomaly_scores(pipeline, val_x)

        threshold, allowed_fp = threshold_at_fixed_fpr(
            val_y, scores, TARGET_FPR
        )

        predictions = scores >= threshold

        metrics = calculate_metrics(
            val_y, predictions, scores
        )

        if metrics["fp"] > allowed_fp:
            raise RuntimeError("Validation FP 예산을 초과했습니다.")

        print(
            f"{variant_name:<18} "
            f"AP={metrics['ap']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"Recall={metrics['recall']:.4f} "
            f"FPR={metrics['fpr']:.4f} "
            f"FP={metrics['fp']}"
        )

        candidates.append({
            "name": variant_name,
            "columns": columns,
            "pipeline": pipeline,
            "threshold": threshold,
            "metrics": metrics,
            "predictions": predictions,
        })

    # 목표 FPR 아래에서 Recall을 우선하여 선정한다.
    best = max(
        candidates,
        key=lambda candidate: (
            candidate["metrics"]["recall"],
            candidate["metrics"]["ap"],
        ),
    )

    print("\n=== 선정된 Isolation Forest ===")
    print("Feature 구성:", best["name"])
    print("Threshold:", f"{best['threshold']:.6f}")

    for name, value in best["metrics"].items():
        if name in ("tn", "fp", "fn", "tp"):
            print(f"{name}: {value}")
        else:
            print(f"{name}: {value:.4f}")

    rf_predictions = get_existing_predictions(
        "random_forest_baseline.joblib",
        features,
        transactions,
        val_indices,
        split_info,
    )

    xgb_predictions = get_existing_predictions(
        "xgboost_candidate.joblib",
        features,
        transactions,
        val_indices,
        split_info,
    )

    y = np.array([
        int(transactions[index]["fraudLabel"])
        for index in val_indices
    ])

    if_predictions = np.asarray(
        best["predictions"],
        dtype=bool,
    )

    # 기존 두 모델이 모두 놓친 사기
    common_misses = (
        (y == 1)
        & ~rf_predictions
        & ~xgb_predictions
    )

    recovered = common_misses & if_predictions

    # 기존 두 모델이 모두 정상으로 판정한 거래 중
    # Isolation Forest가 새로 오탐한 정상 거래
    new_false_positives = (
        (y == 0)
        & ~rf_predictions
        & ~xgb_predictions
        & if_predictions
    )

    print("\n=== RF·XGBoost 공통 미탐 복구 ===")
    print(f"공통 미탐: {int(common_misses.sum())}건")
    print(f"IF 추가 탐지: {int(recovered.sum())}건")
    print(
        "추가 정상 오탐:",
        int(new_false_positives.sum()),
        "건",
    )

    # 세 모델을 단순 OR로 결합하는 참고 실험
    union_predictions = (
        rf_predictions
        | xgb_predictions
        | if_predictions
    )

    union_metrics = calculate_metrics(
        y, union_predictions
    )

    print("\n=== 3개 모델 OR 결합 참고 결과 ===")

    for name in ("precision", "recall", "f1", "fpr"):
        print(f"{name}: {union_metrics[name]:.4f}")

    print(
        f"TN={union_metrics['tn']}, "
        f"FP={union_metrics['fp']}, "
        f"FN={union_metrics['fn']}, "
        f"TP={union_metrics['tp']}"
    )

    print_scenario_analysis(
        transactions,
        features,
        val_indices,
        y,
        if_predictions,
        rf_predictions,
        xgb_predictions,
    )

    # 향후 재현 가능한 비교를 위해 모델 저장
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "isolation_forest_candidate.joblib"

    joblib.dump({
        "pipeline": best["pipeline"],
        "threshold": best["threshold"],
        "feature_columns": best["columns"],
        "target_fpr": TARGET_FPR,
        "random_state": RANDOM_STATE,
        "train_cut": split_info["train_cut"].isoformat(),
        "test_cut": split_info["test_cut"].isoformat(),
    }, model_path)

    print("\n모델 저장:", model_path)
    print("Test 평가: 수행하지 않음")


if __name__ == "__main__":
    main()