
from collections import Counter, defaultdict
from pathlib import Path

import joblib
from xgboost import XGBClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from features import FEATURE_COLUMNS, build_features
from train_random_forest import (
    read_csv,
    split_and_purge,
    make_xy,
    choose_threshold,
)


BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


def create_pipeline(scale_pos_weight):
    return Pipeline([
        (
            "vectorizer",
            DictVectorizer(sparse=False),
        ),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=2,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


def evaluate(y_true, probabilities, threshold):
    predictions = [
        int(prob >= threshold)
        for prob in probabilities
    ]

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "ap": average_precision_score(
            y_true, probabilities
        ),
        "fpr": fpr,
    }


def main():
    transactions = read_csv(
        BASE_DIR / "data" / "transactions.csv"
    )
    episodes = read_csv(
        BASE_DIR / "data" / "fraud_episodes.csv"
    )

    feature_rows = build_features(transactions)
    split_info = split_and_purge(transactions, episodes)

    train_indices = split_info["indices"]["Train"]
    val_indices = split_info["indices"]["Validation"]

    train_labels = [
        int(transactions[i]["fraudLabel"])
        for i in train_indices
    ]

    train_counts = Counter(train_labels)

    # RF class_weight="balanced"와 동일한 옵션은 아니지만
    # XGBoost의 양성 클래스 가중치를 비교하는 실험이다.
    balanced_weight = (
        train_counts[0] / train_counts[1]
    )

    variants = {
        "A_full": list(FEATURE_COLUMNS),
        "B_no_shared_card": [
            name for name in FEATURE_COLUMNS
            if name != "cardUsedByOtherCustomer"
        ],
    }

    weights = {
        "None": 1.0,
        "balanced": balanced_weight,
    }

    print("=== XGBoost Baseline ===")
    print(f"Train: {len(train_indices):,}건")
    print(f"Validation: {len(val_indices):,}건")
    print(f"Episode Purging: {len(split_info['purged_payments'])}건")
    print(f"Balanced scale_pos_weight: {balanced_weight:.4f}")

    candidates = []

    for variant_name, columns in variants.items():
        train_x, train_y = make_xy(
            feature_rows, transactions, train_indices, columns
        )
        val_x, val_y = make_xy(
            feature_rows, transactions, val_indices, columns
        )

        for weight_name, weight_value in weights.items():
            pipeline = create_pipeline(weight_value)
            pipeline.fit(train_x, train_y)

            probabilities = pipeline.predict_proba(
                val_x
            )[:, 1]

            threshold, _ = choose_threshold(
                val_y, probabilities
            )

            metrics = evaluate(
                val_y, probabilities, threshold
            )

            print(
                f"{variant_name:<18} "
                f"weight={weight_name:<9} "
                f"AP={metrics['ap']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"Recall={metrics['recall']:.4f} "
                f"FPR={metrics['fpr']:.4f} "
                f"Threshold={threshold:.2f}"
            )

            candidates.append({
                "name": variant_name,
                "weight_name": weight_name,
                "weight_value": weight_value,
                "columns": columns,
                "pipeline": pipeline,
                "threshold": threshold,
                "metrics": metrics,
            })

    best = max(
        candidates,
        key=lambda candidate: (
            candidate["metrics"]["f1"],
            candidate["metrics"]["ap"],
        ),
    )

    print("\n=== XGBoost Validation 선정 모델 ===")
    print("Feature 구성:", best["name"])
    print("Class weight:", best["weight_name"])
    print("Threshold:", best["threshold"])

    for metric, value in best["metrics"].items():
        print(f"{metric}: {value:.4f}")

    # 선정된 모델의 시나리오별 Validation Recall
    val_x, val_y = make_xy(
        feature_rows,
        transactions,
        val_indices,
        best["columns"],
    )

    probabilities = best["pipeline"].predict_proba(
        val_x
    )[:, 1]

    predictions = [
        int(prob >= best["threshold"])
        for prob in probabilities
    ]

    scenarios = defaultdict(
        lambda: {"total": 0, "detected": 0}
    )

    history_groups = defaultdict(
        lambda: {"total": 0, "detected": 0}
    )

    for index, actual, predicted in zip(
        val_indices, val_y, predictions
    ):
        if actual != 1:
            continue

        scenario = transactions[index]["scenarioType"]

        past_count = feature_rows[index][
            "customerPastTxCount"
        ]

        history_group = (
            "0~2" if past_count <= 2 else "3+"
        )

        for groups, name in [
            (scenarios, scenario),
            (history_groups, history_group),
        ]:
            groups[name]["total"] += 1
            groups[name]["detected"] += predicted

    print("\n=== XGBoost 시나리오별 Validation Recall ===")

    for scenario, counts in sorted(scenarios.items()):
        total = counts["total"]
        detected = counts["detected"]

        print(
            f"{scenario:<23} "
            f"TP={detected:>3} "
            f"FN={total - detected:>3} "
            f"Recall={detected / total:.2%}"
        )

    print("\n=== 고객 이력별 Validation Recall ===")

    for name, counts in sorted(history_groups.items()):
        total = counts["total"]
        detected = counts["detected"]

        print(
            f"{name}: {detected}/{total} "
            f"({detected / total:.2%})"
        )

    print("\n=== RF Validation 비교 기준 ===")
    print("RF F1:     0.8290")
    print("RF AP:     0.8984")
    print("RF Recall: 0.7878")
    print("RF FPR:    0.0196")

    # Test는 평가하지 않고 모델 후보만 저장한다.
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "xgboost_candidate.joblib"

    joblib.dump({
        "pipeline": best["pipeline"],
        "threshold": best["threshold"],
        "feature_columns": best["columns"],
        "weight_name": best["weight_name"],
        "random_state": RANDOM_STATE,
        "train_cut": split_info["train_cut"].isoformat(),
        "test_cut": split_info["test_cut"].isoformat(),
    }, model_path)

    print("\n모델 저장:", model_path)
    print("Test 평가: 수행하지 않음")


if __name__ == "__main__":
    main()