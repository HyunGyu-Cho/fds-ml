
from collections import defaultdict
from pathlib import Path

import joblib
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from features import build_features
from train_random_forest import (
    read_csv,
    split_and_purge,
    make_xy,
)


BASE_DIR = Path(__file__).resolve().parent


def print_group_recall(title, groups):
    print(f"\n[{title}]")

    for name, counts in sorted(groups.items()):
        total = counts["total"]
        detected = counts["detected"]
        missed = total - detected
        recall = detected / total if total else 0.0

        print(
            f"{name:<25} "
            f"전체={total:>4}, "
            f"TP={detected:>4}, "
            f"FN={missed:>4}, "
            f"Recall={recall:.2%}"
        )


def main():
    transactions = read_csv(
        BASE_DIR / "data" / "transactions.csv"
    )
    episodes = read_csv(
        BASE_DIR / "data" / "fraud_episodes.csv"
    )

    features = build_features(transactions)
    split_info = split_and_purge(transactions, episodes)

    # 앞서 직접 학습해 저장한 로컬 모델만 로드한다.
    model_bundle = joblib.load(
        BASE_DIR / "models" / "random_forest_baseline.joblib"
    )

    pipeline = model_bundle["pipeline"]
    threshold = model_bundle["threshold"]
    columns = model_bundle["feature_columns"]

    indices = split_info["indices"]["Validation"]

    x, y = make_xy(
        features, transactions, indices, columns
    )

    probabilities = pipeline.predict_proba(x)[:, 1]
    predictions = [
        int(prob >= threshold)
        for prob in probabilities
    ]

    print("=== Random Forest Validation 오류 분석 ===")
    print(f"Threshold: {threshold:.2f}")
    print(f"검증 거래: {len(indices):,}건")
    print(f"Precision: {precision_score(y, predictions):.4f}")
    print(f"Recall:    {recall_score(y, predictions):.4f}")
    print(f"F1:        {f1_score(y, predictions):.4f}")

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    print(f"\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")

    scenario_groups = defaultdict(
        lambda: {"total": 0, "detected": 0}
    )
    history_groups = defaultdict(
        lambda: {"total": 0, "detected": 0}
    )

    false_negatives = []
    false_positives = []

    for index, actual, predicted, probability in zip(
        indices, y, predictions, probabilities
    ):
        row = transactions[index]
        feature = features[index]

        if actual == 1:
            scenario = row["scenarioType"]
            past_count = feature["customerPastTxCount"]

            history_group = (
                "0~2건"
                if past_count <= 2
                else "3건 이상"
            )

            for groups, key in [
                (scenario_groups, scenario),
                (history_groups, history_group),
            ]:
                groups[key]["total"] += 1
                groups[key]["detected"] += predicted

            if predicted == 0:
                false_negatives.append(
                    (index, probability)
                )

        elif predicted == 1:
            false_positives.append(
                (index, probability)
            )

    print_group_recall(
        "사기 시나리오별 탐지 성능",
        scenario_groups,
    )

    print_group_recall(
        "고객 과거 거래 수에 따른 탐지 성능",
        history_groups,
    )

    print("\n[미탐 사기 사례: Threshold에 가까운 상위 5건]")

    for index, probability in sorted(
        false_negatives,
        key=lambda item: item[1],
        reverse=True,
    )[:5]:
        row = transactions[index]
        feature = features[index]

        print(
            f"scenario={row['scenarioType']:<22} "
            f"score={probability:.3f} "
            f"amount={feature['amount']:.0f} "
            f"pastTx={feature['customerPastTxCount']} "
            f"tx30m={feature['txCount30m']}"
        )

    print("\n[정상 오탐 사례: 점수 상위 5건]")

    for index, probability in sorted(
        false_positives,
        key=lambda item: item[1],
        reverse=True,
    )[:5]:
        feature = features[index]

        print(
            f"score={probability:.3f} "
            f"amount={feature['amount']:.0f} "
            f"ratio={feature['amountRatio']:.2f} "
            f"tx30m={feature['txCount30m']} "
            f"sharedCard="
            f"{feature['cardUsedByOtherCustomer']}"
        )

    print("\n분석 완료: 모델과 Test 데이터는 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
