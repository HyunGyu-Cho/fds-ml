
from collections import Counter, defaultdict
from pathlib import Path

import joblib

from features import build_features
from train_random_forest import (
    read_csv,
    split_and_purge,
    make_xy,
)


BASE_DIR = Path(__file__).resolve().parent

RESULT_NAMES = (
    "BOTH",
    "RF_ONLY",
    "XGB_ONLY",
    "NEITHER",
)


def compare_result(rf_prediction, xgb_prediction):
    """두 모델의 양성(사기) 예측을 비교한다."""

    if rf_prediction == 1 and xgb_prediction == 1:
        return "BOTH"

    if rf_prediction == 1:
        return "RF_ONLY"

    if xgb_prediction == 1:
        return "XGB_ONLY"

    return "NEITHER"


def predict_model(bundle, features, transactions, indices):
    """저장된 모델과 Threshold로 예측한다."""

    x, y = make_xy(
        features,
        transactions,
        indices,
        bundle["feature_columns"],
    )

    probabilities = bundle["pipeline"].predict_proba(
        x
    )[:, 1]

    predictions = [
        int(prob >= bundle["threshold"])
        for prob in probabilities
    ]

    return y, predictions, probabilities


def print_counts(title, counts):
    print(f"\n[{title}]")

    for name in RESULT_NAMES:
        print(f"{name:<12}: {counts[name]:>4}건")


def main():
    transactions = read_csv(
        BASE_DIR / "data" / "transactions.csv"
    )

    episodes = read_csv(
        BASE_DIR / "data" / "fraud_episodes.csv"
    )

    features = build_features(transactions)

    split_info = split_and_purge(
        transactions, episodes
    )

    validation_indices = split_info[
        "indices"
    ]["Validation"]

    rf = joblib.load(
        BASE_DIR / "models" /
        "random_forest_baseline.joblib"
    )

    xgb = joblib.load(
        BASE_DIR / "models" /
        "xgboost_candidate.joblib"
    )

    # 두 모델이 같은 시간순 분할을 사용했는지 확인
    for bundle in (rf, xgb):
        assert bundle["train_cut"] == (
            split_info["train_cut"].isoformat()
        )
        assert bundle["test_cut"] == (
            split_info["test_cut"].isoformat()
        )

    rf_y, rf_predictions, rf_probabilities = (
        predict_model(
            rf, features, transactions, validation_indices
        )
    )

    xgb_y, xgb_predictions, xgb_probabilities = (
        predict_model(
            xgb, features, transactions, validation_indices
        )
    )

    assert rf_y == xgb_y

    fraud_counts = Counter()
    normal_counts = Counter()

    scenario_counts = defaultdict(Counter)
    history_counts = defaultdict(Counter)

    examples = {
        "RF_ONLY": [],
        "XGB_ONLY": [],
        "NEITHER": [],
    }

    for position, index in enumerate(validation_indices):
        actual = rf_y[position]

        rf_prediction = rf_predictions[position]
        xgb_prediction = xgb_predictions[position]

        result = compare_result(
            rf_prediction,
            xgb_prediction,
        )

        row = transactions[index]
        feature = features[index]

        if actual == 1:
            fraud_counts[result] += 1
            scenario_counts[row["scenarioType"]][result] += 1

            history_group = (
                "0~2건"
                if feature["customerPastTxCount"] <= 2
                else "3건 이상"
            )

            history_counts[history_group][result] += 1

            if result != "BOTH":
                examples[result].append({
                    "scenario": row["scenarioType"],
                    "rf_score": rf_probabilities[position],
                    "xgb_score": xgb_probabilities[position],
                    "amount": feature["amount"],
                    "tx30m": feature["txCount30m"],
                    "pastTx": feature["customerPastTxCount"],
                })

        else:
            normal_counts[result] += 1

    print("=== RF vs XGBoost Validation 비교 ===")
    print(f"Validation 거래: {len(validation_indices):,}건")
    print(f"RF Threshold: {rf['threshold']:.2f}")
    print(f"XGB Threshold: {xgb['threshold']:.2f}")

    print_counts(
        "실제 사기 거래의 탐지 조합",
        fraud_counts,
    )

    print_counts(
        "실제 정상 거래의 사기 오탐 조합",
        normal_counts,
    )

    # BOTH, RF_ONLY, XGB_ONLY는 정상 거래에서 오탐
    # NEITHER는 정상 거래에서 올바른 정상 판정

    rf_tp = fraud_counts["BOTH"] + fraud_counts["RF_ONLY"]
    xgb_tp = fraud_counts["BOTH"] + fraud_counts["XGB_ONLY"]

    rf_fp = normal_counts["BOTH"] + normal_counts["RF_ONLY"]
    xgb_fp = normal_counts["BOTH"] + normal_counts["XGB_ONLY"]

    assert rf_tp == 349
    assert xgb_tp == 347
    assert rf_fp == 50
    assert xgb_fp == 49

    print("\n[시나리오별 미탐·추가 탐지 비교]")

    for scenario, counts in sorted(scenario_counts.items()):
        print(
            f"{scenario:<23} "
            f"둘다탐지={counts['BOTH']:>3} "
            f"RF만={counts['RF_ONLY']:>3} "
            f"XGB만={counts['XGB_ONLY']:>3} "
            f"둘다미탐={counts['NEITHER']:>3}"
        )

    print("\n[고객 과거 이력별 비교]")

    for name, counts in sorted(history_counts.items()):
        print(
            f"{name}: "
            f"둘다탐지={counts['BOTH']}, "
            f"RF만={counts['RF_ONLY']}, "
            f"XGB만={counts['XGB_ONLY']}, "
            f"둘다미탐={counts['NEITHER']}"
        )

    print("\n[추가 분석할 사기 거래 예시]")

    for category, records in examples.items():
        print(f"\n{category} — {len(records)}건")

        for item in records[:5]:
            print(
                f"  scenario={item['scenario']:<22} "
                f"RF={item['rf_score']:.3f} "
                f"XGB={item['xgb_score']:.3f} "
                f"amount={item['amount']:.0f} "
                f"tx30m={item['tx30m']} "
                f"pastTx={item['pastTx']}"
            )

    print("\n분석 완료: Validation만 사용했습니다.")


if __name__ == "__main__":
    main()
