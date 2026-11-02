
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from features import FEATURE_COLUMNS, build_features


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

RANDOM_STATE = 42
SPLIT_NAMES = ("Train", "Validation", "Test")


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_time(value):
    timestamp = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    if timestamp.tzinfo is None:
        raise ValueError("거래 시각에 시간대 정보가 없습니다.")

    return timestamp.astimezone(timezone.utc)


def split_and_purge(transactions, episodes):
    """
    시간순 70/15/15 분할 후
    경계를 넘는 Episode 전체를 평가 데이터에서 제외한다.
    """

    if len(transactions) < 3:
        raise ValueError("거래 데이터가 부족합니다.")

    timestamps = [
        parse_time(row["paymentRequestedAt"])
        for row in transactions
    ]

    ordered = sorted(timestamps)
    total = len(ordered)

    train_cut = ordered[int(total * 0.70)]
    test_cut = ordered[int(total * 0.85)]

    if train_cut >= test_cut:
        raise ValueError("시간 분할 경계가 겹칩니다.")

    # 동일 타임스탬프는 반드시 동일 구간에 배정한다.
    assignments = []

    for timestamp in timestamps:
        if timestamp < train_cut:
            assignments.append("Train")
        elif timestamp < test_cut:
            assignments.append("Validation")
        else:
            assignments.append("Test")

    # 거래 ID 유일성을 검증한다.
    payment_to_index = {}

    for index, row in enumerate(transactions):
        payment_id = row["paymentId"]

        if payment_id in payment_to_index:
            raise ValueError(f"중복 paymentId: {payment_id}")

        payment_to_index[payment_id] = index

    fraud_payment_ids = {
        row["paymentId"]
        for row in transactions
        if row["fraudLabel"] == "1"
    }

    episode_by_payment = {}

    for row in episodes:
        payment_id = row["paymentId"]

        if payment_id in episode_by_payment:
            raise ValueError(f"Episode 연결 중복: {payment_id}")

        episode_by_payment[payment_id] = row["attackEventId"]

    if set(episode_by_payment) != fraud_payment_ids:
        raise ValueError(
            "사기 거래와 Episode 연결 정보가 일치하지 않습니다."
        )

    episode_splits = defaultdict(set)
    episode_payments = defaultdict(set)

    for payment_id, episode_id in episode_by_payment.items():
        index = payment_to_index[payment_id]

        episode_splits[episode_id].add(assignments[index])
        episode_payments[episode_id].add(payment_id)

    crossing_episodes = {
        episode_id
        for episode_id, areas in episode_splits.items()
        if len(areas) > 1
    }

    purged_payments = set()

    for episode_id in crossing_episodes:
        purged_payments.update(episode_payments[episode_id])

    split_indices = {}

    for split_name in SPLIT_NAMES:
        split_indices[split_name] = [
            index
            for index, row in enumerate(transactions)
            if assignments[index] == split_name
            and row["paymentId"] not in purged_payments
        ]

    # 각 구간에 정상과 사기가 모두 있는지 확인한다.
    for split_name, indices in split_indices.items():
        labels = {
            int(transactions[i]["fraudLabel"])
            for i in indices
        }

        if labels != {0, 1}:
            raise ValueError(
                f"{split_name}에 정상/사기 클래스가 모두 존재하지 않습니다."
            )

    # Purging 이후 Episode가 서로 다른 구간에 남지 않는지 검사한다.
    remaining_episode_splits = defaultdict(set)

    for split_name, indices in split_indices.items():
        for index in indices:
            payment_id = transactions[index]["paymentId"]

            if payment_id in episode_by_payment:
                episode_id = episode_by_payment[payment_id]
                remaining_episode_splits[episode_id].add(
                    split_name
                )

    if any(
        len(areas) > 1
        for areas in remaining_episode_splits.values()
    ):
        raise RuntimeError("Episode 분할 누출이 발생했습니다.")

    return {
        "indices": split_indices,
        "crossing_episodes": crossing_episodes,
        "purged_payments": purged_payments,
        "episode_by_payment": episode_by_payment,
        "train_cut": train_cut,
        "test_cut": test_cut,
    }


def make_xy(feature_rows, transactions, indices, columns):
    """
    선택한 Feature만 X에 포함한다.
    fraudLabel은 y로만 사용한다.
    """

    x = [
        {
            name: feature_rows[index][name]
            for name in columns
        }
        for index in indices
    ]

    y = [
        int(transactions[index]["fraudLabel"])
        for index in indices
    ]

    return x, y


def choose_threshold(y_true, probabilities):
    """
    Validation F1이 가장 높은 Threshold를 선택한다.
    동률이면 높은 Threshold를 선택한다.
    """

    best_threshold = 0.5
    best_f1 = -1.0

    for step in range(1, 20):
        threshold = step * 0.05

        predictions = [
            int(prob >= threshold)
            for prob in probabilities
        ]

        score = f1_score(
            y_true, predictions, zero_division=0
        )

        if (
            score > best_f1
            or (
                score == best_f1
                and threshold > best_threshold
            )
        ):
            best_threshold = threshold
            best_f1 = score

    return best_threshold, best_f1


def create_pipeline(class_weight):
    """
    DictVectorizer:
    - 문자열 merchantId는 One-Hot Encoding
    - 숫자 Feature는 그대로 전달
    - 인코딩 규칙은 Train 데이터에서만 학습
    """

    return Pipeline([
        (
            "vectorizer",
            DictVectorizer(sparse=False),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight=class_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


def print_split_summary(transactions, split_info):
    print("\n=== 시간순 분할 및 Episode 검증 ===")

    print("Train 경계:", split_info["train_cut"])
    print("Test 경계:", split_info["test_cut"])

    for name, indices in split_info["indices"].items():
        labels = [
            int(transactions[i]["fraudLabel"])
            for i in indices
        ]
        counts = Counter(labels)

        print(
            f"{name}: 전체={len(indices):,}, "
            f"정상={counts[0]:,}, "
            f"사기={counts[1]:,}, "
            f"사기율={counts[1] / len(indices):.2%}"
        )

    print(
        "경계 충돌 Episode:",
        len(split_info["crossing_episodes"]),
    )
    print(
        "제외 거래:",
        len(split_info["purged_payments"]),
    )


def episode_recall(
    transactions,
    test_indices,
    predictions,
    episode_by_payment,
):
    detected = {}

    for index, prediction in zip(test_indices, predictions):
        row = transactions[index]

        if row["fraudLabel"] != "1":
            continue

        episode_id = episode_by_payment[row["paymentId"]]

        if episode_id not in detected:
            detected[episode_id] = False

        if prediction == 1:
            detected[episode_id] = True

    total = len(detected)
    found = sum(detected.values())

    return found, total


def main():
    transactions = read_csv(
        DATA_DIR / "transactions.csv"
    )
    episodes = read_csv(
        DATA_DIR / "fraud_episodes.csv"
    )

    print("=== Random Forest Baseline ===")

    # 계산 단계에서는 전체 거래의 과거 이력을 사용한다.
    # 현재 또는 미래 거래의 Label은 사용하지 않는다.
    feature_rows = build_features(transactions)

    split_info = split_and_purge(
        transactions, episodes
    )

    print_split_summary(transactions, split_info)

    # 전체 Feature와 카드 공유 이력 제외 모델 비교
    variants = {
        "A_full": list(FEATURE_COLUMNS),
        "B_no_shared_card": [
            name for name in FEATURE_COLUMNS
            if name != "cardUsedByOtherCustomer"
        ],
    }

    candidates = []

    print("\n=== Validation 모델 비교 ===")

    for variant_name, columns in variants.items():
        train_x, train_y = make_xy(
            feature_rows,
            transactions,
            split_info["indices"]["Train"],
            columns,
        )

        val_x, val_y = make_xy(
            feature_rows,
            transactions,
            split_info["indices"]["Validation"],
            columns,
        )

        for weight in (None, "balanced"):
            pipeline = create_pipeline(weight)

            pipeline.fit(train_x, train_y)

            val_probabilities = pipeline.predict_proba(
                val_x
            )[:, 1]

            threshold, val_f1 = choose_threshold(
                val_y, val_probabilities
            )

            val_ap = average_precision_score(
                val_y, val_probabilities
            )

            weight_name = (
                "None" if weight is None else "balanced"
            )

            print(
                f"{variant_name:<18} "
                f"weight={weight_name:<9} "
                f"AP={val_ap:.4f} "
                f"F1={val_f1:.4f} "
                f"Threshold={threshold:.2f}"
            )

            candidates.append({
                "name": variant_name,
                "weight": weight_name,
                "columns": columns,
                "pipeline": pipeline,
                "threshold": threshold,
                "val_f1": val_f1,
                "val_ap": val_ap,
            })

    # Test를 보지 않고 Validation 결과만으로 선정한다.
    best = max(
        candidates,
        key=lambda candidate: (
            candidate["val_f1"],
            candidate["val_ap"],
        ),
    )

    print("\n=== 최종 선정 모델 ===")
    print("Feature 구성:", best["name"])
    print("class_weight:", best["weight"])
    print("Threshold:", f"{best['threshold']:.2f}")
    print("Validation F1:", f"{best['val_f1']:.4f}")

    # Test는 모델과 Threshold 선정이 끝난 뒤 평가한다.
    test_indices = split_info["indices"]["Test"]

    test_x, test_y = make_xy(
        feature_rows,
        transactions,
        test_indices,
        best["columns"],
    )

    start_time = perf_counter()

    test_probabilities = best[
        "pipeline"
    ].predict_proba(test_x)[:, 1]

    elapsed = perf_counter() - start_time

    test_predictions = [
        int(prob >= best["threshold"])
        for prob in test_probabilities
    ]

    tn, fp, fn, tp = confusion_matrix(
        test_y,
        test_predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        test_y, test_predictions, zero_division=0
    )
    recall = recall_score(
        test_y, test_predictions, zero_division=0
    )
    f1 = f1_score(
        test_y, test_predictions, zero_division=0
    )
    ap = average_precision_score(
        test_y, test_probabilities
    )
    roc_auc = roc_auc_score(
        test_y, test_probabilities
    )

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    detected, total_episodes = episode_recall(
        transactions,
        test_indices,
        test_predictions,
        split_info["episode_by_payment"],
    )

    print("\n=== 최종 Test 평가 ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"AP:        {ap:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"FPR:       {fpr:.4f}")

    print("\n[Confusion Matrix]")
    print(f"TN={tn}, FP={fp}")
    print(f"FN={fn}, TP={tp}")

    print(
        f"\nEpisode Recall: "
        f"{detected}/{total_episodes} "
        f"({detected / total_episodes:.2%})"
    )

    print(
        "Batch 평균 추론 시간: "
        f"{elapsed * 1000 / len(test_x):.4f} ms/거래"
    )

    print("\n=== Feature Importance 상위 12개 ===")

    vectorizer = best["pipeline"].named_steps["vectorizer"]
    forest = best["pipeline"].named_steps["random_forest"]

    importance_pairs = sorted(
        zip(
            vectorizer.get_feature_names_out(),
            forest.feature_importances_,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    for name, importance in importance_pairs[:12]:
        print(f"{name:<35} {importance:.4f}")

    # 선택된 모델과 추론에 필요한 정보를 저장한다.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "random_forest_baseline.joblib"

    joblib.dump({
        "pipeline": best["pipeline"],
        "threshold": best["threshold"],
        "feature_columns": best["columns"],
        "class_weight": best["weight"],
        "random_state": RANDOM_STATE,
        "train_cut": split_info["train_cut"].isoformat(),
        "test_cut": split_info["test_cut"].isoformat(),
    }, model_path)

    print("\n모델 저장:", model_path)
    print("Random Forest Baseline 완료")


if __name__ == "__main__":
    main()