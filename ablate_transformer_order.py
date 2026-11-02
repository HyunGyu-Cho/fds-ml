
from pathlib import Path

import numpy as np

from features import build_features
from sequences import build_sequences
from train_random_forest import read_csv, split_and_purge
from train_transformer import (
    FraudNeuralNetwork,
    prepare_data,
    make_loader,
    train_model,
    predict_probabilities,
    evaluate,
    set_seed,
)


BASE_DIR = Path(__file__).resolve().parent
SEQ_LENGTH = 5
SHUFFLE_SEED = 42


def shuffle_valid_tokens(sequences, mask, seed=42):
    """
    실제 과거 거래 Token의 순서만 섞는다.

    - Padding 위치 유지
    - 거래 내용 유지
    - Label 미사용
    - 재현 가능한 무작위 순열 사용
    """

    if sequences.shape[:2] != mask.shape:
        raise ValueError("Sequence와 Mask 크기가 다릅니다.")

    rng = np.random.default_rng(seed)
    shuffled = sequences.copy()

    for i in range(len(sequences)):
        valid_positions = np.flatnonzero(mask[i])

        if len(valid_positions) < 2:
            continue

        permutation = rng.permutation(valid_positions)

        shuffled[i, valid_positions] = (
            sequences[i, permutation]
        )

    return shuffled


def run_experiment(
    name,
    tabular,
    sequences,
    mask,
    labels,
    train_indices,
    val_indices,
    positive_weight,
):
    # 학습 시작 조건을 동일하게 설정
    set_seed()

    train_loader = make_loader(
        tabular,
        sequences,
        mask,
        labels,
        train_indices,
        shuffle=True,
    )

    val_loader = make_loader(
        tabular,
        sequences,
        mask,
        labels,
        val_indices,
    )

    model = FraudNeuralNetwork(
        tab_dim=tabular.shape[1],
        seq_dim=sequences.shape[2],
        use_sequence=True,
    )

    print(f"\n=== {name} ===")

    model = train_model(
        model,
        train_loader,
        val_loader,
        labels[val_indices].astype(int),
        positive_weight,
    )

    probabilities = predict_probabilities(
        model,
        val_loader,
    )

    result = evaluate(
        labels[val_indices].astype(int),
        probabilities,
    )

    print(f"\n[{name} Validation 결과]")

    for metric in (
        "threshold",
        "precision",
        "recall",
        "f1",
        "ap",
        "fpr",
    ):
        print(f"{metric}: {result[metric]:.4f}")

    print(
        f"TP={result['tp']} "
        f"FP={result['fp']} "
        f"FN={result['fn']} "
        f"TN={result['tn']}"
    )

    return result


def main():
    set_seed()

    transactions = read_csv(
        BASE_DIR / "data" / "transactions.csv"
    )

    episodes = read_csv(
        BASE_DIR / "data" / "fraud_episodes.csv"
    )

    feature_rows = build_features(transactions)

    windows = build_sequences(
        transactions,
        sequence_length=SEQ_LENGTH,
    )

    split_info = split_and_purge(
        transactions,
        episodes,
    )

    train_indices = split_info["indices"]["Train"]
    val_indices = split_info["indices"]["Validation"]

    labels = np.asarray(
        [
            int(row["fraudLabel"])
            for row in transactions
        ],
        dtype=np.float32,
    )

    (
        tabular,
        sequences,
        mask,
        preprocessing,
    ) = prepare_data(
        feature_rows,
        windows,
        train_indices,
    )

    # 시간 간격으로 원래 순서를 복원하지 못하도록
    # 명시적 상대 시간 Feature를 두 모델에서 모두 제거
    feature_names = list(
        preprocessing["seq_vectorizer"]
        .get_feature_names_out()
    )

    time_feature = "logMinutesBeforeCurrent"

    if time_feature not in feature_names:
        raise ValueError(
            f"Sequence Feature 없음: {time_feature}"
        )

    time_index = feature_names.index(time_feature)

    ordered_sequences = sequences.copy()
    ordered_sequences[:, :, time_index] = 0.0

    shuffled_sequences = shuffle_valid_tokens(
        ordered_sequences,
        mask,
        seed=SHUFFLE_SEED,
    )

    # 무작위 섞기가 실제 Token 값을 제거하지 않았는지 검사
    for i in range(len(mask)):
        valid = mask[i]

        ordered_rows = sorted(
            tuple(row)
            for row in ordered_sequences[i, valid]
        )

        shuffled_rows = sorted(
            tuple(row)
            for row in shuffled_sequences[i, valid]
        )

        if ordered_rows != shuffled_rows:
            raise RuntimeError(
                f"{i}번째 거래에서 Token 정보가 달라졌습니다."
            )

    train_labels = labels[train_indices]

    positive_weight = (
        np.sum(train_labels == 0)
        / np.sum(train_labels == 1)
    )

    print("=== Transformer Order Ablation ===")
    print(f"Train: {len(train_indices):,}")
    print(f"Validation: {len(val_indices):,}")
    print(f"Sequence Length: {SEQ_LENGTH}")
    print(f"제거한 상대 시간 Feature: {time_feature}")
    print("Test 데이터 사용: 없음")

    original = run_experiment(
        "Ordered Sequence",
        tabular,
        ordered_sequences,
        mask,
        labels,
        train_indices,
        val_indices,
        positive_weight,
    )

    shuffled = run_experiment(
        "Shuffled Sequence",
        tabular,
        shuffled_sequences,
        mask,
        labels,
        train_indices,
        val_indices,
        positive_weight,
    )

    print("\n=== 거래 순서 Ablation 비교 ===")

    print(
        f"{'Metric':<12}"
        f"{'Ordered':>12}"
        f"{'Shuffled':>12}"
        f"{'Difference':>14}"
    )

    for name in ("f1", "ap", "recall", "fpr"):
        a = original[name]
        b = shuffled[name]

        print(
            f"{name:<12}"
            f"{a:>12.4f}"
            f"{b:>12.4f}"
            f"{a-b:>+14.4f}"
        )

    print("\n원본 모델 및 데이터는 변경하지 않았습니다.")
    print("Test 평가: 수행하지 않음")


if __name__ == "__main__":
    main()
    