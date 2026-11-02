
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


def run_experiment(
    name,
    use_history,
    tabular,
    sequences,
    mask,
    labels,
    train_indices,
    val_indices,
    positive_weight,
):
    # 동일한 초기화와 데이터 셔플 조건
    set_seed()

    if use_history:
        experiment_sequences = sequences
        experiment_mask = mask
    else:
        # 거래 이력 정보를 완전히 제거
        experiment_sequences = np.zeros_like(sequences)
        experiment_mask = np.zeros_like(mask, dtype=bool)

    train_loader = make_loader(
        tabular,
        experiment_sequences,
        experiment_mask,
        labels,
        train_indices,
        shuffle=True,
    )

    val_loader = make_loader(
        tabular,
        experiment_sequences,
        experiment_mask,
        labels,
        val_indices,
    )

    model = FraudNeuralNetwork(
        tab_dim=tabular.shape[1],
        seq_dim=sequences.shape[2],
        use_sequence=True,
    )

    print(f"\n=== {name} 학습 ===")

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

    print(f"\n[{name} 결과]")

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
        sequence_length=5,
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
        _,
    ) = prepare_data(
        feature_rows,
        windows,
        train_indices,
    )

    train_labels = labels[train_indices]

    positive_weight = (
        np.sum(train_labels == 0)
        / np.sum(train_labels == 1)
    )

    print("=== Transformer Sequence Ablation ===")
    print(f"Train: {len(train_indices):,}")
    print(f"Validation: {len(val_indices):,}")
    print("Test 사용: 없음")

    original = run_experiment(
        "Original Transformer",
        True,
        tabular,
        sequences,
        mask,
        labels,
        train_indices,
        val_indices,
        positive_weight,
    )

    no_history = run_experiment(
        "No History Transformer",
        False,
        tabular,
        sequences,
        mask,
        labels,
        train_indices,
        val_indices,
        positive_weight,
    )

    print("\n=== Ablation 비교 ===")
    print(
        f"{'Metric':<14}"
        f"{'Original':>12}"
        f"{'No History':>14}"
        f"{'Difference':>14}"
    )

    for metric in ("f1", "ap", "recall", "fpr"):
        a = original[metric]
        b = no_history[metric]

        print(
            f"{metric:<14}"
            f"{a:>12.4f}"
            f"{b:>14.4f}"
            f"{a-b:>+14.4f}"
        )

    print("\n기존 Transformer Validation AP: 0.8940")
    print("기존 Transformer Validation F1: 0.8212")
    print("기존 저장 모델: 변경 없음")
    print("Test 데이터: 사용하지 않음")


if __name__ == "__main__":
    main()
