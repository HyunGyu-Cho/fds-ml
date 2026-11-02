
from __future__ import annotations

import copy
import random
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from features import FEATURE_COLUMNS, build_features
from sequences import build_sequences
from train_random_forest import (
    read_csv,
    split_and_purge,
    make_xy,
    choose_threshold,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

SEED = 42
SEQ_LENGTH = 5
D_MODEL = 32
BATCH_SIZE = 128
MAX_EPOCHS = 20
PATIENCE = 4
LEARNING_RATE = 0.001


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 노트북 CPU에서 과도한 스레드 사용을 방지한다.
    torch.set_num_threads(min(4, torch.get_num_threads()))


def prepare_data(feature_rows, windows, train_indices):
    """
    Tabular 및 Sequence 데이터를 숫자 행렬로 변환한다.

    DictVectorizer와 StandardScaler는 Train 데이터에서만
    fit한다. Validation/Test에는 transform만 수행한다.
    """

    # 현재 거래의 기존 Feature 13개
    current_rows = [
        {name: row[name] for name in FEATURE_COLUMNS}
        for row in feature_rows
    ]

    tab_vectorizer = DictVectorizer(sparse=False)
    tab_vectorizer.fit([
        current_rows[i] for i in train_indices
    ])

    tab_raw = tab_vectorizer.transform(current_rows)

    tab_scaler = StandardScaler()
    tab_scaler.fit(tab_raw[train_indices])

    tabular = tab_scaler.transform(
        tab_raw
    ).astype(np.float32)

    # 과거 거래 Token: Padding이 아닌 거래만 인코더 학습에 사용
    valid_train_tokens = [
        token
        for i in train_indices
        for token, valid in zip(
            windows[i].tokens, windows[i].mask
        )
        if valid
    ]

    if not valid_train_tokens:
        raise ValueError("Train의 유효한 과거 거래가 없습니다.")

    seq_vectorizer = DictVectorizer(sparse=False)
    seq_vectorizer.fit(valid_train_tokens)

    all_tokens = [
        token
        for window in windows
        for token in window.tokens
    ]

    n = len(windows)

    seq_raw = seq_vectorizer.transform(
        all_tokens
    ).reshape(n, SEQ_LENGTH, -1)

    mask = np.asarray(
        [window.mask for window in windows],
        dtype=bool,
    )

    seq_scaler = StandardScaler()

    # Train의 실제 과거 Token으로만 정규화 통계 계산
    train_valid_values = seq_raw[train_indices][
        mask[train_indices]
    ]

    seq_scaler.fit(train_valid_values)

    sequences = seq_scaler.transform(
        seq_raw.reshape(-1, seq_raw.shape[-1])
    ).reshape(n, SEQ_LENGTH, -1).astype(np.float32)

    # Padding에는 정규화된 값을 넣지 않는다.
    sequences[~mask] = 0.0

    if not np.isfinite(tabular).all():
        raise ValueError("현재 Feature에 비정상 값이 있습니다.")

    if not np.isfinite(sequences).all():
        raise ValueError("Sequence Feature에 비정상 값이 있습니다.")

    preprocessing = {
        "tab_vectorizer": tab_vectorizer,
        "tab_scaler": tab_scaler,
        "seq_vectorizer": seq_vectorizer,
        "seq_scaler": seq_scaler,
        "sequence_length": SEQ_LENGTH,
    }

    return tabular, sequences, mask, preprocessing


class FraudNeuralNetwork(nn.Module):
    """
    use_sequence=False: Tabular MLP
    use_sequence=True: Transformer + Tabular MLP
    """

    def __init__(
        self,
        tab_dim: int,
        seq_dim: int,
        use_sequence: bool,
    ):
        super().__init__()

        self.use_sequence = use_sequence

        if use_sequence:
            self.input_projection = nn.Linear(
                seq_dim, D_MODEL
            )

            # 이력이 없는 고객도 표현할 수 있는 CLS 토큰
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, D_MODEL)
            )

            self.position_embedding = nn.Embedding(
                SEQ_LENGTH + 1, D_MODEL
            )

            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=4,
                dim_feedforward=64,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )

            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=1,
                enable_nested_tensor=False,
            )

        combined_dim = (
            tab_dim + D_MODEL
            if use_sequence
            else tab_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, tabular, sequences, mask):
        if self.use_sequence:
            batch_size = tabular.shape[0]

            tokens = self.input_projection(sequences)

            cls = self.cls_token.expand(
                batch_size, -1, -1
            )

            tokens = torch.cat(
                [cls, tokens], dim=1
            )

            positions = torch.arange(
                tokens.shape[1],
                device=tokens.device,
            )

            tokens = tokens + self.position_embedding(
                positions
            ).unsqueeze(0)

            # True는 Attention에서 제외되는 위치
            # CLS는 항상 유효하므로 모든 Padding도 안전하게 처리
            cls_mask = torch.zeros(
                (batch_size, 1),
                dtype=torch.bool,
                device=mask.device,
            )

            padding_mask = torch.cat(
                [cls_mask, ~mask], dim=1
            )

            encoded = self.encoder(
                tokens,
                src_key_padding_mask=padding_mask,
            )

            sequence_embedding = encoded[:, 0, :]

            combined = torch.cat(
                [tabular, sequence_embedding],
                dim=1,
            )
        else:
            combined = tabular

        return self.classifier(
            combined
        ).squeeze(-1)


def make_loader(
    tabular,
    sequences,
    mask,
    labels,
    indices,
    shuffle=False,
):
    dataset = TensorDataset(
        torch.from_numpy(tabular[indices]),
        torch.from_numpy(sequences[indices]),
        torch.from_numpy(mask[indices]),
        torch.from_numpy(labels[indices]),
    )

    generator = torch.Generator().manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


@torch.no_grad()
def predict_probabilities(model, loader):
    model.eval()
    outputs = []

    for tabular, sequences, mask, _ in loader:
        logits = model(tabular, sequences, mask)

        probabilities = torch.sigmoid(logits)

        outputs.append(
            probabilities.cpu().numpy()
        )

    return np.concatenate(outputs)


def train_model(
    model,
    train_loader,
    val_loader,
    val_labels,
    positive_weight,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            positive_weight, dtype=torch.float32
        )
    )

    best_ap = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for tabular, sequences, mask, labels in train_loader:
            optimizer.zero_grad()

            logits = model(
                tabular, sequences, mask
            )

            loss = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            count = len(labels)
            total_loss += loss.item() * count
            total_samples += count

        val_probabilities = predict_probabilities(
            model, val_loader
        )

        val_ap = average_precision_score(
            val_labels, val_probabilities
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss={total_loss/total_samples:.4f} | "
            f"Val AP={val_ap:.4f}"
        )

        if val_ap > best_ap + 1e-4:
            best_ap = val_ap
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print("Early Stopping")
            break

    if best_state is None:
        raise RuntimeError("최적 모델을 찾지 못했습니다.")

    model.load_state_dict(best_state)

    print(
        f"선정 Epoch: {best_epoch}, "
        f"Best Validation AP: {best_ap:.4f}"
    )

    return model


def evaluate(y, probabilities):
    threshold, _ = choose_threshold(
        y, probabilities
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
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
            y, probabilities
        ),
        "fpr": fp / (fp + tn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predictions": predictions,
    }


def existing_model_predictions(
    filename,
    feature_rows,
    transactions,
    indices,
    split_info,
):
    bundle = joblib.load(MODEL_DIR / filename)

    if bundle["train_cut"] != split_info["train_cut"].isoformat():
        raise ValueError("기존 모델의 Train 경계가 다릅니다.")

    if bundle["test_cut"] != split_info["test_cut"].isoformat():
        raise ValueError("기존 모델의 Test 경계가 다릅니다.")

    x, _ = make_xy(
        feature_rows,
        transactions,
        indices,
        bundle["feature_columns"],
    )

    return (
        bundle["pipeline"].predict_proba(x)[:, 1]
        >= bundle["threshold"]
    )


def print_scenario_results(
    transactions,
    indices,
    labels,
    predictions,
    rf_predictions,
    xgb_predictions,
):
    groups = defaultdict(
        lambda: {
            "total": 0,
            "detected": 0,
            "recovered": 0,
        }
    )

    for pos, index in enumerate(indices):
        if labels[pos] != 1:
            continue

        scenario = transactions[index]["scenarioType"]
        detected = bool(predictions[pos])

        common_miss = (
            not rf_predictions[pos]
            and not xgb_predictions[pos]
        )

        groups[scenario]["total"] += 1
        groups[scenario]["detected"] += int(detected)
        groups[scenario]["recovered"] += int(
            detected and common_miss
        )

    print("\n[시나리오별 Validation Recall]")

    for name, counts in sorted(groups.items()):
        total = counts["total"]
        detected = counts["detected"]

        print(
            f"{name:<23} "
            f"TP={detected:>3} "
            f"FN={total-detected:>3} "
            f"Recall={detected/total:.2%} "
            f"recovered={counts['recovered']}"
        )


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
        transactions, sequence_length=SEQ_LENGTH
    )

    split_info = split_and_purge(
        transactions, episodes
    )

    train_indices = split_info["indices"]["Train"]
    val_indices = split_info["indices"]["Validation"]

    labels = np.asarray([
        int(row["fraudLabel"])
        for row in transactions
    ], dtype=np.float32)

    (
        tabular,
        sequences,
        mask,
        preprocessing,
    ) = prepare_data(
        feature_rows, windows, train_indices
    )

    train_labels = labels[train_indices]
    val_labels = labels[val_indices].astype(int)

    positive_weight = (
        np.sum(train_labels == 0)
        / np.sum(train_labels == 1)
    )

    print("=== Transformer 실험 ===")
    print(f"Train: {len(train_indices):,}")
    print(f"Validation: {len(val_indices):,}")
    print(f"시퀀스 길이: {SEQ_LENGTH}")
    print(f"Tabular 입력 차원: {tabular.shape[1]}")
    print(f"Sequence 입력 차원: {sequences.shape[2]}")
    print(f"Positive weight: {positive_weight:.4f}")
    print("Device: CPU")

    val_loader = make_loader(
        tabular, sequences, mask,
        labels, val_indices,
    )

    rf_predictions = existing_model_predictions(
        "random_forest_baseline.joblib",
        feature_rows,
        transactions,
        val_indices,
        split_info,
    )

    xgb_predictions = existing_model_predictions(
        "xgboost_candidate.joblib",
        feature_rows,
        transactions,
        val_indices,
        split_info,
    )

    common_misses = (
        (val_labels == 1)
        & ~rf_predictions
        & ~xgb_predictions
    )

    print(
        "RF·XGB 공통 미탐:",
        int(common_misses.sum()),
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for name, use_sequence in (
        ("tabular_mlp", False),
        ("transformer", True),
    ):
        print(f"\n=== {name} 학습 ===")

        # 두 실험의 무작위 초기화와 셔플 재현성 확보
        set_seed()

        train_loader = make_loader(
            tabular, sequences, mask,
            labels, train_indices,
            shuffle=True,
        )

        model = FraudNeuralNetwork(
            tab_dim=tabular.shape[1],
            seq_dim=sequences.shape[2],
            use_sequence=use_sequence,
        )

        model = train_model(
            model,
            train_loader,
            val_loader,
            val_labels,
            positive_weight,
        )

        probabilities = predict_probabilities(
            model, val_loader
        )

        metrics = evaluate(
            val_labels, probabilities
        )

        predictions = metrics["predictions"]

        recovered = int((
            common_misses & (predictions == 1)
        ).sum())

        # 기존 두 모델이 모두 정상이라 판단한
        # 정상 거래 중 새로 오탐한 건수
        new_fp = int((
            (val_labels == 0)
            & ~rf_predictions
            & ~xgb_predictions
            & (predictions == 1)
        ).sum())

        print(f"\n[{name} Validation 결과]")

        for metric in (
            "threshold", "precision", "recall",
            "f1", "ap", "fpr",
        ):
            print(f"{metric}: {metrics[metric]:.4f}")

        print(
            f"TN={metrics['tn']} "
            f"FP={metrics['fp']} "
            f"FN={metrics['fn']} "
            f"TP={metrics['tp']}"
        )

        print("기존 공통 미탐 추가 탐지:", recovered)
        print("기존 OR 대비 추가 정상 오탐:", new_fp)

        print_scenario_results(
            transactions,
            val_indices,
            val_labels,
            predictions,
            rf_predictions,
            xgb_predictions,
        )

        results[name] = {
            "metrics": metrics,
            "recovered": recovered,
            "new_fp": new_fp,
        }

        torch.save({
            "state_dict": model.state_dict(),
            "tab_dim": tabular.shape[1],
            "seq_dim": sequences.shape[2],
            "use_sequence": use_sequence,
            "threshold": metrics["threshold"],
            "sequence_length": SEQ_LENGTH,
            "train_cut": split_info["train_cut"].isoformat(),
            "test_cut": split_info["test_cut"].isoformat(),
        }, MODEL_DIR / f"{name}_candidate.pt")

    joblib.dump(
        preprocessing,
        MODEL_DIR / "neural_preprocessing.joblib",
    )

    print("\n=== 최종 Validation 비교 ===")
    print(
        f"{'Model':<18}"
        f"{'F1':>9}"
        f"{'AP':>9}"
        f"{'Recall':>9}"
        f"{'FPR':>9}"
        f"{'Recovered':>12}"
    )

    for name, result in results.items():
        m = result["metrics"]

        print(
            f"{name:<18}"
            f"{m['f1']:>9.4f}"
            f"{m['ap']:>9.4f}"
            f"{m['recall']:>9.4f}"
            f"{m['fpr']:>9.4f}"
            f"{result['recovered']:>12}"
        )

    print("\n기존 RF Validation F1: 0.8290")
    print("기존 XGBoost Validation F1: 0.8272")
    print("Test 데이터: 사용하지 않음")
    print("학습 완료")


if __name__ == "__main__":
    main()
