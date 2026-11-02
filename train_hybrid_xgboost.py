
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

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
    parse_time,
    split_and_purge,
    make_xy,
    choose_threshold,
)
from train_transformer import (
    FraudNeuralNetwork,
    prepare_data,
    make_loader,
    train_model,
    set_seed,
)
from train_xgboost import create_pipeline


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

SEED = 42
SEQ_LENGTH = 5
TARGET_FPR = 50 / 2555

INNER_ENCODER_FRACTION = 0.55
INNER_EARLYSTOP_FRACTION = 0.15


def split_inner_train(
    transactions,
    outer_train_indices,
    episode_by_payment,
):
    """
    기존 Train을 시간순으로 3개 구간으로 분리한다.

    EncoderTrain: 앞의 약 55%
    EncoderEarlyStop: 다음 약 15%
    XGBTrain: 마지막 약 30%

    내부 경계를 넘는 공격 Episode는 모든 구간에서 제외한다.
    """

    ordered = sorted(
        outer_train_indices,
        key=lambda i: parse_time(
            transactions[i]["paymentRequestedAt"]
        ),
    )

    total = len(ordered)

    if total < 10:
        raise ValueError("내부 분할에 필요한 거래가 부족합니다.")

    timestamps = [
        parse_time(transactions[i]["paymentRequestedAt"])
        for i in ordered
    ]

    first_cut = timestamps[
        int(total * INNER_ENCODER_FRACTION)
    ]

    second_cut = timestamps[
        int(
            total * (
                INNER_ENCODER_FRACTION
                + INNER_EARLYSTOP_FRACTION
            )
        )
    ]

    if first_cut >= second_cut:
        raise ValueError("내부 시간 경계가 겹칩니다.")

    groups = {
        "EncoderTrain": [],
        "EncoderEarlyStop": [],
        "XGBTrain": [],
    }

    episode_groups = defaultdict(set)
    episode_payments = defaultdict(set)

    for index in ordered:
        row = transactions[index]
        timestamp = parse_time(row["paymentRequestedAt"])

        if timestamp < first_cut:
            group_name = "EncoderTrain"
        elif timestamp < second_cut:
            group_name = "EncoderEarlyStop"
        else:
            group_name = "XGBTrain"

        groups[group_name].append(index)

        payment_id = row["paymentId"]
        episode_id = episode_by_payment.get(payment_id)

        if episode_id is not None:
            episode_groups[episode_id].add(group_name)
            episode_payments[episode_id].add(payment_id)

    crossing_episodes = {
        episode_id
        for episode_id, names in episode_groups.items()
        if len(names) > 1
    }

    purged_payments = set()

    for episode_id in crossing_episodes:
        purged_payments.update(
            episode_payments[episode_id]
        )

    for name in groups:
        groups[name] = [
            i
            for i in groups[name]
            if transactions[i]["paymentId"]
            not in purged_payments
        ]

        labels = {
            int(transactions[i]["fraudLabel"])
            for i in groups[name]
        }

        if labels != {0, 1}:
            raise ValueError(
                f"{name}에 정상/사기 클래스가 모두 필요합니다."
            )

    # 내부 Episode 누출이 남지 않았는지 재검증
    remaining_episodes = defaultdict(set)

    for name, indices in groups.items():
        for index in indices:
            payment_id = transactions[index]["paymentId"]
            episode_id = episode_by_payment.get(payment_id)

            if episode_id is not None:
                remaining_episodes[episode_id].add(name)

    if any(
        len(names) > 1
        for names in remaining_episodes.values()
    ):
        raise RuntimeError("내부 Episode 누출이 발생했습니다.")

    return {
        "groups": groups,
        "first_cut": first_cut,
        "second_cut": second_cut,
        "crossing_episodes": crossing_episodes,
        "purged_payments": purged_payments,
    }


@torch.inference_mode()
def extract_sequence_embeddings(
    model,
    sequences,
    mask,
    indices,
    batch_size=256,
):
    """
    학습된 Transformer Encoder의 CLS 임베딩 추출.

    분류기의 마지막 출력이 아니라,
    Encoder가 생성한 32차원 표현을 반환한다.

    모든 구간에서 같은 Encoder 가중치를 사용한다.
    """

    model.eval()
    outputs = []

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[
            start:start + batch_size
        ]

        seq = torch.from_numpy(
            sequences[batch_indices]
        )

        valid_mask = torch.from_numpy(
            mask[batch_indices]
        )

        batch_size_actual = len(batch_indices)

        tokens = model.input_projection(seq)

        cls = model.cls_token.expand(
            batch_size_actual, -1, -1
        )

        tokens = torch.cat(
            [cls, tokens],
            dim=1,
        )

        positions = torch.arange(
            tokens.shape[1],
            device=tokens.device,
        )

        tokens = tokens + model.position_embedding(
            positions
        ).unsqueeze(0)

        cls_mask = torch.zeros(
            (batch_size_actual, 1),
            dtype=torch.bool,
        )

        padding_mask = torch.cat(
            [cls_mask, ~valid_mask],
            dim=1,
        )

        encoded = model.encoder(
            tokens,
            src_key_padding_mask=padding_mask,
        )

        outputs.append(
            encoded[:, 0, :].cpu().numpy()
        )

    if not outputs:
        raise ValueError("임베딩 대상 거래가 없습니다.")

    embeddings = np.concatenate(outputs, axis=0)

    if not np.isfinite(embeddings).all():
        raise ValueError("임베딩에 비정상 값이 존재합니다.")

    return embeddings


def add_embedding_features(base_rows, embeddings):
    """기존 Feature 딕셔너리에 32차원 임베딩 추가."""

    if len(base_rows) != len(embeddings):
        raise ValueError("거래 수와 임베딩 수가 다릅니다.")

    results = []

    for row, vector in zip(base_rows, embeddings):
        combined = dict(row)

        for dimension, value in enumerate(vector):
            combined[
                f"seq_emb_{dimension:02d}"
            ] = float(value)

        results.append(combined)

    return results


def fixed_fpr_predictions(y, scores, target_fpr):
    """
    Validation FPR 상한으로 Threshold를 결정한다.

    경계 동점 때문에 FP 예산을 초과하지 않도록
    경계보다 높은 점수만 양성으로 판정한다.
    """

    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)

    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr 범위가 잘못되었습니다.")

    if not np.isfinite(scores).all():
        raise ValueError("예측 점수에 NaN/Inf가 있습니다.")

    normal_scores = np.sort(
        scores[y == 0]
    )[::-1]

    if len(normal_scores) == 0:
        raise ValueError("Validation 정상 거래가 없습니다.")

    allowed_fp = math.floor(
        target_fpr * len(normal_scores) + 1e-9
    )

    threshold = float(normal_scores[allowed_fp])
    predictions = scores > threshold

    actual_fp = int(
        ((y == 0) & predictions).sum()
    )

    if actual_fp > allowed_fp:
        raise RuntimeError("FPR 예산 초과")

    return threshold, predictions


def calculate_metrics(y, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def evaluate_candidate(name, pipeline, x_train, y_train,
                       x_val, y_val):
    """XGBoost 학습 후 Validation 성능 평가."""

    pipeline.fit(x_train, y_train)

    scores = pipeline.predict_proba(
        x_val
    )[:, 1]

    ap = average_precision_score(
        y_val, scores
    )

    # 기존 모델과 같은 Validation F1 Threshold 탐색
    best_threshold, _ = choose_threshold(
        y_val, scores
    )

    best_predictions = (
        scores >= best_threshold
    )

    best_metrics = calculate_metrics(
        y_val, best_predictions
    )

    # 동일 FPR에서 비교
    fixed_threshold, fixed_predictions = (
        fixed_fpr_predictions(
            y_val, scores, TARGET_FPR
        )
    )

    fixed_metrics = calculate_metrics(
        y_val, fixed_predictions
    )

    print(f"\n[{name}]")

    print(
        f"Validation AP: {ap:.4f}"
    )

    print(
        f"Best F1={best_metrics['f1']:.4f}, "
        f"Recall={best_metrics['recall']:.4f}, "
        f"FPR={best_metrics['fpr']:.4f}, "
        f"Threshold={best_threshold:.4f}"
    )

    print(
        f"Fixed-FPR F1={fixed_metrics['f1']:.4f}, "
        f"Recall={fixed_metrics['recall']:.4f}, "
        f"FPR={fixed_metrics['fpr']:.4f}, "
        f"TP={fixed_metrics['tp']}, "
        f"FP={fixed_metrics['fp']}, "
        f"Threshold={fixed_threshold:.6f}"
    )

    return {
        "name": name,
        "pipeline": pipeline,
        "scores": scores,
        "ap": ap,
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
        "fixed_threshold": fixed_threshold,
        "fixed_predictions": fixed_predictions,
        "fixed_metrics": fixed_metrics,
    }


def report_scenario_recall(
    transactions,
    indices,
    y,
    baseline_predictions,
    hybrid_predictions,
):
    groups = defaultdict(
        lambda: {
            "total": 0,
            "baseline": 0,
            "hybrid": 0,
        }
    )

    for position, index in enumerate(indices):
        if y[position] != 1:
            continue

        scenario = transactions[index]["scenarioType"]
        counts = groups[scenario]

        counts["total"] += 1
        counts["baseline"] += int(
            baseline_predictions[position]
        )
        counts["hybrid"] += int(
            hybrid_predictions[position]
        )

    print("\n=== 시나리오별 Recall (동일 FPR 기준) ===")

    for scenario, counts in sorted(groups.items()):
        total = counts["total"]

        baseline_recall = counts["baseline"] / total
        hybrid_recall = counts["hybrid"] / total

        print(
            f"{scenario:<23} "
            f"XGB={baseline_recall:.2%} "
            f"Hybrid={hybrid_recall:.2%} "
            f"차이={hybrid_recall-baseline_recall:+.2%}"
        )


def compare_with_existing_models(
    feature_rows,
    transactions,
    val_indices,
    val_y,
    outer_split,
    hybrid_predictions,
):
    """
    기존 전체 Train RF·XGB와 비교한다.

    모델 선정에는 사용하지 않고,
    공통 미탐의 추가 탐지 여부만 확인한다.
    """

    predictions = {}

    for name, filename in (
        ("RF", "random_forest_baseline.joblib"),
        ("XGB", "xgboost_candidate.joblib"),
    ):
        bundle = joblib.load(
            MODEL_DIR / filename
        )

        if (
            bundle["train_cut"]
            != outer_split["train_cut"].isoformat()
            or bundle["test_cut"]
            != outer_split["test_cut"].isoformat()
        ):
            raise ValueError(
                f"{name} 모델의 분할 경계가 다릅니다."
            )

        x, _ = make_xy(
            feature_rows,
            transactions,
            val_indices,
            bundle["feature_columns"],
        )

        scores = bundle["pipeline"].predict_proba(
            x
        )[:, 1]

        _, pred = fixed_fpr_predictions(
            val_y, scores, TARGET_FPR
        )

        predictions[name] = pred

    rf = predictions["RF"]
    xgb = predictions["XGB"]

    common_misses = (
        (val_y == 1) & ~rf & ~xgb
    )

    recovered = (
        common_misses & hybrid_predictions
    )

    new_false_positives = (
        (val_y == 0)
        & ~rf
        & ~xgb
        & hybrid_predictions
    )

    print("\n=== 기존 RF·XGB 공통 미탐 복구 ===")
    print("공통 미탐:", int(common_misses.sum()))
    print("Hybrid 추가 탐지:", int(recovered.sum()))
    print(
        "추가 정상 오탐:",
        int(new_false_positives.sum()),
    )

    union = rf | xgb | hybrid_predictions

    metrics = calculate_metrics(
        val_y, union
    )

    print("\n[RF + XGB + Hybrid OR 참고 결과]")
    print(
        f"Recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}, "
        f"FPR={metrics['fpr']:.4f}, "
        f"TP={metrics['tp']}, "
        f"FP={metrics['fp']}"
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
        transactions,
        sequence_length=SEQ_LENGTH,
    )

    outer_split = split_and_purge(
        transactions, episodes
    )

    inner = split_inner_train(
        transactions,
        outer_split["indices"]["Train"],
        outer_split["episode_by_payment"],
    )

    groups = inner["groups"]

    encoder_train = groups["EncoderTrain"]
    encoder_val = groups["EncoderEarlyStop"]
    xgb_train = groups["XGBTrain"]
    val_indices = outer_split["indices"]["Validation"]

    labels = np.asarray(
        [int(row["fraudLabel"]) for row in transactions],
        dtype=np.float32,
    )

    print("=== Transformer Embedding + XGBoost ===")

    for name, indices in groups.items():
        counts = Counter(
            int(labels[i]) for i in indices
        )

        print(
            f"{name}: {len(indices):,}건 "
            f"(정상={counts[0]:,}, 사기={counts[1]:,})"
        )

    print(f"Validation: {len(val_indices):,}건")
    print(
        "내부 경계 충돌 Episode:",
        len(inner["crossing_episodes"]),
    )
    print(
        "내부 제외 거래:",
        len(inner["purged_payments"]),
    )
    print("Test 사용: 없음")

    # 전처리 fit은 반드시 EncoderTrain에서만 수행
    (
        tabular,
        sequences,
        mask,
        preprocessing,
    ) = prepare_data(
        feature_rows,
        windows,
        encoder_train,
    )

    encoder_train_labels = labels[encoder_train]
    encoder_val_labels = labels[encoder_val].astype(int)
    val_y = labels[val_indices].astype(int)

    positive_weight = (
        np.sum(encoder_train_labels == 0)
        / np.sum(encoder_train_labels == 1)
    )

    train_loader = make_loader(
        tabular,
        sequences,
        mask,
        labels,
        encoder_train,
        shuffle=True,
    )

    encoder_val_loader = make_loader(
        tabular,
        sequences,
        mask,
        labels,
        encoder_val,
    )

    encoder = FraudNeuralNetwork(
        tab_dim=tabular.shape[1],
        seq_dim=sequences.shape[2],
        use_sequence=True,
    )

    print("\n=== 1단계: Transformer Encoder 학습 ===")

    encoder = train_model(
        encoder,
        train_loader,
        encoder_val_loader,
        encoder_val_labels,
        positive_weight,
    )

    # 여기서부터 Encoder를 고정한다.
    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    print("\n=== 2단계: 동일 Encoder 임베딩 생성 ===")

    xgb_embeddings = extract_sequence_embeddings(
        encoder,
        sequences,
        mask,
        xgb_train,
    )

    val_embeddings = extract_sequence_embeddings(
        encoder,
        sequences,
        mask,
        val_indices,
    )

    print("XGB Train Embedding:", xgb_embeddings.shape)
    print("Validation Embedding:", val_embeddings.shape)

    # XGBoost 학습에 사용하는 것은 마지막 Train 30%뿐
    xgb_base_x, xgb_y = make_xy(
        feature_rows,
        transactions,
        xgb_train,
        FEATURE_COLUMNS,
    )

    val_base_x, _ = make_xy(
        feature_rows,
        transactions,
        val_indices,
        FEATURE_COLUMNS,
    )

    xgb_hybrid_x = add_embedding_features(
        xgb_base_x,
        xgb_embeddings,
    )

    val_hybrid_x = add_embedding_features(
        val_base_x,
        val_embeddings,
    )

    print("\n=== 3단계: XGBoost 대조 실험 ===")

    # 두 후보의 데이터·파라미터·가중치 설정 동일
    baseline = evaluate_candidate(
        "XGB (동일 Train 부분집합)",
        create_pipeline(1.0),
        xgb_base_x,
        xgb_y,
        val_base_x,
        val_y,
    )

    hybrid = evaluate_candidate(
        "Hybrid (Feature + Embedding)",
        create_pipeline(1.0),
        xgb_hybrid_x,
        xgb_y,
        val_hybrid_x,
        val_y,
    )

    print("\n=== 최종 Validation 비교 ===")
    print(
        f"{'Model':<15}"
        f"{'AP':>9}"
        f"{'F1':>9}"
        f"{'Recall@FPR':>14}"
        f"{'FPR':>10}"
    )

    for name, result in (
        ("Baseline", baseline),
        ("Hybrid", hybrid),
    ):
        fixed = result["fixed_metrics"]

        print(
            f"{name:<15}"
            f"{result['ap']:>9.4f}"
            f"{result['best_metrics']['f1']:>9.4f}"
            f"{fixed['recall']:>14.4f}"
            f"{fixed['fpr']:>10.4f}"
        )

    print("\n[기존 전체 Train XGBoost 참고 기준]")
    print("Validation AP=0.9070")
    print("Validation F1=0.8272")
    print("동일 FPR Recall=0.7833")

    report_scenario_recall(
        transactions,
        val_indices,
        val_y,
        baseline["fixed_predictions"],
        hybrid["fixed_predictions"],
    )

    compare_with_existing_models(
        feature_rows,
        transactions,
        val_indices,
        val_y,
        outer_split,
        hybrid["fixed_predictions"],
    )

    # 실험 결과 재현을 위한 모델 저장
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict": encoder.state_dict(),
        "tab_dim": tabular.shape[1],
        "seq_dim": sequences.shape[2],
        "sequence_length": SEQ_LENGTH,
        "encoder_train_cut": inner["first_cut"].isoformat(),
        "xgb_train_cut": inner["second_cut"].isoformat(),
    }, MODEL_DIR / "hybrid_encoder_candidate.pt")

    joblib.dump(
        preprocessing,
        MODEL_DIR / "hybrid_preprocessing.joblib",
    )

    joblib.dump({
        "pipeline": hybrid["pipeline"],
        "threshold": hybrid["best_threshold"],
        "fixed_fpr_threshold": hybrid["fixed_threshold"],
        "feature_columns": list(FEATURE_COLUMNS),
        "embedding_dim": xgb_embeddings.shape[1],
        "train_cut": outer_split["train_cut"].isoformat(),
        "test_cut": outer_split["test_cut"].isoformat(),
        "random_state": SEED,
    }, MODEL_DIR / "hybrid_xgboost_candidate.joblib")

    print("\n=== 저장 완료 ===")
    print("hybrid_encoder_candidate.pt")
    print("hybrid_preprocessing.joblib")
    print("hybrid_xgboost_candidate.joblib")

    print("\nTest 평가: 수행하지 않음")
    print("기존 모델 파일: 변경 없음")


if __name__ == "__main__":
    main()
