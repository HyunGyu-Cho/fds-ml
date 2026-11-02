from __future__ import annotations

import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from features import FEATURE_COLUMNS, build_features
from rule_engine import ALWAYS_OFF_RULES, REVIEW_MIN_SCORE, DENY_MIN_SCORE, evaluate_rules
from train_random_forest import make_xy, read_csv, split_and_purge

# Rule · RF · XGBoost · 하이브리드를 같은 Validation · 같은 헛경보 수에서 비교하는 성적표

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_PATH = BASE_DIR / "reports" / "scorecard.md"

CONTROL = "XGBoost (30% 데이터)"

# 기존 실험 기준: Validation 정상 2,555건 중 50건까지 헛경보 허용
EXPERIMENT_FPR = 50 / 2555

SCENARIO_NAMES = {
    "CARD_TESTING": "카드 테스팅",
    "ACCOUNT_TAKEOVER": "계정 탈취",
    "STOLEN_CARD": "카드 도용",
    "LOW_AND_SLOW": "저속 반복",
    "SPLIT_PAYMENT": "분할 결제",
    "BEHAVIORAL_ANOMALY": "행동 이상",
    "CARD_DEVICE_CHANGE": "카드·기기 변경",
}


def check_boundaries(bundle, split_info):
    # 모델이 학습 때 쓴 분할 경계와 지금 분할이 같아야 같은 Validation이다
    if bundle["train_cut"] != split_info["train_cut"].isoformat():
        raise ValueError("Train 분할 경계가 모델과 다릅니다. 데이터를 다시 만들었는지 확인하세요.")
    if bundle["test_cut"] != split_info["test_cut"].isoformat():
        raise ValueError("Test 분할 경계가 모델과 다릅니다. 데이터를 다시 만들었는지 확인하세요.")


def tree_scores(filename, features, transactions, indices, split_info):
    bundle = joblib.load(MODEL_DIR / filename)
    check_boundaries(bundle, split_info)
    x, _ = make_xy(features, transactions, indices, bundle["feature_columns"])
    return bundle["pipeline"].predict_proba(x)[:, 1]


def hybrid_scores(features, transactions, indices, split_info):
    # 저장된 Encoder로 최근 5건 시퀀스 임베딩을 만들고, 13개 Feature와 합쳐 XGBoost가 채점한다
    import torch

    from sequences import build_sequences
    from train_hybrid_xgboost import add_embedding_features, extract_sequence_embeddings
    from train_transformer import FraudNeuralNetwork

    checkpoint = torch.load(MODEL_DIR / "hybrid_encoder_candidate.pt", map_location="cpu", weights_only=True)
    preprocessing = joblib.load(MODEL_DIR / "hybrid_preprocessing.joblib")
    bundle = joblib.load(MODEL_DIR / "hybrid_xgboost_candidate.joblib")
    check_boundaries(bundle, split_info)

    length = checkpoint["sequence_length"]
    all_windows = build_sequences(transactions, sequence_length=length)
    windows = [all_windows[i] for i in indices]
    tokens = [token for window in windows for token in window.tokens]
    seq_raw = preprocessing["seq_vectorizer"].transform(tokens)
    seq = preprocessing["seq_scaler"].transform(seq_raw).reshape(len(indices), length, -1).astype(np.float32)
    mask = np.asarray([window.mask for window in windows], dtype=bool)
    seq[~mask] = 0.0

    encoder = FraudNeuralNetwork(tab_dim=checkpoint["tab_dim"], seq_dim=checkpoint["seq_dim"], use_sequence=True)
    encoder.load_state_dict(checkpoint["state_dict"])
    with torch.inference_mode():
        embeddings = extract_sequence_embeddings(encoder, seq, mask, np.arange(len(indices)))

    base_x, _ = make_xy(features, transactions, indices, bundle["feature_columns"])
    x = add_embedding_features(base_x, embeddings)
    return bundle["pipeline"].predict_proba(x)[:, 1]


def same_data_xgb_scores(features, transactions, indices, split_info):
    # 대조군: 하이브리드의 XGBoost와 같은 데이터(Train 마지막 약 30%) · 같은 설정, 순서 임베딩 없이 13개 Feature만
    from train_hybrid_xgboost import split_inner_train
    from train_xgboost import create_pipeline

    inner = split_inner_train(transactions, split_info["indices"]["Train"], split_info["episode_by_payment"])
    train_x, train_y = make_xy(features, transactions, inner["groups"]["XGBTrain"], FEATURE_COLUMNS)
    pipeline = create_pipeline(1.0)
    pipeline.fit(train_x, train_y)
    x, _ = make_xy(features, transactions, indices, FEATURE_COLUMNS)
    return pipeline.predict_proba(x)[:, 1]


def predictions_with_fp_budget(y, scores, fp_budget):
    # 정상 거래 헛경보가 fp_budget건을 넘지 않는 가장 낮은 기준값 (경계와 같은 점수는 경보에서 뺀다)
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    normal_scores = np.sort(scores[y == 0])[::-1]
    if fp_budget >= len(normal_scores):
        return np.ones_like(y, dtype=bool)
    predictions = scores > normal_scores[fp_budget]
    if int(((y == 0) & predictions).sum()) > fp_budget:
        raise RuntimeError("헛경보 예산 초과")
    return predictions


def load_model_scores(features, transactions, indices, split_info):
    # 파일 · 라이브러리가 없는 모델은 건너뛰고 이유를 남긴다
    loaders = {
        "RF": lambda: tree_scores("random_forest_baseline.joblib", features, transactions, indices, split_info),
        "RF v2": lambda: tree_scores("random_forest_v2.joblib", features, transactions, indices, split_info),
        "XGBoost": lambda: tree_scores("xgboost_candidate.joblib", features, transactions, indices, split_info),
        CONTROL: lambda: same_data_xgb_scores(features, transactions, indices, split_info),
        "하이브리드": lambda: hybrid_scores(features, transactions, indices, split_info),
    }
    scores, skipped = {}, {}
    for name, load in loaders.items():
        try:
            scores[name] = load()
        except (FileNotFoundError, ModuleNotFoundError) as error:
            skipped[name] = f"{type(error).__name__}: {error}"
    return scores, skipped


def pad(text, width):
    # 한글은 화면에서 두 칸을 차지하므로 그만큼 덜 채운다
    shown = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - shown)


def count_bar(count, total, width=30):
    filled = round(width * count / total) if total else 0
    return "█" * filled


def build_report(transactions, indices, y, rule_results, model_scores, skipped):
    rule_scores = np.asarray([rule_results[i].score for i in indices], dtype=np.float64)
    scenarios = [transactions[i]["scenarioType"] for i in indices]
    fraud_total = int((y == 1).sum())
    normal_total = int((y == 0).sum())
    lines = []

    lines.append("# FDS 모델 성적표 — Rule · RF · XGBoost · 하이브리드")
    lines.append("")
    lines.append(f"- 만든 시각: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · `python scorecard.py`")
    lines.append(f"- 문제지: 합성 데이터 Validation {len(indices):,}건 (정상 {normal_total:,} · 사기 {fraud_total:,}), 모든 모델이 학습에 쓰지 않은 같은 구간")
    lines.append("- 비교 방법: 정상 손님을 의심한 횟수(헛경보)를 같게 맞춘 뒤, 사기를 몇 건 잡았는지 비교")
    lines.append(f"- {CONTROL}: 하이브리드는 Train 앞 70%를 순서 학습기(Transformer)에 쓰고 마지막 약 30%로만 XGBoost를 배운다. 같은 30% · 같은 설정으로 순서 정보 없이 배운 대조군이라, 하이브리드와의 차이 = 순서 정보(임베딩)의 효과")
    lines.append(f"- Rule은 팀 Rule Engine(R01~R08)을 Python으로 옮긴 `rule_engine.py`. {' · '.join(ALWAYS_OFF_RULES)}는 합성 데이터에 IP · 차단 목록이 없어 항상 꺼짐")
    if skipped:
        for name, reason in skipped.items():
            lines.append(f"- 건너뜀: {name} — {reason}")
    lines.append("- 합성 데이터 결과라 실제 결제 성능의 근거는 아님")
    lines.append("")

    rule_review = rule_scores >= REVIEW_MIN_SCORE
    rule_deny = rule_scores >= DENY_MIN_SCORE
    experiment_budget = int(np.floor(EXPERIMENT_FPR * normal_total + 1e-9))
    points = [
        ("기준 ① Rule 보류(REVIEW 이상, 50점+)와 같은 헛경보", rule_review),
        ("기준 ② Rule 거절(DENY, 80점+)과 같은 헛경보", rule_deny),
        ("기준 ③ 기존 실험 기준 (정상 2,555건 중 50건)", None),
    ]

    for title, rule_predictions in points:
        if rule_predictions is None:
            budget = experiment_budget
            rule_predictions = predictions_with_fp_budget(y, rule_scores, budget)
            rule_label = "Rule (점수 기준값 조정)"
        else:
            budget = int(((y == 0) & rule_predictions).sum())
            rule_label = "Rule"
        predictions = {rule_label: rule_predictions}
        for name, scores in model_scores.items():
            predictions[name] = predictions_with_fp_budget(y, scores, budget)

        lines.append(f"## {title} — 정상 {budget}건 의심까지 허용")
        lines.append("")
        lines.append("```text")
        for name, pred in predictions.items():
            tp = int(((y == 1) & pred).sum())
            fp = int(((y == 0) & pred).sum())
            lines.append(f"{pad(name, 26)} {pad(count_bar(tp, fraud_total), 30)} 사기 {tp:>3}/{fraud_total} ({tp / fraud_total:6.1%}) · 헛경보 {fp}")
        lines.append("```")
        lines.append("")
        if rule_label != "Rule":
            lines.append("- Rule 점수는 0 · 25 · 30 … 처럼 띄엄띄엄이라 헛경보를 정확히 맞출 수 없어, 허용 수를 넘지 않는 가장 가까운 기준값을 썼다")

        rule_miss = (y == 1) & ~rule_predictions
        for name in model_scores:
            if name == CONTROL:
                continue
            caught = int((rule_miss & predictions[name]).sum())
            lines.append(f"- Rule이 놓친 사기 {int(rule_miss.sum())}건 중 {name}가 잡은 것: {caught}건")
        if {"RF", "XGBoost", "하이브리드"} <= set(model_scores):
            common_miss = (y == 1) & ~predictions["RF"] & ~predictions["XGBoost"]
            recovered = int((common_miss & predictions["하이브리드"]).sum())
            extra_fp = int(((y == 0) & ~predictions["RF"] & ~predictions["XGBoost"] & predictions["하이브리드"]).sum())
            lines.append(f"- RF · XGBoost가 둘 다 놓친 사기 {int(common_miss.sum())}건 중 하이브리드만 잡은 것: {recovered}건 (대신 하이브리드만 의심한 정상 {extra_fp}건)")
        if {CONTROL, "하이브리드"} <= set(model_scores):
            control_tp = int(((y == 1) & predictions[CONTROL]).sum())
            hybrid_tp = int(((y == 1) & predictions["하이브리드"]).sum())
            lines.append(f"- 같은 30% 데이터에서 순서 정보(임베딩)를 더한 효과: {CONTROL} {control_tp}건 → 하이브리드 {hybrid_tp}건 ({hybrid_tp - control_tp:+d}건)")
        lines.append("")

        lines.append("| 사기 시나리오 | 건수 | " + " | ".join(predictions) + " |")
        lines.append("|---|---:|" + "---:|" * len(predictions))
        fraud_counts = Counter(s for s, label in zip(scenarios, y) if label == 1)
        for scenario, korean in SCENARIO_NAMES.items():
            total = fraud_counts.get(scenario, 0)
            if total == 0:
                continue
            row_mask = np.asarray([s == scenario for s in scenarios]) & (y == 1)
            cells = [f"{int((row_mask & pred).sum())} ({int((row_mask & pred).sum()) / total:.0%})" for pred in predictions.values()]
            lines.append(f"| {korean} | {total} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## 참고 — 기준값과 무관한 순위 성능 (AP, 1에 가까울수록 좋음)")
    lines.append("")
    lines.append(f"- Rule: {average_precision_score(y, rule_scores):.4f} (점수가 0 · 25 · 30 … 처럼 띄엄띄엄이라 불리함)")
    for name, scores in model_scores.items():
        lines.append(f"- {name}: {average_precision_score(y, scores):.4f}")
    lines.append("")

    lines.append("## 참고 — Validation에서 Rule이 걸린 횟수")
    lines.append("")
    lines.append("| Rule | 사기에서 | 정상에서 |")
    lines.append("|---|---:|---:|")
    hit_counts = {label: Counter() for label in (0, 1)}
    for i, label in zip(indices, y):
        hit_counts[int(label)].update(rule_results[i].hits)
    for rule in ("R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08"):
        lines.append(f"| {rule} | {hit_counts[1][rule]} | {hit_counts[0][rule]} |")
    lines.append("")
    return "\n".join(lines)


def main():
    transactions = read_csv(DATA_DIR / "transactions.csv")
    episodes = read_csv(DATA_DIR / "fraud_episodes.csv")
    features = build_features(transactions)
    split_info = split_and_purge(transactions, episodes)
    indices = split_info["indices"]["Validation"]
    y = np.asarray([int(transactions[i]["fraudLabel"]) for i in indices])

    rule_results = evaluate_rules(transactions)
    model_scores, skipped = load_model_scores(features, transactions, indices, split_info)
    report = build_report(transactions, indices, y, rule_results, model_scores, skipped)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n저장: {REPORT_PATH}")


if __name__ == "__main__":
    main()
