from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from features import _parse_timestamp, build_features
from rule_engine import DENY_MIN_SCORE, REVIEW_MIN_SCORE, evaluate_rules
from scorecard import EXPERIMENT_FPR, pad, predictions_with_fp_budget
from train_random_forest import choose_threshold, create_pipeline, make_xy, read_csv, split_and_purge

# 외부 데이터(IBM TabFormer, Apache 2.0)로 같은 설계를 다시 시험한다
# Rule · RF(TabFormer로 학습, rf-v2와 같은 칸 · 설정) · rf-v2(합성 데이터로 학습한 그대로)를 같은 헛경보 수에서 비교

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "tabformer"
MODEL_DIR = BASE_DIR / "models"
REPORT_PATH = BASE_DIR / "reports" / "tabformer_scorecard.md"
ONLINE = "Online Transaction"
LOCATION_FEATURE = "locationSeenBefore"


def location_seen_before(transactions):
    # 이 손님이 이 결제보다 앞서 같은 장소에서 결제한 적 있으면 1 — 같은 시각 결제끼리는 서로 안 본다 (features.py와 같은 규칙)
    # features.py는 팀 ai-server와 같아야 해서 건드리지 않고, 실험 전용으로 따로 계산한다
    order = sorted(range(len(transactions)), key=lambda i: _parse_timestamp(transactions[i]["paymentRequestedAt"]))
    seen = defaultdict(set)
    result = [0] * len(transactions)
    for _, group in groupby(order, key=lambda i: _parse_timestamp(transactions[i]["paymentRequestedAt"])):
        group = list(group)
        for i in group:
            location = transactions[i].get("locationRef") or ""
            result[i] = int(bool(location) and location in seen[transactions[i]["customerRef"]])
        for i in group:
            location = transactions[i].get("locationRef") or ""
            if location:
                seen[transactions[i]["customerRef"]].add(location)
    return result


def train_tabformer_rf(features, transactions, split_info, columns):
    # rf-v2가 고른 설정(가맹점 번호 뺀 12칸 · class_weight None) 하나만 학습한다 — 거래가 많아 후보 4개는 오래 걸림
    train_x, train_y = make_xy(features, transactions, split_info["indices"]["Train"], columns)
    pipeline = create_pipeline(None)
    started = perf_counter()
    pipeline.fit(train_x, train_y)
    return pipeline, perf_counter() - started


def caught_line(name, y, prediction, total):
    tp = int(((y == 1) & prediction).sum())
    fp = int(((y == 0) & prediction).sum())
    return f"{pad(name, 30)} 사기 {tp:>4}/{total} ({tp / total:6.1%}) · 헛경보 {fp:,}"


def section(lines, title, y, rule_scores, model_scores):
    # 기준 ① Rule 보류와 같은 헛경보 ② Rule 거절과 같은 헛경보 ③ 정상의 약 2% (합성 성적표 ③과 같은 비율)
    total = int((y == 1).sum())
    normal = int((y == 0).sum())
    # ①② Rule은 실제 판정(50점 · 80점 이상) 그대로, ③ Rule은 점수 기준값을 헛경보 수에 맞춰 조정
    points = [
        ("기준 ① Rule 보류(50점+)와 같은 헛경보", "Rule (50점 이상)", rule_scores >= REVIEW_MIN_SCORE),
        ("기준 ② Rule 거절(80점+)과 같은 헛경보", "Rule (80점 이상)", rule_scores >= DENY_MIN_SCORE),
        (f"기준 ③ 정상 {EXPERIMENT_FPR:.2%}까지 헛경보 허용", "Rule (점수 기준값 조정)", None),
    ]
    lines.append(f"## {title} — 거래 {len(y):,}건 (정상 {normal:,} · 사기 {total:,})")
    lines.append("")
    for name, rule_label, rule_prediction in points:
        if rule_prediction is None:
            budget = int(np.floor(EXPERIMENT_FPR * normal + 1e-9))
            rule_prediction = predictions_with_fp_budget(y, rule_scores, budget)
        else:
            budget = int(((y == 0) & rule_prediction).sum())
        lines.append(f"{name} — 정상 {budget:,}건 의심까지 허용")
        lines.append("")
        lines.append("```text")
        lines.append(caught_line(rule_label, y, rule_prediction, total))
        for model, scores in model_scores.items():
            lines.append(caught_line(model, y, predictions_with_fp_budget(y, scores, budget), total))
        lines.append("```")
        lines.append("")
    lines.append("AP (기준값과 무관한 순위 성능, 1에 가까울수록 좋음, 사기 비율이 낮아 값 자체가 작게 나옴)")
    lines.append("")
    lines.append(f"- Rule: {average_precision_score(y, rule_scores):.4f}")
    for model, scores in model_scores.items():
        lines.append(f"- {model}: {average_precision_score(y, scores):.4f}")
    lines.append(f"- 무작위로 고르면: {total / len(y):.4f}")
    lines.append("")


def main():
    transactions = read_csv(DATA_DIR / "transactions.csv")
    episodes = read_csv(DATA_DIR / "fraud_episodes.csv")
    print(f"거래 {len(transactions):,}건 읽음 — Feature 계산 중")
    features = build_features(transactions)
    split_info = split_and_purge(transactions, episodes)
    val = split_info["indices"]["Validation"]
    y = np.asarray([int(transactions[i]["fraudLabel"]) for i in val])

    print("Rule 계산 중")
    rule_results = evaluate_rules(transactions)
    rule_scores = np.asarray([rule_results[i].score for i in val], dtype=np.float64)

    v2 = joblib.load(MODEL_DIR / "random_forest_v2.joblib")
    columns = list(v2["feature_columns"])
    print(f"RF 학습 중 (Train {len(split_info['indices']['Train']):,}건)")
    pipeline, seconds = train_tabformer_rf(features, transactions, split_info, columns)

    val_x, _ = make_xy(features, transactions, val, columns)
    tabformer_scores = pipeline.predict_proba(val_x)[:, 1]
    threshold, val_f1 = choose_threshold(list(y), tabformer_scores)
    v2_scores = v2["pipeline"].predict_proba(val_x)[:, 1]

    # 확인 실험: 같은 RF에 "전에 이 장소에서 결제한 적 있나" 한 칸만 더한다
    for row, seen in zip(features, location_seen_before(transactions)):
        row[LOCATION_FEATURE] = seen
    location_columns = [*columns, LOCATION_FEATURE]
    print("RF + 지역 칸 학습 중")
    location_pipeline, location_seconds = train_tabformer_rf(features, transactions, split_info, location_columns)
    location_x, _ = make_xy(features, transactions, val, location_columns)
    location_scores = location_pipeline.predict_proba(location_x)[:, 1]

    model_scores = {
        "RF (TabFormer로 학습)": tabformer_scores,
        "RF + 지역 칸": location_scores,
        "rf-v2 (합성 학습 그대로)": v2_scores,
    }

    lines = []
    lines.append("# 외부 데이터 성적표 — IBM TabFormer로 같은 설계 다시 시험")
    lines.append("")
    lines.append(f"- 만든 시각: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · `python tabformer_scorecard.py`")
    lines.append("- 데이터 출처: IBM TabFormer 합성 카드 거래 (https://github.com/IBM/TabFormer, Apache 2.0). IBM이 이 실험을 검토 · 보증한 것은 아님. 데이터는 이 저장소에 넣지 않음")
    lines.append("- 변환: `tabformer_convert.py` (고객 단위 표본 · 금액 달러 × 환율 · 환불 제외 · 오류 있으면 거절 · 기기 정보 없음)")
    lines.append(f"- 나누기: 시간순 Train {len(split_info['indices']['Train']):,} · Validation {len(val):,} · Test {len(split_info['indices']['Test']):,} (경계를 넘는 공격 묶음 {len(split_info['crossing_episodes'])}개 제외). 아래 숫자는 모두 Validation")
    lines.append(f"- RF (TabFormer로 학습): rf-v2와 같은 칸 {len(columns)}개 · 같은 설정, 학습 {seconds:.0f}초, Validation F1 최고 기준값 {threshold:.2f} (F1 {val_f1:.4f})")
    lines.append(f"- RF + 지역 칸: 위 RF에 \"이 손님이 앞서 같은 장소(가게가 있는 주 · 나라, 온라인은 ONLINE)에서 결제한 적 있나\" 한 칸만 더함 (학습 {location_seconds:.0f}초). 돈워리의 \"평소와 다른 접속 국가\"를 대신하는 확인 실험 — 원인이 장소 정보 부족인지 확인")
    lines.append("- rf-v2 (합성 학습 그대로): 팀 ai-server 모델과 같은 파일을 다시 학습하지 않고 채점 — 처음 보는 데이터에서도 통하는지")
    lines.append("- Rule: `rule_engine.py` (팀 R01~R08 재현). 기기 · IP · 차단 목록이 없어 R02 · R05 · R07은 꺼짐")
    lines.append("- 합성 데이터 결과라 실제 결제 성능의 근거는 아님")
    lines.append("")
    section(lines, "전체 결제", y, rule_scores, model_scores)

    online = np.asarray([transactions[i].get("useChip") == ONLINE for i in val])
    if online.any() and len(set(y[online])) == 2:
        section(lines, "참고 — 온라인 결제만 (돈워리와 같은 비대면)", y[online], rule_scores[online],
                {name: scores[online] for name, scores in model_scores.items()})

    lines.append("## 참고 — Validation에서 Rule이 걸린 횟수")
    lines.append("")
    lines.append("| Rule | 사기에서 | 정상에서 |")
    lines.append("|---|---:|---:|")
    hits = {0: {}, 1: {}}
    for i, label in zip(val, y):
        for rule in rule_results[i].hits:
            hits[int(label)][rule] = hits[int(label)].get(rule, 0) + 1
    for rule in ("R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08"):
        lines.append(f"| {rule} | {hits[1].get(rule, 0)} | {hits[0].get(rule, 0)} |")
    lines.append("")

    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")
    print(report)
    print(f"\n저장: {REPORT_PATH}")


if __name__ == "__main__":
    main()
