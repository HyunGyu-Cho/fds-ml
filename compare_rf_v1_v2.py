from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from features import build_features
from scorecard import EXPERIMENT_FPR, SCENARIO_NAMES, check_boundaries, predictions_with_fp_budget
from train_random_forest import make_xy, read_csv, split_and_purge

# rf-v1(가맹점 번호 포함)과 rf-v2(제외)를 같은 Validation · 같은 헛경보 수에서 비교하고 통과 여부를 판정한다

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_PATH = BASE_DIR / "reports" / "rf_v1_vs_v2.md"

# 통과 기준 (11/02 합의): 헛경보 50건 기준 잡은 사기가 5% 넘게 줄지 않고, AP가 0.01 넘게 떨어지지 않을 것
MAX_CAUGHT_DROP = 0.05
MAX_AP_DROP = 0.01

# 합성 가맹점 번호 + 학습 때 본 적 없는 번호(실제 가맹점 상황 흉내)
MERCHANT_IDS = ("1001", "1002", "1003", "9999")

# 시연 결제와 비슷한 예시: 처음 온 손님이 60만 원을 첫 결제
EXAMPLE_NEW_CUSTOMER = {
    "amount": 600000.0, "hourOfDay": 14, "customerPastTxCount": 0, "customerAvgAmount": 0.0,
    "amountRatio": 0.0, "txCount5m": 0, "txCount30m": 0, "txCount24h": 0, "minutesSinceLastTx": -1.0,
    "deviceSeenBefore": 0, "cardSeenBefore": 0, "cardUsedByOtherCustomer": 0,
}


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reasons: tuple[str, ...]


def judge(v1_caught, v2_caught, v1_ap, v2_ap):
    # 경계값(정확히 5% · 0.01 하락)은 통과. 소수 오차로 경계가 흔들리지 않게 반올림해 비교
    reasons = []
    drop = (v1_caught - v2_caught) / v1_caught if v1_caught else 0.0
    if round(drop, 10) > MAX_CAUGHT_DROP:
        reasons.append(f"헛경보 50건 기준 잡은 사기가 {drop:.1%} 줄었다 (허용 {MAX_CAUGHT_DROP:.0%})")
    ap_drop = v1_ap - v2_ap
    if round(ap_drop, 10) > MAX_AP_DROP:
        reasons.append(f"AP가 {ap_drop:.4f} 떨어졌다 (허용 {MAX_AP_DROP})")
    return Verdict(passed=not reasons, reasons=tuple(reasons))


def load_bundle(filename, split_info):
    bundle = joblib.load(MODEL_DIR / filename)
    check_boundaries(bundle, split_info)
    return bundle


def scores(bundle, features, transactions, indices, merchant_id=None):
    # merchant_id를 주면 모든 거래의 가맹점 번호를 그 값으로 바꿔 채점한다 (v2는 이 칸을 안 쓰므로 그대로)
    x, _ = make_xy(features, transactions, indices, bundle["feature_columns"])
    if merchant_id is not None and "merchantId" in bundle["feature_columns"]:
        x = [{**row, "merchantId": merchant_id} for row in x]
    return bundle["pipeline"].predict_proba(x)[:, 1]


def caught_at(y, s, budget):
    return int(((y == 1) & predictions_with_fp_budget(y, s, budget)).sum())


def merchant_swing(bundle, features, transactions, indices):
    # 같은 거래에 가맹점 번호만 바꿔 넣었을 때 확률이 얼마나 달라지나 · 기준값 위아래가 뒤집히는 거래 수
    per_merchant = np.vstack([scores(bundle, features, transactions, indices, m) for m in MERCHANT_IDS])
    spread = per_merchant.max(axis=0) - per_merchant.min(axis=0)
    above = per_merchant >= bundle["threshold"]
    flips = int((above.any(axis=0) & ~above.all(axis=0)).sum())
    return float(spread.mean()), float(spread.max()), flips


def example_probabilities(bundle):
    rows = [{**EXAMPLE_NEW_CUSTOMER, "merchantId": m} for m in MERCHANT_IDS]
    x = [{name: row[name] for name in bundle["feature_columns"]} for row in rows]
    return bundle["pipeline"].predict_proba(x)[:, 1]


def build_report(transactions, split_info, features, v1, v2):
    val = split_info["indices"]["Validation"]
    test = split_info["indices"]["Test"]
    y = np.asarray([int(transactions[i]["fraudLabel"]) for i in val])
    y_test = np.asarray([int(transactions[i]["fraudLabel"]) for i in test])
    normal_total = int((y == 0).sum())
    fraud_total = int((y == 1).sum())
    budget50 = int(np.floor(EXPERIMENT_FPR * normal_total + 1e-9))

    s1 = scores(v1, features, transactions, val)
    s2 = scores(v2, features, transactions, val)
    s1_unseen = scores(v1, features, transactions, val, "9999")
    caught = {name: [caught_at(y, s, b) for b in (0, 5, budget50)] for name, s in (("v1", s1), ("v2", s2), ("v1u", s1_unseen))}
    ap = {"v1": average_precision_score(y, s1), "v2": average_precision_score(y, s2), "v1u": average_precision_score(y, s1_unseen)}
    verdict = judge(caught["v1"][2], caught["v2"][2], ap["v1"], ap["v2"])

    lines = []
    lines.append("# RF v1 vs v2 — 가맹점 번호를 빼도 성능이 유지되나")
    lines.append("")
    lines.append(f"- 만든 시각: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · `python compare_rf_v1_v2.py`")
    lines.append(f"- v1: `random_forest_baseline.joblib` (= ai-server rf-v1) · 칸 {len(v1['feature_columns'])}개, 기준값 {v1['threshold']:.2f}")
    lines.append(f"- v2: `random_forest_v2.joblib` · 칸 {len(v2['feature_columns'])}개 (가맹점 번호 제외), 기준값 {v2['threshold']:.2f}")
    lines.append(f"- 문제지: 합성 데이터 Validation {len(val):,}건 (정상 {normal_total:,} · 사기 {fraud_total:,}). 합성 데이터 결과라 실제 결제 성능의 근거는 아님")
    lines.append("")

    lines.append(f"## 판정: {'✅ 통과' if verdict.passed else '❌ 불통과'}")
    lines.append("")
    lines.append(f"- 기준: 헛경보 {budget50}건에서 잡은 사기가 v1보다 {MAX_CAUGHT_DROP:.0%} 넘게 줄지 않고, AP 하락 {MAX_AP_DROP} 이내")
    drop = (caught["v1"][2] - caught["v2"][2]) / caught["v1"][2]
    lines.append(f"- 잡은 사기 {caught['v1'][2]} → {caught['v2'][2]} ({-drop:+.1%}) · AP {ap['v1']:.4f} → {ap['v2']:.4f} ({ap['v2'] - ap['v1']:+.4f})")
    for reason in verdict.reasons:
        lines.append(f"- 불통과 이유: {reason}")
    lines.append("- 헛경보 0~5건 구간은 몇 건 차이로 순위가 바뀌어 불안정해 판정에 넣지 않고 참고로만 본다")
    lines.append("")

    lines.append("## 같은 헛경보 수에서 잡은 사기 (Validation)")
    lines.append("")
    lines.append("| 모델 | 헛경보 0건 | 헛경보 5건 | 헛경보 " + str(budget50) + "건 (판정) | AP |")
    lines.append("|---|---:|---:|---:|---:|")
    labels = {"v1": "v1 (가맹점 번호 포함)", "v2": "v2 (가맹점 번호 제외)", "v1u": "참고: v1에 처음 보는 가맹점 번호(9999)"}
    for key, label in labels.items():
        cells = " | ".join(str(c) for c in caught[key])
        lines.append(f"| {label} | {cells} | {ap[key]:.4f} |")
    lines.append("")

    lines.append("## 시나리오별 — 헛경보 " + str(budget50) + "건")
    lines.append("")
    lines.append("| 사기 시나리오 | 건수 | v1 | v2 |")
    lines.append("|---|---:|---:|---:|")
    scenarios = np.asarray([transactions[i]["scenarioType"] for i in val])
    p1 = predictions_with_fp_budget(y, s1, budget50)
    p2 = predictions_with_fp_budget(y, s2, budget50)
    fraud_counts = Counter(s for s, label in zip(scenarios, y) if label == 1)
    for scenario, korean in SCENARIO_NAMES.items():
        total = fraud_counts.get(scenario, 0)
        if total:
            mask = (scenarios == scenario) & (y == 1)
            lines.append(f"| {korean} | {total} | {int((mask & p1).sum())} | {int((mask & p2).sum())} |")
    lines.append("")

    lines.append("## v2가 필요한 이유 — 가맹점 번호만 바꿔 넣었을 때")
    lines.append("")
    lines.append(f"Validation 거래마다 가맹점 번호를 {' · '.join(MERCHANT_IDS)}로 바꿔 채점 (9999 = 학습 때 본 적 없는 번호, 실제 가맹점 상황)")
    lines.append("")
    lines.append("| 모델 | 평균 확률 차이 | 최대 확률 차이 | 기준값 위아래가 뒤집히는 거래 |")
    lines.append("|---|---:|---:|---:|")
    for label, bundle in (("v1", v1), ("v2", v2)):
        mean_spread, max_spread, flips = merchant_swing(bundle, features, transactions, val)
        lines.append(f"| {label} | {mean_spread:.1%}p | {max_spread:.1%}p | {flips:,}건 / {len(val):,} |")
    lines.append("")
    lines.append("예시 — 처음 온 손님이 60만 원을 첫 결제 (시연 결제와 비슷)")
    lines.append("")
    lines.append("| 모델 | " + " | ".join(f"가맹점 {m}" for m in MERCHANT_IDS) + " |")
    lines.append("|---|" + "---:|" * len(MERCHANT_IDS))
    for label, bundle in (("v1", v1), ("v2", v2)):
        lines.append(f"| {label} | " + " | ".join(f"{p:.0%}" for p in example_probabilities(bundle)) + " |")
    lines.append("")
    lines.append("- v2도 이 손님은 기준값(30%)을 넘는다 → 시연에서 새 손님이 전부 기준 이상인 문제는 가맹점 번호 때문이 아님 (Ⓚ에서 다시 봄)")
    lines.append("")

    lines.append("## 참고 — Test (모델 선정에 안 쓴 구간, 저장된 기준값 그대로)")
    lines.append("")
    lines.append("| 모델 | F1 | AP | 잡은 사기 | 헛경보 |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, bundle in (("v1", v1), ("v2", v2)):
        s = scores(bundle, features, transactions, test)
        pred = s >= bundle["threshold"]
        lines.append(
            f"| {label} | {f1_score(y_test, pred):.4f} | {average_precision_score(y_test, s):.4f} | "
            f"{int(((y_test == 1) & pred).sum())}/{int((y_test == 1).sum())} | {int(((y_test == 0) & pred).sum())} |"
        )
    lines.append("")
    return "\n".join(lines), verdict


def main():
    transactions = read_csv(DATA_DIR / "transactions.csv")
    episodes = read_csv(DATA_DIR / "fraud_episodes.csv")
    features = build_features(transactions)
    split_info = split_and_purge(transactions, episodes)

    v1 = load_bundle("random_forest_baseline.joblib", split_info)
    v2 = load_bundle("random_forest_v2.joblib", split_info)
    report, verdict = build_report(transactions, split_info, features, v1, v2)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n저장: {REPORT_PATH}")
    print("판정:", "통과" if verdict.passed else "불통과")


if __name__ == "__main__":
    main()
