from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from country_features import HOME_COUNTRY_TABFORMER, country_features, tabformer_country
from features import build_features
from generate_data_v3 import (
    ABROAD_RESIDENT,
    ABROAD_TRAVEL,
    FRAUD_CONDITIONS,
    FRAUD_FOREIGN,
    FRAUD_HIGH_RISK,
    HIGH_RISK_COUNTRIES,
    NORMAL_CONDITIONS,
)
from rule_engine import REVIEW_MIN_SCORE, evaluate_rules
from scorecard import EXPERIMENT_FPR, check_boundaries, predictions_with_fp_budget
from train_random_forest import make_xy, read_csv, split_and_purge
from train_random_forest_v3 import COUNTRY_FEATURE, DATA_DIR, MODEL_PATH, NO_COUNTRY_PATH, load_v3

# rf-v2(팀 ai-server 파일 그대로) vs rf-v3를 v3 데이터 Test에서 비교하고 통과 기준 5개를 자동 판정한다 (Ⓥ3 ③)
# 기준 2~4의 비율은 기준 1과 같은 지점(Rule 보류 50점+와 같은 헛경보 수)에서 센다 — 11/03 합의

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TABFORMER_DIR = BASE_DIR / "data" / "tabformer"
REPORT_PATH = BASE_DIR / "reports" / "rf_v2_vs_v3.md"

MIN_FOREIGN_GAIN = 0.20
MAX_ABROAD_FP_RATE = 0.10
HIGH_AMOUNT = 500_000
# Rule 헛경보가 이보다 적으면 비교 지점이 불안정해 기존 실험 비율(50 / 2555)로 바꾼다
MIN_RULE_BUDGET = 5

FOREIGN_FRAUD = (FRAUD_FOREIGN, FRAUD_HIGH_RISK)
ABROAD_NORMAL = (ABROAD_RESIDENT, ABROAD_TRAVEL)

CONDITION_NAMES = {
    "NORMAL": "보통 정상",
    "ODD_BIG_FIRST": "정상 · 새 손님 첫 결제 고액",
    "ODD_NEW_CARD_DEVICE": "정상 · 카드 · 기기 한꺼번에 바꿈",
    "ODD_BURST": "정상 · 몇 분 사이 연달아 결제",
    "ABROAD_RESIDENT": "정상 · 해외 거주",
    "ABROAD_TRAVEL": "정상 · 해외 여행 중",
    "FRAUD_SAME_COUNTRY": "사기 · 평소 국가로 위장",
    "FRAUD_FOREIGN": "사기 · 해외 접속",
    "FRAUD_HIGH_RISK": "사기 · 고위험 국가 접속",
}


@dataclass(frozen=True)
class Metrics:
    caught: int
    ap: float
    foreign_fraud_rate: float
    abroad_normal_fp_rate: float
    high_amount_fp_rate: float
    tabformer_ap: float | None = None


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool | None
    detail: str


@dataclass(frozen=True)
class Verdict:
    passed: bool
    pending: bool
    checks: tuple[Check, ...]

    def label(self) -> str:
        # 하나라도 불통과면 미검증 기준이 남아도 불통과 (나머지 결과로 뒤집히지 않음)
        if self.passed:
            return "✅ 통과"
        if any(c.passed is False for c in self.checks):
            return "❌ 불통과"
        return "⏸ 보류 (미검증 기준 있음)"


def _r(value: float) -> float:
    # 소수 오차로 경계가 흔들리지 않게 반올림해 비교 (예: 0.7 - 0.5 = 0.19999…)
    return round(value, 10)


def judge(v2: Metrics, v3: Metrics) -> Verdict:
    # 경계: 기준 1 잡은 사기와 기준 4는 "더 나아야" 통과(같으면 불통과), 나머지는 경계값이면 통과
    checks = []
    c1 = v3.caught > v2.caught and _r(v3.ap - v2.ap) >= 0
    checks.append(Check("1. 같은 헛경보에서 잡은 사기 v3 > v2 · AP 안 떨어짐", c1,
                        f"잡은 사기 {v2.caught} → {v3.caught} · AP {v2.ap:.4f} → {v3.ap:.4f}"))
    gain = v3.foreign_fraud_rate - v2.foreign_fraud_rate
    checks.append(Check(f"2. 해외 사기 잡은 비율 v3 ≥ v2 + {MIN_FOREIGN_GAIN:.0%}p", _r(gain) >= _r(MIN_FOREIGN_GAIN),
                        f"{v2.foreign_fraud_rate:.1%} → {v3.foreign_fraud_rate:.1%} ({gain * 100:+.1f}%p)"))
    checks.append(Check(f"3. 해외 정상 헛경보 비율 ≤ {MAX_ABROAD_FP_RATE:.0%}", _r(v3.abroad_normal_fp_rate) <= _r(MAX_ABROAD_FP_RATE),
                        f"v3 {v3.abroad_normal_fp_rate:.1%} (참고 v2 {v2.abroad_normal_fp_rate:.1%})"))
    checks.append(Check("4. 고액 정상(50만 원+) 헛경보 비율 v3 < v2", _r(v3.high_amount_fp_rate - v2.high_amount_fp_rate) < 0,
                        f"{v2.high_amount_fp_rate:.1%} → {v3.high_amount_fp_rate:.1%}"))
    if v2.tabformer_ap is None or v3.tabformer_ap is None:
        checks.append(Check("5. TabFormer AP v3 ≥ v2", None, "미검증 — TabFormer 데이터가 없어 건너뜀 (PC에서 실행)"))
    else:
        checks.append(Check("5. TabFormer AP v3 ≥ v2", _r(v3.tabformer_ap - v2.tabformer_ap) >= 0,
                            f"{v2.tabformer_ap:.4f} → {v3.tabformer_ap:.4f}"))
    pending = any(c.passed is None for c in checks)
    return Verdict(passed=all(c.passed is True for c in checks), pending=pending, checks=tuple(checks))


def scores(bundle, features, transactions, indices):
    x, _ = make_xy(features, transactions, indices, bundle["feature_columns"])
    return bundle["pipeline"].predict_proba(x)[:, 1]


def rate(predictions, mask):
    total = int(mask.sum())
    if total == 0:
        raise ValueError("비율을 셀 거래가 0건입니다 — 데이터 · 분할을 확인하세요.")
    return float((predictions & mask).sum()) / total


def operating_budget(y, rule_scores):
    # 비교 지점 = Test에서 Rule 보류(50점+)가 낸 헛경보 수. 너무 적으면 기존 실험 비율로
    rule_fp = int(((y == 0) & (rule_scores >= REVIEW_MIN_SCORE)).sum())
    if rule_fp < MIN_RULE_BUDGET:
        return int(np.floor(EXPERIMENT_FPR * int((y == 0).sum()) + 1e-9)), rule_fp, True
    return rule_fp, rule_fp, False


def metrics_at(predictions, model_scores, y, conditions, amounts, tabformer_ap=None):
    return Metrics(
        caught=int(((y == 1) & predictions).sum()),
        ap=float(average_precision_score(y, model_scores)),
        foreign_fraud_rate=rate(predictions, (y == 1) & np.isin(conditions, FOREIGN_FRAUD)),
        abroad_normal_fp_rate=rate(predictions, (y == 0) & np.isin(conditions, ABROAD_NORMAL)),
        high_amount_fp_rate=rate(predictions, (y == 0) & (amounts >= HIGH_AMOUNT)),
        tabformer_ap=tabformer_ap,
    )


def tabformer_aps(bundles):
    # TabFormer Validation AP — Ⓛ와 같은 구간 · 같은 Feature, v3 국가 칸은 기본 국가 US로 계산 (온라인 · 빈칸은 -1)
    path = TABFORMER_DIR / "transactions.csv"
    if not path.exists():
        return None
    print("TabFormer 읽는 중 (오래 걸림)")
    transactions = read_csv(path)
    episodes = read_csv(TABFORMER_DIR / "fraud_episodes.csv")
    features = build_features(transactions)
    for transaction in transactions:
        transaction["country"] = tabformer_country(transaction.get("locationRef"))
    values = country_features(transactions, home_country=HOME_COUNTRY_TABFORMER, country_key="country")
    for row, value in zip(features, values):
        row[COUNTRY_FEATURE] = value[COUNTRY_FEATURE]
    split_info = split_and_purge(transactions, episodes)
    val = split_info["indices"]["Validation"]
    y = np.asarray([int(transactions[i]["fraudLabel"]) for i in val])
    return {name: float(average_precision_score(y, scores(b, features, transactions, val))) for name, b in bundles.items()}


def build_report(transactions, features, split_info, bundles, tab_aps):
    test = split_info["indices"]["Test"]
    condition_by_payment = {row["paymentId"]: row["condition"] for row in read_csv(DATA_DIR / "conditions.csv")}
    y = np.asarray([int(transactions[i]["fraudLabel"]) for i in test])
    conditions = np.asarray([condition_by_payment[transactions[i]["paymentId"]] for i in test])
    amounts = np.asarray([int(transactions[i]["amount"]) for i in test])

    rule_results = evaluate_rules(transactions, HIGH_RISK_COUNTRIES)
    rule_scores = np.asarray([rule_results[i].score for i in test], dtype=np.float64)
    budget, rule_fp, fallback = operating_budget(y, rule_scores)

    model_scores = {name: scores(b, features, transactions, test) for name, b in bundles.items()}
    preds = {name: predictions_with_fp_budget(y, s, budget) for name, s in model_scores.items()}
    preds["Rule"] = rule_scores >= REVIEW_MIN_SCORE
    model_scores["Rule"] = rule_scores
    tab = tab_aps or {}
    metrics = {name: metrics_at(preds[name], model_scores[name], y, conditions, amounts, tab.get(name)) for name in preds}
    verdict = judge(metrics["v2"], metrics["v3"])

    labels = {
        "Rule": f"Rule 보류(50점+, R05 켬 · 고위험 {' · '.join(HIGH_RISK_COUNTRIES)})",
        "v2": "rf-v2 (팀 ai-server 파일 그대로)",
        "v3_no_country": "참고: v3 데이터 · 국가 칸 없음",
        "v3": "rf-v3 (국가 칸 포함)",
    }
    order = ("Rule", "v2", "v3_no_country", "v3")

    lines = []
    lines.append("# RF v2 vs v3 — 국가 칸 · 현실적인 합성 데이터로 나아졌나")
    lines.append("")
    lines.append(f"- 만든 시각: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · `python compare_v2_v3.py`")
    for key in ("v2", "v3_no_country", "v3"):
        b = bundles[key]
        lines.append(f"- {labels[key]}: 칸 {len(b['feature_columns'])}개, 저장된 기준값 {b['threshold']:.2f}")
    lines.append(f"- 문제지: v3 합성 데이터 Test {len(test):,}건 (정상 {int((y == 0).sum()):,} · 사기 {int((y == 1).sum()):,}). "
                 "비율 · 국가는 실험용 가정이라 실제 결제 성능의 근거는 아님")
    point = f"Rule 보류와 같은 헛경보 {budget}건" if not fallback else f"Rule 헛경보가 {rule_fp}건뿐이라 기존 실험 비율 헛경보 {budget}건"
    lines.append(f"- 비교 지점: {point} (모델은 이 수를 넘지 않는 가장 낮은 기준값)")
    lines.append("")

    lines.append(f"## 판정: {verdict.label()}")
    lines.append("")
    for c in verdict.checks:
        mark = "⏸" if c.passed is None else ("✅" if c.passed else "❌")
        lines.append(f"- {mark} {c.name} — {c.detail}")
    lines.append("")

    lines.append(f"## 같은 헛경보에서 (Test, 비교 지점)")
    lines.append("")
    lines.append("| 모델 | 헛경보 | 잡은 사기 | AP | 해외 사기 잡음 | 해외 정상 헛경보 | 고액 정상 헛경보 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key in order:
        m = metrics[key]
        fp = int(((y == 0) & preds[key]).sum())
        lines.append(f"| {labels[key]} | {fp} | {m.caught}/{int((y == 1).sum())} | {m.ap:.4f} | {m.foreign_fraud_rate:.1%} | "
                     f"{m.abroad_normal_fp_rate:.1%} | {m.high_amount_fp_rate:.1%} |")
    lines.append("")
    lines.append("- Rule의 AP는 점수(0~100)를 순위로 본 값이라 참고만")
    lines.append("- 국가 칸 없음 → 포함의 차이가 국가 칸 효과, v2 → 국가 칸 없음의 차이가 학습 데이터 효과")
    lines.append("")

    lines.append("## 상황별 — 경보 비율 (사기는 잡은 비율, 정상은 헛경보 비율)")
    lines.append("")
    lines.append("| 상황 | 건수 | " + " | ".join(labels[k] for k in order) + " |")
    lines.append("|---|---:|" + "---:|" * len(order))
    for condition in (*NORMAL_CONDITIONS, *FRAUD_CONDITIONS):
        mask = conditions == condition
        total = int(mask.sum())
        if total:
            cells = " | ".join(f"{rate(preds[k], mask):.1%}" for k in order)
            lines.append(f"| {CONDITION_NAMES[condition]} | {total:,} | {cells} |")
    lines.append("")

    lines.append("## 참고 — 각자 저장된 기준값 그대로 (실제 ai-server 동작)")
    lines.append("")
    lines.append("| 모델 | 기준값 | 헛경보 | 잡은 사기 | 해외 사기 잡음 | 해외 정상 헛경보 | 고액 정상 헛경보 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key in ("v2", "v3_no_country", "v3"):
        own = model_scores[key] >= bundles[key]["threshold"]
        m = metrics_at(own, model_scores[key], y, conditions, amounts)
        lines.append(f"| {labels[key]} | {bundles[key]['threshold']:.2f} | {int(((y == 0) & own).sum())} | {m.caught} | "
                     f"{m.foreign_fraud_rate:.1%} | {m.abroad_normal_fp_rate:.1%} | {m.high_amount_fp_rate:.1%} |")
    lines.append("")
    lines.append("- v2 기준값 0.30은 사기 10% 데이터에서 고른 값이라 사기 0.5%인 v3 데이터에선 맞지 않음 → 판정은 위 같은 헛경보 비교로 함")
    lines.append("")

    lines.append("## 참고 — 비교 지점(헛경보 수)에 따라 (판정에 안 씀)")
    lines.append("")
    lines.append("| 헛경보 | v2 잡은 사기 | v3 잡은 사기 | v2 해외 사기 잡음 | v3 해외 사기 잡음 | v3 해외 정상 헛경보 | v3 여행 중 정상 헛경보 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    foreign = (y == 1) & np.isin(conditions, FOREIGN_FRAUD)
    abroad = (y == 0) & np.isin(conditions, ABROAD_NORMAL)
    travel = (y == 0) & (conditions == ABROAD_TRAVEL)
    for points in sorted({int(((y == 0) & (model_scores["v3"] >= bundles["v3"]["threshold"])).sum()), 50, 100, 200, budget}):
        p2 = predictions_with_fp_budget(y, model_scores["v2"], points)
        p3 = predictions_with_fp_budget(y, model_scores["v3"], points)
        lines.append(f"| {points} | {int((p2 & (y == 1)).sum())} | {int((p3 & (y == 1)).sum())} | {rate(p2, foreign):.1%} | "
                     f"{rate(p3, foreign):.1%} | {rate(p3, abroad):.1%} | {rate(p3, travel):.1%} |")
    lines.append("")
    lines.append("- 헛경보를 많이 허용할수록 v3는 남는 경보를 해외 여행 중 정상에 쓴다 — 국가 칸만으로는 여행과 해외 공격을 잘 못 가름")
    lines.append(f"- '정상 · 새 손님 첫 결제 고액'은 Test에 {int((conditions == 'ODD_BIG_FIRST').sum())}건뿐 (첫 결제가 대부분 Train 기간) — 이 상황 비율은 믿기 어려움")
    lines.append("")

    lines.append("## TabFormer (외부 데이터, Validation AP)")
    lines.append("")
    if tab_aps is None:
        lines.append("- 미검증 — data/tabformer 없음 (PC에서 `python compare_v2_v3.py`로 다시 실행)")
    else:
        for key in ("v2", "v3_no_country", "v3"):
            lines.append(f"- {labels[key]}: {tab_aps[key]:.4f}")
        lines.append("- 단서: TabFormer 사기는 해외 · 다른 국가 매장 결제에 몰려 있어 국가 칸 효과가 크게 보인다 (①에서 RF 0.0146 → RF + 국가 B 0.5080). "
                     "v3가 이 기준을 넘어도 v3 실력의 근거로는 약하다. 온라인 결제는 국가가 없어(-1) 국가 칸이 돕지 못한다")
    lines.append("")
    return "\n".join(lines), verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tabformer", action="store_true", help="TabFormer 기준(5)을 건너뛴다 — 판정은 보류")
    args = parser.parse_args()

    transactions, features, split_info = load_v3()
    bundles = {
        "v2": joblib.load(MODEL_DIR / "random_forest_v2.joblib"),
        "v3_no_country": joblib.load(NO_COUNTRY_PATH),
        "v3": joblib.load(MODEL_PATH),
    }
    # v2는 다른 데이터(v1)로 학습했으므로 분할 경계 검사는 v3 두 모델만
    check_boundaries(bundles["v3_no_country"], split_info)
    check_boundaries(bundles["v3"], split_info)

    tab_aps = None if args.skip_tabformer else tabformer_aps(bundles)
    report, verdict = build_report(transactions, features, split_info, bundles, tab_aps)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n저장: {REPORT_PATH}")
    print("판정:", verdict.label())


if __name__ == "__main__":
    main()
