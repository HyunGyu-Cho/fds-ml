from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rule_engine import evaluate_rules
from train_random_forest import read_csv

# "기간이 바뀌어도 룰을 믿을 수 있나"를 TabFormer(IBM, Apache 2.0)로 확인한다
# 룰은 학습하지 않으므로 연도별 5칸을 모두 쓴다 (시험 구간을 아낄 필요 없음)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "tabformer"
REPORT_PATH = BASE_DIR / "reports" / "tabformer_rule_periods.md"

ONLINE = "Online Transaction"
ALL, STORE, WEB = "전체", "매장", "온라인"
CHANNELS = (STORE, WEB)
RULES = ("R01", "R03", "R04", "R06", "R08")
RULE_NAMES = {
    "R01": "고액",
    "R03": "이력 없는 카드",
    "R04": "10분 안 3번",
    "R06": "카드 테스팅",
    "R08": "거절 후 재시도",
}
# 판정 기준(돌리기 전 고정): 위험 배수 2배 이상이면 위험, 사기 걸림이 5건 미만인 해는 판정에서 뺀다
TRUSTED_RATIO = 2.0
MIN_FRAUD_HITS = 5


def channel_of(row):
    # 온라인 결제와 매장 결제(칩 · 카드 긁기)를 나눈다 — 섞어 보면 결제 방식 차이가 룰 효과처럼 보일 수 있어서
    return WEB if row.get("useChip") == ONLINE else STORE


def year_of(row):
    # 변환 때 원래 시각을 한국 시간으로 적었으므로 UTC로 바꾸지 않고 적힌 연도를 그대로 쓴다
    return datetime.fromisoformat(row["paymentRequestedAt"]).year


def new_cell():
    return {"total": 0, "fraud": 0, "hit_total": Counter(), "hit_fraud": Counter()}


def tally(transactions, results):
    # 범위(전체 · 매장 · 온라인) → 연도 → 결제 수 · 사기 수 · 룰별 걸린 수
    table = defaultdict(lambda: defaultdict(new_cell))
    for row, result in zip(transactions, results):
        fraud = int(row["fraudLabel"])
        for scope in (ALL, channel_of(row)):
            cell = table[scope][year_of(row)]
            cell["total"] += 1
            cell["fraud"] += fraud
            for rule in result.hits:
                cell["hit_total"][rule] += 1
                cell["hit_fraud"][rule] += fraud
    return table


def merge(cells):
    # 여러 해를 합친 칸 (5년 전체 위험 배수용)
    merged = new_cell()
    for cell in cells:
        merged["total"] += cell["total"]
        merged["fraud"] += cell["fraud"]
        merged["hit_total"].update(cell["hit_total"])
        merged["hit_fraud"].update(cell["hit_fraud"])
    return merged


def risk_ratio(cell, rule):
    # 룰에 걸린 결제의 사기 비율 ÷ 안 걸린 결제의 사기 비율 (계산할 수 없으면 None)
    hit_total, hit_fraud = cell["hit_total"][rule], cell["hit_fraud"][rule]
    miss_total, miss_fraud = cell["total"] - hit_total, cell["fraud"] - hit_fraud
    if hit_total == 0 or miss_total == 0:
        return None
    hit_rate, miss_rate = hit_fraud / hit_total, miss_fraud / miss_total
    if miss_rate == 0:
        return math.inf if hit_rate > 0 else None
    return hit_rate / miss_rate


def ratio_text(ratio):
    if ratio is None:
        return "계산 불가"
    if math.isinf(ratio):
        return "사기만 걸림"
    return f"{ratio:.1f}배"


def judge(by_year, rule):
    # 연도별 위험 배수로 한 줄 판정 — 사기 걸림이 적은 해는 우연일 수 있어 뺀다
    cells = list(by_year.values())
    fraud_hits = sum(c["hit_fraud"][rule] for c in cells)
    normal_hits = sum(c["hit_total"][rule] - c["hit_fraud"][rule] for c in cells)
    if fraud_hits + normal_hits == 0:
        return f"판단 보류 — {len(cells)}년 동안 한 번도 안 걸림"
    if fraud_hits < MIN_FRAUD_HITS:
        return f"판단 보류 — {len(cells)}년 합쳐 사기 걸림 {fraud_hits}건 · 정상 걸림 {normal_hits:,}건 (사기를 거의 못 잡음)"
    judged = [risk_ratio(c, rule) for c in cells if c["hit_fraud"][rule] >= MIN_FRAUD_HITS]
    if not judged:
        return f"판단 보류 — 해마다 사기 걸림이 {MIN_FRAUD_HITS}건 미만 ({len(cells)}년 합쳐 {fraud_hits}건)"
    strong = sum(1 for r in judged if r is not None and r >= TRUSTED_RATIO)
    if strong == len(judged):
        verdict = "믿을 만함"
    elif strong == 0:
        verdict = "효과 약함"
    else:
        verdict = "기간 따라 흔들림"
    skipped = len(cells) - len(judged)
    note = f" · 건수 적은 해 {skipped}개 제외" if skipped else ""
    return f"{verdict} — {strong}/{len(judged)}년 {TRUSTED_RATIO:.0f}배 넘음{note}"


def mixing_warning(table, rule):
    # 섞어 보면 위험해 보이는데 같은 방식끼리는 아니면(또는 반대) 결제 방식 함정
    overall = risk_ratio(merge(table[ALL].values()), rule)
    by_channel = [risk_ratio(merge(table[c].values()), rule) for c in CHANNELS if table[c]]
    if overall is None or not by_channel or any(r is None for r in by_channel):
        return ""
    overall_strong = overall >= TRUSTED_RATIO
    channel_strong = [r >= TRUSTED_RATIO for r in by_channel]
    if (overall_strong and not any(channel_strong)) or (not overall_strong and all(channel_strong)):
        return " → 섞어 보면 다르게 보임 (결제 방식 함정)"
    return ""


def build_report(table, years):
    lines = [
        "# TabFormer 룰 기간별 점검 — 기간이 바뀌어도 룰을 믿을 수 있나",
        "",
        f"- 만든 시각: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · `python tabformer_rule_periods.py`",
        "- 데이터 출처: IBM TabFormer 합성 카드 거래 (https://github.com/IBM/TabFormer, Apache 2.0). IBM이 이 실험을 검토 · 보증한 것은 아님. 데이터는 이 저장소에 넣지 않음",
        f"- 기간: {years[0]}~{years[-1]}년을 연도별로 나눔. 룰은 학습하지 않아 모든 해를 그대로 씀",
        f"- 위험 배수: 룰에 걸린 결제의 사기 비율 ÷ 안 걸린 결제의 사기 비율. {TRUSTED_RATIO:.0f}배 이상이면 위험, 사기 걸림 {MIN_FRAUD_HITS}건 미만인 해는 판정에서 뺌 (돌리기 전 고정)",
        "- 결제 방식: 매장(칩 · 카드 긁기)과 온라인을 따로 봄 — 섞어 보면 결제 방식 차이가 룰 효과처럼 보이는 함정을 피하려고",
        "- Rule: `rule_engine.py` (팀 R01~R08 재현). 기기 · IP · 차단 목록이 없어 R02 · R05 · R07은 꺼짐",
        "- 합성 데이터라 단독 결론은 내지 않고, 다른 데이터의 결론을 확인하는 용도로만 씀",
        "",
        "## 결론 — 같은 결제 방식끼리",
        "",
    ]
    for rule in RULES:
        lines.append(f"- {rule} {RULE_NAMES[rule]}")
        for channel in CHANNELS:
            if table[channel]:
                lines.append(f"  - {channel}: {judge(table[channel], rule)}")

    lines += ["", f"## 함정 확인 — 섞어 보면 달라지나 ({len(years)}년 합친 위험 배수)", ""]
    for rule in RULES:
        parts = [f"{scope} {ratio_text(risk_ratio(merge(table[scope].values()), rule))}" for scope in (ALL, *CHANNELS) if table[scope]]
        lines.append(f"- {rule} {RULE_NAMES[rule]}: {' · '.join(parts)}{mixing_warning(table, rule)}")

    lines += ["", "## 참고 — 연도별 숫자", "", "```text"]
    for scope in (ALL, *CHANNELS):
        lines.append(f"결제 · 사기 ({scope})")
        for year in years:
            cell = table[scope].get(year, new_cell())
            share = cell["fraud"] / cell["total"] * 100 if cell["total"] else 0.0
            lines.append(f"  {year}  결제 {cell['total']:>9,} · 사기 {cell['fraud']:>5,} ({share:.2f}%)")
    for rule in RULES:
        for channel in CHANNELS:
            lines.append("")
            lines.append(f"{rule} {RULE_NAMES[rule]} — {channel}")
            for year in years:
                cell = table[channel].get(year, new_cell())
                hits, frauds = cell["hit_total"][rule], cell["hit_fraud"][rule]
                lines.append(f"  {year}  위험 {ratio_text(risk_ratio(cell, rule)):>8} · 걸린 결제 {hits:>7,}건 중 사기 {frauds:>4,}건")
    lines += ["```", ""]
    return "\n".join(lines)


def main():
    transactions = read_csv(DATA_DIR / "transactions.csv")
    print(f"거래 {len(transactions):,}건 읽음 — Rule 계산 중")
    results = evaluate_rules(transactions)
    table = tally(transactions, results)
    years = sorted(table[ALL])

    report = build_report(table, years)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")
    print(report)
    print(f"\n저장: {REPORT_PATH}")


if __name__ == "__main__":
    main()
