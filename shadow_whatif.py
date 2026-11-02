# Ⓚ 만약 비교 — Shadow 기록(sql/shadow_export.sql CSV)으로 "ML이 판정했다면"을 집계해 reports/shadow_whatif.md에 쓴다
# 공개 저장소라 리포트에는 집계 숫자만 쓴다 (손님 번호 · 이메일 · IP · 결제 번호 안 씀)

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
RULE_RESULTS = ("ALLOW", "REVIEW", "DENY")
ML_LEVELS = ("HIGH", "LOW")
SWEEP_THRESHOLDS = (0.30, 0.40, 0.50, 0.60)
HIGH_AMOUNT = 500_000
HOME_COUNTRY = "KR"
# HeidiSQL은 NULL을 \N으로 내보낸다
NULLS = {"", "NULL", "null", "<null>", "\\N"}


@dataclass(frozen=True)
class ShadowRow:
    payment_id: int
    model_version: str
    rule_result: str
    rule_score: int | None
    probability: float
    threshold: float
    ml_level: str
    features: dict
    country: str | None
    matched_rules: tuple[str, ...]
    staff_decision: str | None


@dataclass
class Loaded:
    rows: list[ShadowRow] = field(default_factory=list)
    # 확률이 없는 기록(FAILED · SKIPPED 등)은 표에 넣지 않고 상태별 개수만 센다
    unscored: Counter = field(default_factory=Counter)


def _value(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.strip()
    return None if text in NULLS else text


def load_rows(path: Path, model_version: str) -> Loaded:
    loaded = Loaded()
    with open(path, encoding="utf-8-sig", newline="") as file:
        for record in csv.DictReader(file):
            if _value(record.get("model_version")) != model_version:
                continue
            status = _value(record.get("status")) or "UNKNOWN"
            probability = _value(record.get("fraud_probability"))
            if status != "SUCCEEDED" or probability is None:
                loaded.unscored[status] += 1
                continue
            score = _value(record.get("rule_risk_score"))
            features = _value(record.get("features"))
            rules = _value(record.get("matched_rules"))
            loaded.rows.append(
                ShadowRow(
                    payment_id=int(_value(record["payment_id"])),
                    model_version=model_version,
                    rule_result=_value(record.get("rule_result")) or "UNKNOWN",
                    rule_score=int(score) if score is not None else None,
                    probability=float(probability),
                    threshold=float(_value(record.get("threshold"))),
                    ml_level=_value(record.get("ml_risk_level")) or "UNKNOWN",
                    features=json.loads(features) if features else {},
                    country=_value(record.get("ip_country_code")),
                    matched_rules=tuple(rules.split()) if rules else (),
                    staff_decision=_value(record.get("staff_decision")),
                )
            )
    return loaded


def load_labels(path: Path) -> dict[int, tuple[str, str]]:
    labels = {}
    with open(path, encoding="utf-8-sig", newline="") as file:
        for record in csv.DictReader(file):
            payment_id = _value(record.get("payment_id"))
            if payment_id:
                labels[int(payment_id)] = (record["scenario"], record["label"])
    return labels


# ACCEPT = 매입 허용(직원이 정상으로 봄), REJECT = 승인 취소(직원이 사기로 봄)
def staff_label(decision: str | None) -> str | None:
    return {"ACCEPT": "정상", "REJECT": "사기"}.get(decision or "")


def rule_flagged(row: ShadowRow) -> bool:
    return row.rule_result in ("REVIEW", "DENY")


def confusion_table(rows: list[ShadowRow]) -> Counter:
    return Counter((row.rule_result, row.ml_level) for row in rows)


def threshold_sweep(rows: list[ShadowRow], thresholds=SWEEP_THRESHOLDS) -> dict[float, int]:
    return {t: sum(1 for row in rows if row.probability >= t) for t in thresholds}


def group_high(rows: list[ShadowRow]) -> dict[str, tuple[int, int]]:
    groups = {
        "새 손님 (이전 결제 0건)": lambda r: r.features.get("customerPastTxCount") == 0,
        "이전 결제 있는 손님": lambda r: (r.features.get("customerPastTxCount") or 0) > 0,
        f"고액 ({HIGH_AMOUNT:,}원 이상)": lambda r: (r.features.get("amount") or 0) >= HIGH_AMOUNT,
        f"{HIGH_AMOUNT:,}원 미만": lambda r: (r.features.get("amount") or 0) < HIGH_AMOUNT,
    }
    result = {}
    for name, match in groups.items():
        members = [row for row in rows if match(row)]
        result[name] = (sum(1 for row in members if row.ml_level == "HIGH"), len(members))
    return result


# 해외 접속인데 ML이 LOW면 "국가 칸이 없어 놓쳤을 수 있는 결제"로 센다 (Ⓥ3 근거)
def country_gap(rows: list[ShadowRow], home: str = HOME_COUNTRY) -> dict[str, int]:
    foreign = [row for row in rows if row.country and row.country != home]
    return {
        "foreign": len(foreign),
        "rule_r05": sum(1 for row in foreign if "R05" in row.matched_rules),
        "ml_high": sum(1 for row in foreign if row.ml_level == "HIGH"),
        "ml_missed": sum(1 for row in foreign if row.ml_level == "LOW"),
        "r05_but_ml_low": sum(1 for row in foreign if "R05" in row.matched_rules and row.ml_level == "LOW"),
    }


# 같은 결제를 두 모델이 모두 채점했을 때 HIGH · LOW가 뒤집힌 건수
def version_pairs(base: list[ShadowRow], other: list[ShadowRow]) -> dict[str, float]:
    other_by_payment = {row.payment_id: row for row in other}
    pairs = [(row, other_by_payment[row.payment_id]) for row in base if row.payment_id in other_by_payment]
    flips = [(a, b) for a, b in pairs if a.ml_level != b.ml_level]
    return {
        "pairs": len(pairs),
        "flips": len(flips),
        "high_to_low": sum(1 for a, b in flips if a.ml_level == "HIGH"),
        "low_to_high": sum(1 for a, b in flips if a.ml_level == "LOW"),
        "max_diff": max((abs(a.probability - b.probability) for a, b in pairs), default=0.0),
    }


# 정답(정상 · 사기)별로 Rule · ML이 각각 몇 건을 의심했는지
def answer_table(rows: list[ShadowRow], answer_of) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        answer = answer_of(row)
        if answer is None:
            continue
        cell = table.setdefault(answer, {"total": 0, "rule": 0, "ml": 0, "both_miss": 0})
        cell["total"] += 1
        cell["rule"] += rule_flagged(row)
        cell["ml"] += row.ml_level == "HIGH"
        cell["both_miss"] += not rule_flagged(row) and row.ml_level != "HIGH"
    return table


def scenario_table(rows: list[ShadowRow], labels: dict[int, tuple[str, str]]) -> dict[str, dict]:
    table: dict[str, dict] = {}
    for row in rows:
        if row.payment_id not in labels:
            continue
        scenario, label = labels[row.payment_id]
        cell = table.setdefault(scenario, {"label": label, "total": 0, "rule": 0, "ml": 0, "probs": []})
        cell["total"] += 1
        cell["rule"] += rule_flagged(row)
        cell["ml"] += row.ml_level == "HIGH"
        cell["probs"].append(row.probability)
    return table


def _pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "0/0"


def render_report(loaded: Loaded, labels: dict[int, tuple[str, str]], pair_stats: dict | None, compare_version: str | None) -> str:
    rows = loaded.rows
    version = rows[0].model_version if rows else "(기록 없음)"
    threshold = rows[0].threshold if rows else 0.0
    table = confusion_table(rows)
    lines = [
        f"# 만약 비교 — 돈워리 Shadow 기록 ({version})",
        "",
        "> 돈워리 로컬 DB의 `ml_shadow_scores`를 집계한 결과다. ML은 결제 판정에 영향을 주지 않았다 (ADR-0061).",
        "> 시연 · 시나리오 결제라 표본이 작고, 결제를 초 단위로 이어 보내 `minutesSinceLastTx`가 실제보다 작다. 성능 주장이 아니라 경향 확인용이다.",
        "",
        f"- 채점된 결제: {len(rows)}건 · ML 기준값 {threshold:.2f}",
        f"- 확률 없는 기록(표에서 제외): {sum(loaded.unscored.values())}건"
        + (f" ({', '.join(f'{k} {v}' for k, v in sorted(loaded.unscored.items()))})" if loaded.unscored else ""),
        "",
        "## 1. 일치표 — Rule 판정 × ML",
        "",
        "| Rule \\ ML | HIGH | LOW |",
        "|---|---|---|",
    ]
    for result in RULE_RESULTS:
        lines.append(f"| {result} | {table[(result, 'HIGH')]} | {table[(result, 'LOW')]} |")
    ml_only = table[("ALLOW", "HIGH")]
    rule_only = table[("REVIEW", "LOW")] + table[("DENY", "LOW")]
    lines += [
        "",
        f"- ML만 의심(Rule ALLOW · ML HIGH): **{ml_only}건** — ML이 판정했다면 추가로 보류됐을 결제",
        f"- Rule만 의심(Rule REVIEW · DENY · ML LOW): **{rule_only}건** — ML이 판정했다면 풀렸을 결제",
        "",
        "## 2. 정답과 대조",
        "",
    ]
    staff = answer_table(rows, lambda r: staff_label(r.staff_decision))
    lines += _answer_lines("직원 판정 (REVIEW 사건만 있음)", staff)
    if labels:
        by_label = answer_table(rows, lambda r: labels.get(r.payment_id, (None, None))[1])
        lines += _answer_lines("시나리오 라벨", by_label)
        lines += ["| 시나리오 | 정답 | 결제 | Rule 의심 | ML HIGH | ML 확률 범위 |", "|---|---|---|---|---|---|"]
        for name, cell in sorted(scenario_table(rows, labels).items()):
            probs = cell["probs"]
            lines.append(
                f"| {name} | {cell['label']} | {cell['total']} | {cell['rule']} | {cell['ml']} | {min(probs):.2f}~{max(probs):.2f} |"
            )
    lines += ["", "## 3. ML HIGH가 몰리는 곳 · 기준값을 바꿨다면", "", "| 묶음 | ML HIGH |", "|---|---|"]
    for name, (high, total) in group_high(rows).items():
        lines.append(f"| {name} | {_pct(high, total)} |")
    lines += ["", "| 기준값 | HIGH 건수 |", "|---|---|"]
    for t, count in threshold_sweep(rows).items():
        lines.append(f"| {t:.2f} | {_pct(count, len(rows))} |")
    gap = country_gap(rows)
    lines += [
        "",
        "## 4. 국가 공백 (ML 입력에 접속 국가가 없음)",
        "",
        f"- 해외 접속({HOME_COUNTRY} 아님) 결제 {gap['foreign']}건 — Rule R05(고위험 국가) 적중 {gap['rule_r05']}건, ML HIGH {gap['ml_high']}건",
        f"- 해외 접속인데 ML LOW: **{gap['ml_missed']}건** (그중 R05가 잡은 결제 {gap['r05_but_ml_low']}건) → Ⓥ3 국가 칸 근거",
    ]
    if pair_stats is not None:
        lines += [
            "",
            f"## 5. 같은 결제 {compare_version} ↔ {version}",
            "",
            f"- 두 모델이 모두 채점한 결제 {pair_stats['pairs']}건 중 HIGH · LOW가 뒤집힌 결제 **{pair_stats['flips']}건**"
            f" (HIGH→LOW {pair_stats['high_to_low']} · LOW→HIGH {pair_stats['low_to_high']}), 확률 차이 최대 {pair_stats['max_diff']:.3f}",
        ]
    return "\n".join(lines) + "\n"


def _answer_lines(title: str, table: dict[str, dict[str, int]]) -> list[str]:
    if not table:
        return [f"- {title}: 대조할 결제 없음"]
    lines = [f"**{title}**", "", "| 정답 | 결제 | Rule 의심 | ML HIGH | 둘 다 놓침 |", "|---|---|---|---|---|"]
    for answer in ("사기", "정상"):
        if answer in table:
            c = table[answer]
            lines.append(f"| {answer} | {c['total']} | {c['rule']} | {c['ml']} | {c['both_miss']} |")
    return lines + [""]


def main() -> int:
    parser = argparse.ArgumentParser(description="Shadow 기록 Rule vs ML 만약 비교")
    parser.add_argument("--export", type=Path, default=ML_DIR / "data" / "shadow" / "shadow_export.csv")
    parser.add_argument("--labels", type=Path, default=None, help="shadow_scenarios.py가 남긴 라벨 CSV")
    parser.add_argument("--model", default="rf-v2")
    parser.add_argument("--compare", default="rf-v1", help="같은 결제를 비교할 다른 모델 버전 (빈 값이면 생략)")
    parser.add_argument("--out", type=Path, default=ML_DIR / "reports" / "shadow_whatif.md")
    args = parser.parse_args()
    loaded = load_rows(args.export, args.model)
    labels = load_labels(args.labels) if args.labels else {}
    pair_stats = None
    if args.compare:
        other = load_rows(args.export, args.compare)
        pair_stats = version_pairs(other.rows, loaded.rows) if other.rows else None
    report = render_report(loaded, labels, pair_stats, args.compare)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
