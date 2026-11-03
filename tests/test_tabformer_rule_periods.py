import math
import sys
from collections import Counter
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from rule_engine import RuleResult
from tabformer_rule_periods import (
    ALL, STORE, WEB, build_report, channel_of, judge, mixing_warning, risk_ratio, tally, year_of,
)


def row(year=2016, fraud=0, chip="Chip Transaction", at=None):
    return {"paymentRequestedAt": at or f"{year}-03-05T14:20:00+09:00", "fraudLabel": str(fraud), "useChip": chip}


def hit(*rules):
    return RuleResult(score=0, decision="ALLOW", hits=tuple(rules))


def cell(total, fraud, hit_total, hit_fraud, rule="R01"):
    return {"total": total, "fraud": fraud, "hit_total": Counter({rule: hit_total}), "hit_fraud": Counter({rule: hit_fraud})}


# 새해 첫 10분 결제도 적힌 연도 그대로 (UTC로 바꾸면 전년도가 됨)
def test_year_keeps_written_year():
    assert year_of(row(at="2016-01-01T00:10:00+09:00")) == 2016


# 온라인만 온라인, 칩 · 카드 긁기는 매장
def test_channel_online_vs_store():
    assert channel_of(row(chip="Online Transaction")) == WEB
    assert channel_of(row(chip="Chip Transaction")) == STORE
    assert channel_of(row(chip="Swipe Transaction")) == STORE


# 걸린 10건 중 사기 2건(20%) · 안 걸린 90건 중 사기 1건 → 18배, 걸린 게 없으면 계산 불가, 안 걸린 쪽 사기 0이면 무한
def test_risk_ratio():
    assert math.isclose(risk_ratio(cell(100, 3, 10, 2), "R01"), 18.0)
    assert risk_ratio(cell(100, 3, 0, 0), "R01") is None
    assert math.isinf(risk_ratio(cell(100, 2, 10, 2), "R01"))


# 전체에는 두 방식이 다 들어가고, 온라인 결제는 매장 칸에 안 들어감
def test_tally_splits_by_channel_and_year():
    rows = [row(2015, 1, "Online Transaction"), row(2015, 0), row(2016, 1)]
    table = tally(rows, [hit("R01"), hit(), hit("R01", "R04")])

    assert table[ALL][2015]["total"] == 2 and table[ALL][2015]["hit_fraud"]["R01"] == 1
    assert table[WEB][2015]["total"] == 1 and table[STORE][2015]["total"] == 1
    assert table[STORE][2015]["hit_total"]["R01"] == 0
    assert table[STORE][2016]["hit_fraud"]["R04"] == 1


# 해마다 2배 넘으면 믿을 만함, 일부만 넘으면 흔들림, 사기 걸림이 적으면 판단 보류
def test_judge_verdicts():
    strong = cell(1000, 20, 50, 10)
    weak = cell(1000, 20, 500, 10)
    few = cell(1000, 20, 50, 1)

    assert judge({2015: strong, 2016: strong}, "R01").startswith("믿을 만함 — 2/2년")
    assert judge({2015: strong, 2016: weak}, "R01").startswith("기간 따라 흔들림 — 1/2년")
    assert judge({2015: few, 2016: few}, "R01").startswith("판단 보류")
    assert "건수 적은 해 1개 제외" in judge({2015: strong, 2016: strong, 2017: few}, "R01")
    assert "한 번도 안 걸림" in judge({2015: cell(1000, 20, 0, 0)}, "R01")


# 섞으면 위험해 보이는데 같은 방식끼리는 아니면 함정 경고
def test_mixing_warning():
    # 매장: 사기 적고 고액 드묾 · 온라인: 사기 많고 고액 많음 → 섞으면 고액이 위험해 보임
    table = {
        STORE: {2015: cell(1000, 10, 10, 0)},
        WEB: {2015: cell(100, 50, 90, 45)},
    }
    table[ALL] = {2015: cell(1100, 60, 100, 45)}
    assert "결제 방식 함정" in mixing_warning(table, "R01")

    table[ALL] = {2015: cell(1100, 60, 100, 45)}
    table[STORE] = {2015: cell(1000, 10, 10, 5)}
    assert mixing_warning(table, "R01") == ""


# 결과 문서는 결론이 맨 앞, 연도별 숫자는 뒤쪽 참고
def test_report_puts_conclusion_first():
    rows = [row(y, f, c) for y in (2015, 2016) for f in (0, 1) for c in ("Chip Transaction", "Online Transaction")]
    table = tally(rows, [hit("R01")] * len(rows))
    report = build_report(table, [2015, 2016])

    assert report.index("## 결론") < report.index("## 함정 확인") < report.index("## 참고")
    assert "- R01 고액" in report and "  - 매장:" in report and "  - 온라인:" in report
