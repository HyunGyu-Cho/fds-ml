import csv
import json
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from shadow_whatif import (
    confusion_table,
    country_gap,
    load_rows,
    render_report,
    staff_label,
    version_pairs,
)

COLUMNS = [
    "payment_id", "model_version", "status", "rule_result", "rule_risk_score", "fraud_probability",
    "threshold", "ml_risk_level", "features", "ip_country_code", "matched_rules", "staff_decision",
]


def row(payment_id, version="rf-v2", status="SUCCEEDED", rule="ALLOW", prob="0.500000", level="HIGH",
        country="KR", rules="", decision="", **extra):
    features = json.dumps({"amount": 600000, "customerPastTxCount": 0})
    record = dict(zip(COLUMNS, [payment_id, version, status, rule, "30", prob, "0.300000", level, features,
                                country, rules, decision]))
    record.update(extra)
    return record


def write_export(tmp_path, records, extra_columns=()):
    path = tmp_path / "shadow_export.csv"
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS + list(extra_columns))
        writer.writeheader()
        writer.writerows(records)
    return path


# 공개 저장소에 올라가는 리포트에 CSV의 손님 번호 · 이메일 · IP가 섞여 나오지 않는다
def test_render_report_anyRows_containsNoCustomerRefEmailOrIp(tmp_path):
    secret = {"customer_ref": "884213907761", "buyer_email": "demo-884213907761@example.test",
              "ip_address": "198.51.100.77"}
    path = write_export(tmp_path, [row(1, **secret), row(2, country="NG", rules="R05", **secret)],
                        extra_columns=secret.keys())

    report = render_report(load_rows(path, "rf-v2"), {}, None, None)

    for value in secret.values():
        assert value not in report


# ACCEPT(매입 허용)는 정상, REJECT(승인 취소)는 사기, 판정 없음은 대조에서 빠진다
def test_staff_label_acceptReject_acceptIsNormalRejectIsFraud():
    assert staff_label("ACCEPT") == "정상"
    assert staff_label("REJECT") == "사기"
    assert staff_label(None) is None
    assert staff_label("") is None


# 결제 하나는 일치표 한 칸에만 한 번 들어간다
def test_confusion_table_ruleAndMlLevels_eachRowCountedOnce(tmp_path):
    path = write_export(tmp_path, [
        row(1, rule="ALLOW", level="HIGH"),
        row(2, rule="ALLOW", level="HIGH"),
        row(3, rule="ALLOW", level="LOW", prob="0.100000"),
        row(4, rule="REVIEW", level="HIGH"),
        row(5, rule="DENY", level="LOW", prob="0.200000"),
    ])

    table = confusion_table(load_rows(path, "rf-v2").rows)

    assert table[("ALLOW", "HIGH")] == 2
    assert table[("ALLOW", "LOW")] == 1
    assert table[("REVIEW", "HIGH")] == 1
    assert table[("DENY", "LOW")] == 1
    assert sum(table.values()) == 5


# 확률이 없는 FAILED · SKIPPED는 LOW로 세지 않고 따로 센다
def test_load_rows_failedOrSkipped_excludedAndCountedSeparately(tmp_path):
    path = write_export(tmp_path, [
        row(1),
        row(2, status="FAILED", prob="", level=""),
        row(3, status="SKIPPED", prob="<null>", level="<null>"),
    ])

    loaded = load_rows(path, "rf-v2")

    assert [r.payment_id for r in loaded.rows] == [1]
    assert confusion_table(loaded.rows)[("ALLOW", "LOW")] == 0
    assert loaded.unscored == {"FAILED": 1, "SKIPPED": 1}


# 같은 CSV에 rf-v1 · rf-v2가 섞여 있어도 고른 버전만 센다
def test_load_rows_mixedModelVersions_onlySelectedVersionCounted(tmp_path):
    path = write_export(tmp_path, [
        row(1, version="rf-v1", level="HIGH"),
        row(1, version="rf-v2", level="LOW", prob="0.200000"),
        row(2, version="rf-v1", status="FAILED", prob="", level=""),
    ])

    loaded = load_rows(path, "rf-v2")

    assert [(r.payment_id, r.model_version, r.ml_level) for r in loaded.rows] == [(1, "rf-v2", "LOW")]
    assert not loaded.unscored


# 해외 접속인데 ML LOW면 놓친 결제로 세고, 국내 · 국가 없음은 세지 않는다
def test_country_gap_foreignIpMlLow_countedAsMissed(tmp_path):
    path = write_export(tmp_path, [
        row(1, country="NG", rules="R05", level="LOW", prob="0.100000"),
        row(2, country="US", level="LOW", prob="0.100000"),
        row(3, country="NG", rules="R05", level="HIGH"),
        row(4, country="KR", level="LOW", prob="0.100000"),
        row(5, country="", level="LOW", prob="0.100000"),
    ])

    gap = country_gap(load_rows(path, "rf-v2").rows)

    assert gap["foreign"] == 3
    assert gap["ml_missed"] == 2
    assert gap["r05_but_ml_low"] == 1
    assert gap["rule_r05"] == 2


# 두 모델이 모두 채점한 결제만 짝짓고, HIGH · LOW가 다른 결제를 뒤집힘으로 센다
def test_version_pairs_samePayment_flipAcrossThresholdCounted(tmp_path):
    path = write_export(tmp_path, [
        row(1, version="rf-v1", level="HIGH", prob="0.350000"),
        row(1, version="rf-v2", level="LOW", prob="0.250000"),
        row(2, version="rf-v1", level="HIGH", prob="0.600000"),
        row(2, version="rf-v2", level="HIGH", prob="0.550000"),
        row(3, version="rf-v1", level="LOW", prob="0.100000"),
        row(4, version="rf-v2", level="HIGH", prob="0.900000"),
    ])

    stats = version_pairs(load_rows(path, "rf-v1").rows, load_rows(path, "rf-v2").rows)

    assert stats["pairs"] == 2
    assert stats["flips"] == 1
    assert stats["high_to_low"] == 1
    assert stats["low_to_high"] == 0
    assert abs(stats["max_diff"] - 0.10) < 1e-9


# HeidiSQL이 NULL을 \N으로 내보내도 국가 · 적중 규칙 · 직원 판정이 없는 값으로 읽힌다 (#Ⓚ 재현: 해외 45건으로 부풀려짐)
def test_load_rows_heidiSqlNullMarker_treatedAsMissing(tmp_path):
    path = write_export(tmp_path, [row(1, country="\\N", rules="\\N", decision="\\N")])

    loaded = load_rows(path, "rf-v2")

    assert loaded.rows[0].country is None
    assert loaded.rows[0].matched_rules == ()
    assert loaded.rows[0].staff_decision is None
    assert country_gap(loaded.rows)["foreign"] == 0


# mariadb --batch 출력(탭 구분 · NULL 글자 · \t 이스케이프)을 CSV로 바꾸면 load_rows가 칸을 밀리지 않고 읽는다
def test_batch_to_csv_nullAndEscapedTab_convertedForLoadRows(tmp_path):
    from shadow_export import batch_to_rows, write_csv

    features = '{"amount": 600000, "customerPastTxCount": 0, "note": "a\\\\tb"}'
    values = ["7", "rf-v2", "SUCCEEDED", "REVIEW", "60", "0.510000", "0.300000", "HIGH", features,
              "NULL", "R01 R04", "NULL"]
    lines = ["\t".join(COLUMNS), "\t".join(values)]
    path = tmp_path / "shadow_export.csv"

    write_csv(batch_to_rows(lines), path)
    loaded = load_rows(path, "rf-v2")

    assert len(loaded.rows) == 1
    shadow = loaded.rows[0]
    assert shadow.country is None
    assert shadow.staff_decision is None
    assert shadow.matched_rules == ("R01", "R04")
    assert shadow.features["note"] == "a\tb"
    assert shadow.probability == 0.51
