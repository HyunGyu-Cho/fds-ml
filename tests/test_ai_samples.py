import json
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from ai_samples import Index, build_sample, export, has_rare_signal
from score_ai_samples import build_report, grade


def tx(pid, day, customer, card, device, fraud="0", condition="NORMAL", result="APPROVED", country="KR"):
    return {
        "paymentId": str(pid),
        "merchantId": "1001",
        "customerRef": customer,
        "amount": "30000",
        "paymentRequestedAt": f"2026-04-{day:02d}T03:00:00Z",
        "deviceId": f"SIM-DEVICE-{device}",
        "cardToken": f"SIM-CARD-{card}",
        "cardDisplayLabel": "SIM **** 0000",
        "buyerEmail": f"{customer}@example.invalid",
        "authorizationResult": result,
        "fraudLabel": fraud,
        "scenarioType": "STOLEN_CARD" if fraud == "1" else "NORMAL",
        "ipCountryCode": country,
        "_condition": condition,
    }


# C1이 C2가 먼저 쓴 카드 · 기기로 결제(3번, REVIEW) → C2가 나중에 같은 카드 사용(4번) → C1이 또 새 카드 · 기기(5번, REVIEW)
def rows():
    return [
        tx(1, 1, "C1", "A", "A"),
        tx(2, 2, "C2", "B", "B"),
        tx(3, 3, "C1", "B", "B", fraud="1", condition="FRAUD_SAME_COUNTRY"),
        tx(4, 4, "C2", "B", "B"),
        tx(5, 5, "C1", "C", "C"),
    ]


def sample_of(pid):
    data = rows()
    index = Index(data)
    row = next(r for r in data if r["paymentId"] == str(pid))
    return row, build_sample(index, row, "1001")


# 샘플 JSON에는 정답 · 상황 · 원문 식별자가 없다 — 정답은 labels.csv에만
def test_sample_has_no_answer_or_raw_identifiers():
    row, sample = sample_of(3)
    text = json.dumps(sample, ensure_ascii=False)

    assert row["_decision"] == "REVIEW"
    for leaked in ("fraudLabel", "scenarioType", "STOLEN_CARD", "FRAUD_", "C1", "SIM-CARD-", "SIM-DEVICE-", "example.invalid"):
        assert leaked not in text


# 고객 이력은 같은 고객의 앞선 결제만 — 지난 보류 결제는 사기여도 "만료"로 보여 정답이 새지 않는다
def test_history_uses_only_earlier_payments_and_hides_review_outcome():
    _, sample = sample_of(5)
    history = sample["history"]["payments"]

    assert [p["paymentId"] for p in history] == ["3", "1"]
    assert [p["paymentStatus"] for p in history] == ["EXPIRED", "CAPTURED"]
    assert sample["history"]["statusCounts"] == [
        {"paymentStatus": "CAPTURED", "count": 1},
        {"paymentStatus": "EXPIRED", "count": 1},
    ]


# 같은 카드 · 기기를 쓴 다른 고객은 이 결제 전 30일만 센다 — 나중 결제(4번)는 빠진다
def test_linked_counts_only_earlier_other_customers():
    _, sample = sample_of(3)
    linked = sample["linked"]

    assert linked["card"]["otherCustomerCount"] == 1
    assert linked["card"]["paymentCount"] == 1
    assert linked["device"]["otherCustomerCount"] == 1
    assert linked["ip"]["provided"] is False
    assert has_rare_signal(sample)
    assert not has_rare_signal(sample_of(5)[1])


# 규칙 근거는 Spring과 같은 이름 · 문장 — R02 · R03은 이전 승인 건수
def test_rule_hits_match_spring_names():
    _, sample = sample_of(3)
    hits = sample["fdsEvidence"]["payments"][0]["ruleHits"]

    assert [(h["code"], h["name"], h["appliedScore"]) for h in hits] == [("R02", "신규 단말기", 25), ("R03", "신규 카드", 25)]
    assert hits[0]["observed"] == "이전 승인 1건 중 같은 단말기 없음"


# 같은 seed면 같은 사건이 뽑히고, 할당보다 적으면 있는 만큼만
def test_export_is_deterministic(tmp_path):
    quotas = {"NORMAL": 5, "FRAUD_SAME_COUNTRY": 5}
    first = export(rows(), tmp_path / "a", quotas)
    second = export(rows(), tmp_path / "b", quotas)

    assert [r["paymentId"] for r in first] == [r["paymentId"] for r in second]
    assert sorted(r["paymentId"] for r in first) == ["3", "5"]
    assert (tmp_path / "a" / "labels.csv").exists()


def test_grade_separates_dangerous_misses_from_holds():
    assert grade(True, "REJECT_REVIEW") == "맞음"
    assert grade(True, "ACCEPT_REVIEW") == "위험한 놓침"
    assert grade(False, "REJECT_REVIEW") == "잘못된 취소"
    assert grade(False, "ACCEPT_REVIEW") == "맞음"
    assert grade(True, "NEEDS_CHECK") == grade(False, "NEEDS_CHECK") == "보류"


def test_report_counts_by_condition_and_compares_runs():
    labels = [
        {"caseId": "1", "fraudLabel": "1", "condition": "FRAUD_FOREIGN", "rareSignal": "0"},
        {"caseId": "2", "fraudLabel": "0", "condition": "ABROAD_TRAVEL", "rareSignal": "0"},
    ]
    runs = {
        "a": [
            {"caseId": "1", "status": "ok", "recommendation": "ACCEPT_REVIEW", "seconds": "10"},
            {"caseId": "2", "status": "ok", "recommendation": "NEEDS_CHECK", "seconds": "20"},
        ],
        "b": [
            {"caseId": "1", "status": "ok", "recommendation": "REJECT_REVIEW", "seconds": "10"},
            {"caseId": "2", "status": "fail", "recommendation": "", "seconds": "1"},
        ],
    }

    report = build_report(labels, runs)

    assert "맞음 0 · 보류 1 · 위험한 놓침 1 · 잘못된 취소 0" in report
    assert "| 사기 · 해외 | 없음 | 1 | 0 | 0 | 1 | 0 |" in report
    assert "- a ↔ b: 0 / 1건" in report
