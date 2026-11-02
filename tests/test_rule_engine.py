import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from rule_engine import evaluate_rules


BASE_TIME = datetime(2026, 5, 1, 3, 0, tzinfo=timezone.utc)


def tx(payment_id, minutes, amount=50_000, merchant=1001, customer="C1",
       device="D1", card="K1", result="APPROVED"):
    at = BASE_TIME + timedelta(minutes=minutes)
    return {
        "paymentId": str(payment_id),
        "merchantId": str(merchant),
        "customerRef": customer,
        "amount": str(amount),
        "paymentRequestedAt": at.isoformat().replace("+00:00", "Z"),
        "deviceId": device,
        "cardToken": card,
        "authorizationResult": result,
    }


# R04: 현재 결제 포함 10분 안 3건이면 발동
def test_r04_fires_on_third_payment_within_10min():
    results = evaluate_rules([tx(1, 0), tx(2, 4), tx(3, 9)])

    assert "R04" not in results[1].hits
    assert "R04" in results[2].hits
    assert results[2].score == 30


# R02·R03: 승인 이력이 없는 첫 손님에게는 발동 안 함
def test_r02_r03_skip_customer_without_approved_history():
    results = evaluate_rules([
        tx(1, 0, device="D1", card="K1", result="DECLINED"),
        tx(2, 60, device="D2", card="K2"),
        tx(3, 120, device="D3", card="K3"),
    ])

    assert "R02" not in results[0].hits and "R03" not in results[0].hits
    assert "R02" not in results[1].hits and "R03" not in results[1].hits
    assert {"R02", "R03"} <= set(results[2].hits)


# 점수는 100 상한, 50 이상 보류 · 80 이상 거절
def test_score_capped_at_100_with_review_50_deny_80():
    review = evaluate_rules([
        tx(1, 0, device="D1", card="K1"),
        tx(2, 60, device="D2", card="K2"),
    ])[1]
    deny = evaluate_rules([
        tx(1, 0),
        tx(2, 1),
        tx(3, 2, amount=600_000, device="D9", card="K9"),
    ])[2]
    allow = evaluate_rules([tx(1, 0, amount=500_000)])[0]

    assert (review.score, review.decision) == (50, "REVIEW")
    assert (allow.score, allow.decision) == (30, "ALLOW")
    assert deny.hits == ("R01", "R04", "R02", "R03")
    assert (deny.score, deny.decision) == (100, "DENY")


# 뒤 시각 거래 · 다른 가맹점 이력은 보지 않음
def test_ignores_later_and_other_merchant_history():
    results = evaluate_rules([
        tx(10, 30, device="D2", card="K2"),
        tx(1, 0, merchant=1002),
        tx(2, 4, merchant=1002),
        tx(3, 9),
        tx(4, 20),
        tx(5, 21),
    ])

    assert results[3].hits == ()
    assert "R04" not in results[4].hits
    assert "R04" not in results[5].hits


# R06: 소액 5회 · 카드 3장 · 거절 3회를 모두 만족할 때만 발동
def test_r06_requires_all_three_conditions():
    declined_small = [
        tx(i, i, amount=5_000, card=f"K{i}", result="DECLINED")
        for i in range(1, 5)
    ]
    full = evaluate_rules(declined_small + [tx(5, 5, amount=5_000, card="K5")])
    one_card = evaluate_rules(
        [tx(i, i, amount=5_000, card="K1", result="DECLINED") for i in range(1, 5)]
        + [tx(5, 5, amount=5_000, card="K1")]
    )
    big_amount = evaluate_rules(declined_small + [tx(5, 5, amount=20_000, card="K5")])

    assert "R06" in full[4].hits
    assert "R06" not in one_card[4].hits
    assert "R06" not in big_amount[4].hits


# R05: 고위험 국가 접속일 때만 +40, 다른 국가 · 국가 없음은 안 켜지고, 목록을 안 주면 기존 성적표처럼 꺼진다
def test_rule_engine_R05_highRiskCountryOnly():
    rows = [
        {**tx(1, 0, customer="C1"), "ipCountryCode": "NG"},
        {**tx(2, 30, customer="C2"), "ipCountryCode": "KR"},
        {**tx(3, 60, customer="C3"), "ipCountryCode": ""},
        {**tx(4, 90, customer="C4", amount=600_000), "ipCountryCode": "NG"},
    ]

    with_list = evaluate_rules(rows, ("NG",))
    assert ["R05" in r.hits for r in with_list] == [True, False, False, True]
    assert (with_list[0].score, with_list[0].decision) == (40, "ALLOW")
    assert (with_list[3].score, with_list[3].decision) == (70, "REVIEW")
    assert [r.score for r in with_list[1:3]] == [0, 0]

    without = evaluate_rules(rows)
    assert all("R05" not in r.hits for r in without)
    assert [r.score for r in without] == [0, 0, 0, 30]

    # 국가 칸이 없는 v1 · v2 데이터 모양은 목록을 줘도 안 켜진다
    assert evaluate_rules([tx(5, 0)], ("NG",))[0].hits == ()
