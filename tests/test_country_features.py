import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from country_features import (
    DIFFERS_FROM_USUAL,
    SEEN_BEFORE,
    UNKNOWN,
    country_features,
    tabformer_country,
)


def tx(at, country, customer="C1"):
    return {"paymentRequestedAt": at, "customerRef": customer, "ipCountryCode": country}


def column(rows, name, home="KR"):
    return [row[name] for row in country_features(rows, home_country=home)]


# 앞선 결제만 본다 — 나중 결제 · 같은 시각 결제 · 다른 손님의 국가는 평소 국가 계산에 안 들어간다
def test_country_differs_from_usual_usesOnlyEarlierTransactions():
    rows = [
        tx("2026-03-04T10:00:00+09:00", "US"),
        tx("2026-03-01T10:00:00+09:00", "US"),
        tx("2026-03-02T10:00:00+09:00", "KR"),
        tx("2026-03-02T10:00:00+09:00", "KR"),
        tx("2026-03-02T10:00:00+09:00", "KR"),
        tx("2026-03-03T10:00:00+09:00", "KR"),
        tx("2026-02-27T10:00:00+09:00", "US", customer="C2"),
        tx("2026-02-28T10:00:00+09:00", "US", customer="C2"),
    ]

    # C1: US(3/1) → 같은 시각 KR 3건은 평소 US로 판단 → KR(3/3)부터 평소 KR → US(3/4)는 다름
    assert column(rows, DIFFERS_FROM_USUAL) == [1, 1, 1, 1, 1, 0, 1, 0]
    assert column(rows, SEEN_BEFORE) == [1, 0, 0, 0, 0, 1, 0, 1]


# 횟수가 같으면 먼저 나온 국가가 평소 국가 — PC와 ai-server가 같은 답을 내야 한다
def test_country_differs_from_usual_tieBreak_firstSeenCountryWins():
    rows = [
        tx("2026-03-01T10:00:00+09:00", "US"),
        tx("2026-03-02T10:00:00+09:00", "JP"),
        tx("2026-03-03T10:00:00+09:00", "US"),
        tx("2026-03-04T10:00:00+09:00", "JP"),
        tx("2026-03-05T10:00:00+09:00", "JP"),
        tx("2026-03-06T10:00:00+09:00", "JP"),
    ]

    # 3번째: US 1 · JP 1 동률 → 먼저 나온 US(알파벳 · 최근 순이면 JP) / 5번째: 2 · 2 동률 → US / 6번째: JP 3 → JP
    assert column(rows, DIFFERS_FROM_USUAL) == [1, 1, 0, 1, 1, 0]


# 국가 없는 결제는 -1(판단 불가)이고, 평소 국가 계산에도 안 들어간다
def test_country_feature_missingCountry_markedUnknownNotSameAsUsual():
    rows = [
        tx("2026-03-01T10:00:00+09:00", "KR"),
        tx("2026-03-02T10:00:00+09:00", None),
        tx("2026-03-03T10:00:00+09:00", " "),
        tx("2026-03-04T10:00:00+09:00", "NG"),
        tx("2026-03-05T10:00:00+09:00", "KR"),
    ]

    assert column(rows, DIFFERS_FROM_USUAL) == [0, UNKNOWN, UNKNOWN, 1, 0]
    assert column(rows, SEEN_BEFORE) == [1, UNKNOWN, UNKNOWN, 0, 1]


# 이력 없는 새 손님은 서비스 기본 국가와 비교한다 — 고위험 국가 새 손님(F3)도 "다름"
def test_country_feature_noHistory_comparesWithHomeCountry():
    rows = [
        tx("2026-03-01T10:00:00+09:00", "NG", customer="NEW1"),
        tx("2026-03-01T10:00:00+09:00", "KR", customer="NEW2"),
        tx("2026-03-02T10:00:00+09:00", "NG", customer="NEW1"),
    ]

    assert column(rows, DIFFERS_FROM_USUAL) == [1, 0, 0]
    assert column(rows, SEEN_BEFORE) == [0, 1, 1]
    assert column(rows[:2], DIFFERS_FROM_USUAL, home="US") == [1, 1]


# TabFormer 장소 칸 → 국가: 미국 주는 US 하나, 해외는 나라 이름, 온라인(앞 공백 포함) · 빈칸은 국가 없음
def test_tabformer_country_usStatesCollapsedAndOnlineMissing():
    assert tabformer_country("CA") == "US"
    assert tabformer_country("NY") == "US"
    assert tabformer_country("Italy") == "Italy"
    assert tabformer_country("United Kingdom") == "United Kingdom"
    assert tabformer_country(" ONLINE") is None
    assert tabformer_country("ONLINE") is None
    assert tabformer_country("") is None
    assert tabformer_country(None) is None

    rows = [
        {"paymentRequestedAt": "2016-03-01T10:00:00+09:00", "customerRef": "C1", "locationRef": "CA"},
        {"paymentRequestedAt": "2016-03-02T10:00:00+09:00", "customerRef": "C1", "locationRef": "NY"},
        {"paymentRequestedAt": "2016-03-03T10:00:00+09:00", "customerRef": "C1", "locationRef": " ONLINE"},
        {"paymentRequestedAt": "2016-03-04T10:00:00+09:00", "customerRef": "C1", "locationRef": "Italy"},
    ]
    for row in rows:
        row["country"] = tabformer_country(row["locationRef"])
    result = country_features(rows, home_country="US", country_key="country")

    # 다른 주(NY)는 다른 국가가 아니다 · 온라인은 판단 불가
    assert [r[DIFFERS_FROM_USUAL] for r in result] == [0, 0, UNKNOWN, 1]
