from __future__ import annotations

from collections import defaultdict
from itertools import groupby
from typing import Mapping, Sequence

from features import _parse_timestamp

# 국가 칸 두 정의 (Ⓥ3) — 국가 이름 자체는 Feature로 넣지 않고 "앞선 결제와 비교한 결과"만 넣는다
# A countrySeenBefore: 앞서 이 국가에서 결제한 적 있나 / B countryDiffersFromUsual: 평소 국가와 다른가
# 값: 1 · 0, 이번 결제 국가가 없으면 -1(판단 불가 — "같음"으로 세지 않는다)

SEEN_BEFORE = "countrySeenBefore"
DIFFERS_FROM_USUAL = "countryDiffersFromUsual"
UNKNOWN = -1

HOME_COUNTRY_DONWORRY = "KR"
HOME_COUNTRY_TABFORMER = "US"


class _CountryHistory:
    # 손님별 앞선 결제의 국가 횟수와 처음 나온 순서 (동률이면 먼저 나온 국가가 평소 국가)
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.first_seen: dict[str, int] = {}

    def add(self, country: str) -> None:
        if country not in self.counts:
            self.first_seen[country] = len(self.first_seen)
            self.counts[country] = 0
        self.counts[country] += 1

    def usual(self, home_country: str) -> str:
        # 앞선 국가 기록이 없으면 서비스 기본 국가를 평소 국가로 본다 (새 손님 해외 결제도 "다름")
        if not self.counts:
            return home_country
        return min(self.counts, key=lambda c: (-self.counts[c], self.first_seen[c]))

    def seen(self, country: str, home_country: str) -> bool:
        # B와 같은 기준: 앞선 국가 기록이 없으면 기본 국가만 "본 적 있음"
        if not self.counts:
            return country == home_country
        return country in self.counts


def _country(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    text = "" if value is None else str(value).strip()
    return text or None


def country_features(
    transactions: Sequence[Mapping[str, object]],
    *,
    home_country: str,
    country_key: str = "ipCountryCode",
) -> list[dict[str, int]]:
    # 원래 순서대로 두 칸을 돌려준다 — 같은 시각 결제끼리는 서로 안 본다 (features.py와 같은 규칙)
    times = [_parse_timestamp(row["paymentRequestedAt"]) for row in transactions]
    order = sorted(range(len(transactions)), key=lambda i: times[i])
    histories: dict[str, _CountryHistory] = defaultdict(_CountryHistory)
    result: list[dict[str, int]] = [{} for _ in transactions]

    for _, group in groupby(order, key=lambda i: times[i]):
        group = list(group)
        for i in group:
            country = _country(transactions[i], country_key)
            history = histories[str(transactions[i]["customerRef"])]
            if country is None:
                result[i] = {SEEN_BEFORE: UNKNOWN, DIFFERS_FROM_USUAL: UNKNOWN}
            else:
                result[i] = {
                    SEEN_BEFORE: int(history.seen(country, home_country)),
                    DIFFERS_FROM_USUAL: int(country != history.usual(home_country)),
                }
        for i in group:
            country = _country(transactions[i], country_key)
            if country is not None:
                histories[str(transactions[i]["customerRef"])].add(country)

    return result


def tabformer_country(location_ref: object) -> str | None:
    # TabFormer 장소 칸을 돈워리 접속 국가처럼 국가 단위로 바꾼다
    # 미국 주 약자(2글자 대문자) → US, 해외는 나라 이름 그대로, 온라인(" ONLINE") · 빈칸 → 국가 없음
    text = "" if location_ref is None else str(location_ref).strip()
    if not text or text.upper() == "ONLINE":
        return None
    if len(text) == 2 and text.isalpha() and text.isupper():
        return HOME_COUNTRY_TABFORMER
    return text
