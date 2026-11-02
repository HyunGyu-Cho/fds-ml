from __future__ import annotations

import csv
import random
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from csv_export import CSV_COLUMNS
from episode_export import save_episode_links_csv
from generate_data import normalize_identifiers
from profiles import generate_customer_profiles
from scenarios import generate_fraud_transactions
from transactions import END_DATE, generate_normal_transactions, to_utc_string

# v3 합성 데이터 (Ⓥ3) — 기존 생성기(v1 · v2 데이터)는 그대로 두고, 그 결과 위에 접속 국가와 현실적인 상황 4가지를 입힌다
# 비율 · 국가는 모두 실제 통계가 아닌 실험용 가정. 국가는 IP 원본 없이 국가 코드만 둔다

SEED = 42
CUSTOMER_COUNT = 10_000
NORMAL_COUNT = 199_000
FRAUD_COUNT = 1_000
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "v3"

V3_COLUMNS = (*CSV_COLUMNS, "ipCountryCode")
CONDITION_COLUMNS = ("paymentId", "condition")

HOME_COUNTRY = "KR"
FOREIGN_COUNTRIES = ("US", "JP", "VN", "TH", "FR")
HIGH_RISK_COUNTRIES = ("NG",)

# 해외인데 정상: 해외 거주 손님(손님 비율) + 여행 중 결제(정상 결제 비율) — 합쳐서 정상의 약 2%
ABROAD_RESIDENT_CUSTOMER_RATE = 0.008
ABROAD_TRAVEL_TX_RATE = 0.012
# 정상인데 이상해 보임: 정상 결제의 약 4%를 세 가지로 나눔
ODD_TX_RATE = 0.04
# 사기 공격 묶음 중 피해 손님의 평소 국가로 접속(정상인 척) / 해외 중 고위험 국가
FRAUD_SAME_COUNTRY_RATE = 0.5
FRAUD_HIGH_RISK_RATE = 0.4

NORMAL = "NORMAL"
ODD_BIG_FIRST = "ODD_BIG_FIRST"
ODD_NEW_CARD_DEVICE = "ODD_NEW_CARD_DEVICE"
ODD_BURST = "ODD_BURST"
ABROAD_RESIDENT = "ABROAD_RESIDENT"
ABROAD_TRAVEL = "ABROAD_TRAVEL"
FRAUD_SAME_COUNTRY = "FRAUD_SAME_COUNTRY"
FRAUD_FOREIGN = "FRAUD_FOREIGN"
FRAUD_HIGH_RISK = "FRAUD_HIGH_RISK"
NORMAL_CONDITIONS = (NORMAL, ODD_BIG_FIRST, ODD_NEW_CARD_DEVICE, ODD_BURST, ABROAD_RESIDENT, ABROAD_TRAVEL)
FRAUD_CONDITIONS = (FRAUD_SAME_COUNTRY, FRAUD_FOREIGN, FRAUD_HIGH_RISK)


def _time(tx: dict) -> datetime:
    return datetime.fromisoformat(tx["paymentRequestedAt"].replace("Z", "+00:00"))


def _by_customer(normal: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for tx in normal:
        groups[tx["customerRef"]].append(tx)
    for group in groups.values():
        group.sort(key=lambda tx: (tx["paymentRequestedAt"], tx["amount"]))
    return groups


def _apply_normal_conditions(rng: random.Random, customers: list[dict], normal: list[dict]) -> dict[str, str]:
    # 정상 결제에 국가 · 상황을 입히고, 손님별 평소 국가를 돌려준다 (사기 묶음의 "정상인 척" 국가로 씀)
    homes = {c["customerRef"]: HOME_COUNTRY for c in customers}
    for customer in rng.sample(customers, round(len(customers) * ABROAD_RESIDENT_CUSTOMER_RATE)):
        homes[customer["customerRef"]] = rng.choice(FOREIGN_COUNTRIES)
    for tx in normal:
        tx["ipCountryCode"] = homes[tx["customerRef"]]
        tx["_condition"] = ABROAD_RESIDENT if homes[tx["customerRef"]] != HOME_COUNTRY else NORMAL

    odd_each = round(len(normal) * ODD_TX_RATE / 3)
    groups = _by_customer(normal)
    refs = sorted(ref for ref, group in groups.items() if homes[ref] == HOME_COUNTRY and len(group) >= 4)

    # 선물 묶음: 같은 손님 3건을 첫 결제 뒤 1~5분 간격으로 붙인다 (시각이 바뀌므로 다른 상황보다 먼저)
    burst = 0
    for ref in rng.sample(refs, len(refs)):
        if burst >= odd_each:
            break
        group = [tx for tx in groups[ref] if tx["_condition"] == NORMAL]
        if len(group) < 3:
            continue
        first = rng.randrange(len(group) - 2)
        base = _time(group[first])
        if base >= END_DATE - timedelta(minutes=30):
            continue
        when = base
        for tx in group[first:first + 3]:
            if tx is not group[first]:
                when += timedelta(seconds=rng.randint(60, 300))
                tx["paymentRequestedAt"] = to_utc_string(when)
                tx["orderCreatedAt"] = to_utc_string(when - timedelta(minutes=rng.randint(0, 3)))
                tx["cardToken"] = group[first]["cardToken"]
                tx["deviceId"] = group[first]["deviceId"]
            tx["_condition"] = ODD_BURST
        burst += 3

    groups = _by_customer(normal)
    # 여행: 평소 국가에서 결제하던 단골이 2~5일 동안 해외에서 결제
    travel_target = round(len(normal) * ABROAD_TRAVEL_TX_RATE)
    travel = 0
    for ref in rng.sample(refs, len(refs)):
        if travel >= travel_target:
            break
        group = groups[ref]
        start = _time(group[rng.randrange(1, len(group))])
        end = start + timedelta(days=rng.randint(2, 5))
        country = rng.choice(FOREIGN_COUNTRIES)
        for tx in group:
            if start <= _time(tx) <= end and tx["_condition"] == NORMAL:
                tx["ipCountryCode"] = country
                tx["_condition"] = ABROAD_TRAVEL
                travel += 1

    # 새 손님 첫 결제가 고액(50만 원 이상) / 단골이 카드 · 기기를 한꺼번에 바꿈(그 뒤로 새 카드 · 기기 사용)
    big = changed = 0
    for ref in rng.sample(sorted(groups), len(groups)):
        group = groups[ref]
        # 두 상황이 손님을 번갈아 나눠 갖도록 매번 어느 쪽부터 볼지 무작위로 정한다
        want_big = big < odd_each and (changed >= odd_each or rng.random() < 0.5)
        if want_big and group[0]["_condition"] == NORMAL:
            group[0]["amount"] = rng.randrange(500_000, 900_001, 100)
            group[0]["_condition"] = ODD_BIG_FIRST
            big += 1
        elif changed < odd_each and homes[ref] == HOME_COUNTRY and len(group) >= 3:
            switch = rng.randrange(2, len(group))
            if group[switch]["_condition"] != NORMAL:
                continue
            for tx in group[switch:]:
                tx["cardToken"] = f"SIM-CARD-{ref}-NEW"
                tx["deviceId"] = f"SIM-DEVICE-{ref}-NEW"
            group[switch]["_condition"] = ODD_NEW_CARD_DEVICE
            changed += 1
        if big >= odd_each and changed >= odd_each:
            break
    return homes


def _apply_fraud_conditions(rng: random.Random, fraud: list[dict], homes: dict[str, str]) -> None:
    # 공격 묶음 하나는 한 곳에서 접속한다 — 피해 손님 평소 국가(정상인 척) · 해외 · 고위험 국가
    episodes = defaultdict(list)
    for tx in fraud:
        episodes[tx["_attackEventId"]].append(tx)
    for key in sorted(episodes):
        home = homes[episodes[key][0]["customerRef"]]
        if rng.random() < FRAUD_SAME_COUNTRY_RATE:
            country, condition = home, FRAUD_SAME_COUNTRY
        elif rng.random() < FRAUD_HIGH_RISK_RATE:
            country, condition = rng.choice(HIGH_RISK_COUNTRIES), FRAUD_HIGH_RISK
        else:
            country = rng.choice([c for c in (HOME_COUNTRY, *FOREIGN_COUNTRIES) if c != home])
            condition = FRAUD_FOREIGN
        for tx in episodes[key]:
            tx["ipCountryCode"] = country
            tx["_condition"] = condition


def build_dataset_v3(seed: int = SEED, customer_count: int = CUSTOMER_COUNT,
                     normal_count: int = NORMAL_COUNT, fraud_count: int = FRAUD_COUNT):
    # 기존 생성기와 같은 순서로 손님 · 정상 · 사기를 만든 뒤, v3 상황은 별도 난수로 입힌다 (기존 난수 흐름 유지)
    rng = random.Random(seed)
    customers = generate_customer_profiles(rng=rng, count=customer_count)
    normal = generate_normal_transactions(rng=rng, customers=customers, count=normal_count)
    fraud = generate_fraud_transactions(rng=rng, customers=customers, normal_transactions=normal, count=fraud_count)

    v3_rng = random.Random(seed * 1000 + 3)
    homes = _apply_normal_conditions(v3_rng, customers, normal)
    _apply_fraud_conditions(v3_rng, fraud, homes)

    transactions = [dict(tx) for tx in normal + fraud]
    normalize_identifiers(customers, transactions)
    transactions.sort(key=lambda tx: (tx["paymentRequestedAt"], tx["merchantId"], tx["customerRef"], tx["amount"]))

    episode_links, conditions = [], []
    for index, tx in enumerate(transactions, start=1):
        attack_event_id = tx.pop("_attackEventId", None)
        condition = tx.pop("_condition")
        tx["paymentId"] = index
        tx["orderRef"] = f"SIM-ORDER-{index:08d}"
        tx["evaluationRequestId"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"donworry-v3-{seed}-{index}-{tx['paymentRequestedAt']}"))
        conditions.append({"paymentId": index, "condition": condition})
        if tx["fraudLabel"] == 1:
            episode_links.append({"attackEventId": attack_event_id, "scenarioType": tx["scenarioType"], "paymentId": index})
    return transactions, episode_links, conditions


def _write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_dataset_v3(transactions, episode_links, conditions, output_dir: Path = OUTPUT_DIR) -> None:
    # 상황 기록은 거래 파일과 따로 둔다 — 거래 파일에 있으면 학습 때 정답이 섞일 수 있다
    _write_csv(output_dir / "transactions.csv", V3_COLUMNS, transactions)
    _write_csv(output_dir / "conditions.csv", CONDITION_COLUMNS, conditions)
    save_episode_links_csv(episode_links, output_dir / "fraud_episodes.csv")


def main():
    transactions, episode_links, conditions = build_dataset_v3()
    save_dataset_v3(transactions, episode_links, conditions)
    labels = Counter(tx["fraudLabel"] for tx in transactions)
    print(f"seed {SEED} · 손님 {CUSTOMER_COUNT:,} · 거래 {len(transactions):,} (정상 {labels[0]:,} · 사기 {labels[1]:,}, 사기율 {labels[1] / len(transactions):.2%})")
    print("상황별 거래 수:")
    for condition, count in sorted(Counter(c["condition"] for c in conditions).items()):
        print(f"  {condition:<22} {count:>7,}")
    countries = Counter(tx["ipCountryCode"] for tx in transactions)
    print("접속 국가:", " · ".join(f"{k} {v:,}" for k, v in countries.most_common()))
    print(f"저장: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
