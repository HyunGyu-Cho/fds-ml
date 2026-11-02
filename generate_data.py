"""DonWorry FDS 합성 거래 데이터 및 공격 사건 메타데이터 생성."""

import random
import uuid
from collections import Counter, defaultdict
from datetime import datetime

from profiles import generate_customer_profiles
from transactions import (
    KST,
    START_DATE,
    END_DATE,
    generate_normal_transactions,
)
from scenarios import SCENARIOS, generate_fraud_transactions
from csv_export import CSV_COLUMNS, save_transactions_csv
from episode_export import save_episode_links_csv


SEED = 42
CUSTOMER_COUNT = 1500
NORMAL_COUNT = 18000
FRAUD_COUNT = 2000
TOTAL_COUNT = NORMAL_COUNT + FRAUD_COUNT


def display_kst(utc_string: str) -> str:
    """UTC 문자열을 한국 표준시로 변환한다."""
    utc_time = datetime.fromisoformat(utc_string.replace("Z", "+00:00"))
    return utc_time.astimezone(KST).isoformat()


def make_opaque_identifier(kind: str, source: str) -> str:
    """동일한 가상 카드·기기 ID를 항상 동일한 UUID5 형식으로 변환한다."""
    if kind not in {"CARD", "DEVICE"}:
        raise ValueError(f"지원하지 않는 식별자 유형: {kind}")

    value = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"donworry-fds-ml:synthetic:v1:{kind}:{source}",
    )
    return f"SIM-{kind}-{value.hex.upper()}"


def normalize_identifiers(customers, transactions):
    """고객 프로필과 거래의 카드·기기 식별자 형식을 통일한다."""
    for customer in customers:
        customer["cardTokens"] = [
            make_opaque_identifier("CARD", card)
            for card in customer["cardTokens"]
        ]
        customer["deviceIds"] = [
            make_opaque_identifier("DEVICE", device)
            for device in customer["deviceIds"]
        ]

    for tx in transactions:
        tx["cardToken"] = make_opaque_identifier("CARD", tx["cardToken"])
        tx["deviceId"] = make_opaque_identifier("DEVICE", tx["deviceId"])


def build_dataset_with_episodes(seed: int):
    """거래 데이터와 각 사기 거래의 공격 사건 연결 정보를 생성한다."""
    rng = random.Random(seed)

    customers = generate_customer_profiles(rng=rng, count=CUSTOMER_COUNT)
    normal = generate_normal_transactions(
        rng=rng, customers=customers, count=NORMAL_COUNT
    )
    fraud = generate_fraud_transactions(
        rng=rng,
        customers=customers,
        normal_transactions=normal,
        count=FRAUD_COUNT,
    )
    transactions = [dict(tx) for tx in normal + fraud]

    # 정답 메타데이터를 분리하기 전에 식별자의 사용 관계를 유지한다.
    normalize_identifiers(customers, transactions)

    transactions.sort(
        key=lambda tx: (
            tx["paymentRequestedAt"],
            tx["merchantId"],
            tx["customerRef"],
            tx["amount"],
        )
    )

    episode_links = []

    for index, tx in enumerate(transactions, start=1):
        # 사기 생성 단계의 내부 사건 ID는 일반 거래 파일에서 제거한다.
        attack_event_id = tx.pop("_attackEventId", None)
        tx["paymentId"] = index
        tx["orderRef"] = f"SIM-ORDER-{index:08d}"
        evaluation_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"donworry-{seed}-{index}-{tx['paymentRequestedAt']}",
        )
        tx["evaluationRequestId"] = str(evaluation_id)

        if tx["fraudLabel"] == 1:
            if not attack_event_id:
                raise ValueError(f"사기 거래 {index}: 사건 ID가 없습니다.")
            episode_links.append(
                {
                    "attackEventId": attack_event_id,
                    "scenarioType": tx["scenarioType"],
                    "paymentId": index,
                }
            )
        elif attack_event_id is not None:
            raise ValueError(f"정상 거래 {index}: 사건 ID가 존재합니다.")

    return customers, transactions, episode_links


def build_dataset(seed: int):
    """기존 코드와 테스트에서 사용한 반환값 2개 인터페이스를 유지한다."""
    customers, transactions, _ = build_dataset_with_episodes(seed)
    return customers, transactions


def validate_dataset(customers, transactions):
    """기존 고객·거래 건수, 시각, 식별자, Label 검증."""
    assert len(customers) == CUSTOMER_COUNT
    assert len(transactions) == TOTAL_COUNT

    labels = Counter(tx["fraudLabel"] for tx in transactions)
    assert labels[0] == NORMAL_COUNT
    assert labels[1] == FRAUD_COUNT

    scenarios = Counter(
        tx["scenarioType"]
        for tx in transactions
        if tx["fraudLabel"] == 1
    )
    assert set(scenarios) == set(SCENARIOS)
    assert sum(scenarios.values()) == FRAUD_COUNT

    assert len({tx["paymentId"] for tx in transactions}) == TOTAL_COUNT
    assert len({tx["orderRef"] for tx in transactions}) == TOTAL_COUNT
    assert len({tx["evaluationRequestId"] for tx in transactions}) == TOTAL_COUNT

    payment_times = [tx["paymentRequestedAt"] for tx in transactions]
    assert payment_times == sorted(payment_times)

    assert all(
        tx["orderCreatedAt"] <= tx["paymentRequestedAt"]
        for tx in transactions
    )
    assert all(
        START_DATE
        <= datetime.fromisoformat(
            tx["paymentRequestedAt"].replace("Z", "+00:00")
        ).astimezone(KST)
        < END_DATE
        for tx in transactions
    )

    assert all(tx["amount"] > 0 for tx in transactions)
    assert all(
        tx["authorizationResult"] in {"APPROVED", "DECLINED"}
        for tx in transactions
    )
    assert all(
        "ipAddress" not in tx and "ipCountryCode" not in tx
        for tx in transactions
    )
    assert all(
        tx["buyerEmail"].endswith("@example.invalid")
        for tx in transactions
    )
    assert all(set(tx) == set(CSV_COLUMNS) for tx in transactions)

    return scenarios


def validate_episode_links(transactions, episode_links):
    """각 사기 거래가 정확히 하나의 사건에 소속되는지 검증한다."""
    assert len(episode_links) == FRAUD_COUNT
    by_payment = {tx["paymentId"]: tx for tx in transactions}
    fraud_ids = {
        tx["paymentId"] for tx in transactions if tx["fraudLabel"] == 1
    }
    linked_ids = [link["paymentId"] for link in episode_links]
    assert len(linked_ids) == len(set(linked_ids))
    assert set(linked_ids) == fraud_ids

    episodes = defaultdict(list)
    for link in episode_links:
        tx = by_payment[link["paymentId"]]
        assert tx["fraudLabel"] == 1
        assert tx["scenarioType"] == link["scenarioType"]
        episodes[link["attackEventId"]].append(tx)

    for attack_id, group in episodes.items():
        scenario = group[0]["scenarioType"]
        assert scenario in SCENARIOS, attack_id
        assert len(group) >= SCENARIOS[scenario], attack_id
        assert len({tx["merchantId"] for tx in group}) == 1, attack_id
        assert len({tx["customerRef"] for tx in group}) == 1, attack_id
        assert len({tx["scenarioType"] for tx in group}) == 1, attack_id

    return len(episodes)


def main():
    customers, transactions, episode_links = build_dataset_with_episodes(SEED)
    scenarios = validate_dataset(customers, transactions)
    episode_count = validate_episode_links(transactions, episode_links)

    labels = Counter(tx["fraudLabel"] for tx in transactions)
    print(f"Random Seed: {SEED}")
    print(f"생성된 고객 수: {len(customers)}")
    print(f"전체 거래 수: {len(transactions)}")
    print(f"정상 거래 수: {labels[0]}")
    print(f"사기 거래 수: {labels[1]}")

    print("\n[사기 시나리오별 거래 수]")
    for name in SCENARIOS:
        print(f"{name}: {scenarios[name]}건")

    merchant_counts = Counter(tx["merchantId"] for tx in transactions)
    print("\n[가맹점별 전체 거래 수]")
    for merchant_id, count in sorted(merchant_counts.items()):
        print(f"merchantId={merchant_id}: {count}건")

    authorization_counts = Counter(
        tx["authorizationResult"] for tx in transactions
    )
    print("\n[카드 승인 결과]")
    for result, count in sorted(authorization_counts.items()):
        print(f"{result}: {count}건")

    first_time = transactions[0]["paymentRequestedAt"]
    last_time = transactions[-1]["paymentRequestedAt"]
    print("\n[거래 기간]")
    print(f"첫 거래(KST): {display_kst(first_time)}")
    print(f"마지막 거래(KST): {display_kst(last_time)}")
    print("\n기본 데이터 검증: PASS")
    print(f"공격 사건 연결 검증: PASS ({episode_count}개 사건)")

    replay_customers, replay_transactions, replay_links = (
        build_dataset_with_episodes(SEED)
    )
    assert customers == replay_customers
    assert transactions == replay_transactions
    assert episode_links == replay_links
    print("Seed 재현성 검증: PASS")

    csv_path, saved_count = save_transactions_csv(transactions)
    assert saved_count == TOTAL_COUNT
    print("\n[CSV 저장 결과]")
    print(f"저장 경로: {csv_path}")
    print(f"저장 거래 수: {saved_count}")
    print("CSV 저장: PASS")

    episode_path, saved_links = save_episode_links_csv(episode_links)
    assert saved_links == FRAUD_COUNT
    print("\n[사건 메타데이터 저장 결과]")
    print(f"저장 경로: {episode_path}")
    print(f"연결 거래 수: {saved_links}")
    print(f"고유 공격 사건 수: {episode_count}")
    print("사건 CSV 저장: PASS")

    print("\n[정상 거래 예시]")
    print(next(tx for tx in transactions if tx["fraudLabel"] == 0))
    print("\n[사기 거래 예시]")
    print(next(tx for tx in transactions if tx["fraudLabel"] == 1))


if __name__ == "__main__":
    main()