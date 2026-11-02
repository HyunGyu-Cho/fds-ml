
"""FDS 사기 시나리오별 거래 시퀀스 분석."""

import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median


CSV_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "transactions.csv"
)

KST = timezone(timedelta(hours=9))


def new_stats():
    """시나리오별 통계 저장 공간을 생성한다."""
    return {
        "count": 0,
        "customers": set(),
        "new_card": 0,
        "new_device": 0,
        "other_customer_card": 0,
        "extra_card": 0,
        "extra_device": 0,
        "night": 0,
        "amounts": [],
        "gaps": [],
    }


def load_transactions():
    """CSV를 불러와 거래 발생 순서대로 정렬한다."""

    with CSV_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    # 동일 시각의 거래는 paymentId 순서로 처리
    rows.sort(
        key=lambda tx: (
            tx["paymentRequestedAt"],
            int(tx["paymentId"]),
        )
    )

    return rows


def analyze_sequences(transactions):
    """과거 카드·기기 사용 이력과 거래 간격을 분석한다."""

    stats = defaultdict(new_stats)

    # 고객별 과거에 관측된 카드·기기
    seen_cards = defaultdict(set)
    seen_devices = defaultdict(set)

    # 특정 카드의 과거 사용자
    card_owners = defaultdict(set)

    # 동일 고객·시나리오의 마지막 사기 거래
    last_scenario_time = {}

    for tx in transactions:
        key = (
            tx["merchantId"],
            tx["customerRef"],
        )

        card = tx["cardToken"]
        device = tx["deviceId"]

        current_time = datetime.fromisoformat(
            tx["paymentRequestedAt"].replace(
                "Z", "+00:00"
            )
        )

        # 반드시 현재 거래를 이력에 추가하기 전에 계산
        is_new_card = card not in seen_cards[key]
        is_new_device = device not in seen_devices[key]

        # 다른 고객이 과거에 사용한 카드인지 확인
        used_by_other = bool(
            card_owners[card] - {key}
        )

        if int(tx["fraudLabel"]) == 1:
            scenario = tx["scenarioType"]
            item = stats[scenario]

            item["count"] += 1
            item["customers"].add(key)

            item["new_card"] += is_new_card
            item["new_device"] += is_new_device
            item["other_customer_card"] += used_by_other

            item["extra_card"] += (
                "EXTRA" in card.upper()
            )
            item["extra_device"] += (
                "EXTRA" in device.upper()
            )

            hour = current_time.astimezone(KST).hour
            item["night"] += 0 <= hour < 6

            item["amounts"].append(float(tx["amount"]))

            # 동일 고객·시나리오의 이전 사기 거래와 비교
            scenario_key = (*key, scenario)

            if scenario_key in last_scenario_time:
                previous = last_scenario_time[
                    scenario_key
                ]

                gap = (
                    current_time - previous
                ).total_seconds() / 60

                item["gaps"].append(gap)

            last_scenario_time[scenario_key] = (
                current_time
            )

        # 현재 거래를 처리한 후 이력에 반영
        seen_cards[key].add(card)
        seen_devices[key].add(device)
        card_owners[card].add(key)

    return stats


def display_results(stats):
    """시나리오별 통계를 사람이 읽기 쉽게 출력한다."""

    print("\n[사기 시나리오별 거래 행동 분석]")

    for scenario, item in sorted(stats.items()):
        count = item["count"]

        if count == 0:
            continue

        print(f"\n--- {scenario} ---")
        print(f"사기 거래 수: {count}")
        print(f"고유 고객 수: {len(item['customers'])}")

        print(
            "최초 관측 카드 비율: "
            f"{item['new_card'] / count:.2%}"
        )
        print(
            "최초 관측 기기 비율: "
            f"{item['new_device'] / count:.2%}"
        )
        print(
            "타 고객의 기존 카드 사용 비율: "
            f"{item['other_customer_card'] / count:.2%}"
        )

        print(
            "EXTRA 카드 비율: "
            f"{item['extra_card'] / count:.2%}"
        )
        print(
            "EXTRA 기기 비율: "
            f"{item['extra_device'] / count:.2%}"
        )

        print(
            "새벽 00~05시 거래 비율: "
            f"{item['night'] / count:.2%}"
        )
        print(
            "거래 금액 중앙값: "
            f"{median(item['amounts']):,.0f}원"
        )

        gaps = item["gaps"]

        if gaps:
            print(
                "동일 고객·시나리오 연속 거래 간격 "
                f"중앙값: {median(gaps):.1f}분"
            )
            print(
                "10분 이내 간격 비율: "
                f"{sum(g <= 10 for g in gaps) / len(gaps):.2%}"
            )
            print(
                "20~50분 간격 비율: "
                f"{sum(20 <= g <= 50 for g in gaps) / len(gaps):.2%}"
            )
        else:
            print("동일 고객·시나리오의 후속 거래 없음")


def main():
    transactions = load_transactions()

    print(f"전체 거래 수: {len(transactions):,}")

    stats = analyze_sequences(transactions)
    display_results(stats)

    print(
        "\n주의: 동일 고객·시나리오의 거래가 "
        "서로 다른 공격 사건일 수도 있습니다."
    )


if __name__ == "__main__":
    main()