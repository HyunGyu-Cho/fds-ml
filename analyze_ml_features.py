
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import math

from features import FEATURE_COLUMNS, load_feature_rows


BASE_DIR = Path(__file__).resolve().parent
TRANSACTIONS_PATH = BASE_DIR / "data" / "transactions.csv"
EPISODES_PATH = BASE_DIR / "data" / "fraud_episodes.csv"

KST = timezone(timedelta(hours=9))

SPLIT_NAMES = ("Train", "Validation", "Test")

# 분포를 집중적으로 확인할 Feature
NUMERIC_FEATURES = [
    "amount",
    "customerPastTxCount",
    "customerAvgAmount",
    "amountRatio",
    "txCount5m",
    "txCount30m",
    "txCount24h",
    "minutesSinceLastTx",
    "deviceSeenBefore",
    "cardSeenBefore",
    "cardUsedByOtherCustomer",
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_time(text):
    return datetime.fromisoformat(
        text.replace("Z", "+00:00")
    ).astimezone(timezone.utc)


def percentage(part, total):
    if total == 0:
        return 0.0
    return part / total * 100


def percentile(values, fraction):
    values = sorted(values)

    if not values:
        return 0.0

    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)

    if low == high:
        return values[low]

    weight = position - low

    return (
        values[low] * (1 - weight)
        + values[high] * weight
    )


def analyze_feature_distribution(features, labels):
    print("\n=== 1. Feature 품질 분석 ===")

    print(f"Feature 수: {len(FEATURE_COLUMNS)}")
    print(f"Feature 생성 행 수: {len(features)}")

    normal_indices = [
        i for i, label in enumerate(labels) if label == 0
    ]
    fraud_indices = [
        i for i, label in enumerate(labels) if label == 1
    ]

    print(f"정상 거래: {len(normal_indices):,}건")
    print(f"사기 거래: {len(fraud_indices):,}건")

    # NaN 및 무한대 검사
    for index, row in enumerate(features):
        for name in NUMERIC_FEATURES:
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError(
                    f"비정상 Feature: {index}행, {name}"
                )

    print("숫자형 Feature 유한값 검사: PASS")

    print("\n[정상 vs 사기 Feature 평균]")
    print(
        f"{'Feature':<29}"
        f"{'정상 평균':>14}"
        f"{'사기 평균':>14}"
    )

    for name in NUMERIC_FEATURES:
        normal_values = [
            float(features[i][name])
            for i in normal_indices
        ]
        fraud_values = [
            float(features[i][name])
            for i in fraud_indices
        ]

        normal_mean = (
            sum(normal_values) / len(normal_values)
            if normal_values else 0.0
        )
        fraud_mean = (
            sum(fraud_values) / len(fraud_values)
            if fraud_values else 0.0
        )

        print(
            f"{name:<29}"
            f"{normal_mean:>14.3f}"
            f"{fraud_mean:>14.3f}"
        )

    print("\n[극단값 확인: 전체 거래]")
    for name in ("amount", "amountRatio", "txCount5m"):
        values = [float(row[name]) for row in features]

        print(
            f"{name}: "
            f"중앙값={percentile(values, 0.50):.2f}, "
            f"P95={percentile(values, 0.95):.2f}, "
            f"P99={percentile(values, 0.99):.2f}, "
            f"최대={max(values):.2f}"
        )

    print("\n[과거 이력 0건인 거래]")
    for label, indices in [
        (0, normal_indices),
        (1, fraud_indices),
    ]:
        cold_start = sum(
            features[i]["customerPastTxCount"] == 0
            for i in indices
        )

        print(
            f"fraudLabel={label}: "
            f"{cold_start:,}건 "
            f"({percentage(cold_start, len(indices)):.2f}%)"
        )


def determine_time_splits(transactions):
    """
    시간순 70% / 15% / 15% 후보 경계를 찾는다.

    동일 타임스탬프는 동일 분할에 배정한다.
    이 함수는 데이터를 이동하거나 삭제하지 않는다.
    """

    timestamps = [
        parse_time(row["paymentRequestedAt"])
        for row in transactions
    ]

    ordered_times = sorted(timestamps)
    total = len(ordered_times)

    if total < 3:
        raise ValueError("분할에 필요한 거래 수가 부족합니다.")

    train_boundary = ordered_times[int(total * 0.70)]
    test_boundary = ordered_times[int(total * 0.85)]

    if train_boundary >= test_boundary:
        raise ValueError(
            "시간 경계가 겹칩니다. 분할 비율을 재검토하세요."
        )

    splits = []

    for timestamp in timestamps:
        if timestamp < train_boundary:
            splits.append("Train")
        elif timestamp < test_boundary:
            splits.append("Validation")
        else:
            splits.append("Test")

    print("\n=== 2. 시간순 분할 후보 ===")
    print(
        "Train 종료 경계:",
        train_boundary.astimezone(KST).isoformat(),
    )
    print(
        "Test 시작 경계:",
        test_boundary.astimezone(KST).isoformat(),
    )

    return splits


def analyze_split_distribution(transactions, splits):
    print("\n[분할별 거래 및 사기 비율]")

    for split_name in SPLIT_NAMES:
        indices = [
            i for i, name in enumerate(splits)
            if name == split_name
        ]

        fraud_indices = [
            i for i in indices
            if transactions[i]["fraudLabel"] == "1"
        ]

        print(
            f"{split_name}: "
            f"전체 {len(indices):,}건 / "
            f"사기 {len(fraud_indices):,}건 / "
            f"사기율 "
            f"{percentage(len(fraud_indices), len(indices)):.2f}%"
        )

    print("\n[분할별 사기 시나리오 수]")

    scenario_names = sorted({
        row["scenarioType"]
        for row in transactions
        if row["fraudLabel"] == "1"
    })

    for scenario in scenario_names:
        counts = Counter(
            splits[i]
            for i, row in enumerate(transactions)
            if row["scenarioType"] == scenario
            and row["fraudLabel"] == "1"
        )

        print(
            f"{scenario:<23} "
            f"Train={counts['Train']:>4}, "
            f"Val={counts['Validation']:>4}, "
            f"Test={counts['Test']:>4}"
        )


def analyze_episode_boundaries(transactions, episodes, splits):
    print("\n=== 3. Episode 경계 검사 ===")

    payment_to_episode = {}

    for row in episodes:
        payment_id = row["paymentId"]
        episode_id = row["attackEventId"]

        if payment_id in payment_to_episode:
            raise ValueError(
                f"Episode 중복 연결: {payment_id}"
            )

        payment_to_episode[payment_id] = episode_id

    fraud_payments = {
        row["paymentId"]
        for row in transactions
        if row["fraudLabel"] == "1"
    }

    missing = fraud_payments - set(payment_to_episode)
    unexpected = set(payment_to_episode) - fraud_payments

    if missing or unexpected:
        raise ValueError(
            f"Episode 연결 오류: "
            f"누락={len(missing)}, "
            f"잘못된 연결={len(unexpected)}"
        )

    episode_splits = defaultdict(set)
    episode_payments = defaultdict(list)

    for i, row in enumerate(transactions):
        payment_id = row["paymentId"]

        if payment_id not in payment_to_episode:
            continue

        episode_id = payment_to_episode[payment_id]

        episode_splits[episode_id].add(splits[i])
        episode_payments[episode_id].append(payment_id)

    crossing_ids = [
        episode_id
        for episode_id, areas in episode_splits.items()
        if len(areas) > 1
    ]

    affected_payments = sum(
        len(episode_payments[episode_id])
        for episode_id in crossing_ids
    )

    print(f"전체 Episode: {len(episode_splits):,}개")
    print(f"경계를 넘는 Episode: {len(crossing_ids):,}개")
    print(f"관련 사기 거래: {affected_payments:,}건")

    print("\n[경계를 넘는 Episode 예시: 최대 10개]")

    for episode_id in crossing_ids[:10]:
        areas = episode_splits[episode_id]

        print(
            f"{episode_id}: "
            f"{', '.join(sorted(areas))} / "
            f"{len(episode_payments[episode_id])}건"
        )

    if not crossing_ids:
        print("없음: Episode 경계 충돌 미발견")

    print(
        "\n※ 실제 분할 시에는 경계를 넘는 Episode의 "
        "처리 방식을 확정해야 합니다."
    )


def main():
    transactions = read_csv(TRANSACTIONS_PATH)
    episodes = read_csv(EPISODES_PATH)
    features = load_feature_rows(TRANSACTIONS_PATH)

    if len(transactions) != len(features):
        raise ValueError("거래 수와 Feature 수가 다릅니다.")

    labels = [
        int(row["fraudLabel"])
        for row in transactions
    ]

    if not set(labels).issubset({0, 1}):
        raise ValueError("fraudLabel은 0 또는 1이어야 합니다.")

    analyze_feature_distribution(features, labels)

    splits = determine_time_splits(transactions)

    analyze_split_distribution(transactions, splits)

    analyze_episode_boundaries(
        transactions,
        episodes,
        splits,
    )

    print("\n분석 완료. CSV와 기존 소스는 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
