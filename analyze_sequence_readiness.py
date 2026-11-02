
from collections import Counter
from pathlib import Path
from statistics import median

from features import build_features
from train_random_forest import read_csv, split_and_purge


BASE_DIR = Path(__file__).resolve().parent
SEQUENCE_LENGTHS = (3, 5, 10, 20)


def summarize(name, indices, features):
    past_counts = [
        int(features[i]["customerPastTxCount"])
        for i in indices
    ]

    if not past_counts:
        print(f"{name}: 거래 없음")
        return

    print(f"\n[{name}]")
    print(f"거래 수: {len(indices):,}")
    print(f"과거 거래 수 중앙값: {median(past_counts):.1f}")
    print(f"평균 과거 거래 수: {sum(past_counts)/len(past_counts):.2f}")

    print(
        f"{'시퀀스 길이':<14}"
        f"{'충분한 거래':>14}"
        f"{'비율':>12}"
        f"{'평균 채움률':>14}"
    )

    for length in SEQUENCE_LENGTHS:
        enough = sum(
            count >= length for count in past_counts
        )
        fill_rate = (
            sum(min(count, length) for count in past_counts)
            / (len(past_counts) * length)
        )

        print(
            f"{length:<14}"
            f"{enough:>14,}"
            f"{enough/len(past_counts):>11.2%}"
            f"{fill_rate:>13.2%}"
        )


def main():
    transactions = read_csv(
        BASE_DIR / "data" / "transactions.csv"
    )
    episodes = read_csv(
        BASE_DIR / "data" / "fraud_episodes.csv"
    )

    features = build_features(transactions)
    splits = split_and_purge(transactions, episodes)["indices"]

    print("=== Transformer 시퀀스 데이터 적합성 ===")

    for split_name, indices in splits.items():
        print(f"\n{'=' * 45}")
        print(split_name)

        unique_customers = {
            transactions[i]["customerRef"]
            for i in indices
        }
        print(f"고유 고객 수: {len(unique_customers):,}")

        for label, name in [
            ("0", "정상 거래"),
            ("1", "사기 거래"),
        ]:
            selected = [
                i for i in indices
                if transactions[i]["fraudLabel"] == label
            ]
            summarize(name, selected, features)

    print("\n=== Validation 사기 시나리오별 시퀀스 ===")

    groups = {}

    for index in splits["Validation"]:
        row = transactions[index]

        if row["fraudLabel"] != "1":
            continue

        groups.setdefault(row["scenarioType"], []).append(
            int(features[index]["customerPastTxCount"])
        )

    for scenario, counts in sorted(groups.items()):
        enough_5 = sum(c >= 5 for c in counts)
        enough_10 = sum(c >= 10 for c in counts)

        print(
            f"{scenario:<23}"
            f"전체={len(counts):>3} "
            f"과거5건+={enough_5:>3} "
            f"과거10건+={enough_10:>3}"
        )

    print("\n분석 완료: 기존 데이터와 모델은 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
