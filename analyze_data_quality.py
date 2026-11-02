
"""FDS 합성 거래 데이터의 기본 분포 및 시퀀스 품질 분석."""

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import median


CSV_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "transactions.csv"
)

KST = timezone(timedelta(hours=9))

# 정답 정보를 암시할 수 있는 식별자 문자열
LEAK_MARKERS = (
    "EXTRA",
    "FRAUD",
    "ATTACK",
    "STOLEN",
    "TEST",
    "ATO",
)


def load_transactions():
    """CSV를 읽고 분석에 필요한 자료형으로 변환한다."""

    with CSV_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        transactions = list(csv.DictReader(file))

    for tx in transactions:
        tx["amount"] = float(tx["amount"])
        tx["fraudLabel"] = int(tx["fraudLabel"])

        tx["_time"] = datetime.fromisoformat(
            tx["paymentRequestedAt"].replace("Z", "+00:00")
        )

    return transactions


def percentile(values, percent):
    """정렬된 금액 배열의 백분위수를 구한다."""

    values = sorted(values)

    index = int((len(values) - 1) * percent / 100)

    return values[index]


def analyze_distribution(transactions):
    """정상·사기별 거래 금액 및 승인 결과를 비교한다."""

    print("\n[1. 정상·사기 거래 분포]")

    for label, name in [(0, "정상"), (1, "사기")]:
        group = [
            tx for tx in transactions
            if tx["fraudLabel"] == label
        ]

        amounts = [tx["amount"] for tx in group]

        declined = sum(
            tx["authorizationResult"] == "DECLINED"
            for tx in group
        )

        print(f"\n{name} 거래: {len(group):,}건")
        print(f"금액 중앙값: {median(amounts):,.0f}원")
        print(f"금액 95백분위: {percentile(amounts, 95):,.0f}원")
        print(f"금액 최솟값: {min(amounts):,.0f}원")
        print(f"금액 최댓값: {max(amounts):,.0f}원")
        print(f"승인 거절률: {declined / len(group):.2%}")


def analyze_merchants(transactions):
    """가맹점별 정상·사기 거래 분포를 확인한다."""

    print("\n[2. 가맹점별 거래 분포]")

    counts = defaultdict(Counter)

    for tx in transactions:
        counts[tx["merchantId"]][tx["fraudLabel"]] += 1

    for merchant_id, labels in sorted(counts.items()):
        normal = labels[0]
        fraud = labels[1]
        total = normal + fraud

        print(
            f"merchantId={merchant_id}: "
            f"정상={normal:,}, 사기={fraud:,}, "
            f"사기 비율={fraud / total:.2%}"
        )


def analyze_hours(transactions):
    """KST 기준 정상·사기 거래 발생 시간대를 비교한다."""

    print("\n[3. 거래 시간대 분포(KST)]")

    counts = defaultdict(Counter)

    for tx in transactions:
        hour = tx["_time"].astimezone(KST).hour
        counts[tx["fraudLabel"]][hour] += 1

    print("시간 | 정상 | 사기")

    for hour in range(24):
        print(
            f"{hour:02d}시 | "
            f"{counts[0][hour]:5d} | "
            f"{counts[1][hour]:5d}"
        )


def analyze_history(transactions):
    """사기 거래 이전에 정상 이력이 존재하는지 확인한다."""

    print("\n[4. 고객별 과거 거래 이력]")

    normal_history = Counter()
    fraud_total = 0
    fraud_with_three = 0

    # 동일 시각은 원본 CSV 순서를 유지
    ordered = sorted(
        transactions,
        key=lambda tx: tx["_time"]
    )

    for tx in ordered:
        key = (
            tx["merchantId"],
            tx["customerRef"],
        )

        if tx["fraudLabel"] == 0:
            normal_history[key] += 1

        else:
            fraud_total += 1

            if normal_history[key] >= 3:
                fraud_with_three += 1

    print(f"사기 거래 수: {fraud_total:,}건")
    print(
        "이전 정상 거래 3건 이상 보유: "
        f"{fraud_with_three:,}건"
    )
    print(
        "이전 정상 거래 3건 미만: "
        f"{fraud_total - fraud_with_three:,}건"
    )


def analyze_identifier_leakage(transactions):
    """식별자에 정답을 암시하는 문자열이 있는지 점검한다."""

    print("\n[5. 식별자 문자열 누수 위험]")

    fields = ("cardToken", "deviceId")
    found = False

    for field in fields:
        for marker in LEAK_MARKERS:
            counts = Counter(
                tx["fraudLabel"]
                for tx in transactions
                if marker in tx[field].upper()
            )

            if not counts:
                continue

            found = True

            print(
                f"{field} / {marker}: "
                f"정상={counts[0]}, 사기={counts[1]}"
            )

    if not found:
        print("검사 대상 문자열 패턴은 발견되지 않았습니다.")

    print(
        "주의: 이 검사는 일부 문자열 패턴만 확인하며, "
        "누수 부재를 증명하지 않습니다."
    )


def main():
    transactions = load_transactions()

    print(f"분석 대상: {CSV_PATH}")
    print(f"거래 건수: {len(transactions):,}")

    analyze_distribution(transactions)
    analyze_merchants(transactions)
    analyze_hours(transactions)
    analyze_history(transactions)
    analyze_identifier_leakage(transactions)


if __name__ == "__main__":
    main()