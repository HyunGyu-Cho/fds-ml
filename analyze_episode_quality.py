"""DonWorry FDS 합성 데이터의 사건별 품질을 탐색적으로 분석한다.

입력(읽기 전용): fds-ml/data/transactions.csv, fraud_episodes.csv
출력: 콘솔 통계. 원본 데이터/생성기는 수정하지 않는다.
"""

import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import groupby
from pathlib import Path
from statistics import median


DATA_DIR = Path(__file__).resolve().parent / "data"
KST = timezone(timedelta(hours=9))
MIN_HISTORY = 3


def read_csv(path, required_columns):
    """CSV를 읽고 필수 컬럼이 있는지 확인한다."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV를 찾을 수 없습니다: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = set(required_columns) - columns
        if missing:
            raise ValueError(f"{path.name} 필수 컬럼 누락: {sorted(missing)}")
        return list(reader)


def parse_time(value):
    """저장된 UTC ISO 8601 문자열을 datetime으로 변환한다."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_data():
    """거래 CSV와 사건 매핑 CSV를 paymentId 기준으로 연결한다."""
    transactions = read_csv(
        DATA_DIR / "transactions.csv",
        ("paymentId", "merchantId", "customerRef", "amount",
         "paymentRequestedAt", "fraudLabel", "scenarioType"),
    )
    links = read_csv(
        DATA_DIR / "fraud_episodes.csv",
        ("attackEventId", "scenarioType", "paymentId"),
    )

    payments = {}
    for tx in transactions:
        tx["paymentId"] = int(tx["paymentId"])
        tx["amount"] = float(tx["amount"])
        tx["fraudLabel"] = int(tx["fraudLabel"])
        tx["_time"] = parse_time(tx["paymentRequestedAt"])
        if tx["paymentId"] in payments:
            raise ValueError(f"중복 paymentId: {tx['paymentId']}")
        payments[tx["paymentId"]] = tx

    episodes = defaultdict(list)
    mapped_ids = set()
    for link in links:
        payment_id = int(link["paymentId"])
        if payment_id not in payments:
            raise ValueError(f"존재하지 않는 paymentId: {payment_id}")
        if payment_id in mapped_ids:
            raise ValueError(f"여러 사건에 중복 연결된 paymentId: {payment_id}")
        mapped_ids.add(payment_id)

        tx = payments[payment_id]
        if tx["fraudLabel"] != 1 or tx["scenarioType"] != link["scenarioType"]:
            raise ValueError(f"사건과 거래의 시나리오/Label 불일치: {payment_id}")
        episodes[link["attackEventId"]].append(tx)

    expected = {tx["paymentId"] for tx in transactions if tx["fraudLabel"] == 1}
    if mapped_ids != expected:
        raise ValueError("사기 거래와 공격 사건 연결 정보가 일치하지 않습니다.")

    return transactions, episodes


def summarize_numbers(values):
    """숫자 배열의 최소·중앙·최대값을 문자열로 나타낸다."""
    return (f"최소 {min(values):,.1f} / 중앙 {median(values):,.1f} / "
            f"최대 {max(values):,.1f}")


def analyze_split_payment(transactions, episodes):
    """SPLIT_PAYMENT 사건의 건수·누적금액·시간 간격을 검사한다."""
    split_groups = []
    for group in episodes.values():
        if group[0]["scenarioType"] == "SPLIT_PAYMENT":
            split_groups.append(sorted(group, key=lambda tx: tx["_time"]))

    if not split_groups:
        raise ValueError("SPLIT_PAYMENT 사건이 없습니다.")

    totals = [sum(tx["amount"] for tx in group) for group in split_groups]
    sizes = [len(group) for group in split_groups]
    durations = [
        (group[-1]["_time"] - group[0]["_time"]).total_seconds() / 60
        for group in split_groups
    ]
    all_below_500k = sum(all(tx["amount"] < 500_000 for tx in group)
                         for group in split_groups)
    totals_over_500k = sum(total > 500_000 for total in totals)
    all_gaps_in_range = 0
    for group in split_groups:
        gaps = [
            (after["_time"] - before["_time"]).total_seconds() / 60
            for before, after in zip(group, group[1:])
        ]
        all_gaps_in_range += all(12 <= gap <= 25 for gap in gaps)

    # 개별 금액의 중첩 정도: 정상 거래의 5~95백분위 구간을 기준으로 비교
    normal_amounts = sorted(tx["amount"] for tx in transactions
                            if tx["fraudLabel"] == 0)
    lower = normal_amounts[int((len(normal_amounts) - 1) * 0.05)]
    upper = normal_amounts[int((len(normal_amounts) - 1) * 0.95)]
    split_amounts = [tx["amount"] for group in split_groups for tx in group]
    overlap = sum(lower <= amount <= upper for amount in split_amounts)

    print("\n[1. SPLIT_PAYMENT — 사건 단위]")
    print(f"사건 수: {len(split_groups):,}개 / 거래 수: {len(split_amounts):,}건")
    print(f"사건별 거래 수: {summarize_numbers(sizes)}")
    print(f"사건별 총액(원): {summarize_numbers(totals)}")
    print(f"사건 진행시간(분): {summarize_numbers(durations)}")
    print(f"총액 50만원 초과 사건: {totals_over_500k}/{len(split_groups)}")
    print(f"개별 거래 전부 50만원 미만인 사건: {all_below_500k}/{len(split_groups)}")
    print(f"모든 연속 간격이 12~25분인 사건: {all_gaps_in_range}/{len(split_groups)}")
    print(f"정상 거래 금액 5~95백분위: {lower:,.0f}~{upper:,.0f}원")
    print(f"해당 구간 안의 SPLIT_PAYMENT 거래: {overlap}/{len(split_amounts)} "
          f"({overlap / len(split_amounts):.1%})")
    print("참고: 총액이 50만원을 넘는다고 실제 원 결제를 분할한 것은 아닙니다.")


def hour_of_day(time):
    """KST 기준 소수 시간(예: 13:30 → 13.5)을 계산한다."""
    local_time = time.astimezone(KST)
    return local_time.hour + local_time.minute / 60 + local_time.second / 3600


def circular_distance(a, b):
    """자정 경계까지 고려한 두 시간대 사이의 최소 시간 차이."""
    diff = abs(a - b) % 24
    return min(diff, 24 - diff)


def typical_hour(history):
    """이전 정상 거래와 총 시간대 거리가 가장 작은 대표 시각을 고른다."""
    hours = [hour_of_day(tx["_time"]) for tx in history]
    return min(range(24), key=lambda h: sum(circular_distance(h, v) for v in hours))


def analyze_behavioral_anomaly(transactions):
    """각 거래 시점보다 과거의 정상 이력만 사용해 행동 이탈을 측정한다."""
    ordered = sorted(transactions, key=lambda tx: (tx["_time"], tx["paymentId"]))
    normal_history = defaultdict(list)
    comparison = {"NORMAL": [], "BEHAVIORAL_ANOMALY": []}
    skipped = {"NORMAL": 0, "BEHAVIORAL_ANOMALY": 0}

    # 같은 시각의 거래는 한 묶음으로 처리해 동시 거래를 과거로 사용하지 않는다.
    for _, same_time in groupby(ordered, key=lambda tx: tx["_time"]):
        same_time = list(same_time)
        for tx in same_time:
            category = ("NORMAL" if tx["fraudLabel"] == 0 else tx["scenarioType"])
            if category not in comparison:
                continue

            key = (tx["merchantId"], tx["customerRef"])
            past = normal_history[key]
            if len(past) < MIN_HISTORY:
                skipped[category] += 1
                continue

            baseline = median(item["amount"] for item in past)
            ratio = tx["amount"] / baseline
            # 2배 이상 증가나 1/2 이하 감소를 같은 강도로 취급하는 배율
            fold_change = max(ratio, 1 / ratio)
            hour_gap = circular_distance(
                hour_of_day(tx["_time"]), typical_hour(past)
            )
            comparison[category].append((fold_change, hour_gap))

        # 현 시각 거래가 끝난 뒤에야 정상 거래를 과거 이력에 추가한다.
        for tx in same_time:
            if tx["fraudLabel"] == 0:
                key = (tx["merchantId"], tx["customerRef"])
                normal_history[key].append(tx)

    print("\n[2. BEHAVIORAL_ANOMALY — 과거 정상 이력 대비]")
    print("비교 기준: 동일 가맹점·고객의 현재 시각 이전 정상 거래 3건 이상")
    print("금액 변화: max(현재금액/과거중앙값, 과거중앙값/현재금액)")
    print("시간대 차이: 과거 정상 거래의 대표 KST 시간에서 원형 거리(0~12시간)")

    for category, label in (("NORMAL", "정상 거래(비교군)"),
                            ("BEHAVIORAL_ANOMALY", "행동 이상 거래")):
        points = comparison[category]
        print(f"\n{label}: 비교 가능 {len(points):,}건 / 과거 정상 3건 미만으로 제외 {skipped[category]:,}건")
        if not points:
            continue
        folds = [point[0] for point in points]
        hours = [point[1] for point in points]
        high_amount = sum(fold >= 2 for fold in folds)
        unusual_hour = sum(hour >= 6 for hour in hours)
        either = sum(fold >= 2 or hour >= 6 for fold, hour in points)
        both = sum(fold >= 2 and hour >= 6 for fold, hour in points)
        print(f"금액 변화 배율 중앙값: {median(folds):.2f}배")
        print(f"시간대 차이 중앙값: {median(hours):.2f}시간")
        print(f"금액 2배 이상 이탈: {high_amount}/{len(points)} ({high_amount / len(points):.1%})")
        print(f"시간대 6시간 이상 이탈: {unusual_hour}/{len(points)} ({unusual_hour / len(points):.1%})")
        print(f"둘 중 하나 이상: {either}/{len(points)} ({either / len(points):.1%})")
        print(f"둘 다 해당: {both}/{len(points)} ({both / len(points):.1%})")

    print("\n참고: 이 분석은 사후 정답 Label을 사용해 정상 이력을 고른 품질 진단입니다.")
    print("실제 ML 입력에서는 과거 거래의 확정되지 않은 fraudLabel을 사용하면 안 됩니다.")


def main():
    transactions, episodes = load_data()
    print(f"분석 파일: {DATA_DIR}")
    print(f"전체 거래: {len(transactions):,}건 / 공격 사건: {len(episodes):,}개")
    analyze_split_payment(transactions, episodes)
    analyze_behavioral_anomaly(transactions)


if __name__ == "__main__":
    main()