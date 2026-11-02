from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# IBM TabFormer 합성 카드 거래(Apache 2.0)를 돈워리 transactions.csv 모양으로 바꾼다
# 표준 라이브러리만 쓴다 — 원본(2.35GB)이 있는 PC에서 그대로 돌리고, 원본은 git에 올리지 않는다(external/)

BASE_DIR = Path(__file__).resolve().parent
KST = timezone(timedelta(hours=9))
MERCHANT_ID = "9001"
EPISODE_GAP = timedelta(days=7)
OUTPUT_COLUMNS = (
    "paymentId", "merchantId", "customerRef", "amount", "paymentRequestedAt", "deviceId",
    "cardToken", "authorizationResult", "fraudLabel", "scenarioType", "useChip", "locationRef",
)


def choose_users(user_ids, count, seed):
    # 고객 단위로 뽑는다 — 뽑힌 고객의 거래는 전부 넣어야 이력 Feature가 맞다
    ordered = sorted(user_ids, key=int)
    return set(random.Random(seed).sample(ordered, min(count, len(ordered))))


def convert_row(row, usd_krw):
    # 원본 한 줄 → 돈워리 칸 (paymentId 제외). 환불(음수 금액)은 None
    dollars = float(row["Amount"].replace("$", "").replace(",", ""))
    if dollars < 0:
        return None
    hour, minute = row["Time"].split(":")
    # 원본 시각을 한국 시간으로 읽어 hourOfDay가 원래 시각과 같게 한다
    at = datetime(int(row["Year"]), int(row["Month"]), int(row["Day"]), int(hour), int(minute), tzinfo=KST)
    fraud = row["Is Fraud?"] == "Yes"
    return {
        "merchantId": MERCHANT_ID,
        "customerRef": f"TF-USER-{row['User']}",
        "amount": round(dollars * usd_krw),
        "paymentRequestedAt": at.isoformat(),
        "deviceId": "",
        "cardToken": f"TF-CARD-{row['User']}-{row['Card']}",
        "authorizationResult": "DECLINED" if row["Errors?"] else "APPROVED",
        "fraudLabel": int(fraud),
        "scenarioType": "TABFORMER_FRAUD" if fraud else "NORMAL",
        "useChip": row["Use Chip"],
        # 가게가 있는 주(해외면 나라 이름). 온라인은 도시 칸이 "ONLINE" — 돈워리의 접속 국가 대신 쓰는 실험용 칸
        "locationRef": row["Merchant State"] or row["Merchant City"],
    }


def group_episodes(rows):
    # 같은 카드의 사기가 7일 안에 이어지면 한 공격(Episode)으로 묶는다 — 학습 · 검증에 나뉘어 들어가지 않게
    by_card = defaultdict(list)
    for row in rows:
        if row["fraudLabel"] == 1:
            by_card[row["cardToken"]].append(row)
    episodes = []
    for card, frauds in sorted(by_card.items()):
        frauds.sort(key=lambda r: (r["paymentRequestedAt"], r["paymentId"]))
        number, last = 0, None
        for row in frauds:
            at = datetime.fromisoformat(row["paymentRequestedAt"])
            if last is None or at - last > EPISODE_GAP:
                number += 1
            last = at
            episodes.append({"paymentId": row["paymentId"], "attackEventId": f"{card}-EP{number}"})
    return episodes


def read_users(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)
        return {row[0] for row in reader}


def convert(path, users, years, usd_krw):
    rows, dropped = [], Counter()
    with path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            if raw["User"] not in users:
                continue
            if not years[0] <= int(raw["Year"]) <= years[1]:
                dropped["기간 밖"] += 1
                continue
            row = convert_row(raw, usd_krw)
            if row is None:
                dropped["환불(음수 금액)"] += 1
                continue
            rows.append({"paymentId": len(rows) + 1, **row})
    return rows, dropped


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="TabFormer → 돈워리 transactions.csv")
    parser.add_argument("--input", default="external/card_transaction.v1.csv")
    parser.add_argument("--out", default="data/tabformer")
    parser.add_argument("--users", type=int, default=250)
    parser.add_argument("--years", default="2015-2019")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--usd-krw", type=int, default=1300)
    args = parser.parse_args()

    source = BASE_DIR / args.input
    out = BASE_DIR / args.out
    years = tuple(int(y) for y in args.years.split("-"))
    users = choose_users(read_users(source), args.users, args.seed)
    rows, dropped = convert(source, users, years, args.usd_krw)
    episodes = group_episodes(rows)

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "transactions.csv", OUTPUT_COLUMNS, rows)
    write_csv(out / "fraud_episodes.csv", ("paymentId", "attackEventId"), episodes)

    frauds = sum(r["fraudLabel"] for r in rows)
    online = sum(1 for r in rows if r["useChip"] == "Online Transaction")
    print(f"고객 {len(users)}명 · {years[0]}~{years[1]}년 · 1달러 {args.usd_krw}원 · seed {args.seed}")
    print(f"거래 {len(rows):,}건 (온라인 {online:,}) · 사기 {frauds:,}건 ({frauds / len(rows):.2%}) · 공격 묶음 {len({e['attackEventId'] for e in episodes}):,}개")
    print(f"뺀 거래: {dict(dropped)}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
