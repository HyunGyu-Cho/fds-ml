import csv
import sys
from datetime import datetime
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from tabformer_convert import choose_users, convert, convert_row, group_episodes
from tabformer_scorecard import location_seen_before

COLUMNS = ["User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip", "Merchant Name",
           "Merchant City", "Merchant State", "Zip", "MCC", "Errors?", "Is Fraud?"]


def raw(user="7", card="1", year="2016", time="14:20", amount="$45.10", chip="Chip Transaction",
        city="La Verne", state="CA", errors="", fraud="No", day="5"):
    return {"User": user, "Card": card, "Year": year, "Month": "3", "Day": day, "Time": time,
            "Amount": amount, "Use Chip": chip, "Merchant Name": "1", "Merchant City": city,
            "Merchant State": state, "Zip": "91750.0", "MCC": "5411", "Errors?": errors, "Is Fraud?": fraud}


def tx(payment_id, at, customer="C1", location="CA", fraud=0, card="K1"):
    return {"paymentId": payment_id, "paymentRequestedAt": at, "customerRef": customer,
            "locationRef": location, "fraudLabel": fraud, "cardToken": card}


# 뽑힌 손님은 기간 안 거래가 전부 들어가고, 안 뽑힌 손님은 하나도 안 들어간다
def test_sample_keeps_all_transactions_of_chosen_users(tmp_path):
    path = tmp_path / "card_transaction.v1.csv"
    rows = [raw(user=str(u), day=str(d), year=str(y)) for u in range(6) for y in (2014, 2016) for d in range(1, 11)]
    rows.append(raw(user="0", amount="$-3.00"))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    users = choose_users({str(u) for u in range(6)}, 3, seed=42)
    converted, dropped = convert(path, users, (2015, 2019), 1300)

    kept = {r["customerRef"] for r in converted}
    assert kept == {f"TF-USER-{u}" for u in users}
    assert len(converted) == 3 * 10
    assert users == choose_users({str(u) for u in range(6)}, 3, seed=42)
    assert [r["paymentId"] for r in converted] == list(range(1, 31))


# 금액은 달러 × 환율(쉼표 포함) · 환불 제외, 시각은 원래 시각 그대로(한국 시간), 오류가 있으면 거절
def test_convert_row_amount_time_declined():
    normal = convert_row(raw(), 1300)
    big = convert_row(raw(amount="$1,250.00", errors="Bad PIN,", fraud="Yes", chip="Online Transaction",
                          city="ONLINE", state=""), 1300)

    assert normal["amount"] == 58630
    assert big["amount"] == 1625000
    assert convert_row(raw(amount="$-45.10"), 1300) is None
    assert datetime.fromisoformat(normal["paymentRequestedAt"]).hour == 14
    assert normal["paymentRequestedAt"] == "2016-03-05T14:20:00+09:00"
    assert normal["authorizationResult"] == "APPROVED" and big["authorizationResult"] == "DECLINED"
    declined_only = convert_row(raw(errors="Insufficient Balance,"), 1300)
    fraud_only = convert_row(raw(fraud="Yes"), 1300)
    assert declined_only["authorizationResult"] == "DECLINED" and declined_only["fraudLabel"] == 0
    assert fraud_only["authorizationResult"] == "APPROVED" and fraud_only["fraudLabel"] == 1
    assert normal["fraudLabel"] == 0 and big["fraudLabel"] == 1
    assert normal["cardToken"] == "TF-CARD-7-1" and normal["customerRef"] == "TF-USER-7"
    assert normal["locationRef"] == "CA" and big["locationRef"] == "ONLINE"


# 같은 카드의 사기가 7일 안에 이어지면 한 묶음, 7일 넘게 떨어지거나 다른 카드면 다른 묶음
def test_episodes_group_fraud_by_user_card():
    rows = [
        tx(1, "2016-03-01T10:00:00+09:00", fraud=1),
        tx(2, "2016-03-04T10:00:00+09:00", fraud=1),
        tx(3, "2016-03-05T10:00:00+09:00", fraud=0),
        tx(4, "2016-03-20T10:00:00+09:00", fraud=1),
        tx(5, "2016-03-02T10:00:00+09:00", fraud=1, card="K2"),
    ]

    episodes = {e["paymentId"]: e["attackEventId"] for e in group_episodes(rows)}

    assert set(episodes) == {1, 2, 4, 5}
    assert episodes[1] == episodes[2]
    assert episodes[4] != episodes[1]
    assert episodes[5] not in {episodes[1], episodes[4]}


# 앞선 거래만 본다 — 같은 시각 · 나중 거래 · 다른 손님의 장소는 "가 본 곳"으로 세지 않는다
def test_location_seen_before_uses_only_earlier_transactions():
    rows = [
        tx(1, "2016-03-05T12:00:00+09:00", location="NY"),
        tx(2, "2016-03-01T10:00:00+09:00", location="CA"),
        tx(3, "2016-03-02T10:00:00+09:00", location="NY"),
        tx(4, "2016-03-02T10:00:00+09:00", location="NY"),
        tx(5, "2016-03-03T10:00:00+09:00", location="CA"),
        tx(6, "2016-03-04T10:00:00+09:00", customer="C2", location="CA"),
        tx(7, "2016-03-06T10:00:00+09:00", location=""),
    ]

    assert location_seen_before(rows) == [1, 0, 0, 0, 1, 0, 0]
