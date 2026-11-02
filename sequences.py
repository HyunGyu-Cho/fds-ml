
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import groupby
from typing import Iterable, Mapping


UTC = timezone.utc
KST = timezone(timedelta(hours=9))

SEQUENCE_LENGTH = 5
PAD_MERCHANT = "__PAD__"

SEQUENCE_FEATURES = (
    "logAmount",
    "logMinutesBeforeCurrent",
    "hourSin",
    "hourCos",
    "merchantId",
    "sameDeviceAsCurrent",
    "sameCardAsCurrent",
)


@dataclass(frozen=True)
class Transaction:
    index: int
    timestamp: datetime
    customer_ref: str
    merchant_id: str
    amount: float
    device_id: str
    card_token: str


@dataclass(frozen=True)
class SequenceWindow:
    """
    tokens: 과거 거래를 오래된 순서부터 저장
    mask: 1은 실제 거래, 0은 Padding
    """
    tokens: tuple[dict, ...]
    mask: tuple[int, ...]


def parse_timestamp(value):
    text = str(value).strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    timestamp = datetime.fromisoformat(text)

    if timestamp.tzinfo is None:
        raise ValueError("거래 시각에 시간대가 없습니다.")

    return timestamp.astimezone(UTC)


def parse_transaction(row: Mapping, index: int):
    timestamp = parse_timestamp(row["paymentRequestedAt"])
    amount = float(row["amount"])

    if not math.isfinite(amount) or amount < 0:
        raise ValueError("amount는 0 이상의 유한한 값이어야 합니다.")

    customer = str(row["customerRef"]).strip()
    merchant = str(row["merchantId"]).strip()

    if not customer or not merchant:
        raise ValueError("customerRef 또는 merchantId가 비었습니다.")

    return Transaction(
        index=index,
        timestamp=timestamp,
        customer_ref=customer,
        merchant_id=merchant,
        amount=amount,
        device_id=str(row.get("deviceId") or ""),
        card_token=str(row.get("cardToken") or ""),
    )


def padding_token():
    """Padding용 가상 토큰. 실제 거래를 의미하지 않는다."""

    return {
        "logAmount": 0.0,
        "logMinutesBeforeCurrent": 0.0,
        "hourSin": 0.0,
        "hourCos": 0.0,
        "merchantId": PAD_MERCHANT,
        "sameDeviceAsCurrent": 0.0,
        "sameCardAsCurrent": 0.0,
    }


def transaction_to_token(
    past: Transaction,
    current: Transaction,
):
    """
    과거 거래 1건을 Transformer 입력용 Token으로 변환한다.

    현재 거래의 시각과 카드/기기 정보는
    평가 시점에 이용 가능하다고 가정한다.
    """

    delta_minutes = (
        current.timestamp - past.timestamp
    ).total_seconds() / 60.0

    if delta_minutes <= 0:
        raise ValueError("과거 거래 시각은 현재보다 빨라야 합니다.")

    hour = past.timestamp.astimezone(KST).hour
    angle = 2 * math.pi * hour / 24

    same_device = int(
        bool(current.device_id)
        and bool(past.device_id)
        and current.device_id == past.device_id
    )

    same_card = int(
        bool(current.card_token)
        and bool(past.card_token)
        and current.card_token == past.card_token
    )

    return {
        "logAmount": math.log1p(past.amount),
        "logMinutesBeforeCurrent": math.log1p(delta_minutes),
        "hourSin": math.sin(angle),
        "hourCos": math.cos(angle),
        "merchantId": past.merchant_id,
        "sameDeviceAsCurrent": float(same_device),
        "sameCardAsCurrent": float(same_card),
    }


def build_sequences(
    transactions: Iterable[Mapping],
    sequence_length: int = SEQUENCE_LENGTH,
) -> list[SequenceWindow]:
    """
    각 거래의 과거 거래 시퀀스를 생성한다.

    중요한 규칙:
    - 같은 고객의 과거 거래만 사용
    - 현재 거래는 제외
    - 동일 타임스탬프 거래는 서로 이력에 포함하지 않음
    - Label과 승인 결과는 읽지 않음
    - 미래 거래를 참조하지 않음
    - 결과 순서는 원본 입력 순서와 동일
    """

    if sequence_length < 1:
        raise ValueError("sequence_length는 1 이상이어야 합니다.")

    records = [
        parse_transaction(row, i)
        for i, row in enumerate(transactions)
    ]

    records.sort(key=lambda tx: tx.timestamp)

    histories = defaultdict(
        lambda: deque(maxlen=sequence_length)
    )

    results: list[SequenceWindow | None] = [
        None
    ] * len(records)

    for timestamp, group in groupby(
        records,
        key=lambda tx: tx.timestamp,
    ):
        same_time_transactions = list(group)

        # 1단계: 모든 현재 거래의 시퀀스를 먼저 생성한다.
        for current in same_time_transactions:
            past_transactions = list(
                histories[current.customer_ref]
            )

            tokens = [
                transaction_to_token(past, current)
                for past in past_transactions
            ]

            missing_count = sequence_length - len(tokens)

            padded_tokens = (
                [padding_token() for _ in range(missing_count)]
                + tokens
            )

            mask = (
                [0] * missing_count
                + [1] * len(tokens)
            )

            results[current.index] = SequenceWindow(
                tokens=tuple(padded_tokens),
                mask=tuple(mask),
            )

        # 2단계: 동일 시각의 Feature 계산을 모두 완료한 뒤
        # 해당 거래들을 이후 거래의 이력에 추가한다.
        for current in same_time_transactions:
            histories[current.customer_ref].append(current)

    if any(result is None for result in results):
        raise RuntimeError("일부 거래의 시퀀스가 생성되지 않았습니다.")

    return [
        result
        for result in results
        if result is not None
    ]
