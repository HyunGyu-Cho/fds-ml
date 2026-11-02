"""합성 FDS 거래 데이터를 재현 가능한 CSV로 저장한다."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


CSV_COLUMNS = (
    "paymentId",
    "evaluationRequestId",
    "orderRef",
    "merchantId",
    "customerRef",
    "amount",
    "orderCreatedAt",
    "paymentRequestedAt",
    "deviceId",
    "cardToken",
    "cardDisplayLabel",
    "buyerEmail",
    "authorizationResult",
    "fraudLabel",
    "scenarioType",
)

DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "transactions.csv"
)


def save_transactions_csv(
    transactions: Iterable[Mapping[str, Any]],
    output_path: str | Path | None = None,
) -> tuple[Path, int]:
    """거래 데이터를 정해진 컬럼 순서로 CSV에 저장한다."""

    csv_path = (
        Path(output_path)
        if output_path is not None
        else DEFAULT_CSV_PATH
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = csv_path.with_name(
        csv_path.name + ".tmp"
    )

    expected_columns = set(CSV_COLUMNS)
    count = 0

    try:
        with temporary_path.open(
            "w", encoding="utf-8", newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_COLUMNS,
                extrasaction="raise",
                lineterminator="\n",
            )

            writer.writeheader()

            for line_number, transaction in enumerate(
                transactions, start=2
            ):
                if not isinstance(transaction, Mapping):
                    raise TypeError(
                        f"CSV {line_number}행: "
                        "거래 데이터는 dict 형태여야 합니다."
                    )

                actual_columns = set(transaction)

                missing = expected_columns - actual_columns
                extra = actual_columns - expected_columns

                if missing or extra:
                    raise ValueError(
                        f"CSV {line_number}행의 컬럼 불일치: "
                        f"누락={sorted(missing)}, "
                        f"추가={sorted(extra)}"
                    )

                writer.writerow({
                    column: transaction[column]
                    for column in CSV_COLUMNS
                })

                count += 1

        os.replace(temporary_path, csv_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return csv_path, count