"""공격 사건과 결제 ID의 관계를 검증 전용 CSV로 저장한다."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


EPISODE_COLUMNS = (
    "attackEventId",
    "scenarioType",
    "paymentId",
)

DEFAULT_EPISODE_PATH = (
    Path(__file__).resolve().parent / "data" / "fraud_episodes.csv"
)


def save_episode_links_csv(
    episode_links: Iterable[Mapping[str, Any]],
    output_path: str | Path | None = None,
) -> tuple[Path, int]:
    """공격 사건별 거래 관계를 고정 컬럼 순서의 CSV로 저장한다."""

    csv_path = Path(output_path) if output_path is not None else DEFAULT_EPISODE_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = csv_path.with_name(csv_path.name + ".tmp")
    expected_columns = set(EPISODE_COLUMNS)
    count = 0

    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=EPISODE_COLUMNS,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()

            for line_number, link in enumerate(episode_links, start=2):
                if not isinstance(link, Mapping):
                    raise TypeError(
                        f"사건 CSV {line_number}행: dict 형태여야 합니다."
                    )

                missing = expected_columns - set(link)
                extra = set(link) - expected_columns
                if missing or extra:
                    raise ValueError(
                        f"사건 CSV {line_number}행 컬럼 불일치: "
                        f"누락={sorted(missing)}, 추가={sorted(extra)}"
                    )

                if not str(link["attackEventId"]).strip():
                    raise ValueError(f"사건 CSV {line_number}행: attackEventId 누락")
                if not str(link["scenarioType"]).strip():
                    raise ValueError(f"사건 CSV {line_number}행: scenarioType 누락")
                if not isinstance(link["paymentId"], int) or link["paymentId"] <= 0:
                    raise ValueError(f"사건 CSV {line_number}행: paymentId가 올바르지 않습니다.")

                writer.writerow({column: link[column] for column in EPISODE_COLUMNS})
                count += 1

        os.replace(temporary_path, csv_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return csv_path, count