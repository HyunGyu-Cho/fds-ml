import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

import generate_data_v3 as v3

# 테스트는 작은 크기로 만든다 (실제 크기 20만 건은 스크립트 실행 때만) — 사기 비율은 실제와 같은 0.5%
SMALL = dict(customer_count=800, normal_count=39_800, fraud_count=200)
IPV4 = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    transactions, episodes, conditions = v3.build_dataset_v3(42, **SMALL)
    out = tmp_path_factory.mktemp("v3")
    v3.save_dataset_v3(transactions, episodes, conditions, out)
    return transactions, conditions, out


def file_hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("*.csv"))}


# 공개 저장소 · 팀 계약에는 IP 원본 없이 국가 코드(대문자 2글자)만 — 어느 칸에도 IP 모양 값이 없어야 한다
def test_generate_v3_outputs_noIpAddressColumn(small):
    _, _, out = small
    with (out / "transactions.csv").open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames
        rows = list(reader)

    assert "ipCountryCode" in header
    assert [c for c in header if "ip" in c.lower() and c != "ipCountryCode"] == []
    assert all(re.fullmatch(r"[A-Z]{2}", row["ipCountryCode"]) for row in rows)
    assert not any(IPV4.search(value) for row in rows for value in row.values())


# 사기 비율 0.5% 이하 — 설정값과 실제로 만든 데이터 둘 다
def test_generate_v3_fraudRate_atMostHalfPercent(small):
    transactions, _, _ = small
    labels = Counter(tx["fraudLabel"] for tx in transactions)

    assert v3.FRAUD_COUNT / (v3.NORMAL_COUNT + v3.FRAUD_COUNT) <= 0.005
    assert labels[1] / len(transactions) <= 0.005
    assert labels[1] == SMALL["fraud_count"]


# 정상 6가지 · 사기 3가지 상황이 모두 한 건 이상 있어야 통과 기준(해외 정상 헛경보 등)을 잴 수 있다
def test_generate_v3_eachCondition_hasRows(small):
    _, conditions, _ = small
    counts = Counter(row["condition"] for row in conditions)

    for condition in (*v3.NORMAL_CONDITIONS, *v3.FRAUD_CONDITIONS):
        assert counts[condition] > 0, condition


# 같은 seed면 세 파일이 바이트까지 같고, seed가 다르면 달라진다 — PC · 클라우드 숫자가 같아야 한다
def test_generate_v3_sameSeed_sameHash(tmp_path):
    tiny = dict(customer_count=300, normal_count=13_930, fraud_count=70)
    for name, seed in (("a", 7), ("b", 7), ("c", 8)):
        v3.save_dataset_v3(*v3.build_dataset_v3(seed, **tiny), tmp_path / name)

    assert file_hashes(tmp_path / "a") == file_hashes(tmp_path / "b")
    assert file_hashes(tmp_path / "a") != file_hashes(tmp_path / "c")
    assert len(file_hashes(tmp_path / "a")) == 3


# 상황 기록의 사기 상황 3종 = fraudLabel 1인 결제와 정확히 같아야 해외 사기 · 해외 정상 비율의 분모가 맞다 (저장 파일 기준)
def test_generate_v3_conditions_matchFraudLabel(small):
    _, _, out = small
    with (out / "transactions.csv").open(encoding="utf-8", newline="") as file:
        labels = {row["paymentId"]: row["fraudLabel"] for row in csv.DictReader(file)}
    with (out / "conditions.csv").open(encoding="utf-8", newline="") as file:
        conditions = {row["paymentId"]: row["condition"] for row in csv.DictReader(file)}

    assert set(conditions) == set(labels)
    assert set(conditions.values()) <= {*v3.NORMAL_CONDITIONS, *v3.FRAUD_CONDITIONS}
    for payment_id, condition in conditions.items():
        assert (condition in v3.FRAUD_CONDITIONS) == (labels[payment_id] == "1"), payment_id
