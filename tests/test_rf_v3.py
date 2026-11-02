import sys
from pathlib import Path

import pytest

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

import generate_data_v3 as gen
from compare_v2_v3 import Metrics, judge
from country_features import DIFFERS_FROM_USUAL
from features import FEATURE_COLUMNS
from train_random_forest import create_pipeline, make_xy
from train_random_forest_v3 import build_bundle, load_v3, v3_variants

# 정답 · 상황 · 원본 국가 · 가맹점 번호는 어떤 경우에도 입력 칸이 되면 안 된다
FORBIDDEN = {"condition", "fraudLabel", "scenarioType", "ipCountryCode", "merchantId", "attackEventId", "paymentId"}


# v3 칸 = v2 칸 + 국가 칸 하나 (국가 칸 없음 참고 모델은 v2 칸 그대로), 실제 데이터로 만든 X에도 다른 칸이 섞이지 않는다
def test_train_v3_featureColumns_excludeConditionAndLabel(tmp_path):
    gen.save_dataset_v3(*gen.build_dataset_v3(42, customer_count=800, normal_count=39_800, fraud_count=200), tmp_path)
    transactions, features, split_info = load_v3(tmp_path)
    allowed = {*FEATURE_COLUMNS, DIFFERS_FROM_USUAL}

    for with_country in (True, False):
        for columns in v3_variants(with_country).values():
            assert not FORBIDDEN & set(columns)
            assert set(columns) <= allowed
            assert (DIFFERS_FROM_USUAL in columns) is with_country
            x, _ = make_xy(features, transactions, split_info["indices"]["Train"][:50], columns)
            assert all(set(row) == set(columns) for row in x)

    columns = v3_variants(True)["A_full_no_merchant_country"]
    rows = [{name: float(i % 5) for name in columns} for i in range(40)]
    pipeline = create_pipeline(None).fit(rows, [int(i % 4 == 0) for i in range(40)])
    bundle = build_bundle(pipeline, 0.2, columns, "None", split_info)
    assert bundle["feature_columns"] == columns
    assert bundle["model_version"] == "rf-v3"
    assert bundle["home_country"] == "KR"


def m(**changes):
    base = dict(caught=100, ap=0.50, foreign_fraud_rate=0.50, abroad_normal_fp_rate=0.05,
                high_amount_fp_rate=0.20, tabformer_ap=0.0021)
    return Metrics(**{**base, **changes})


V2 = m()
# 모든 기준이 딱 경계: AP 같음 · 해외 사기 +20%p(0.7 - 0.5 = 0.1999… 소수 오차) · 해외 정상 10% · TabFormer AP 같음
EDGE = dict(caught=101, foreign_fraud_rate=0.70, abroad_normal_fp_rate=0.10, high_amount_fp_rate=0.19)


# 경계값: 기준 1 잡은 사기 · 기준 4는 "더 나아야" 통과, AP · 기준 2 · 3 · 5는 경계값이면 통과
@pytest.mark.parametrize("changes, failed", [
    ({}, None),
    ({"caught": 100}, 0),
    ({"ap": 0.4999}, 0),
    ({"foreign_fraud_rate": 0.6999}, 1),
    ({"abroad_normal_fp_rate": 0.1001}, 2),
    ({"high_amount_fp_rate": 0.20}, 3),
    ({"tabformer_ap": 0.0020}, 4),
])
def test_compare_v2_v3_verdict_boundaryValuesPass(changes, failed):
    verdict = judge(V2, m(**{**EDGE, **changes}))

    assert verdict.passed is (failed is None)
    assert not verdict.pending
    assert [c.passed for c in verdict.checks] == [i != failed for i in range(5)]

