import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from compare_rf_v1_v2 import judge
from features import FEATURE_COLUMNS
from train_random_forest import create_pipeline
from train_random_forest_v2 import build_bundle, v2_variants


# v2 칸 구성에는 가맹점 번호가 없고, 그 밖의 칸은 v1 구성과 같다
def test_v2_columns_exclude_merchant_id():
    variants = v2_variants()

    assert set(variants) == {"A_full_no_merchant", "B_no_shared_card_no_merchant"}
    for columns in variants.values():
        assert "merchantId" not in columns
    assert variants["A_full_no_merchant"] == [c for c in FEATURE_COLUMNS if c != "merchantId"]
    assert variants["B_no_shared_card_no_merchant"] == [
        c for c in FEATURE_COLUMNS if c not in ("merchantId", "cardUsedByOtherCustomer")
    ]


# 저장한 v2 파일을 ai-server model.py 검사(필수 키 · 기준값 0~1 · 아는 칸 · predict_proba)대로 읽을 수 있다
def test_v2_bundle_has_keys_ai_server_requires(tmp_path):
    columns = v2_variants()["A_full_no_merchant"]
    rows = [{name: float(i % 7) for name in columns} for i in range(40)]
    labels = [int(i % 5 == 0) for i in range(40)]
    pipeline = create_pipeline(None).fit(rows, labels)
    cut = datetime(2026, 7, 1, tzinfo=timezone.utc)
    split_info = {"train_cut": cut, "test_cut": cut}

    path = tmp_path / "random_forest_v2.joblib"
    joblib.dump(build_bundle(pipeline, 0.3, columns, "None", split_info), path)
    bundle = joblib.load(path)

    assert {"pipeline", "threshold", "feature_columns"} <= set(bundle)
    assert 0 < float(bundle["threshold"]) < 1
    assert set(bundle["feature_columns"]) <= set(FEATURE_COLUMNS)
    assert "merchantId" not in bundle["feature_columns"]
    assert bundle["model_version"] == "rf-v2"
    probability = bundle["pipeline"].predict_proba([rows[0]])[0][1]
    assert 0.0 <= probability <= 1.0


# 판정: 잡은 사기 5% 넘게 감소 또는 AP 0.01 넘게 하락이면 불통과, 경계값은 통과
def test_verdict_fails_over_5pct_drop_or_ap_drop():
    assert judge(350, 350, 0.90, 0.90).passed
    assert judge(350, 333, 0.90, 0.90).passed
    assert not judge(350, 332, 0.90, 0.90).passed
    assert judge(100, 95, 0.90, 0.90).passed
    assert judge(350, 350, 0.8984, 0.8884).passed
    assert not judge(350, 350, 0.8984, 0.8883).passed
    assert judge(350, 360, 0.8984, 0.9100).passed

    both = judge(350, 300, 0.90, 0.80)
    assert not both.passed and len(both.reasons) == 2
