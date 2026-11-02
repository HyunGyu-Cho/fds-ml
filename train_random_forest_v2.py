from __future__ import annotations

import joblib
from sklearn.metrics import average_precision_score, f1_score

from features import FEATURE_COLUMNS, build_features
from train_random_forest import (
    DATA_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    choose_threshold,
    create_pipeline,
    episode_recall,
    make_xy,
    read_csv,
    split_and_purge,
)

# rf-v2: v1과 같은 절차로 학습하되 가맹점 번호(merchantId)를 모든 칸 구성에서 뺀다
# v1은 합성 가맹점 1001~1003만 배워서 실제 가맹점 번호가 들어오면 확률이 흔들린다

MODEL_VERSION = "rf-v2"
EXCLUDED_FEATURES = ("merchantId",)
MODEL_PATH = MODEL_DIR / "random_forest_v2.joblib"


def v2_variants():
    # v1의 두 칸 구성(A_full · B_no_shared_card)에서 가맹점 번호만 뺀다 (나머지 칸은 v1과 같음)
    base = {
        "A_full": list(FEATURE_COLUMNS),
        "B_no_shared_card": [name for name in FEATURE_COLUMNS if name != "cardUsedByOtherCustomer"],
    }
    return {
        f"{name}_no_merchant": [column for column in columns if column not in EXCLUDED_FEATURES]
        for name, columns in base.items()
    }


def build_bundle(pipeline, threshold, columns, weight_name, split_info):
    # ai-server model.py가 읽는 키(pipeline · threshold · feature_columns)는 v1과 같은 모양으로 둔다
    return {
        "pipeline": pipeline,
        "threshold": threshold,
        "feature_columns": list(columns),
        "class_weight": weight_name,
        "random_state": RANDOM_STATE,
        "train_cut": split_info["train_cut"].isoformat(),
        "test_cut": split_info["test_cut"].isoformat(),
        "model_version": MODEL_VERSION,
        "excluded_features": list(EXCLUDED_FEATURES),
    }


def train_candidates(features, transactions, split_info):
    # 칸 구성 2개 × class_weight 2개를 Train으로 학습하고 Validation F1 · AP를 잰다
    candidates = []
    for name, columns in v2_variants().items():
        train_x, train_y = make_xy(features, transactions, split_info["indices"]["Train"], columns)
        val_x, val_y = make_xy(features, transactions, split_info["indices"]["Validation"], columns)
        for weight in (None, "balanced"):
            pipeline = create_pipeline(weight)
            pipeline.fit(train_x, train_y)
            probabilities = pipeline.predict_proba(val_x)[:, 1]
            threshold, val_f1 = choose_threshold(val_y, probabilities)
            candidates.append({
                "name": name,
                "weight": "None" if weight is None else "balanced",
                "columns": columns,
                "pipeline": pipeline,
                "threshold": threshold,
                "val_f1": val_f1,
                "val_ap": average_precision_score(val_y, probabilities),
            })
    return candidates


def main():
    transactions = read_csv(DATA_DIR / "transactions.csv")
    episodes = read_csv(DATA_DIR / "fraud_episodes.csv")
    features = build_features(transactions)
    split_info = split_and_purge(transactions, episodes)

    print(f"=== Random Forest {MODEL_VERSION} (제외: {', '.join(EXCLUDED_FEATURES)}) ===")
    print("\n=== Validation 모델 비교 ===")
    candidates = train_candidates(features, transactions, split_info)
    for c in candidates:
        print(f"{c['name']:<30} weight={c['weight']:<9} AP={c['val_ap']:.4f} F1={c['val_f1']:.4f} Threshold={c['threshold']:.2f}")

    # v1과 같은 선정 규칙: Test를 보지 않고 Validation F1, 같으면 AP
    best = max(candidates, key=lambda c: (c["val_f1"], c["val_ap"]))
    print("\n=== 최종 선정 ===")
    print(f"Feature 구성: {best['name']} · class_weight: {best['weight']} · Threshold: {best['threshold']:.2f}")

    test_indices = split_info["indices"]["Test"]
    test_x, test_y = make_xy(features, transactions, test_indices, best["columns"])
    probabilities = best["pipeline"].predict_proba(test_x)[:, 1]
    predictions = [int(p >= best["threshold"]) for p in probabilities]
    false_alarms = sum(1 for y, p in zip(test_y, predictions) if y == 0 and p == 1)
    caught = sum(1 for y, p in zip(test_y, predictions) if y == 1 and p == 1)
    detected, total = episode_recall(transactions, test_indices, predictions, split_info["episode_by_payment"])

    print("\n=== 최종 Test 평가 ===")
    print(f"F1: {f1_score(test_y, predictions, zero_division=0):.4f} · AP: {average_precision_score(test_y, probabilities):.4f}")
    print(f"잡은 사기 {caught}/{sum(test_y)} · 헛경보 {false_alarms} · Episode {detected}/{total}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = build_bundle(best["pipeline"], best["threshold"], best["columns"], best["weight"], split_info)
    joblib.dump(bundle, MODEL_PATH)
    print("\n모델 저장:", MODEL_PATH)


if __name__ == "__main__":
    main()
