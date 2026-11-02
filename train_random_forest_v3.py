from __future__ import annotations

import joblib
from sklearn.metrics import average_precision_score, f1_score

from country_features import DIFFERS_FROM_USUAL, HOME_COUNTRY_DONWORRY, country_features
from features import build_features
from train_random_forest import (
    BASE_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    choose_threshold,
    create_pipeline,
    episode_recall,
    make_xy,
    read_csv,
    split_and_purge,
)
from train_random_forest_v2 import EXCLUDED_FEATURES, v2_variants

# rf-v3 (Ⓥ3): v3 합성 데이터(사기 0.5% · 접속 국가)로 v2와 같은 절차로 학습하고, 칸은 v2 + "평소 국가와 다른가" 하나
# 참고 모델 "v3 국가 칸 없음"도 같은 절차로 학습한다 — 성적 차이가 데이터 덕인지 국가 칸 덕인지 나눠 보려고

MODEL_VERSION = "rf-v3"
DATA_DIR = BASE_DIR / "data" / "v3"
MODEL_PATH = MODEL_DIR / "random_forest_v3.joblib"
NO_COUNTRY_PATH = MODEL_DIR / "random_forest_v3_no_country.joblib"
COUNTRY_FEATURE = DIFFERS_FROM_USUAL
HOME_COUNTRY = HOME_COUNTRY_DONWORRY


def v3_variants(with_country=True):
    # v2 칸 구성 두 개(가맹점 번호 없음)에 국가 칸만 더한다 — 국가 이름 · 상황 기록 · 정답은 칸에 넣지 않는다
    variants = {}
    for name, columns in v2_variants().items():
        if with_country:
            variants[f"{name}_country"] = [*columns, COUNTRY_FEATURE]
        else:
            variants[name] = list(columns)
    return variants


def load_v3(data_dir=DATA_DIR):
    # v3 거래 · Feature(+ 국가 칸) · 분할을 한 번에 — 학습과 비교가 같은 방법으로 계산하게
    transactions = read_csv(data_dir / "transactions.csv")
    episodes = read_csv(data_dir / "fraud_episodes.csv")
    features = build_features(transactions)
    for row, values in zip(features, country_features(transactions, home_country=HOME_COUNTRY)):
        row[COUNTRY_FEATURE] = values[COUNTRY_FEATURE]
    split_info = split_and_purge(transactions, episodes)
    return transactions, features, split_info


def build_bundle(pipeline, threshold, columns, weight_name, split_info, model_version=MODEL_VERSION):
    # ai-server model.py가 읽는 키(pipeline · threshold · feature_columns)는 v1 · v2와 같은 모양
    # home_country: 팀 단계에서 ai-server가 국가 칸을 같은 기준(이력 없으면 KR과 비교)으로 계산하게 남긴다
    return {
        "pipeline": pipeline,
        "threshold": threshold,
        "feature_columns": list(columns),
        "class_weight": weight_name,
        "random_state": RANDOM_STATE,
        "train_cut": split_info["train_cut"].isoformat(),
        "test_cut": split_info["test_cut"].isoformat(),
        "model_version": model_version,
        "excluded_features": list(EXCLUDED_FEATURES),
        "home_country": HOME_COUNTRY,
    }


def train_candidates(features, transactions, split_info, variants):
    # 칸 구성 2개 × class_weight 2개를 Train으로 학습하고 Validation F1 · AP를 잰다 (v2와 같은 절차)
    candidates = []
    for name, columns in variants.items():
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


def train_and_save(label, variants, path, model_version, features, transactions, split_info):
    print(f"\n=== {label} — Validation 후보 ===")
    candidates = train_candidates(features, transactions, split_info, variants)
    for c in candidates:
        print(f"{c['name']:<38} weight={c['weight']:<9} AP={c['val_ap']:.4f} F1={c['val_f1']:.4f} Threshold={c['threshold']:.2f}")

    # v1 · v2와 같은 선정 규칙: Test를 보지 않고 Validation F1, 같으면 AP
    best = max(candidates, key=lambda c: (c["val_f1"], c["val_ap"]))
    print(f"선정: {best['name']} · class_weight {best['weight']} · Threshold {best['threshold']:.2f}")

    test_indices = split_info["indices"]["Test"]
    test_x, test_y = make_xy(features, transactions, test_indices, best["columns"])
    probabilities = best["pipeline"].predict_proba(test_x)[:, 1]
    predictions = [int(p >= best["threshold"]) for p in probabilities]
    false_alarms = sum(1 for y, p in zip(test_y, predictions) if y == 0 and p == 1)
    caught = sum(1 for y, p in zip(test_y, predictions) if y == 1 and p == 1)
    detected, total = episode_recall(transactions, test_indices, predictions, split_info["episode_by_payment"])
    print(f"Test: F1 {f1_score(test_y, predictions, zero_division=0):.4f} · AP {average_precision_score(test_y, probabilities):.4f} · "
          f"잡은 사기 {caught}/{sum(test_y)} · 헛경보 {false_alarms} · Episode {detected}/{total}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(build_bundle(best["pipeline"], best["threshold"], best["columns"], best["weight"], split_info, model_version), path)
    print("저장:", path)


def main():
    transactions, features, split_info = load_v3()
    print(f"=== Random Forest {MODEL_VERSION} — v3 데이터 {len(transactions):,}건 · 국가 칸 {COUNTRY_FEATURE} (기본 국가 {HOME_COUNTRY}) ===")
    train_and_save("rf-v3 (국가 칸 포함)", v3_variants(True), MODEL_PATH, MODEL_VERSION, features, transactions, split_info)
    train_and_save("참고: v3 데이터 · 국가 칸 없음", v3_variants(False), NO_COUNTRY_PATH, "rf-v3-no-country", features, transactions, split_info)


if __name__ == "__main__":
    main()
