from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from features import build_features
from train_random_forest import read_csv

# 팀 ai-server에 넣을 파일 두 개를 만든다: 압축한 모델(<버전>.joblib) · golden 정답(ml_golden_<버전>.json)
# golden 사례(대상 결제 · 이력)는 기존 ai-server golden에서 그대로 가져오고, 정답만 이 모델로 다시 계산한다

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports"
CASE_KEYS = ("name", "target", "customerHistory", "sharedCardHistory")


def features_by_payment(transactions):
    # fds-ml이 거래 2만 건 전체로 계산한 Feature (ai-server는 대상 + 이전 이력만으로 같은 값을 내야 한다)
    rows = build_features(transactions)
    return {int(tx["paymentId"]): row for tx, row in zip(transactions, rows)}


def golden_cases(cases, features, bundle):
    # 1건 예측은 ai-server(model.py)와 같이 n_jobs=1로 계산한다
    columns = list(bundle["feature_columns"])
    pipeline = bundle["pipeline"]
    pipeline[-1].set_params(n_jobs=1)
    result = []
    for case in cases:
        row = features[int(case["target"]["paymentId"])]
        expected = {name: row[name] for name in columns}
        probability = float(pipeline.predict_proba([expected])[0][1])
        result.append({**{key: case[key] for key in CASE_KEYS}, "expectedFeatures": expected, "expectedProbability": probability})
    return result


def main():
    parser = argparse.ArgumentParser(description="ai-server용 모델 파일 · golden 정답 만들기")
    parser.add_argument("--model", required=True, help="fds-ml 모델 파일 (예: models/random_forest_v2.joblib)")
    parser.add_argument("--version", required=True, help="ai-server 모델 버전 = 파일 이름 (예: rf-v2)")
    parser.add_argument("--cases", required=True, help="사례를 가져올 기존 golden (예: ../ai-server/tests/fixtures/ml_golden.json)")
    args = parser.parse_args()

    model_path = BASE_DIR / args.model
    golden = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    transactions = read_csv(DATA_DIR / "transactions.csv")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    model_out = EXPORT_DIR / f"{args.version}.joblib"
    golden_out = EXPORT_DIR / f"ml_golden_{args.version}.json"

    # 모델은 학습 파일 그대로 압축만 한다 (n_jobs는 ai-server가 읽을 때 1로 바꾼다)
    joblib.dump(joblib.load(model_path), model_out, compress=3)

    bundle = joblib.load(model_path)
    output = {
        "source": f"fds-ml/features.py + {args.model} (n_jobs=1) · export_ai_server_model.py",
        "threshold": bundle["threshold"],
        "featureColumns": list(bundle["feature_columns"]),
        "cases": golden_cases(golden["cases"], features_by_payment(transactions), bundle),
    }
    # Windows에서도 줄바꿈을 LF로 (팀 저장소 .gitattributes eol=lf · 기존 golden과 같게)
    golden_out.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")

    print(f"모델: {model_out} ({model_out.stat().st_size / 1_000_000:.1f}MB)")
    print(f"golden: {golden_out} (사례 {len(output['cases'])}건, 칸 {len(output['featureColumns'])}개, 기준값 {output['threshold']})")


if __name__ == "__main__":
    main()
