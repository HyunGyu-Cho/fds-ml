# DonWorry FDS ML — 이상거래 탐지 모델 실험

DonWorry(돈워리) FDS(Fraud Detection System, 이상거래 탐지)를 고도화하기 위한 ML 실험 저장소입니다.
합성 거래 데이터를 만들고, 여러 모델(Random Forest · XGBoost · Isolation Forest · Transformer · Transformer+XGBoost 하이브리드)을 같은 기준으로 비교합니다.

> 모든 데이터는 가상으로 생성됩니다.
> 합성 데이터에서 얻은 ML 성능은 실제 금융거래의 사기 탐지 성능을 보장하지 않습니다.

## 1. 팀 저장소와의 관계

- 팀 저장소: [team-sweethan/DonWorry](https://github.com/team-sweethan/DonWorry)
- 결제 판정은 팀 저장소의 Rule Engine(R01~R08)만 합니다. 이 저장소의 모델은 결제 결과를 바꾸지 않습니다.
- 팀 저장소의 ai-server가 쓰는 모델은 `ai-server/models/rf-v1.joblib` 하나입니다. 이 저장소의 `train_random_forest.py`가 만든 `models/random_forest_baseline.joblib`을 joblib `compress=3`으로 다시 저장한 파일입니다(16MB → 3.6MB).
- ai-server는 그 모델로 사기 확률을 계산해 Rule 결과 옆에 기록만 합니다(Shadow Mode, 그림자 평가). 결정 배경은 팀 저장소 `docs/decisions/0061-fds-ml-shadow-inference.md`(ADR-0061)에 있습니다.
- `features.py`는 팀 저장소 `ai-server/app/ml/features.py`와 계산이 같아야 합니다(그쪽은 코드 정렬만 다름). 한쪽을 바꾸면 다른 쪽과 ai-server `tests/fixtures/ml_golden.json`도 함께 바꿉니다.

## 2. 개발 환경

- Python 3.12
- Windows / PowerShell (Linux에서도 같은 명령으로 실행됨)
- 외부 데이터셋 및 실제 개인정보 사용 없음

## 3. 설치

이 저장소 루트(`fds-ml/`)에서 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

설치는 두 가지입니다.

| 설치 | 명령 | 할 수 있는 것 |
|---|---|---|
| 기본 | `python -m pip install -r requirements.txt` | 데이터 생성 · Random Forest 학습 · 데이터 분석 · 테스트 |
| 실험 | `python -m pip install -r requirements-experiments.txt` | 위 전부 + XGBoost · Isolation Forest · Transformer · 하이브리드 |

- `scikit-learn==1.9.1`은 ai-server와 같은 버전입니다. 버전이 다르면 rf-v1 모델을 그대로 불러오지 못할 수 있습니다.
- 실험 설치의 torch는 CPU 전용 판입니다(GPU 불필요).

## 4. 실행 순서

`data/`와 `models/`는 Git에 올리지 않습니다. 아래 순서대로 다시 만들 수 있습니다.

```text
generate_data ─▶ train_random_forest ─▶ train_xgboost ─┬─▶ train_isolation_forest
   (데이터)          (RF, 기본 설치)      (실험 설치)     ├─▶ train_transformer ─▶ compare_models_fixed_fpr
                                                         └─▶ train_hybrid_xgboost
```

| 순서 | 명령 | 설치 | 만드는 파일 · 하는 일 |
|---|---|---|---|
| 1 | `python generate_data.py` | 기본 | `data/transactions.csv` · `data/fraud_episodes.csv` |
| 2 | `python train_random_forest.py` | 기본 | `models/random_forest_baseline.joblib` (rf-v1의 원본) |
| 3 | `python train_xgboost.py` | 실험 | `models/xgboost_candidate.joblib` |
| 4 | `python train_isolation_forest.py` | 실험 | `models/isolation_forest_candidate.joblib` (RF · XGB 파일 필요) |
| 5 | `python train_transformer.py` | 실험 | `models/transformer_candidate.pt` · `tabular_mlp_candidate.pt` · `neural_preprocessing.joblib` (RF · XGB 파일 필요) |
| 6 | `python train_hybrid_xgboost.py` | 실험 | `models/hybrid_encoder_candidate.pt` · `hybrid_xgboost_candidate.joblib` · `hybrid_preprocessing.joblib` (RF · XGB 파일 필요) |
| 7 | `python compare_models_fixed_fpr.py` | 실험 | RF · XGB · Transformer를 같은 헛경보 수준에서 비교 (5번까지 필요) |

- 데이터는 거래 시각 순으로 Train 70% · Validation 15% · Test 15%로 나눕니다. 경계를 넘는 공격 사건은 평가 데이터에서 뺍니다(`split_and_purge`).
- 같은 헛경보 기준: Validation 정상 거래 2,555건 중 50건까지만 헛경보를 허용하도록(`TARGET_FPR = 50 / 2555`) 모델마다 기준값을 맞춘 뒤 탐지 수를 비교합니다.
- Test 평가는 `train_random_forest.py`만 합니다. 나머지 모델은 Validation으로만 비교합니다.

그 밖의 분석 스크립트:

| 명령 | 설치 | 하는 일 |
|---|---|---|
| `python analyze_ml_features.py` | 기본 | Feature 분포 분석 |
| `python analyze_rf_validation.py` | 기본 | RF Validation 결과를 시나리오별로 분석 |
| `python analyze_sequence_readiness.py` | 기본 | 거래 순서(시퀀스) 학습에 쓸 수 있는지 점검 |
| `python compare_rf_xgb_validation.py` | 실험 | RF · XGB 예측 비교 |
| `python ablate_transformer_history.py` · `ablate_transformer_order.py` | 실험 | Transformer에서 과거 거래 · 순서를 없앴을 때의 차이 (제거 실험) |

## 5. 재현성

- 데이터 생성과 학습은 seed 42로 고정되어 있습니다.
- 11/02 확인: 새 Linux 환경(Python 3.12, `requirements.txt`)에서 다시 만든 결과가 Windows PC 원본과 같았습니다.
  - `data/transactions.csv` · `data/fraud_episodes.csv`: SHA-256 일치
  - `models/random_forest_baseline.joblib`: 거래 2만 건의 사기 확률 차이 최대 2.2e-16, 기준값 0.30 동일 (팀 ai-server의 rf-v1과도 같음)
- 신경망(Transformer · 하이브리드)은 seed를 고정해도 CPU · 라이브러리 환경에 따라 결과가 조금 다를 수 있습니다.

## 6. Windows 주의

- uv가 받은 Python이 Windows 스마트 앱 컨트롤에 막힌 적이 있어, 이 저장소는 python.org에서 설치한 Python 3.12로 만든 `.venv`를 씁니다.

## 7. 데이터셋 구성

기본 설정:

| 항목 | 값 |
|---|---|
| Random Seed | 42 |
| 가상 가맹점 | 3개 |
| 가상 고객 | 1,500명 |
| 정상 거래 | 18,000건 |
| 사기 거래 | 2,000건 |
| 전체 거래 | 20,000건 |
| 사기 비율 | 10% |
| 거래 기간 | 2026-04-01 ~ 2026-08-31 (KST) |

사기 비율과 고객 소비 분포는 실험용 가정이며
실제 금융거래 통계에 맞춰 보정된 값이 아닙니다.

거래 시각은 UTC ISO 8601로 저장하고,
분석할 때 KST로 변환합니다.

### 가상 가맹점

- 1001: SIM_CAKESHOP
- 1002: SIM_SHOP_B
- 1003: SIM_SHOP_C

이 가맹점 ID는 합성 데이터 전용이며
실제 DonWorry 운영 DB의 가맹점 ID와 무관합니다.

## 8. 사기 시나리오

| 시나리오 | 설명 |
|---|---|
| CARD_TESTING | 단시간 여러 카드로 소액 결제 |
| ACCOUNT_TAKEOVER | 기존 고객의 기기·카드·금액 변화 |
| STOLEN_CARD | 다른 가상 고객의 기존 카드 사용 |
| LOW_AND_SLOW | 일정한 시간 간격의 반복 거래 |
| SPLIT_PAYMENT | 목표 총액을 여러 거래로 분할 |
| BEHAVIORAL_ANOMALY | 금액·시간대·복합·미묘한 행동 이탈 |
| CARD_DEVICE_CHANGE | 평소와 다른 카드·기기 조합 사용 |

시나리오별 거래 수는 기본적으로 약 285~286건입니다.

여러 사기 거래가 하나의 공격 사건에 속할 수 있습니다.

## 9. 생성 파일

### transactions.csv

전체 20,000건의 거래를 저장합니다.

컬럼:

```text
paymentId
evaluationRequestId
orderRef
merchantId
customerRef
amount
orderCreatedAt
paymentRequestedAt
deviceId
cardToken
cardDisplayLabel
buyerEmail
authorizationResult
fraudLabel
scenarioType
```

정상 거래는 `fraudLabel=0`,
사기 거래는 `fraudLabel=1`입니다.

`scenarioType`은 사기 시나리오를 구분하며
정상 거래에는 `NORMAL`을 사용합니다.

### fraud_episodes.csv

사기 거래 2,000건과 공격 사건의 연결 정보를 저장합니다.

주요 필드:

```text
attackEventId
scenarioType
paymentId
```

하나의 공격 사건은 여러 거래를 포함할 수 있습니다.

`paymentId`를 이용해 `transactions.csv`와 연결합니다.

이 파일은 학습 정답 및 데이터 품질 검증용입니다.

## 10. 카드·기기 식별자

카드와 기기는 실제 결제 정보와 관계없는
가상 식별자를 사용합니다.

예시 형식:

```text
SIM-CARD-<UUID5 기반 식별자>
SIM-DEVICE-<UUID5 기반 식별자>
```

정상·사기 거래 모두 동일한 형식을 사용합니다.

같은 카드·기기는 거래가 달라도 같은 ID를 유지하므로
고객별 사용 이력을 분석할 수 있습니다.

원본 식별자 문자열 자체는 ML Feature로 사용하지 않습니다.

## 11. 자동 테스트

```powershell
python -m pytest tests -q
```

- 기본 설치: 48개 통과, torch가 필요한 테스트 파일 2개(`test_transformer.py` · `test_hybrid_xgboost.py`)는 자동으로 건너뜀
- 실험 설치: 위 2개 파일의 9개까지 모두 실행

주요 검증: 거래 · 사기 비율, 7종 사기 시나리오, ID 중복, 시간순 정합성, CSV 재읽기, 같은 seed 재현성, 공격 사건 연결, Feature 계산(현재 거래보다 이전 거래만 사용), 시간순 분할과 경계 사건 제외, 시퀀스 생성, 모델 출력 모양.

테스트 통과는 데이터 구조와 생성 규칙의 정합성을 확인하는 것이며, 실제 금융거래에 대한 ML 성능을 보증하지 않습니다.

## 12. 데이터 품질 분석

### 전체 거래 분포

```powershell
python analyze_data_quality.py
```

금액, 승인 결과, 가맹점, 시간대,
식별자 문자열 패턴 등을 분석합니다.

### 시나리오별 거래 시퀀스

```powershell
python analyze_scenario_sequences.py
```

카드·기기의 최초 관측 여부,
타 고객의 카드 사용 여부,
거래 간 시간 간격 등을 분석합니다.

### 공격 사건별 품질

```powershell
python analyze_episode_quality.py
```

SPLIT_PAYMENT의 사건별 총액과 거래 간격,
BEHAVIORAL_ANOMALY의 금액·시간대 이탈을 분석합니다.

이 분석들은 데이터 품질을 점검하기 위한 도구이며
모델 평가 결과를 의미하지 않습니다.

## 13. 개인정보 및 데이터 안전성

- 실제 고객 정보 사용 금지
- 실제 카드번호 및 결제 토큰 사용 금지
- 실제 IP 주소 및 IP 국가 정보 생성 금지
- 가상 카드·기기 ID 사용
- 가상 이메일은 `example.invalid` 사용
- 실제 개인정보에서 파생한 데이터 사용 금지

완전 합성 데이터를 사용하더라도
전체 서비스의 개인정보 처리 의무가
자동으로 면제되는 것은 아닙니다.

## 14. ML 데이터 누수 방지

다음 필드는 결제 전 사기 예측 모델의
입력 Feature로 사용하지 않습니다.

- fraudLabel
- scenarioType
- attackEventId
- 현재 거래의 authorizationResult

과거 거래의 승인 결과 등은 실제 평가 시점에
이미 관측 가능한 경우에만 활용할 수 있습니다.

식별자와 주문번호 자체를 모델에 직접 입력하지 않고,
과거 거래를 기반으로 계산한 행동 Feature를 사용합니다.

Feature Engineering에서는 반드시 현재 거래보다
이전에 발생한 거래 정보만 사용해야 합니다.

사후 분석에서 정답 Label을 사용해 정상 이력을
구분한 결과를 온라인 추론 Feature로 사용하지 않습니다.
