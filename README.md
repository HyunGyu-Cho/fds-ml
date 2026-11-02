# DonWorry FDS ML — 이상거래 탐지 모델 실험

DonWorry(돈워리) FDS(Fraud Detection System, 이상거래 탐지)를 고도화하기 위한 ML 실험 저장소입니다.
합성 거래 데이터를 만들고, 여러 모델(Random Forest · XGBoost · Isolation Forest · Transformer · Transformer+XGBoost 하이브리드)을 같은 기준으로 비교합니다.

> 모든 데이터는 가상으로 생성됩니다.
> 합성 데이터에서 얻은 ML 성능은 실제 금융거래의 사기 탐지 성능을 보장하지 않습니다.

## 1. 팀 저장소와의 관계

- 팀 저장소: [team-sweethan/DonWorry](https://github.com/team-sweethan/DonWorry)
- 결제 판정은 팀 저장소의 Rule Engine(R01~R08)만 합니다. 이 저장소의 모델은 결제 결과를 바꾸지 않습니다.
- 팀 저장소 ai-server의 모델 파일은 `ai-server/models/<버전>.joblib`입니다. rf-v1은 `models/random_forest_baseline.joblib`, rf-v2는 `models/random_forest_v2.joblib`을 joblib `compress=3`으로 다시 저장한 파일입니다(약 16MB → 3.5MB). 팀 저장소 #450에서 기본 모델을 rf-v2로 바꾸고 rf-v1은 되돌리기용으로 남겼습니다.
- ai-server에 넣는 모델 파일과 golden 정답(`tests/fixtures/ml_golden_<버전>.json`)은 `export_ai_server_model.py`로 만듭니다. 이 스크립트로 rf-v1 golden을 다시 만들면 팀 저장소의 기존 `ml_golden.json`과 같습니다(11/02 확인, 30건 Feature · 확률 차이 0).
- 다음 후보는 rf-v2(`train_random_forest_v2.py`, 가맹점 번호 `merchantId`를 뺀 모델)입니다. v1은 합성 가맹점 1001~1003만 배워서 실제 가맹점 번호가 들어오면 확률이 흔들립니다. 성능 유지 판정(`compare_rf_v1_v2.py`)을 통과한 뒤에만 팀 저장소 ai-server에 올립니다.
- ai-server는 그 모델로 사기 확률을 계산해 Rule 결과 옆에 기록만 합니다(Shadow Mode, 그림자 평가). 결정 배경은 팀 저장소 `docs/decisions/0061-fds-ml-shadow-inference.md`(ADR-0061)에 있습니다.
- `features.py`는 팀 저장소 `ai-server/app/ml/features.py`와 계산이 같아야 합니다(그쪽은 코드 정렬만 다름). 한쪽을 바꾸면 다른 쪽과 ai-server `tests/fixtures/ml_golden.json`도 함께 바꿉니다.

## 2. 개발 환경

- Python 3.12
- Windows / PowerShell (Linux에서도 같은 명령으로 실행됨)
- 실제 개인정보 사용 없음. 외부 데이터는 IBM TabFormer 합성 데이터 하나만 선택적으로 사용 (아래 "외부 데이터 출처 · 라이선스")

## 외부 데이터 출처 · 라이선스

이 저장소는 외부 데이터를 **포함하지 않습니다**. 아래 데이터는 각자 내려받아 `external/`에 두고 쓰며, 원본과 변환 결과(`external/` · `data/`)는 Git에 올리지 않습니다.

- 데이터: IBM TabFormer 합성 카드 거래 데이터 (`card_transaction.v1.csv`)
- 출처: https://github.com/IBM/TabFormer (IBM), 내려받기 https://ibm.box.com/v/tabformer-data
- 논문: Padhi et al., "Tabular Transformers for Modeling Multivariate Time Series", ICASSP 2021 (https://arxiv.org/abs/2011.01843)
- 라이선스: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0) — 위 저장소의 `LICENSE`. 압축 파일 안에 데이터 외 다른 약관 파일은 없습니다 (11/02 확인)
- 이 저장소에서 바꾼 것: `tabformer_convert.py`로 칸 이름 · 형식을 돈워리 결제 모양으로 바꾸고(금액 달러 → 원, 시각 형식, 거절 여부 등), 고객 250명 · 2015~2019년 표본만 사용합니다
- IBM이 이 실험을 검토하거나 보증한 것이 아닙니다. 성적표에는 집계 숫자만 싣고 원본 거래 줄은 싣지 않습니다
- 변환한 데이터를 다른 사람에게 건넬 때는 Apache 2.0 라이선스 사본과 위 "바꾼 것"을 함께 전달해야 합니다 (라이선스 4조)

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

`data/` · `models/` · `exports/` · `external/`은 Git에 올리지 않습니다. 아래 순서대로 다시 만들 수 있습니다.

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
| 8 | `python scorecard.py` | 기본 · 실험 | 모델 성적표 `reports/scorecard.md` — Rule(R01~R08 재현) · RF · RF v2 · XGB · 하이브리드 · 대조군(하이브리드와 같은 30% 데이터로 배운 XGB)을 같은 헛경보 수에서 비교. 없는 모델 파일 · 라이브러리는 건너뜀 |
| 9 | `python train_random_forest_v2.py` | 기본 | `models/random_forest_v2.joblib` (rf-v2 후보) — v1과 같은 절차, 가맹점 번호만 뺌 (1번 필요). 만든 뒤 8번을 다시 돌리면 성적표에 RF v2 줄이 생김 |
| 10 | `python compare_rf_v1_v2.py` | 기본 | v1 vs v2 성적표 `reports/rf_v1_vs_v2.md` + 통과 판정(헛경보 50건 기준 잡은 사기 5% 넘게 감소 없음 · AP 하락 0.01 이내), 가맹점 번호만 바꿨을 때 확률 흔들림 비교 (2 · 9번 필요) |
| 11 | `python export_ai_server_model.py --model models/random_forest_v2.joblib --version rf-v2 --cases ../ai-server/tests/fixtures/ml_golden.json` | 기본 | `exports/rf-v2.joblib`(압축 모델) · `exports/ml_golden_rf-v2.json`(golden 정답) — 팀 저장소 ai-server에 넣을 파일. 사례(대상 결제 · 이력)는 기존 golden에서 가져오고 정답만 이 모델로 다시 계산 (1 · 9번 필요, 팀 저장소 폴더 안에서 실행) |
| 12 | `python tabformer_convert.py` | 기본 | 외부 데이터(IBM TabFormer) → `data/tabformer/transactions.csv` · `fraud_episodes.csv`. 기본: 고객 250명(seed 42) · 2015~2019년 · 1달러 1,300원 (아래 "외부 데이터" 절, 원본 필요) |
| 13 | `python tabformer_scorecard.py` | 기본 | 외부 데이터 성적표 `reports/tabformer_scorecard.md` — Rule · RF(TabFormer로 학습, rf-v2와 같은 칸 · 설정) · RF + 지역 칸(확인 실험) · RF + 국가 A · B(Ⓥ3 국가 칸 정의 고르기, `country_features.py`) · rf-v2(합성 학습 그대로)를 같은 헛경보 수에서 비교 (9 · 12번 필요) |
| 14 | `python shadow_scenarios.py` | 기본 | 돈워리 로컬 서버(팀 저장소 실행 중)에 정상 · 사기 시나리오 결제 50건을 보내고 정답 라벨 `data/shadow/scenario_labels.csv`를 남김. 서명 · 전송은 팀 저장소 `scripts/demo/cakeshop_pay.py`를 그대로 씀 (`--dry-run`으로 목록만 보기) |
| 15 | `python shadow_export.py` → `python shadow_whatif.py --labels data/shadow/scenario_labels.csv` | 기본 | 만약 비교 `reports/shadow_whatif.md` — Shadow 기록(`sql/shadow_export.sql` 결과 CSV)으로 Rule × ML 일치표 · 직원 판정 · 시나리오 라벨 대조 · 기준값 바꿔 보기 · 국가 공백 · 같은 결제 rf-v1 ↔ rf-v2 (아래 "Shadow 만약 비교" 절) |

- 데이터는 거래 시각 순으로 Train 70% · Validation 15% · Test 15%로 나눕니다. 경계를 넘는 공격 사건은 평가 데이터에서 뺍니다(`split_and_purge`).
- 같은 헛경보 기준: Validation 정상 거래 2,555건 중 50건까지만 헛경보를 허용하도록(`TARGET_FPR = 50 / 2555`) 모델마다 기준값을 맞춘 뒤 탐지 수를 비교합니다.
- Test 평가는 `train_random_forest.py`만 합니다. 나머지 모델은 Validation으로만 비교합니다.
- `rule_engine.py`는 팀 저장소 `FdsEvaluationTransactionService.java`의 Rule 점수(R01 고액 30 · R04 10분 3회 30 · R02 새 기기 25 · R03 새 카드 25 · R06 카드 테스팅 45 · R08 거절 후 재시도 35, 100점 상한, 50점 보류 · 80점 거절)를 그대로 옮긴 것입니다. 합성 데이터에 IP · 차단 목록이 없어 R05 · R07은 항상 꺼집니다. 서버 Rule을 바꾸면 이 파일도 함께 바꿉니다.

그 밖의 분석 스크립트:

| 명령 | 설치 | 하는 일 |
|---|---|---|
| `python analyze_ml_features.py` | 기본 | Feature 분포 분석 |
| `python analyze_rf_validation.py` | 기본 | RF Validation 결과를 시나리오별로 분석 |
| `python analyze_sequence_readiness.py` | 기본 | 거래 순서(시퀀스) 학습에 쓸 수 있는지 점검 |
| `python compare_rf_xgb_validation.py` | 실험 | RF · XGB 예측 비교 |
| `python ablate_transformer_history.py` · `ablate_transformer_order.py` | 실험 | Transformer에서 과거 거래 · 순서를 없앴을 때의 차이 (제거 실험) |

### 외부 데이터 — IBM TabFormer (12 · 13번)

- 출처: [IBM TabFormer](https://github.com/IBM/TabFormer)의 합성 카드 거래 데이터 (Apache 2.0). 2,000명 · 약 2,439만 건 · 사기 0.12%. IBM이 이 실험을 검토 · 보증한 것은 아닙니다
- 받기: README가 안내하는 Box 링크(https://ibm.box.com/v/tabformer-data)에서 `transactions.tgz`를 받아 압축을 풀면 `card_transaction.v1.csv` 하나가 나옵니다 → `external/card_transaction.v1.csv`에 둡니다. GitHub Download ZIP은 Git LFS라 실제 파일 대신 작은 안내 파일만 들어올 수 있습니다
- 라이선스 확인(11/02): 저장소 LICENSE가 Apache 2.0이고, 압축 파일 안에도 CSV 말고 다른 약관 파일이 없습니다. 원본 · 변환 결과는 이 저장소에 올리지 않습니다(`external/` · `data/` 제외)
- 돈워리 칸으로 바꾸기: User → 고객, User + Card → 카드, 시각은 한국 시간으로 읽음(시간대 Feature가 밀리지 않게), 금액은 달러 × 환율 · 환불(음수) 제외, `Errors?`가 있으면 거절, 기기 · IP 정보는 없음(R02 · R05 · 기기 Feature 꺼짐). 가게가 있는 주(해외면 나라, 온라인은 ONLINE)를 `locationRef`로 남겨 "RF + 지역 칸" 확인 실험에만 씀 — 돈워리의 "평소와 다른 접속 국가"를 대신하는 칸이고 `features.py`는 바꾸지 않음. Ⓥ3에서는 같은 칸을 국가 단위(미국 주 → US, 온라인 → 국가 없음)로 바꿔 A "처음 본 국가인가" · B "평소 국가와 다른가"를 비교하고, 고르는 규칙(Validation 기준 ③ 잡은 사기, 차이 3건 이하면 B)과 고른 쪽의 Test 한 번 결과를 성적표 "국가 칸 고르기" 절에 남김
- 같은 카드의 사기가 7일 안에 이어지면 한 공격 묶음으로 봅니다 (학습 · 검증 구간에 나뉘어 들어가지 않게)
- 결과 (11/02, `reports/tabformer_scorecard.md`, 검증 16만 건 · 사기 252건 · 사기율 0.16%)
  - 정상 1.96% 헛경보에서 잡은 사기: Rule 14 · RF(TabFormer로 학습) 47 · RF + 지역 칸 57 · rf-v2(합성 학습 그대로) 8. AP는 무작위 0.0016 대비 RF 0.0146 · RF + 지역 칸 0.0169 · rf-v2 0.0021
  - 해석: TabFormer의 사기는 "어디서 · 무슨 업종에서"(평소와 다른 주 · 해외 사기율 37배)로 드러나는데, 우리 12칸은 행동 이력만 봐서 약하다. rf-v2가 무작위 수준인 것은 합성 데이터가 우리 Feature가 잡는 신호로 사기를 만들었기 때문 — 합성 성적표의 "Rule 대비 약 3.8배"는 우리가 설계한 사기 시나리오 안에서만 맞는 결론
  - 장소 칸("앞서 이 장소에서 결제한 적 있나")은 조금 도움(+10건). 돈워리에 "평소와 다른 접속 국가" 칸을 넣을지는 더 현실적인 합성 데이터로 다시 학습하는 v3 때 다시 본다
  - Ⓥ3 국가 칸 비교 (11/03): 장소를 국가 단위로 바꿔 비교한 결과 B "평소 국가와 다른가"를 고름. 정상 1.96% 헛경보에서 잡은 사기 A 51 · B 245/252, AP A 0.0359 · B 0.5080, Test 한 번 B 170/174 · AP 0.3875. TabFormer 사기가 해외 매장 결제에 몰린 합성 데이터라 크게 보이고, 온라인 결제(사기 9건)에서는 효과가 없음 → 돈워리(전부 온라인) 성능의 근거는 아님

### 돈워리 Shadow 만약 비교 (14 · 15번)

- 목적: 팀 저장소 ai-server가 결제에 영향 없이 남긴 Shadow 기록(`ml_shadow_scores`)으로 "ML이 판정했다면 무엇이 달라졌나"를 본다. AI 조사 보고서에 ML 점수를 넣을지는 이 결과로 판단한다
- 순서: 팀 저장소에서 Spring(`run-local.local.ps1 -MlShadow`) · ai-server 실행 → 14번 → 1~2분 기다림 → REVIEW 사건 몇 건을 화면에서 직접 판정 → `python shadow_export.py`(DB 비밀번호 입력)로 `data/shadow/shadow_export.csv` 만들기 — PC에 설치된 mariadb 클라이언트가 `sql/shadow_export.sql`을 실행하고 결과를 CSV로 바꿈(기본 127.0.0.1:3307 · root · donworry, 클라이언트는 PATH → `C:\Program Files\MariaDB *\bin` 순으로 찾음). HeidiSQL에서 실행해 CSV(머리글 포함)로 내보내도 됨 → 15번 `shadow_whatif.py`
- 고위험 국가 시나리오(F3)는 로컬 DB `fds_high_risk_countries`에 `--risk-country`(기본 NG)가 있어야 Rule R05가 걸린다
- 리포트에는 집계 숫자만 쓴다(손님 번호 · 이메일 · IP · 결제 번호 없음). 원본 CSV는 `data/`라 Git에 올라가지 않는다
- 한계: 시연 · 시나리오 결제라 표본이 작고, 결제를 초 단위로 이어 보내 `minutesSinceLastTx`가 실제보다 작다
- 결과 (11/03, `reports/shadow_whatif.md`, rf-v2 채점 99건 중 시나리오 50건)
  - 시나리오 정상 27건: 헛경보 Rule 0 · ML 0. 사기 23건: Rule 의심 5 · ML HIGH 13 · 둘 다 놓침 10(고위험 국가 3 · 소액 반복 4 · 고액 연속 2 · 같은 카드 1)
  - 일치표: Rule만 의심 0건 · ML만 의심 30건 — 이 기록에서 Rule이 잡은 결제는 ML도 모두 HIGH
  - ML HIGH는 금액에 몰림(50만 원 이상 74% · 미만 20%). 소액 새 손님은 확률 0.00 — 예전 시연의 "새 손님마다 HIGH"는 60만 원 결제만 보내서 생긴 착시
  - 해외 접속 5건 모두 ML LOW. 고위험 국가(NG) 사기는 Rule도 R05 40점(50점 미만)이라 통과 → 접속 국가 칸(v3) 근거
  - 같은 결제 46건 rf-v1 ↔ rf-v2: HIGH · LOW 뒤집힘 0건(확률 차이 최대 0.07)
  - 단서: 시나리오를 학습용 합성 데이터와 같은 발상으로 만들어 ML에 유리하다. 외부 데이터(TabFormer)에서 rf-v2는 무작위 수준이었다. 직원 판정 5건은 예전 시연 때 누른 판정이라 정답으로 약하다

## 5. 재현성

- 데이터 생성과 학습은 seed 42로 고정되어 있습니다.
- 11/02 확인: 새 Linux 환경(Python 3.12, `requirements.txt`)에서 다시 만든 결과가 Windows PC 원본과 같았습니다.
  - `data/transactions.csv` · `data/fraud_episodes.csv`: SHA-256 일치
  - `models/random_forest_baseline.joblib`: 거래 2만 건의 사기 확률 차이 최대 2.2e-16, 기준값 0.30 동일 (팀 ai-server의 rf-v1과도 같음)
- 신경망(Transformer · 하이브리드)은 seed를 고정해도 CPU · 라이브러리 환경에 따라 결과가 조금 다를 수 있습니다.
- XGBoost도 새로 학습하면 운영체제 · CPU 스레드에 따라 결과가 조금 다를 수 있습니다. 예: 성적표 대조군(헛경보 5건) Windows PC 205건 · Linux 215건. 같은 컴퓨터에서는 다시 돌려도 같습니다.
- 성적표 정본은 PC(학습한 모델 파일이 있는 곳)에서 만든 `reports/scorecard.md`입니다.

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

- 기본 설치: 60개 통과, torch가 필요한 테스트 파일 2개(`test_transformer.py` · `test_hybrid_xgboost.py`)는 자동으로 건너뜀
- 실험 설치: 위 2개 파일의 9개까지 모두 실행

주요 검증: 거래 · 사기 비율, 7종 사기 시나리오, ID 중복, 시간순 정합성, CSV 재읽기, 같은 seed 재현성, 공격 사건 연결, Feature 계산(현재 거래보다 이전 거래만 사용), 시간순 분할과 경계 사건 제외, 시퀀스 생성, 모델 출력 모양, rf-v2 칸 구성(가맹점 번호 제외) · ai-server가 읽는 모델 파일 모양 · v1 vs v2 통과 판정, TabFormer 변환(고객 단위 표본 · 한 줄 변환 · 공격 묶음) · 장소 칸이 앞선 거래만 보는지.

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
