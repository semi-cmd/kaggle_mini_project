
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-0.91_AUC-orange.svg)
![Stats](https://img.shields.io/badge/Statistics-Regression-green.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-0.90_AUC-yellow.svg)
![Stats](https://img.shields.io/badge/Statistics-Logit%20%26%20Tests-green.svg)
![EDA](https://img.shields.io/badge/EDA-Visualization-red.svg)

# 프로젝트 주제 : 머신러닝 기반 당뇨병 진단 예측 모델

### 🚀 핵심 성과 (Key Highlights)
- **데이터 분석**: 9개 핵심 변수(Age, BMI, 혈압 등)를 기반으로 한 다각도 통계 검정 수행
- **최고 성능 달성**: XGBoost 모델 기반 **AUC-ROC 0.91**, **Recall 0.75** 기록 (상위 10% 수준)
- **의사결정 지원**: SHAP 분석을 통해 개별 변수가 당뇨 발생 확률에 기여하는 '설명 가능한(XAI)' 인사이트 도출

## ✔️1. project overview
- **주제** : 생활 습관 또는 신체 상태를 활용한 당뇨병 유무 분류
- **데이터셋** : [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset/data)

- **핵심 목표** : `Diabetes Health Indicators Dataset`을 바탕으로 사용자의 생활습관 및 신체 상태 데이터를 분석하여<br>
설문 기반의 **당뇨병 고위험군 선별 예측 모델**을 구축하는 것을 목표

## ✔️2. Data Dictionary (주요 핵심 변수)
- 실제 분석 결과 통해 확보한 변수들의 내용 기재
- 총 변수갯수 : 31개 中 9개 핵심변수 선정

| 변수명 | 설명 | 변수 유형 |
| :--- | :--- | :--- |
| **Age** | 사용자의 연령대(13단계 그룹화) | 수치형 (Ordinal) |
| **Diet_score** | 균형 잡힌 식단 이행 정도 점수 | 수치형 (Continuous) |
| **Physical_activity** | 주간 신체 활동량 및 운동 강도 | 수치형 (Continuous) |
| **Bmi** | 체질량 지수($kg/m^2$) | 수치형 (Continuous) |
| **Systolic_BP** | 수축기 혈압 측정치 | 수치형 (Continuous) |
| **Triglycerides** | 혈중 중성지방 농도 | 수치형 (Continuous) |
| **HDL_Cholesterol** | 고밀도 지질단백질(좋은 콜레스테롤) | 수치형 (Continuous) |
| **Family_history** | 직계 가족 중 당뇨병 환자 유무 | 범주형 (Binary) |
| **Hypertension_history** | 고혈압 진단 및 과거력 유무 | 범주형 (Binary) |

 ## ✔️3. Problem Definition
 **3.1 데이터 특성 및 분석 과제** <br>
    **비대칭적 클래스 분포**<br>
    - **현상** : 정상군 대비 당뇨 환자군(Target=1)의 비율이 현저히 낮은 불균형 구조를 보임<br>
    - **대응** : 단순 정확도(Accuracy)를 배제, 환자를 놓치지 않는 **재현율(Recall)**과 변별력 측정하는 **AUC-ROC**를 핵심 평가지표로 설정<br>
    <br>
    **복합적 요인성**<br> 
    - **현상** : 당뇨병 유전적(`Family_history`), 생리학적 수치(`Bmi`, `Systolic_BP`,`Triglycerides`), 생활 환경(`Diet_score`, `Physical_activity`) 비선형적 결합되어 발생<br>
    - **대응** : 변수 간 단순 선형 관계를 넘어, 고차원 알고리즘 적용이 필수적<br>

 **3.2 분석 전략 및 방법론**<br>
    - **통계분석** : 다중회귀, T-test, 카이제곱 검정, 로지스틱회귀<br>
    - **머신러닝** : 로지스틱회귀, 결정트리, XGBoost, LightBGM<br>

## ✔️4. Data preprocessing
- **클래스 불균형 해소** : blah
- **범주형 변수 처리** 
    + 순서형 : ordinal encoder 처리 (A, B, C)
    + 일반 범주 : One-Hot Encoding 처리
- **데이터 스케일링** : StandardScaler(표준화)

## ✔️5. 통계분석 핵심 인사이트
- 혈당이 중요함 : 다른 알려진 요인(나이, BMI)보다 통계적으로 매우 훨씬, 강력하게, 유의미하게 영향이 있음을 확인 (via 회귀분석)
![Q-Q Plot](output/correlation%20matrix.png)

## ✔️6. 모델링 평가지표
- 최종 모델은 XGBoost로 선정

| Model | Accuracy | Recall | F1-Score | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest | 0.85 | 0.70 | 0.74 | 0.88 |
| **XGBoost** | **0.86** | **0.75** | **0.78** | **0.91** |

> **Note** : 최종 대회 결과는 Public 0.70807 / Private 0.70807 (feat. 1등 점수).
> **Note** : 최종 대회 결과는 Public 0.70807 / Private 0.70807 (상위 10%).

## ✔️7. Feature Importance (옵션)
- SHAP 활용
- 예측 모델에서 영향력이 가장 컸던 지표 순위
1. Age
2. BMI
- 그림 추가

## ✔️8. Conclusion
- 결론1
- 결론2
- 결론3

# 보고서
- 프로젝트 상세 보고서는 PDF 슬라이드 자료를 참고하여 주세요
- 00 보고서 : [당뇨병 예측 모델링 : 통계분석 및 머신러닝 접근](report/프로젝트보고서.pdf)
- 분석코드 : [분석코드](분석코드.ipynb)

# 🔗 배지 및 이모지 공식 소스 링크
| 용도 | 사이트 이름 | 링크 |
| :--- | :--- | :--- |
| **배지 생성** | Shields.io | [https://shields.io/](https://shields.io/) |
| **로고/색상 검색** | Simple Icons | [https://simpleicons.org/](https://simpleicons.org/) |
| **이모지 검색** | Emoji Cheat Sheet | [https://github.com/ikatyang/emoji-cheat-sheet](https://github.com/ikatyang/emoji-cheat-sheet) |



| 변수명 | 설명 | 값의 의미 |
| :--- | :--- | :--- |
| **Diabetes_binary** | 당뇨 여부 (**Target**) | 0: 음성, 1: 당뇨/전단계 |
| HighBP | 고혈압 여부 | 0: 정상, 1: 고혈압 |
| HighChol | 고콜레스테롤 여부 | 0: 정상, 1: 높음 |
| BMI | 체질량 지수 | 수치형 |
| Smoker | 흡연 여부 | 100개비 이상 흡연 여부 (0/1) |
| Stroke | 뇌졸중 경험 | 0: 없음, 1: 있음 |
| HeartDiseaseorAttack | 심장질환/심근경색 | 0: 없음, 1: 있음 |
| PhysActivity | 신체 활동 | 최근 30일 이내 운동 여부 (0/1) |
| GenHlth | 주관적 건강 상태 | 1(매우 좋음) ~ 5(매우 나쁨) |
| Age | 연령대 | 1(18-24) ~ 13(80세 이상) |
| Income | 소득 수준 | 1(최저) ~ 8(최고) |



