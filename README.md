# B. 이동선택권 지수와 고령자 이동 사각지대

서울 행정동마다 "실제로 쓸 수 있는 대중교통 수단(지하철·버스·따릉이)이 몇 개인가"를 일반 성인 기준과 고령자(65세 이상) 기준으로 각각 계산하고, 고령자 기준으로 선택권이 낮은 지역이 어디에 몰려 있는지 찾는다. 택시는 요금 때문에 매일 쓸 수 있는 선택지로 보기 어려워 지수에서 빼고 참고값으로만 계산한다. 공유 전동킥보드는 고령자의 수단이 아니라 보행을 막는 요인으로 따로 본다.

## 실행

이 폴더(저장소 최상위)에서 실행한다. Python 3.11 이상.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/01_download.py      # 원본 데이터 14종 (약 140MB)
.venv/bin/python scripts/02_build_index.py   # 100m 격자 → 행정동 지수
.venv/bin/python scripts/03_analysis.py      # 격차 분해, LISA, 민감도, 택시 참고값, 그림
.venv/bin/python scripts/04_pm_obstruction.py # 공유 전동킥보드 보행 방해
.venv/bin/python scripts/05_taxi_call_trend.py # 동행 온다 콜택시 이용 추이 (기사 수치)
.venv/bin/python scripts/06_build_usage.py    # A: 고령자 실제 이동(생활이동)·외출·경사
.venv/bin/python scripts/07_model_usage.py    # A: B 검증, 기대 대비 실제(랜덤포레스트), SHAP, 외출 회귀 (약 4분)
.venv/bin/python scripts/08_verify_elderly.py # A: 교통카드 실측(역별 무임 승차)·자동차 등록으로 고령자 특유 불리함 재검정
.venv/bin/python scripts/09_cluster.py        # C: 행정동 유형 군집(K-평균 5개)과 정책 처방
```

Google Colab에서는 `!pip install -r requirements.txt` 후 같은 순서로 `!python scripts/...` 를 실행한다.

## 데이터

| id | 데이터 | 제공 | 기준시점 |
|---|---|---|---|
| population | 행정동 성별·1세별 주민등록 인구 | 행정안전부 / 공공데이터포털 | 2026-08-31 |
| dong_boundary | 행정동 경계 (vuski/admdongkor, 원자료 통계청 SGIS) | GitHub | 2026-07-01 |
| subway_stations | 서울시 역사마스터 정보 (수도권 784개 역) | 서울 열린데이터광장 OA-21232 | 다운로드일 |
| bus_stops | 서울시 버스정류소 위치정보 | 서울 열린데이터광장 OA-15067 | 2026-09-02 |
| bus_route_stops | 서울시 버스노선별 정류소 정보 | 서울 열린데이터광장 OA-1095 | 2026-09-02 |
| bike_stations | 공공자전거 대여소 정보 | 서울 열린데이터광장 OA-13252 | 2026-06 |
| bike_usage_monthly | 공공자전거 이용정보(월별, 연령대 포함) | 서울 열린데이터광장 OA-15248 | 2026-01~06 |
| osm_nonresidential | 산림·공원·수면·묘지·군사시설·공항 폴리곤 | OpenStreetMap (ODbL) | 다운로드일 |
| osm_major_roads | 간선도로 (trunk·primary·secondary·tertiary) | OpenStreetMap (ODbL) | 다운로드일 |
| pm_towing_2025h1/h2 | 전동킥보드 견인 현황 (주소·유형) | 서울 열린데이터광장 OA-21304 | 2025 |
| pm_devices_* | 공유 전동킥보드 업체별 기기 대수 | 서울 열린데이터광장 OA-22199 | 2023.10, 2025.2, 2025.12 |

데이터 파일이 아닌 수치 1개는 기사에서 가져왔다: 60세 이상의 택시 앱 호출 비율 19% (서울연구원 「2024년 택시 이용 시민만족도 조사」, 택시 하차 시민 4,000명 면접조사, 데이터솜·경향신문 보도 재인용, 원문 확인 필요).

원본은 `data/raw/`에 저장되며 Git에 올리지 않는다. 출처와 SHA-256은 `data/raw/manifest.csv`에 남는다.

## 방법

1. 서울을 100m 격자로 나누고, OSM 산림·공원·수면 등에 들어가는 격자는 뺀다 (60,614개 → 거주 가능 38,952개).
2. 격자점마다 수단별 이용가능도(0~1)를 구한다.
   - 지하철: 10분 보행권 안에 역이 있으면 1
   - 버스: 5분 보행권 안 정류소를 지나는 서로 다른 노선 수 ÷ 기준 노선 수(서울 중앙값 7개), 최대 1
   - 따릉이: 5분 보행권 안에 대여소가 있으면 1
3. 이동선택권 지수 MCI = 세 값의 합(0~3).
4. 고령자 기준은 두 가지만 바꾼다.
   - 보행속도: 1.2 → 0.8 m/s
   - 따릉이 연령 가중치 β: 65세 이상 인구 1인당 이용건수 ÷ 20~59세 인구 1인당 이용건수 = **0.115**
5. 행정동 값은 거주 가능 격자점의 평균이다. 사각지대는 MCI가 1 미만인 곳, 즉 제대로 쓸 수 있는 대중교통 수단이 하나도 없는 곳이다. 동 안에서 고령자가 고르게 산다고 보고, 사각지대 격자 비율로 사각지대 고령인구를 추정한다.
6. 공간 분석은 Queen 인접 가중치를 쓴다. 전역 Moran's I, 국지적 Moran(LISA), 이변량 LISA(고령인구 비율 × 이동선택권)를 계산하고, 999회 순열검정에서 p<0.05인 경우만 군집으로 본다.

가정값은 `02_build_index.py`의 `PARAMS`에 모아 두었고, 바꿔 가며 확인한 결과는 `outputs/tables/b3_sensitivity.csv`에 있다.

## 산출물

| 파일 | 내용 |
|---|---|
| `outputs/figures/b1_mci_general_vs_elderly.png` | 일반 성인과 고령자 기준 지수 지도 |
| `outputs/figures/b2_gap_decomposition.png` | 고령자 지수가 어디서 줄어드는지 단계별로 분해 |
| `outputs/figures/b3_lisa_clusters.png` | LISA 군집 지도와 이변량 LISA 지도 |
| `outputs/tables/b0_dong_mci_all.csv` | 427개 행정동 전체 지표 |
| `outputs/tables/b1_gap_decomposition.csv` | 격차 분해 표 |
| `outputs/tables/b2_priority_dongs.csv` | 우선지역 25개 동 |
| `outputs/tables/b3_sensitivity.csv` | 민감도 분석 |
| `outputs/tables/b4_taxi_reference.csv` | 참고: 택시를 넣을 경우 (앱 가능/불가 고령자) |
| `outputs/figures/b5_pm_obstruction.png`, `outputs/tables/b5_pm_*.csv` | 공유 전동킥보드 기기 추이와 견인 유형 |
| `outputs/figures/b6_taxi_call_trend.png`, `outputs/tables/b6_taxi_call.json` | 동행 온다 콜택시 월 이용 추이와 비교 수치 (공개 데이터가 없어 기사 수치, 출처는 스크립트에 기록) |
| `outputs/tables/b_summary.json` | 주요 수치 요약 |
| `data/processed/dong_mci_lisa.gpkg` | 행정동 경계와 전체 지표 (QGIS에서 바로 열림) |

## 한계 (다음 단계 후보)

- 거리는 직선거리에 우회계수 1.3을 곱해 근사했다. 실제 도로망과 경사는 반영하지 않았다.
- 버스는 노선 수만 보고 배차간격은 보지 않았다.
- 지하철역의 엘리베이터 유무, 저상버스 비율 같은 고령자 접근 조건도 빠져 있다.
- 동 안의 고령인구 분포를 균등하다고 가정했다.
- 따릉이 β는 서울 전체 값 하나다. 대여소마다 고령자 이용률이 다를 수 있다.
- 택시 참고값은 간선도로(trunk·primary·secondary) 근처면 길에서 잡을 수 있다고 보았다. 실제 빈 택시 통행량, 배차 실패, 요금 부담은 반영하지 않았다.
- 킥보드 견인 건수는 자치구의 견인 시행 여부와 기록 방식에 좌우된다 (예: 강남구는 2,401건 모두 '차도', 종로·성북구는 거의 0건).

## A. 고령자의 실제 이동으로 본 검증 (06·07)

- 데이터: 서울시·KT 수도권 생활이동 2026년 9월 평일 20일 (추석 연휴 제외)
  - 수단: OA-22655
  - 목적: OA-22298
  - 수단·목적 코드는 시간대 패턴으로 판별했다. 지하철=6(새벽 운행 없음), 버스=4·5, 도보=7, 차량=8, 귀가=3
- 결과변수: 70대 이상 대중교통 분담률 = (지하철+버스)÷(지하철+버스+차량), 1인당 하루 외출(귀가 이동÷인구)
- 소득 변수: 65세 이상 기초생계급여 수급률(OA-22227, 2024.5, 동 이름으로 매칭), 아파트 평균 시가(OA-22163, 최신 분기)
- 기대값: 공급 조건 10개 → 랜덤포레스트 5겹 교차검증 표본 밖 예측(5회 평균). 잔차는 동네 전체 몫(청장년 잔차)과 고령자만의 몫으로 나눈다.
- 산출물: `outputs/figures/a1_validation.png`, `a2_shap.png`, `a3_residual_map.png`, `a4_income_outing.png`, `outputs/tables/a*.csv`
  (소득 없는 모델 결과는 `a2_expected_vs_actual_no_income.csv`, 우선지역 견고성은 `a5_priority_robustness.csv`)

### A 검증 (08)

- **역 단위 검정:** 거주 격자를 가장 가까운 역에 배정해 역세권 인구를 만든다(1.5km 이내).
  - 고령자 상대 이용률 = (무임÷유임 승차) ÷ (역세권 65세 이상÷65세 미만)
  - 이 값을 역세권 고령자의 거리 구간(369~554m, 554m 밖)과 경사 8% 이상 비율로 회귀한다. 표본 6가지, HC1.
- **부촌 검정:** 행정동별 자가용 승용차(인구 1,000명당)와 '고령자만 덜 탐' 잔차의 관계를 본다.
- **산출물:** `outputs/figures/a5_station_test.png`, `outputs/tables/a7_*.csv`, `a8_*.csv`

## C. 행정동 유형 (09)

- **변수 11개:** 공급, 병·의원 접근(OA-20337), 행동, 인구, 소득(자동차 등록 포함), 급경사 비율. 표준화 후 K-평균으로 나눈다.
- **군집 수 선택:** 실루엣, Ward와의 일치도, 부트스트랩 안정성으로 5개를 골랐다.
- **유형별 안정성:** 부트스트랩 Jaccard, 그리고 군집 4·6개나 Ward로 바꿨을 때의 일치도로 확인한다.
- **산출물:** `outputs/figures/c1_cluster_map.png`, `c2_cluster_profile.png`, `outputs/tables/c*.csv`
