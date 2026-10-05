"""B 분석(이동선택권 지수)에 필요한 원본 데이터를 내려받는다.

실행: python scripts/01_download.py

- 받은 파일은 data/raw/ 에 원래 파일명 그대로 저장하고 수정하지 않는다.
- data/raw/manifest.csv 에 출처, 기준시점, 내려받은 날짜, SHA-256 을 기록한다.
- 이미 받은 파일은 다시 받지 않는다(--force 로 강제).
- 파일 버전(seq, 기준일)은 고정해 두었다. 최신본으로 바꿀 때는 아래 목록만 고친다.
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.csv"

SEOUL_FILE_URL = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false"
SEOUL_SHEET_URL = "https://datafile.seoul.go.kr/bigfile/iot/sheet/csv/download.do"
DATA_GO_KR = "https://www.data.go.kr"
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter",
                 "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]

# 서울 열린데이터광장 파일형 데이터: (id, 데이터셋 이름, infId, infSeq, seq, 저장 파일명, 기준시점)
SEOUL_FILES = [
    ("bus_stops", "서울시 버스정류소 위치정보", "OA-15067", "1", "58",
     "서울시버스정류소위치정보(20260902).xlsx", "2026-09-02"),
    ("bus_route_stops", "서울시 버스노선별 정류소 정보", "OA-1095", "2", "58",
     "서울시버스노선별정류소정보(20260902).xlsx", "2026-09-02"),
    ("bike_stations", "서울시 공공자전거 대여소 정보", "OA-13252", "2", "24",
     "공공자전거 대여소 정보(26.6월 기준).xlsx", "2026-06"),
    ("bike_usage_monthly", "서울특별시 공공자전거 이용정보(월별)", "OA-15248", "1", "47",
     "서울특별시 공공자전거 이용정보(월별)_26.1-6.csv", "2026-01~06"),
    # 개인형 이동장치(공유 전동킥보드): 지수에는 넣지 않고 보행 방해 요인으로 따로 본다
    ("pm_towing_2025h1", "서울시 전동킥보드 견인 현황", "OA-21304", "1", "9",
     "서울특별시_전동킥보드_견인_현황(25.1~6월).xlsx", "2025-01~06"),
    ("pm_towing_2025h2", "서울시 전동킥보드 견인 현황", "OA-21304", "1", "10",
     "서울특별시_전동킥보드_견인_현황(25.7~12월).xlsx", "2025-07~12"),
    ("pm_devices_2023", "서울시 공유 전동킥보드 운영 현황", "OA-22199", "1", "2",
     "서울시 공유 전동킥보드 운영 현황_20231018.csv", "2023-10-18"),
    ("pm_devices_2025feb", "서울시 공유 전동킥보드 운영 현황", "OA-22199", "1", "3",
     "서울시 민간대여 공유 전동킥보드 기기 현황_25.2월기준.csv", "2025-02"),
    ("pm_devices_2025dec", "서울시 공유 전동킥보드 운영 현황", "OA-22199", "1", "4",
     "서울시 민간대여 공유 전동킥보드 기기 현황_25.12월기준.csv", "2025-12"),
    # D 보강: KT 생활이동 연령 × 수단 (겨울·여름 17개월) → 고령자 버스·차량·도보, 외출 포기 vs 수단 전환
    ("life_move_mode_202301", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202301",
     "seoul_trans_admdong1_in_202301.zip", "2023-01"),
    ("life_move_mode_202302", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202302",
     "seoul_trans_admdong1_in_202302.zip", "2023-02"),
    ("life_move_mode_202307", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202307",
     "seoul_trans_admdong1_in_202307.zip", "2023-07"),
    ("life_move_mode_202308", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202308",
     "seoul_trans_admdong1_in_202308.zip", "2023-08"),
    ("life_move_mode_202312", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202312",
     "seoul_trans_admdong1_in_202312.zip", "2023-12"),
    ("life_move_mode_202401", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202401",
     "seoul_trans_admdong1_in_202401.zip", "2024-01"),
    ("life_move_mode_202402", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202402",
     "seoul_trans_admdong1_in_202402.zip", "2024-02"),
    ("life_move_mode_202407", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202407",
     "seoul_trans_admdong1_in_202407.zip", "2024-07"),
    ("life_move_mode_202408", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202408",
     "seoul_trans_admdong1_in_202408.zip", "2024-08"),
    ("life_move_mode_202412", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202412",
     "seoul_trans_admdong1_in_202412.zip", "2024-12"),
    ("life_move_mode_202501", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202501",
     "seoul_trans_admdong1_in_202501.zip", "2025-01"),
    ("life_move_mode_202502", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202502",
     "seoul_trans_admdong1_in_202502.zip", "2025-02"),
    ("life_move_mode_202507", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202507",
     "seoul_trans_admdong1_in_202507.zip", "2025-07"),
    ("life_move_mode_202508", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202508",
     "seoul_trans_admdong1_in_202508.zip", "2025-08"),
    ("life_move_mode_202512", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202512",
     "seoul_trans_admdong1_in_202512.zip", "2025-12"),
    ("life_move_mode_202601", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202601",
     "seoul_trans_admdong1_in_202601.zip", "2026-01"),
    ("life_move_mode_202602", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))", "OA-22655", "1", "202602",
     "seoul_trans_admdong1_in_202602.zip", "2026-02"),
    # D 보강: S-DoT 환경센서 (서울 전역 약 1,170곳, 시간별 기온) → 관측소 1곳 대신 동네별 기온
    ("sdot_2023", "스마트서울 도시데이터 센서(S-DoT) 환경정보", "OA-15969", "3", "2023",
     "S-DoT_NATURE_2023.zip", "2023"),
    ("sdot_2024", "스마트서울 도시데이터 센서(S-DoT) 환경정보", "OA-15969", "3", "2024",
     "S-DoT_NATURE_2024.zip", "2024"),
    ("sdot_2025", "스마트서울 도시데이터 센서(S-DoT) 환경정보", "OA-15969", "3", "2025",
     "S-DoT_NATURE_2025.zip", "2025"),
    ("sdot_20260119", "스마트서울 도시데이터 센서(S-DoT) 환경정보", "OA-15969", "3", "20260119",
     "S-DoT_NATURE_2026.01.05-01.11.csv", "2026.01.05-01.11"),
    ("sdot_20260126", "스마트서울 도시데이터 센서(S-DoT) 환경정보", "OA-15969", "3", "20260126",
     "S-DoT_NATURE_2026.01.12-01.18.csv", "2026.01.12-01.18"),
    ("sdot_20260202", "스마트서울 도시데이터 센서(S-DoT) 환경정보", "OA-15969", "3", "20260202",
     "S-DoT_NATURE_2026.01.19-01.25.csv", "2026.01.19-01.25"),
    ("sdot_20260209", "스마트서울 도시데이터 센서(S-DoT) 환경정보", "OA-15969", "3", "20260209",
     "S-DoT_NATURE_2026.01.26-02.01.csv", "2026.01.26-02.01"),
    ("sdot_20260216", "스마트서울 도시데이터 센서(S-DoT) 환경정보", "OA-15969", "3", "20260216",
     "S-DoT_NATURE_2026.02.02-02.08.csv", "2026.02.02-02.08"),
    ("sdot_20260223", "스마트서울 도시데이터 센서(S-DoT) 환경정보", "OA-15969", "3", "20260223",
     "S-DoT_NATURE_2026.02.09-02.15.csv", "2026.02.09-02.15"),
    ("sdot_20260302", "스마트서울 도시데이터 센서(S-DoT) 환경정보", "OA-15969", "3", "20260302",
     "S-DoT_NATURE_2026.02.16-02.22.csv", "2026.02.16-02.22"),
    ("sdot_20260309", "스마트서울 도시데이터 센서(S-DoT) 환경정보", "OA-15969", "3", "20260309",
     "S-DoT_NATURE_2026.02.23-03.01.csv", "2026.02.23-03.01"),
    ("sdot_20260316", "스마트서울 도시데이터 센서(S-DoT) 환경정보", "OA-15969", "3", "20260316",
     "S-DoT_NATURE_2026.03.02-03.08.csv", "2026.03.02-03.08"),
    ("sdot_20260323", "스마트서울 도시데이터 센서(S-DoT) 환경정보", "OA-15969", "3", "20260323",
     "S-DoT_NATURE_2026.03.09-03.15.csv", "2026.03.09-03.15"),
    ("sdot_20260330", "스마트서울 도시데이터 센서(S-DoT) 환경정보", "OA-15969", "3", "20260330",
     "S-DoT_NATURE_2026.03.16-03.22.csv", "2026.03.16-03.22"),
    ("sdot_locations", "스마트서울 도시데이터 센서(S-DoT) 환경정보 설치 위치", "OA-15969", "3", "100000001",
     "서울시 도시데이터 센서(S-DoT) 환경정보 설치 위치정보.xlsx", "2026-03"),
    # D 보강: 서울시 강우량계 (동네별 강수)
    ("rain_2021_2024", "서울시 강우량 데이터", "OA-22715", "1", "3", "서울시 강우량 데이터(2021~2024).zip", "2021~2024"),
    ("rain_2025", "서울시 강우량 데이터", "OA-22715", "1", "4", "서울시 강우량 데이터(2025).zip", "2025"),
    ("rain_gauges", "서울시 강우량계 위치데이터", "OA-22824", "1", "2", "서울시 강우량계 위치정보.xlsx", "2026-09"),
    # A 분석: 도착 행정동 × 시간대 × 성·연령 × 이동수단 (서울시·KT 수도권 생활이동, 2026년 9월 일별 파일 묶음)
    ("life_move_mode_202609", "수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단 데이터 (내국인))",
     "OA-22655", "1", "202609", "seoul_trans_admdong1_in_202609.zip", "2026-09"),
    # A 분석: 도착 행정동 × 시간대 × 성·연령 × 이동목적 (귀가 이동으로 거주자의 외출 횟수를 본다)
    ("life_move_purpose_202609", "수도권 생활이동 (성 연령별, 도착지 기준)-내국인",
     "OA-22298", "1", "202609", "seoul_purpose_admdong1_in_202609.zip", "2026-09"),
    # A 분석 소득 변수: 동별·연령별 기초생활 수급자
    ("welfare_recipients", "서울시 국민기초생활 수급자 동별 현황", "OA-22227", "1", "2",
     "서울시 국민기초생활 수급자 동별 현황(202405).xlsx", "2024-05"),
    # A 검증: 행정동별 자동차 등록 (자가용 승용차 = 차를 쓸 여유)
    ("car_registration", "서울시 행정동별 연료별 자동차 등록현황", "OA-21236", "3", "50",
     "서울시 자치구 읍면동별 연료별 자동차 등록현황(행정동)(26년8월).csv", "2026-08"),
    # E 분석: 차량 이동 평균거리. 출발·도착 행정동 × 수단 × 이동거리(move_dist, m), 연령 없음 (하루 파일 약 90MB)
    # 2026년 9월 평일 5일 (주마다 요일을 바꿔 고른다: 목 9.3, 월 9.7, 수 9.9, 화 9.15, 금 9.18)
    *[(f"life_move_od_mode_{d}", "수도권 생활이동 (출도착 행정동별 수단 데이터)", "OA-22657", "1", d[2:],
       f"seoul_trans_admdong3_final_{d}.zip", f"{d[:4]}-{d[4:6]}-{d[6:]}")
      for d in ["20260903", "20260907", "20260909", "20260915", "20260918"]],
    # E 분석: 고령자 이동거리 보정. 출발·도착 행정동 × 연령대(60세 이상) × 이동거리, 수단 없음 (같은 5일)
    *[(f"life_move_od_age_{d}", "수도권 생활이동 (연령대별 출도착 행정동 데이터)", "OA-22658", "1", d[2:],
       f"seoul_trans_admdong4_in_{d}.zip", f"{d[:4]}-{d[4:6]}-{d[6:]}")
      for d in ["20260903", "20260907", "20260909", "20260915", "20260918"]],
]

# 서울 열린데이터광장 시트형 데이터: (id, 데이터셋 이름, infId, 정렬 컬럼, 저장 파일명)
SEOUL_SHEETS = [
    ("subway_stations", "서울시 역사마스터 정보", "OA-21232", "BLDN_ID DESC", "서울시 역사마스터 정보.csv"),
    # A 분석 소득 변수: 행정동별 아파트 평균 시가 (분기별, 최신 분기를 쓴다)
    ("apartment_price", "서울시 상권분석서비스(아파트-행정동)", "OA-22163", "", "서울시 상권분석서비스(아파트-행정동).csv"),
    # A 검증: 역별·월별 유임/무임 승하차 (교통카드 실측, 무임 = 65세 이상·장애인·국가유공자)
    ("subway_free_ride", "서울시 지하철 호선별 역별 유/무임 승하차 인원 정보", "OA-12251", "",
     "서울시 지하철 호선별 역별 유무임 승하차 인원 정보.csv"),
    # C 분석: 고령자의 일상 목적지인 병·의원 위치 (좌표 포함)
    ("clinics", "서울시 병의원 위치 정보", "OA-20337", "", "서울시 병의원 위치 정보.csv"),
    # D 분석: 무더위·한파 쉼터 (좌표 EPSG:5186)
    ("climate_shelters", "서울시 기후동행쉼터", "OA-22386", "", "서울시 기후동행쉼터.csv"),
]

# 공공데이터포털 파일데이터: (id, 데이터셋 이름, publicDataPk, publicDataDetailPk, fileDetailSn, 저장 파일명, 기준시점)
# 과거 버전은 데이터 화면의 '주기성 과거 데이터' 목록에 있는 detailPk·detailSn 으로 받는다
STP = ("15099330", "서울교통공사_1_8호선 역별 일별 시간대별 승객유형별 승하차인원")
DATA_GO_KR_FILES = [
    ("population", "행정안전부_지역별(행정동) 성별 연령별 주민등록 인구수", "15097972",
     "uddi:5beebd9e-8733-44f8-817f-9cfa03548b7a", "1",
     "지역별(행정동) 성별 연령별 주민등록 인구수_20260831.csv", "2026-08-31"),
    # D 분석: 역별·일별·시간대별 승객유형(우대권 = 65세 이상 등)별 승하차, 반기별 파일 8개 (2022.7~2026.6)
    # 파일 번호(detailSn)는 버전마다 CSV/JSON 순서가 달라 "auto" 로 CSV 가 나올 때까지 시도한다
    ("subway_type_2022h2", STP[1], STP[0], "uddi:ef6b14a9-f9f5-4965-a49b-94f409115ca6", "auto",
     "서울교통공사_역별일별시간대별승객유형별승하차_20221231.csv", "2022-07~12"),
    ("subway_type_2023h1", STP[1], STP[0], "uddi:e78d89ab-938f-42ae-bc81-84477b03c764", "auto",
     "서울교통공사_역별일별시간대별승객유형별승하차_20230630.csv", "2023-01~06"),
    ("subway_type_2023h2", STP[1], STP[0], "uddi:b08fb029-4c0b-4e85-bf8c-ff96b0e155ca", "auto",
     "서울교통공사_역별일별시간대별승객유형별승하차_20231231.csv", "2023-07~12"),
    ("subway_type_2024h1", STP[1], STP[0], "uddi:45aa66a5-25a8-4ede-81b8-cce5140b619c", "auto",
     "서울교통공사_역별일별시간대별승객유형별승하차_20240630.csv", "2024-01~06"),
    ("subway_type_2024h2", STP[1], STP[0], "uddi:b069b014-093b-4780-912d-e8aec3e4cfe4", "1",
     "서울교통공사_역별일별시간대별승객유형별승하차_20241231.csv", "2024-07~12"),
    ("subway_type_2025h1", STP[1], STP[0], "uddi:bef34229-dc1c-4509-8eb4-afd60a9b4be8", "1",
     "서울교통공사_역별일별시간대별승객유형별승하차_20250630.csv", "2025-01~06"),
    ("subway_type_2025h2", STP[1], STP[0], "uddi:f7f77ce0-550a-4510-b2f7-cb7892a96e5b", "1",
     "서울교통공사_역별일별시간대별승객유형별승하차_20251231.csv", "2025-07~12"),
    ("subway_type_2026h1", STP[1], STP[0], "uddi:5845041d-3a37-4c30-9142-e62d3b8816b8", "1",
     "서울교통공사_역별일별시간대별승객유형별승하차_20260630.csv", "2026-01~06"),
    # D 분석: 9호선 2·3단계(언주~중앙보훈병원) 역별 일별 승객유형별 승하차
    ("subway9_type", "서울교통공사_9호선 2_3단계 역별 일별 승객유형별 승하차인원", "15108374",
     "uddi:310e9867-52b7-41fa-8318-a59b1f6799d6", "auto", "서울교통공사_9호선2_3단계_역별일별승객유형별승하차.zip", "2023~2026-07 (연도별 CSV 묶음)"),
    # E 분석: 시도 × 차종(승용·승합·화물·특수) 도로부문 온실가스 배출량 (천 tCO2eq, 2012~2024)
    ("ghg_road_by_type", "한국교통안전공단_지역별 차종별 도로부문 온실가스 배출량", "15106288",
     "uddi:3a061bfc-f00e-48c6-8688-fca740e42669", "1", "한국교통안전공단_지역별차종별도로부문온실가스배출량_20241231.csv", "2012~2024"),
    # E 분석: 같은 자료의 가스별(CO2·CH4·N2O) 판 - CO2eq 중 CO2 비중 확인용
    ("ghg_road_by_gas", "한국교통안전공단_지역별 온실가스별 도로부문 온실가스 배출량", "15087285",
     "uddi:d469c178-b244-44c0-b5c5-90f732cc93d0", "1", "한국교통안전공단_지역별온실가스별도로부문온실가스배출량_20241231.csv", "2012~2024"),
]
DATA_GO_KR_PROVIDER = {"population": "행정안전부", "ghg_road_by_type": "한국교통안전공단", "ghg_road_by_gas": "한국교통안전공단"}

# E 분석: 시도 × 차종 × 연료 연간 자동차 주행거리 (천 km). 교통안전정보관리시스템(TMACS) 화면이 부르는 조회 주소를 그대로 쓴다
# (화면: 자동차 주행거리통계 > 차종별 연료별 주행거리). 연간 주행거리(Tg1700_04)와 1일 주행거리(Tg1700_03), 2021~2024년
TMACS = ("tmacs_mileage", "자동차 주행거리통계 - 차종별 연료별 주행거리 (시도별, 연간·1일)",
         "https://tmacs.kotsa.or.kr/web/TG/TG200/TG2200/Tg2119_AJAX.jsp", "tmacs_mileage_2021_2024.json", "2021~2024")
TMACS_PAGE = "https://tmacs.kotsa.or.kr/web/TG/TG200/TG2200/Tg1700_02.jsp?mid=S3080"

# D 분석: 서울 기상관측소(기상청 ASOS 108 = WMO 47108) 일자료. 기상자료개방포털은 로그인이 필요해
# 같은 관측값을 NOAA 를 거쳐 제공하는 Meteostat 공개 파일을 쓴다 (컬럼: 날짜, 평균·최저·최고기온, 강수, 적설, ...)
WEATHER = ("weather_seoul_daily", "서울 기상관측소 일자료 (Meteostat, WMO 47108)",
           "https://bulk.meteostat.net/v2/daily/47108.csv.gz", "meteostat_47108_daily.csv.gz", "")
# 10년 월별 분석용: Meteostat 서울 관측소는 2017~2019년 최고기온·2022년 이전 강수가 대부분 비어 있어
# 재분석 자료(ERA5, Open-Meteo, 키 불필요)를 받고 2021년 이후 관측값과 겹치는 구간으로 보정해 쓴다
WEATHER_ERA5 = ("weather_seoul_era5", "서울 일별 기온·강수 재분석(ERA5, Open-Meteo) - 서울 관측소 좌표",
                "https://archive-api.open-meteo.com/v1/archive?latitude=37.5714&longitude=126.9658"
                "&start_date=2015-01-01&end_date=2026-09-30"
                "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum&timezone=Asia%2FSeoul",
                "openmeteo_era5_seoul_daily.json", "2015-01-01~2026-09-30")
WEATHER2 = ("weather_gimpo_daily", "김포공항 기상관측소 일자료 (Meteostat, WMO 47110) - 교차 확인용",
            "https://bulk.meteostat.net/v2/daily/47110.csv.gz", "meteostat_47110_daily.csv.gz", "")

# 행정동 경계: 통계청 SGIS 경계를 행정안전부 10자리 코드와 맞춰 정리한 공개본
BOUNDARY = ("dong_boundary", "대한민국 행정동 경계(admdongkor)",
            "https://raw.githubusercontent.com/vuski/admdongkor/master/ver20260701/HangJeongDong_ver20260701.geojson",
            "HangJeongDong_ver20260701.geojson", "2026-07-01")

# 산림·공원·수면: 사람이 살지 않는 땅을 격자에서 빼기 위한 마스크 (OpenStreetMap, 서울 경계 사각형으로 받은 뒤 분석 단계에서 자른다)
OSM_MASK_QUERY = """
[out:json][timeout:300][bbox:37.41,126.76,37.72,127.19];
(
  way["natural"~"^(wood|water|scrub|heath|bare_rock)$"];
  relation["natural"~"^(wood|water)$"];
  way["landuse"~"^(forest|cemetery|military)$"];
  relation["landuse"~"^(forest|military)$"];
  way["leisure"~"^(park|nature_reserve|golf_course)$"];
  relation["leisure"~"^(park|nature_reserve|golf_course)$"];
  relation["boundary"="national_park"];
  way["waterway"="riverbank"];
  way["aeroway"="aerodrome"];
  relation["aeroway"="aerodrome"];
);
out geom;
"""
OSM_MASK = ("osm_nonresidential", "OpenStreetMap 산림·공원·수면 폴리곤 (서울)", "osm_nonresidential_seoul.json")

# 큰길(간선도로): 길에서 택시를 잡을 수 있는 곳의 근사치. 도시고속도로(motorway)는 정차할 수 없어 뺀다
OSM_ROADS_QUERY = """
[out:json][timeout:300][bbox:37.41,126.76,37.72,127.19];
way["highway"~"^(trunk|primary|secondary|tertiary)$"];
out geom tags;
"""
# A 분석: 경사 계산용 수치표고모델 (Copernicus DEM GLO-30, AWS 공개 버킷, 서울을 덮는 2개 타일)
DEM_TILES = [
    (f"dem_{t}", f"Copernicus DEM GLO-30 {t}",
     f"https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t}_00_DEM/Copernicus_DSM_COG_10_{t}_00_DEM.tif",
     f"Copernicus_DSM_COG_10_{t}_00_DEM.tif")
    for t in ["N37_00_E126", "N37_00_E127"]
]

OSM_ROADS = ("osm_major_roads", "OpenStreetMap 간선도로 (trunk·primary·secondary·tertiary)", "osm_major_roads_seoul.json")

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (research; AI-transportation-solution)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_stream(resp: requests.Response, path: Path) -> None:
    resp.raise_for_status()
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(1 << 20):
            f.write(chunk)
    tmp.rename(path)


def get_seoul_file(inf_id, inf_seq, seq, path):
    resp = session.post(SEOUL_FILE_URL, data={"infId": inf_id, "seqNo": "", "seq": seq, "infSeq": inf_seq},
                        stream=True, timeout=600)
    save_stream(resp, path)


def get_seoul_sheet(inf_id, order_by, path):
    resp = session.post(SEOUL_SHEET_URL, data={
        "srvType": "S", "infId": inf_id, "serviceKind": "1", "pageNo": "1", "gridTotalCnt": "999999",
        "ssUserId": "SAMPLE_VIEW", "strWhere": "", "strOrderby": order_by, "filterCol": "필터선택", "txtFilter": "",
    }, stream=True, timeout=600)
    save_stream(resp, path)


def get_data_go_kr(pk, detail_pk, path, detail_sn="1"):
    """포털 화면의 '다운로드' 버튼과 같은 순서로 요청한다. 캡차가 요구되면 멈추고 수동 다운로드를 안내한다.
    detail_sn="auto" 면 1~4번 파일을 차례로 받아 CSV(첫 글자가 '[' 나 '{' 가 아닌 것)를 고른다."""
    if detail_sn == "auto":
        for sn in ["1", "2", "3", "4"]:
            try:
                get_data_go_kr(pk, detail_pk, path, sn)
            except Exception:
                continue
            if path.read_bytes()[:1] not in (b"[", b"{"):
                return
        raise RuntimeError("CSV 파일을 찾지 못했다")
    info = session.post(f"{DATA_GO_KR}/tcs/dss/selectFileDataDownload.do", data={
        "publicDataDetailPk": detail_pk, "publicDataPk": pk, "atchFileId": "", "fileDetailSn": detail_sn,
        "publicDataTyCode": "PR0051"}, timeout=60).json()
    if not info.get("status"):
        raise RuntimeError(f"파일 정보 조회 실패: {info.get('error')}")
    atch, sn = info["atchFileId"], info["fileDetailSn"]
    limit = session.post(f"{DATA_GO_KR}/cmm/cmm/check-limit.json",
                         data={"atchFileId": atch, "fileDetailSn": sn}, timeout=60).json()
    if limit.get("needCaptcha"):
        raise RuntimeError(f"다운로드 횟수 제한으로 보안문자가 필요합니다. 브라우저에서 직접 받아 {path} 로 저장하세요: "
                           f"{DATA_GO_KR}/data/{pk}/fileData.do")
    resp = session.get(f"{DATA_GO_KR}/cmm/cmm/fileDownload.do",
                       params={"atchFileId": atch, "fileDetailSn": sn}, stream=True, timeout=600)
    save_stream(resp, path)


def get_url(url, path):
    save_stream(session.get(url, stream=True, timeout=600), path)


def get_tmacs(url, path):
    """연도 × (연간, 1일) 주행거리를 모두 받아 JSON 하나로 저장한다. 행: 차종 × 연료, 열: 전국·시도."""
    out = []
    for year in range(2021, 2025):
        for gubun, item in [("Tg1700_04", "annual_kkm"), ("Tg1700_03", "daily_km")]:
            resp = session.post(url, data={"gubun": gubun, "year": str(year), "carUse": "전체"}, timeout=120)
            resp.raise_for_status()
            out += [{**r, "item": item} for r in resp.json()]
    path.write_text(json.dumps(out, ensure_ascii=False))


def get_osm(query, path):
    # Overpass 는 브라우저형 User-Agent 를 거부하고(406), 서버가 바쁘면 504 를 준다 → 미러 순서대로 시도
    headers = {"User-Agent": "AI-transportation-solution/0.1 (data analysis contest)"}
    errors = []
    for url in OVERPASS_URLS:
        try:
            resp = requests.post(url, data={"data": query}, headers=headers, timeout=900)
            resp.raise_for_status()
            path.write_text(json.dumps(resp.json(), ensure_ascii=False))
            return
        except Exception as e:
            errors.append(f"{url}: {e}")
    raise RuntimeError("; ".join(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="이미 받은 파일도 다시 받는다")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    jobs = []
    for key, name, inf_id, inf_seq, seq, fname, ref in SEOUL_FILES:
        jobs.append((key, name, "서울특별시", f"https://data.seoul.go.kr/dataList/{inf_id}/F/1/datasetView.do",
                     fname, ref, lambda p, a=(inf_id, inf_seq, seq): get_seoul_file(*a, p)))
    for key, name, inf_id, order_by, fname in SEOUL_SHEETS:
        jobs.append((key, name, "서울특별시", f"https://data.seoul.go.kr/dataList/{inf_id}/S/1/datasetView.do",
                     fname, "", lambda p, a=(inf_id, order_by): get_seoul_sheet(*a, p)))
    for key, name, pk, detail_pk, sn, fname, ref in DATA_GO_KR_FILES:
        provider = DATA_GO_KR_PROVIDER.get(key, "서울교통공사")
        jobs.append((key, name, provider, f"{DATA_GO_KR}/data/{pk}/fileData.do",
                     fname, ref, lambda p, a=(pk, detail_pk, sn): get_data_go_kr(a[0], a[1], p, a[2])))
    for key, name, url, fname, ref in [WEATHER, WEATHER2, WEATHER_ERA5]:
        jobs.append((key, name, "Meteostat (원자료: 기상청 관측, NOAA ISD/GHCN)", url, fname, ref,
                     lambda p, u=url: get_url(u, p)))
    key, name, url, fname, ref = BOUNDARY
    jobs.append((key, name, "vuski/admdongkor (원자료: 통계청 SGIS)", url, fname, ref, lambda p, u=url: get_url(u, p)))
    key, name, fname = OSM_MASK
    jobs.append((key, name, "OpenStreetMap contributors", OVERPASS_URLS[0], fname, "",
                 lambda p: get_osm(OSM_MASK_QUERY, p)))
    for key, name, url, fname in DEM_TILES:
        jobs.append((key, name, "ESA Copernicus (AWS Open Data)", url, fname, "2021 release",
                     lambda p, u=url: get_url(u, p)))
    key, name, url, fname, ref = TMACS
    jobs.append((key, name, "한국교통안전공단 교통안전정보관리시스템(TMACS)", TMACS_PAGE, fname, ref,
                 lambda p, u=url: get_tmacs(u, p)))
    key, name, fname = OSM_ROADS
    jobs.append((key, name, "OpenStreetMap contributors", OVERPASS_URLS[0], fname, "",
                 lambda p: get_osm(OSM_ROADS_QUERY, p)))

    old = {}
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8") as f:
            old = {r["dataset_id"]: r for r in csv.DictReader(f)}

    rows, failed = [], []
    for key, name, provider, url, fname, ref, fetch in jobs:
        path = RAW / fname
        if path.exists() and not args.force and key in old:
            print(f"[skip] {fname}")
            rows.append(old[key])
            continue
        print(f"[get ] {fname} ...", flush=True)
        try:
            fetch(path)
        except Exception as e:  # 한 파일이 실패해도 나머지는 받는다
            print(f"[FAIL] {fname}: {e}", file=sys.stderr)
            failed.append(key)
            continue
        rows.append({
            "dataset_id": key, "dataset_name": name, "provider": provider, "source_url": url,
            "raw_path": f"data/raw/{fname}", "reference_date": ref, "download_date": date.today().isoformat(),
            "size_bytes": path.stat().st_size, "sha256": sha256(path),
        })
        print(f"       {path.stat().st_size / 1e6:.1f} MB")

    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nmanifest: {MANIFEST} ({len(rows)}건)")
    if failed:
        print(f"실패: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
