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
]

# 서울 열린데이터광장 시트형 데이터: (id, 데이터셋 이름, infId, 정렬 컬럼, 저장 파일명)
SEOUL_SHEETS = [
    ("subway_stations", "서울시 역사마스터 정보", "OA-21232", "BLDN_ID DESC", "서울시 역사마스터 정보.csv"),
    # A 분석 소득 변수: 행정동별 아파트 평균 시가 (분기별, 최신 분기를 쓴다)
    ("apartment_price", "서울시 상권분석서비스(아파트-행정동)", "OA-22163", "", "서울시 상권분석서비스(아파트-행정동).csv"),
    # A 검증: 역별·월별 유임/무임 승하차 (교통카드 실측, 무임 = 65세 이상·장애인·국가유공자)
    ("subway_free_ride", "서울시 지하철 호선별 역별 유/무임 승하차 인원 정보", "OA-12251", "",
     "서울시 지하철 호선별 역별 유무임 승하차 인원 정보.csv"),
]

# 공공데이터포털 파일데이터: (id, 데이터셋 이름, publicDataPk, publicDataDetailPk, 저장 파일명, 기준시점)
DATA_GO_KR_FILES = [
    ("population", "행정안전부_지역별(행정동) 성별 연령별 주민등록 인구수", "15097972",
     "uddi:5beebd9e-8733-44f8-817f-9cfa03548b7a",
     "지역별(행정동) 성별 연령별 주민등록 인구수_20260831.csv", "2026-08-31"),
]

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


def get_data_go_kr(pk, detail_pk, path):
    """포털 화면의 '다운로드' 버튼과 같은 순서로 요청한다. 캡차가 요구되면 멈추고 수동 다운로드를 안내한다."""
    info = session.post(f"{DATA_GO_KR}/tcs/dss/selectFileDataDownload.do", data={
        "publicDataDetailPk": detail_pk, "publicDataPk": pk, "atchFileId": "", "fileDetailSn": "1",
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
    for key, name, pk, detail_pk, fname, ref in DATA_GO_KR_FILES:
        jobs.append((key, name, "행정안전부", f"{DATA_GO_KR}/data/{pk}/fileData.do",
                     fname, ref, lambda p, a=(pk, detail_pk): get_data_go_kr(*a, p)))
    key, name, url, fname, ref = BOUNDARY
    jobs.append((key, name, "vuski/admdongkor (원자료: 통계청 SGIS)", url, fname, ref, lambda p, u=url: get_url(u, p)))
    key, name, fname = OSM_MASK
    jobs.append((key, name, "OpenStreetMap contributors", OVERPASS_URLS[0], fname, "",
                 lambda p: get_osm(OSM_MASK_QUERY, p)))
    for key, name, url, fname in DEM_TILES:
        jobs.append((key, name, "ESA Copernicus (AWS Open Data)", url, fname, "2021 release",
                     lambda p, u=url: get_url(u, p)))
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
