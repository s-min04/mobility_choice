"""D 보강 입력: 관측소 1곳 대신 동네별 날씨를 만든다.

실행: python scripts/13_build_local_weather.py   (약 20~40분)

1. S-DoT 환경센서 (서울 전역 약 1,170곳, 1시간 단위 최고·평균·최저 기온), 2023.1 ~ 2026.3
   - 센서 × 날짜별 일 최고·최저 기온을 만든다 (날짜는 24시간제인 '등록일시' 기준).
   - 하루 18시간 이상 기록된 센서만 쓰고, 그날 서울 중앙값과 10℃ 넘게 다른 값(고장·직사광선)은 버린다.
2. 서울시 강우량계 (구청·빗물펌프장 등 수십 곳, 1분~10분 단위 누적), 2023 ~ 2025 → 관측소 × 날짜 일 강수량
3. 역마다 반경 1km 안 센서(없으면 가장 가까운 3개)의 중앙값으로 '역세권 날씨'를 만든다. 강수는 가장 가까운 강우량계.
"""

import glob
import io
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
USE = ["시리얼", "온도 최대(℃)", "온도 최소(℃)", "등록일시"]
station_key = lambda s: re.sub(r"\(.*?\)", "", str(s)).strip()


def sdot_daily_from_csv(src) -> pd.DataFrame:
    """src: 파일 경로 또는 바이트. 파일마다 cp949 / UTF-8 이 섞여 있어 차례로 시도한다."""
    raw = Path(src).read_bytes() if not isinstance(src, (bytes, bytearray)) else src
    for enc in ["cp949", "utf-8-sig"]:
        try:
            d = pd.read_csv(io.BytesIO(raw), encoding=enc, usecols=USE, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    d.columns = ["serial", "tmax_h", "tmin_h", "reg"]
    d[["tmax_h", "tmin_h"]] = d[["tmax_h", "tmin_h"]].apply(pd.to_numeric, errors="coerce")  # 일부 파일은 문자형
    ts = pd.to_datetime(d["reg"].astype(str), errors="coerce")
    d["date"] = ts.dt.normalize()
    d["hour"] = ts.dt.hour
    d = d.dropna(subset=["date", "tmax_h", "tmin_h"])
    d = d[d["tmax_h"].between(-30, 45) & d["tmin_h"].between(-30, 45)]  # -40 등 센서 오류값 제외
    g = d.groupby(["serial", "date"]).agg(tmax=("tmax_h", "max"), tmin=("tmin_h", "min"), hours=("hour", "nunique"))
    return g.reset_index()


def load_sdot() -> pd.DataFrame:
    parts = []
    for zf in sorted(glob.glob(str(RAW / "S-DoT_NATURE_20??.zip"))):
        z = zipfile.ZipFile(zf)
        for n in sorted(i for i in z.namelist() if i.endswith(".csv")):
            parts.append(sdot_daily_from_csv(z.read(n)))
        print(f"  {Path(zf).name} 완료", flush=True)
    for f in sorted(glob.glob(str(RAW / "S-DoT_NATURE_2026.*.csv"))):
        parts.append(sdot_daily_from_csv(f))
    # 주 단위 파일 경계에 걸친 날은 두 파일에 나뉘어 있어 다시 합친다 (시간 수는 큰 쪽)
    d = pd.concat(parts).groupby(["serial", "date"]).agg(tmax=("tmax", "max"), tmin=("tmin", "min"),
                                                         hours=("hours", "max")).reset_index()
    d = d[d["hours"] >= 18]
    med = d.groupby("date")[["tmax", "tmin"]].transform("median")
    ok = ((d["tmax"] - med["tmax"]).abs() <= 10) & ((d["tmin"] - med["tmin"]).abs() <= 10)
    print(f"  센서-날짜 {len(d):,}개 중 이상값 {(~ok).sum():,}개 제외")
    return d[ok]


def sensor_locations() -> pd.DataFrame:
    loc = pd.read_excel(RAW / "서울시 도시데이터 센서(S-DoT) 환경정보 설치 위치정보.xlsx")
    cols = {c: c for c in loc.columns}
    ser = next(c for c in loc.columns if "시리얼" in str(c))
    lat = next(c for c in loc.columns if "위도" in str(c))
    lon = next(c for c in loc.columns if "경도" in str(c))
    loc = loc[[ser, lat, lon]].set_axis(["serial", "lat", "lon"], axis=1)
    loc[["lat", "lon"]] = loc[["lat", "lon"]].apply(pd.to_numeric, errors="coerce")
    return loc.dropna().drop_duplicates("serial")


def stations() -> pd.DataFrame:
    st = pd.read_csv(RAW / "서울시 역사마스터 정보.csv", encoding="cp949")
    st["key"] = st["역사명"].map(station_key)
    st = st.groupby("key")[["위도", "경도"]].mean()
    p = gpd.GeoSeries(gpd.points_from_xy(st["경도"], st["위도"]), crs="EPSG:4326").to_crs("EPSG:5179")
    return pd.DataFrame({"x": p.x.to_numpy(), "y": p.y.to_numpy()}, index=st.index)


def to_xy(lon, lat):
    p = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs("EPSG:5179")
    return np.c_[p.x, p.y]


def station_weather(sd: pd.DataFrame, loc: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    loc = loc[loc["serial"].isin(sd["serial"].unique())].reset_index(drop=True)
    tree = cKDTree(to_xy(loc["lon"], loc["lat"]))
    near = {}
    for k, (x, y) in st[["x", "y"]].iterrows():
        idx = tree.query_ball_point([x, y], 1000)
        if len(idx) < 3:
            idx = list(tree.query([x, y], 3)[1])
        near[k] = loc.loc[idx, "serial"].tolist()
    rows = [(k, s) for k, ss in near.items() for s in ss]
    m = pd.DataFrame(rows, columns=["station", "serial"]).merge(sd, on="serial")
    out = m.groupby(["station", "date"]).agg(tmax_local=("tmax", "median"), tmin_local=("tmin", "median"),
                                              n_sensors=("serial", "nunique")).reset_index()
    print(f"  역 {out['station'].nunique()}개, 역당 센서 중앙값 {pd.Series({k: len(v) for k, v in near.items()}).median():.0f}개")
    return out


def read_any(raw: bytes, name: str, skip: int = 0) -> pd.DataFrame:
    for enc in ["cp949", "utf-8-sig"]:
        try:
            d = pd.read_csv(io.BytesIO(raw), encoding=enc, skiprows=skip) if name.lower().endswith(".csv") \
                else pd.read_excel(io.BytesIO(raw), skiprows=skip)
        except UnicodeDecodeError:
            continue
        # 2025년 파일은 첫 줄이 '2025년' 제목이라 머리글이 둘째 줄에 있다
        if skip == 0 and len(d.columns) == 1 and "년" in str(d.columns[0]):
            return read_any(raw, name, skip=1)
        return d
    raise ValueError(name)


def load_rain() -> tuple[pd.DataFrame, pd.DataFrame]:
    """강우량계 × 날짜 일 강수량(mm). 원자료는 10분 단위 강수량(보정하지 않은 원시자료)."""
    g = pd.read_excel(RAW / "서울시 강우량계 위치정보.xlsx")
    g = g[["강우량계명", "경도", "위도"]].dropna().drop_duplicates("강우량계명")
    parts = []
    for zf in sorted(glob.glob(str(RAW / "서울시 강우량 데이터(20*).zip"))):
        z = zipfile.ZipFile(zf)
        for n in z.namelist():
            if not n.lower().endswith((".csv", ".xlsx")):
                continue
            d = read_any(z.read(n), n)
            name_col = next((c for c in d.columns if "강우량계명" in str(c)), None)
            time_col = next((c for c in d.columns if str(c).strip() in ("시간", "자료수집 시각", "일시")), None)
            rain_col = next((c for c in d.columns if "10분" in str(c)), None)
            if not (name_col and time_col and rain_col):
                continue
            d = d[[name_col, time_col, rain_col]].set_axis(["gauge", "time", "rain10"], axis=1)
            d["date"] = pd.to_datetime(d["time"].astype(str).str[:10], errors="coerce")
            d["rain10"] = pd.to_numeric(d["rain10"], errors="coerce")
            d = d[d["date"] >= "2022-07-01"]
            parts.append(d.groupby(["gauge", "date"])["rain10"].sum().reset_index())
    r = pd.concat(parts).groupby(["gauge", "date"])["rain10"].sum().rename("prcp").reset_index()
    r = r[r["prcp"].between(0, 400)]  # 하루 400mm 초과는 계측 오류로 본다
    print(f"  강우량계 {r['gauge'].nunique()}곳 (위치 있는 곳 {r['gauge'].isin(g['강우량계명']).sum() and r.loc[r['gauge'].isin(g['강우량계명']), 'gauge'].nunique()}), "
          f"{r['date'].min().date()} ~ {r['date'].max().date()}")
    return g, r


def station_rain(g: pd.DataFrame, r: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    g = g[g["강우량계명"].isin(r["gauge"])].reset_index(drop=True)
    d, i = cKDTree(to_xy(g["경도"], g["위도"])).query(st[["x", "y"]].to_numpy())
    near = pd.DataFrame({"station": st.index, "gauge": g.loc[i, "강우량계명"].to_numpy(), "gauge_dist_m": d})
    out = near.merge(r, on="gauge").rename(columns={"prcp": "prcp_local"})
    print(f"  역-강우량계 거리 중앙값 {near['gauge_dist_m'].median():.0f}m")
    return out[["station", "date", "prcp_local", "gauge_dist_m"]]


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    if (PROC / "sdot_sensor_daily.parquet").exists():  # S-DoT 집계는 오래 걸려 한 번 만든 결과를 다시 쓴다
        sd = pd.read_parquet(PROC / "sdot_sensor_daily.parquet")
    else:
        sd = load_sdot()
        sd.to_parquet(PROC / "sdot_sensor_daily.parquet")
    city = sd.groupby("date")[["tmax", "tmin"]].median().add_suffix("_sdot").reset_index()
    city.to_parquet(PROC / "sdot_city_daily.parquet")
    loc = sensor_locations()
    print(f"  위치가 있는 센서 {loc['serial'].isin(sd['serial']).sum()}개 / 자료가 있는 센서 {sd['serial'].nunique()}개")
    sw = station_weather(sd, loc, stations())
    sw.to_parquet(PROC / "sdot_station_daily.parquet")
    g, r = load_rain()
    r.to_parquet(PROC / "rain_gauge_daily.parquet")
    station_rain(g, r, stations()).to_parquet(PROC / "rain_station_daily.parquet")
    print("저장 완료")


if __name__ == "__main__":
    main()
