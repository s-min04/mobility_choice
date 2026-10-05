"""D 분석 입력: 역별·일별 고령자·일반 승차와 서울 날씨를 맞춘다.

실행: python scripts/10_build_weather.py

- 서울교통공사 1~8호선 역별·일별·시간대별 승객유형별 승하차 (2022.7~2026.6, 반기별 파일 8개)
  - 승차만 쓴다. 우대권 = 65세 이상 경로(장애인·국가유공자 포함), 일반 = 일반 요금 성인
  - 환승역은 호선별로 나뉘어 있어 역 이름으로 합친다
- 서울교통공사 9호선 2·3단계(언주~중앙보훈병원, 12개 역) 역별·일별 승객유형별 승하차 (2023~2026.7)
  - 무임권 = 고령자, 일반 + 기후동행카드 + 정기권 = 일반 (1~8호선 자료는 기후동행카드가 일반에 들어 있다)
- 날씨 (서울 전체 하루 값):
  - 서울 기상관측소(WMO 47108 = 기상청 ASOS 108), Meteostat. 기본값.
  - 김포공항 관측소(WMO 47110), Meteostat. 교차 확인용.
  - ERA5 재분석(Open-Meteo, 서울 관측소 좌표): 2021년 이후 서울 관측소와 겹치는 날로 선형 보정해,
    관측값이 빈 2015~2020년 월별 분석에 쓴다.
- 공휴일: 관공서 공휴일·대체·임시공휴일·선거일 목록 (HOLIDAYS). 일반 승차가 같은 달 평일 중앙값의 75% 미만인
  평일을 데이터로 다시 찾아 목록과 맞는지 확인한다.
"""

import glob
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

HOLIDAYS = pd.to_datetime([
    "2022-08-15", "2022-09-09", "2022-09-12", "2022-10-03", "2022-10-10",
    "2023-01-23", "2023-01-24", "2023-03-01", "2023-05-05", "2023-05-29", "2023-06-06", "2023-08-15", "2023-09-28",
    "2023-09-29", "2023-10-02", "2023-10-03", "2023-10-09", "2023-12-25",
    "2024-01-01", "2024-02-09", "2024-02-12", "2024-03-01", "2024-04-10", "2024-05-06", "2024-05-15", "2024-06-06",
    "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18", "2024-10-01", "2024-10-03", "2024-10-09", "2024-12-25",
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-01", "2025-03-03", "2025-05-05",
    "2025-05-06", "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03", "2025-10-05", "2025-10-06", "2025-10-07",
    "2025-10-08", "2025-10-09", "2025-12-25",
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-03-01", "2026-03-02",
    "2026-05-05", "2026-05-25", "2026-06-03",
    # 공휴일은 아니지만 일반 승차가 공휴일 수준으로 줄어드는 근로자의 날 (데이터로 확인)
    "2023-05-01", "2024-05-01", "2025-05-01", "2026-05-01",
])
TYPES = {"우대권": "elderly", "일반": "general"}
L9_TYPES = {"무임권": "elderly", "일반": "general", "기후동행카드": "general", "정기권": "general"}
station_key = lambda s: re.sub(r"\(.*?\)", "", str(s)).strip()
METEO_COLS = ["date", "tavg", "tmin", "tmax", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"]


def load_ridership() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily, hourly = [], []
    for f in sorted(glob.glob(str(RAW / "서울교통공사_역별일별시간대별승객유형별승하차_*.csv"))):
        d = pd.read_csv(f, encoding="cp949")
        d = d[(d["승하차구분"] == "승차") & d["승객유형"].isin(TYPES)]
        hours = [c for c in d.columns if "시간대" in c]
        d["date"] = pd.to_datetime(d["수송일자"].astype(str))
        d["station"] = d["역명"].map(station_key)
        d["type"] = d["승객유형"].map(TYPES)
        d["total"] = d[hours].sum(axis=1)
        daily.append(d.groupby(["date", "station", "type"])["total"].sum())
        hourly.append(d.groupby(["date", "type"])[hours].sum())
        print(f"  {Path(f).name}: {d['date'].min().date()} ~ {d['date'].max().date()}, 역 {d['station'].nunique()}개")
    daily = pd.concat(daily).groupby(level=[0, 1, 2]).sum().unstack("type").reset_index().assign(line="1~8호선")
    hourly = pd.concat(hourly).groupby(level=[0, 1]).sum()
    hourly.columns = [f"h{i:02d}" for i in range(len(hourly.columns))]  # h00 = 06시 이전, h01 = 06~07시, ...
    return daily, hourly.reset_index()


def load_line9() -> pd.DataFrame:
    z = zipfile.ZipFile(RAW / "서울교통공사_9호선2_3단계_역별일별승객유형별승하차.zip")
    parts = []
    for i in z.infolist():
        d = pd.read_csv(io.BytesIO(z.read(i)), encoding="cp949")
        d = d[(d["승하차구분"] == "승차") & d["승객유형"].isin(L9_TYPES)]
        d["date"] = pd.to_datetime(d["수송일자"].astype(str))
        d["station"] = d["역명"].str.replace(r"역$", "", regex=True).map(station_key)
        d["type"] = d["승객유형"].map(L9_TYPES)
        parts.append(d.groupby(["date", "station", "type"])["인원수"].sum())
    out = pd.concat(parts).groupby(level=[0, 1, 2]).sum().unstack("type").reset_index().assign(line="9호선 2·3단계")
    print(f"  9호선 2·3단계: {out['date'].min().date()} ~ {out['date'].max().date()}, 역 {out['station'].nunique()}개")
    return out


def meteostat(station: str) -> pd.DataFrame:
    w = pd.read_csv(RAW / f"meteostat_{station}_daily.csv.gz", header=None, names=METEO_COLS)
    w["date"] = pd.to_datetime(w["date"])
    return w[["date", "tmin", "tmax", "prcp"]]


def era5_calibrated(asos: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    js = json.loads((RAW / "openmeteo_era5_seoul_daily.json").read_text())["daily"]
    e = pd.DataFrame({"date": pd.to_datetime(js["time"]), "tmax": js["temperature_2m_max"],
                      "tmin": js["temperature_2m_min"], "prcp": js["precipitation_sum"]})
    m = e.merge(asos, on="date", suffixes=("_era5", "_asos"))
    m = m[m["date"] >= "2021-01-01"]
    fit = {}
    for v in ["tmax", "tmin"]:
        ok = m[[f"{v}_era5", f"{v}_asos"]].dropna()
        b, a = np.polyfit(ok[f"{v}_era5"], ok[f"{v}_asos"], 1)
        e[v] = a + b * e[v]
        fit[v] = {"기울기": b, "절편": a, "상관": float(ok.corr().iloc[0, 1]),
                  "보정 후 평균오차(℃)": float(np.mean(np.abs(a + b * ok[f"{v}_era5"] - ok[f"{v}_asos"]))), "n": len(ok)}
    ok = m[["prcp_era5", "prcp_asos"]].dropna()
    scale = ok["prcp_asos"].sum() / ok["prcp_era5"].sum()
    e["prcp"] = e["prcp"] * scale
    fit["prcp"] = {"배율": scale, "상관": float(ok.corr().iloc[0, 1]), "n": len(ok)}
    return e, fit


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    daily, hourly = load_ridership()
    daily = pd.concat([daily, load_line9()], ignore_index=True)
    asos = meteostat("47108")
    gimpo = meteostat("47110").rename(columns={"tmin": "tmin_gimpo", "tmax": "tmax_gimpo", "prcp": "prcp_gimpo"})
    era5, fit = era5_calibrated(asos)
    print("ERA5 보정:", json.dumps(fit, ensure_ascii=False, default=float))

    city = (daily[daily["line"] == "1~8호선"].groupby("date")[["elderly", "general"]].sum().reset_index()
            .merge(asos, on="date", how="left").merge(gimpo, on="date", how="left"))
    city["dow"] = city["date"].dt.dayofweek
    city["holiday"] = city["date"].isin(HOLIDAYS)
    wk = city[(city["dow"] < 5)]
    med = wk.groupby(wk["date"].dt.to_period("M"))["general"].transform("median")
    low = wk[wk["general"] < 0.75 * med]
    print(f"데이터로 찾은 저이용 평일 {len(low)}일 중 목록에 없는 날: {low.loc[~low['holiday'], 'date'].dt.date.tolist()}")
    print("목록에 있으나 이용이 줄지 않은 평일:",
          city.loc[city["holiday"] & (city["dow"] < 5) & ~city["date"].isin(low["date"]), "date"].dt.date.tolist())

    city = city[city["tmax"].notna() & city["tmin"].notna()]
    end = city["date"].max()
    print(f"분석 기간: {city['date'].min().date()} ~ {end.date()} ({len(city)}일)")
    daily = daily[daily["date"] <= end].merge(city[["date", "holiday"]], on="date")
    hourly = hourly[hourly["date"] <= end]
    city.to_parquet(PROC / "weather_city_daily.parquet")
    daily.to_parquet(PROC / "weather_station_daily.parquet")
    hourly.to_parquet(PROC / "weather_city_hourly.parquet")
    era5.to_parquet(PROC / "weather_era5_calibrated.parquet")
    (PROC / "weather_era5_fit.json").write_text(json.dumps(fit, ensure_ascii=False, indent=2, default=float))
    print(f"폭염일(최고 33℃ 이상) {(city['tmax'] >= 33).sum()}일, 한파일(최저 -10℃ 이하) {(city['tmin'] <= -10).sum()}일, "
          f"큰비(10mm 이상) {(city['prcp'] >= 10).sum()}일, 역 {daily['station'].nunique()}개")


if __name__ == "__main__":
    main()
