"""D 분석 입력: 역별·일별 고령자(우대권)·일반 승차와 서울 날씨를 맞춘다.

실행: python scripts/10_build_weather.py

- 서울교통공사 1~8호선 역별·일별·시간대별 승객유형별 승하차 (2024.7~2026.6, 반기별 파일 4개)
  - 승차만 쓴다. 우대권 = 65세 이상 경로(장애인·국가유공자 포함), 일반 = 일반 요금 성인
  - 환승역은 호선별로 나뉘어 있어 역 이름으로 합친다
- 날씨: 서울 기상관측소(WMO 47108 = 기상청 ASOS 108) 일자료, Meteostat. 2026.3.19 까지 있어 분석 기간도 그때까지다.
- 공휴일: 관공서 공휴일·대체공휴일·임시공휴일 목록 (아래 HOLIDAYS). 일반 승차가 같은 달 평일 중앙값의 75% 미만인
  평일을 데이터로 다시 찾아 목록과 맞는지 확인한다.
"""

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

HOLIDAYS = pd.to_datetime([
    "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18", "2024-10-01", "2024-10-03", "2024-10-09", "2024-12-25",
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-01", "2025-03-03", "2025-05-05",
    "2025-05-06", "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03", "2025-10-05", "2025-10-06", "2025-10-07",
    "2025-10-08", "2025-10-09", "2025-12-25",
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-03-01", "2026-03-02",
    "2026-05-05", "2026-05-25", "2026-06-03",
    # 공휴일은 아니지만 일반 승차가 공휴일 수준으로 줄어드는 근로자의 날 (데이터로 확인)
    "2025-05-01", "2026-05-01",
])
TYPES = {"우대권": "elderly", "일반": "general"}
station_key = lambda s: re.sub(r"\(.*?\)", "", str(s)).strip()


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
    daily = pd.concat(daily).unstack("type").reset_index()
    hourly = pd.concat(hourly)
    hourly.columns = [f"h{i:02d}" for i in range(len(hourly.columns))]  # h00 = 06시 이전, h01 = 06~07시, ...
    return daily, hourly.reset_index()


def load_weather() -> pd.DataFrame:
    w = pd.read_csv(RAW / "meteostat_47108_daily.csv.gz", header=None,
                    names=["date", "tavg", "tmin", "tmax", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"])
    w["date"] = pd.to_datetime(w["date"])
    return w[["date", "tavg", "tmin", "tmax", "prcp", "snow"]]


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    daily, hourly = load_ridership()
    w = load_weather()

    city = daily.groupby("date")[["elderly", "general"]].sum().reset_index().merge(w, on="date", how="left")
    city["dow"] = city["date"].dt.dayofweek
    city["holiday"] = city["date"].isin(HOLIDAYS)
    # 데이터로 공휴일 확인: 평일인데 일반 승차가 같은 달 평일 중앙값의 75% 미만
    wk = city[(city["dow"] < 5)]
    med = wk.groupby(wk["date"].dt.to_period("M"))["general"].transform("median")
    low = wk[wk["general"] < 0.75 * med]
    unlisted = low.loc[~low["holiday"], "date"].dt.date.tolist()
    listed_not_low = city.loc[city["holiday"] & (city["dow"] < 5) & ~city["date"].isin(low["date"]), "date"].dt.date.tolist()
    print(f"데이터로 찾은 저이용 평일 {len(low)}일 중 목록에 없는 날: {unlisted}")
    print(f"목록에 있으나 이용이 줄지 않은 평일: {listed_not_low}")

    city = city[city["tmax"].notna()]
    end = city["date"].max()
    print(f"분석 기간: {city['date'].min().date()} ~ {end.date()} ({len(city)}일)")
    daily = daily[daily["date"] <= end].merge(city[["date", "holiday"]], on="date")
    hourly = hourly[hourly["date"] <= end]
    city.to_parquet(PROC / "weather_city_daily.parquet")
    daily.to_parquet(PROC / "weather_station_daily.parquet")
    hourly.to_parquet(PROC / "weather_city_hourly.parquet")
    print(city[["tmax", "tmin", "prcp"]].describe().round(1).to_string())
    print(f"폭염일(최고 33℃ 이상) {(city['tmax'] >= 33).sum()}일, 한파일(최저 -12℃ 이하) {(city['tmin'] <= -12).sum()}일, "
          f"역 {daily['station'].nunique()}개")


if __name__ == "__main__":
    main()
