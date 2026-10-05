"""E 분석 입력: 승용차·승합차 배출계수(g/km)와 행정동별 차량 이동 평균거리를 만든다.

실행: python scripts/14_build_carbon_inputs.py   (01_download.py 다음, 약 3분)

1. 배출계수 = 한국교통안전공단 시도별 차종별 도로부문 온실가스 배출량(천 tCO2eq)
             ÷ 교통안전정보관리시스템(TMACS) 시도별 차종별 연간 주행거리(천 km)
   - 두 자료 모두 같은 기관(한국교통안전공단)이 차종(승용·승합·화물·특수)과 시도를 같은 기준으로 나눈다.
   - 승용차에는 택시(영업용 승용)가 들어 있다. KT 생활이동의 '차량'도 자가용·택시를 구분하지 않으므로 맞는 짝이다.
   - 배기관 배출만이다. 전기차 충전 전력의 배출은 들어 있지 않다.
2. 차량 이동 평균거리 = 수도권 생활이동 출도착 × 수단(OA-22657)의 move_dist, 2026년 9월 평일 5일
   - 내국인만 쓴다 (A의 OA-22655 도 내국인 자료다). 서울 행정동에 도착한 차량(코드 8) 이동만 본다.
   - move_dist 는 출발·도착 동 중심점 사이 직선거리와 거의 같다(아래에서 비율을 계산해 출력한다).
     그래서 도로 거리로 쓰려면 우회계수를 곱해야 한다 → 15번에서 가정으로 다룬다.
   - 출발지가 서울인 이동만 따로 구한다 (마을버스·수요응답형 버스가 대신할 수 있는 이동에 가깝다).
3. 고령자 거리 보정값 = 연령대별 출도착(OA-22658)에서 60세 이상 평균 이동거리 ÷ 전체 평균 이동거리
   - 이 자료는 수단이 없다. 그래서 차량만의 비율이 아니라 모든 수단을 합친 비율이다 → 민감도에만 쓴다.
4. 점검: 수도권 안 차량 인·km(KT) ÷ 수도권(서울·인천·경기) 승용차 차·km(TMACS) = 대략의 재차인원
"""

import json
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TAB = ROOT / "outputs" / "tables"

DAYS = ["20260903", "20260907", "20260909", "20260915", "20260918"]
CAR = 8
CAPITAL = {"11": "SEOUL", "28": "INCHEON", "41": "GYEONGGI"}  # 행정기관코드 앞 2자리
TYPES = {"승용": "승용차", "승합": "승합차"}


def emission_factors() -> pd.DataFrame:
    e = pd.read_csv(RAW / "한국교통안전공단_지역별차종별도로부문온실가스배출량_20241231.csv", encoding="cp949")
    e.columns = ["year", "region", "승용", "승합", "화물", "특수"]
    e["region"] = e["region"].str.replace(" ", "")
    m = pd.DataFrame(json.loads((RAW / "tmacs_mileage_2021_2024.json").read_text()))
    m = m[(m["item"] == "annual_kkm") & (m["FUEL_CLS_NM"] == "계") & (m["CAR_USE_NM"] == "전체")]
    m["year"] = m["YEAR"].astype(int)
    rows = []
    for year in sorted(m["year"].unique()):
        for t, tm in TYPES.items():
            mm = m[(m["year"] == year) & (m["CAR_CLS_NM"] == tm)].iloc[0]
            ee = e[e["year"] == year]
            for region, kt, kkm in [("서울", ee.loc[ee["region"] == "서울", t].sum(), mm["SEOUL"]),
                                    ("전국(17개 시도 합)", ee[t].sum(), mm["ALL"])]:
                rows.append({"year": year, "region": region, "차종": t, "배출량_천tCO2eq": kt,
                             "주행거리_백만km": kkm / 1e3, "g_per_km": kt * 1e9 / (kkm * 1e3)})
    f = pd.DataFrame(rows)

    g = pd.read_csv(RAW / "한국교통안전공단_지역별온실가스별도로부문온실가스배출량_20241231.csv", encoding="cp949")
    g.columns = ["year", "region", "CO2", "CH4", "N2O"]
    g["region"] = g["region"].str.replace(" ", "")
    s = g[g["region"] == "서울"].set_index("year")
    f["서울_CO2비중"] = f["year"].map(s["CO2"] / s[["CO2", "CH4", "N2O"]].sum(axis=1))
    return f


def read_day(z: zipfile.ZipFile, cols: list[str]) -> pd.DataFrame:
    return pd.read_csv(z.open(z.namelist()[0]), usecols=cols)


def car_distance(centroids: gpd.GeoSeries) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """도착 동별 차량 이동 건수와 평균 직선거리 (전체 / 서울 출발), 날짜별 서울 합계, 재차인원 점검."""
    per_dong, per_day, capital = [], [], []
    for day in DAYS:
        d = read_day(zipfile.ZipFile(RAW / f"seoul_trans_admdong3_final_{day}.zip"),
                     ["o_admdong_cd", "d_admdong_cd", "in_forn_div_nm", "move_trans", "move_dist", "cnt"])
        d = d[(d["in_forn_div_nm"] == "내국인") & (d["move_trans"] == CAR)]
        o2, d2 = d["o_admdong_cd"].astype(str).str[:2], d["d_admdong_cd"].astype(str).str[:2]
        d = d.assign(pkm=d["cnt"] * d["move_dist"] / 1000, o_seoul=o2 == "11", d_seoul=d2 == "11",
                     o_cap=o2.isin(CAPITAL), d_cap=d2.isin(CAPITAL))
        capital.append({"date": day, "수도권_안_차량_인km": d.loc[d["o_cap"] & d["d_cap"], "pkm"].sum(),
                        "서울_관련_차량_인km": d.loc[d["o_seoul"] | d["d_seoul"], "pkm"].sum()})
        s = d[d["d_seoul"]]
        g_all = s.groupby("d_admdong_cd")[["cnt", "pkm"]].sum()
        g_seo = s[s["o_seoul"]].groupby("d_admdong_cd")[["cnt", "pkm"]].sum()
        per_dong.append(g_all.join(g_seo, rsuffix="_seoul", how="left").fillna(0))
        per_day.append({"date": day, "서울도착_차량이동": s["cnt"].sum(), "평균거리_km": s["pkm"].sum() / s["cnt"].sum(),
                        "서울출발_비중": s.loc[s["o_seoul"], "cnt"].sum() / s["cnt"].sum(),
                        "서울출발_평균거리_km": s.loc[s["o_seoul"], "pkm"].sum() / s.loc[s["o_seoul"], "cnt"].sum()})
        if day == DAYS[0]:  # move_dist 가 직선거리인지: 서로 다른 동 사이 이동을 동 중심점 거리와 비교
            x = s[s["o_seoul"] & (s["o_admdong_cd"] != s["d_admdong_cd"])]
            x = x[x["o_admdong_cd"].isin(centroids.index) & x["d_admdong_cd"].isin(centroids.index)]
            cd = centroids.loc[x["o_admdong_cd"]].distance(centroids.loc[x["d_admdong_cd"]], align=False).values
            ratio = np.average(x["move_dist"], weights=x["cnt"]) / np.average(cd, weights=x["cnt"])
            print(f"  move_dist ÷ 동 중심점 직선거리 (서울 안 동 사이 차량 이동, 가중평균) = {ratio:.3f}")
        print(f"  {day} 완료", flush=True)
    g = pd.concat(per_dong).groupby(level=0).sum() / len(DAYS)
    out = pd.DataFrame({
        "car_trips_od": g["cnt"], "car_km_mean": g["pkm"] / g["cnt"],
        "car_trips_seoul_origin": g["cnt_seoul"], "car_km_mean_seoul_origin": g["pkm_seoul"] / g["cnt_seoul"]})
    out.index = out.index.astype(str).str.ljust(10, "0")
    out.index.name = "adm_cd10"
    return out, pd.DataFrame(per_day), pd.DataFrame(capital)


def elderly_ratio() -> pd.DataFrame:
    """도착 동별 60세 이상 평균 이동거리 ÷ 전체 평균 이동거리 (모든 수단)."""
    parts = []
    for day in DAYS:
        d = read_day(zipfile.ZipFile(RAW / f"seoul_trans_admdong4_in_{day}.zip"),
                     ["d_admdong_cd", "move_dist", "60plus_cnt", "total_cnt"])
        d = d[d["d_admdong_cd"].astype(str).str.startswith("11")]
        d = d.assign(km60=d["60plus_cnt"] * d["move_dist"] / 1000, kmall=d["total_cnt"] * d["move_dist"] / 1000)
        parts.append(d.groupby("d_admdong_cd")[["60plus_cnt", "total_cnt", "km60", "kmall"]].sum())
    g = pd.concat(parts).groupby(level=0).sum()
    out = pd.DataFrame({"dist_ratio_60p": (g["km60"] / g["60plus_cnt"]) / (g["kmall"] / g["total_cnt"])})
    out.index = out.index.astype(str).str.ljust(10, "0")
    out.index.name = "adm_cd10"
    city = (g["km60"].sum() / g["60plus_cnt"].sum()) / (g["kmall"].sum() / g["total_cnt"].sum())
    print(f"  60세 이상 평균 이동거리 ÷ 전체 (서울 도착, 모든 수단) = {city:.3f}")
    return out, city


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    f = emission_factors()
    f.to_csv(TAB / "e1_emission_factors.csv", index=False, encoding="utf-8-sig")
    print(f.round(3).to_string())

    dongs = gpd.read_file(PROC / "dong_mci.gpkg")[["adm_cd10", "geometry"]]
    cen = dongs.set_index(dongs["adm_cd10"].str[:8].astype(int)).geometry.centroid
    dist, per_day, capital = car_distance(cen)
    ratio, city_ratio = elderly_ratio()
    out = dist.join(ratio, how="left")
    missing = set(dongs["adm_cd10"]) - set(out.index)
    if missing:
        raise SystemExit(f"출도착 자료에 없는 행정동: {sorted(missing)}")
    out.to_parquet(PROC / "car_dist_dong.parquet")
    per_day.to_csv(TAB / "e2_car_distance_days.csv", index=False, encoding="utf-8-sig")
    print(per_day.round(3).to_string())

    # 재차인원 점검: KT 수도권 안 차량 인·km (평일 하루) vs TMACS 서울·인천·경기 승용차 차·km (2024년 하루 평균)
    m = pd.DataFrame(json.loads((RAW / "tmacs_mileage_2021_2024.json").read_text()))
    m = m[(m["item"] == "annual_kkm") & (m["FUEL_CLS_NM"] == "계") & (m["CAR_CLS_NM"] == "승용차") & (m["YEAR"] == "2024")]
    vkm = m[["SEOUL", "INCHEON", "GYEONGGI"]].iloc[0].sum() * 1e3 / 365
    pkm = capital["수도권_안_차량_인km"].mean()
    chk = pd.DataFrame([{"KT_수도권안_차량_인km_평일하루(직선)": pkm, "TMACS_수도권_승용_차km_하루": vkm,
                         "인km÷차km (직선거리)": pkm / vkm, "인km÷차km (우회계수 1.3)": pkm * 1.3 / vkm,
                         "60세이상_거리비(서울도착, 모든 수단)": city_ratio}])
    chk.to_csv(TAB / "e3_occupancy_check.csv", index=False, encoding="utf-8-sig")
    print(chk.T.round(3).to_string())
    print(f"저장: {PROC / 'car_dist_dong.parquet'} ({len(out)}개 동)")


if __name__ == "__main__":
    main()
