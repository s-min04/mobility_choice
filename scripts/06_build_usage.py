"""A 분석 입력: 행정동별 고령자의 실제 이동수단 구성과 경사를 만든다.

실행: python scripts/06_build_usage.py   (02_build_index.py 다음)

1. 수도권 생활이동(도착 행정동 × 시간대 × 성·연령 × 수단, 2026년 9월)
   - 평일만 쓰고 추석 연휴(9.24~25)는 뺀다 → 20일
   - 수단 코드: 명세서를 찾지 못해 데이터로 판별했다.
       6 = 지하철 (01~04시 이동이 0, 출근 시간에 가장 많음), 4·5 = 버스(광역·일반, 심야에도 이동 있음),
       7 = 도보, 8 = 차량, 1·2·3·9 = 항공·기차·고속버스·기타
   - 연령: 70대 이상(70_cnt) = 고령자, 20~50대 = 청장년
   - 대중교통 분담률 = (지하철+버스) ÷ (지하철+버스+차량). 도보와 기타는 뺀다.
   - 도착 기준이므로 그 동으로 들어오는 모든 이동이다 (귀가 이동 + 방문 이동).
2. 외출 횟수: 같은 생활이동의 목적별 자료에서 '귀가'(코드 3: 밤 8~10시에 가장 많음, 데이터로 판별) 이동을 세어
   거주 인구로 나눈다 = 1인당 하루 외출 횟수 (70대 이상, 20~50대)
3. 신뢰도 확인용: 평일을 홀수일/짝수일로 나눠 대중교통 분담률을 따로 구한다
4. 소득: (가난한 쪽) 기초생계급여 수급자 비율 — 2024.5 동별·연령별 자료를 이름으로 맞춘다
         (부유한 쪽) 아파트 평균 시가 — 서울시 상권분석서비스 최신 분기
5. 경사: Copernicus DEM(30m, 건물을 포함한 표면 고도)을 150m 중앙값으로 다듬어 건물 경계를 줄인 뒤 경사(%)를 구하고,
   B의 거주 가능 격자점에서 뽑아 행정동 평균을 낸다.
"""

import zipfile
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

HOLIDAYS = {"20260924", "20260925"}  # 추석 연휴 평일
MODES = {"subway": [6], "bus": [4, 5], "walk": [7], "car": [8], "other": [1, 2, 3, 9]}
AGES = {"70p": ["male_70_cnt", "feml_70_cnt"],
        "adult": [f"{s}_{a}_cnt" for s in ["male", "feml"] for a in ["20", "30", "40", "50"]]}
CITY_HALL = (126.9780, 37.5665)


def weekdays(z: zipfile.ZipFile) -> list[str]:
    names = []
    for n in sorted(z.namelist()):
        day = n[-12:-4]
        if datetime.strptime(day, "%Y%m%d").weekday() < 5 and day not in HOLIDAYS:
            names.append(n)
    return names


def daily_mean(z: zipfile.ZipFile, names: list[str], key: str, cols: list[str], where=None) -> pd.DataFrame:
    parts = []
    for n in names:
        d = pd.read_csv(z.open(n), usecols=["d_admdong_cd", key, *cols])
        d = d[d["d_admdong_cd"].astype(str).str.startswith("11")]
        if where is not None:
            d = d[where(d)]
        parts.append(d.groupby(["d_admdong_cd", key]).sum())
    return pd.concat(parts).groupby(level=[0, 1]).sum() / len(names)


def mode_shares(t: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=t.index.get_level_values(0).unique())
    for age, cols in AGES.items():
        per_mode = t[cols].sum(axis=1).unstack(fill_value=0)
        for mode, codes in MODES.items():
            out[f"trips_{age}_{mode}"] = per_mode[[c for c in codes if c in per_mode.columns]].sum(axis=1)
        motor = out[[f"trips_{age}_{m}" for m in ["subway", "bus", "car"]]].sum(axis=1)
        out[f"pt_share_{age}"] = (out[f"trips_{age}_subway"] + out[f"trips_{age}_bus"]) / motor
        out[f"car_share_{age}"] = out[f"trips_{age}_car"] / motor
        out[f"motor_trips_{age}"] = motor
    out.index = out.index.astype(str).str.ljust(10, "0")  # 8자리 → 행정기관코드 10자리
    out.index.name = "adm_cd10"
    return out


def load_trips() -> pd.DataFrame:
    z = zipfile.ZipFile(RAW / "seoul_trans_admdong1_in_202609.zip")
    names = weekdays(z)
    cols = [*AGES["70p"], *AGES["adult"]]
    print(f"평일 {len(names)}일 사용: {names[0][-12:-4]} ~ {names[-1][-12:-4]}")
    out = mode_shares(daily_mean(z, names, "move_trans", cols))
    for half, sub in [("odd", names[0::2]), ("even", names[1::2])]:
        h = mode_shares(daily_mean(z, sub, "move_trans", cols))
        out[f"pt_share_70p_{half}"] = h["pt_share_70p"]
        out[f"pt_share_adult_{half}"] = h["pt_share_adult"]
    return out


def load_outings(pop: pd.DataFrame) -> pd.DataFrame:
    """귀가 이동 ÷ 거주 인구 = 1인당 하루 외출 횟수."""
    z = zipfile.ZipFile(RAW / "seoul_purpose_admdong1_in_202609.zip")
    cols = [*AGES["70p"], *AGES["adult"]]
    t = daily_mean(z, weekdays(z), "move_purpose", cols, where=lambda d: d["move_purpose"] == 3).droplevel(1)
    t.index = t.index.astype(str).str.ljust(10, "0")
    out = pd.DataFrame({"home_trips_70p": t[AGES["70p"]].sum(axis=1), "home_trips_adult": t[AGES["adult"]].sum(axis=1)})
    out = out.join(pop)
    out["out_rate_70p"] = out["home_trips_70p"] / out["pop_70_plus"]
    out["out_rate_adult"] = out["home_trips_adult"] / out["pop_20_59"]
    return out[["home_trips_70p", "home_trips_adult", "out_rate_70p", "out_rate_adult"]]


def norm_dong(name: str) -> str:
    """수급자 자료와 경계 자료의 동 이름 표기 차이를 맞춘다 (상계3.4동 → 상계3·4동, 자양제4동 → 자양4동)."""
    import re
    name = re.sub(r"[.,\-]", "·", str(name).strip())
    return re.sub(r"제(?=\d)", "", name)


def load_income(dongs: gpd.GeoDataFrame) -> pd.DataFrame:
    w = pd.read_excel(RAW / "서울시 국민기초생활 수급자 동별 현황(202405).xlsx", header=None, skiprows=3)
    w = w.iloc[:, [0, 1, 2, 4]].set_axis(["name", "kind", "age", "n"], axis=1)
    w[["name", "kind"]] = w[["name", "kind"]].ffill()
    gus = set(dongs["gu"])
    w["gu"] = w["name"].where(w["name"].isin(gus)).ffill()
    w = w[~w["name"].isin(gus) & (w["kind"] == "기초생계급여")]
    w = w.pivot_table(index=["gu", "name"], columns="age", values="n", aggfunc="sum").fillna(0)
    w = w.rename(columns={"65세이상": "welfare_65p", "18~64세": "welfare_18_64", "18세미만": "welfare_u18"}).reset_index()
    w["key"] = w["name"].map(norm_dong)
    # 동대문구 용신동은 2024년 이후 신설동·용두동으로 나뉘었다 → 두 동에 같은 수급률을 쓰도록 합쳐서 나눈다
    split = {("동대문구", "용신동"): ["신설동", "용두동"]}
    rows = []
    for (gu, old), news in split.items():
        r = w[(w["gu"] == gu) & (w["key"] == old)]
        for new in news:
            rows.append(r.assign(key=new, split_from=old))
    w = pd.concat([w, *rows])
    d = dongs[["adm_cd10", "gu", "dong", "pop_total", "pop_65_plus"]].assign(key=dongs["dong"].map(norm_dong))
    d = d.merge(w.drop(columns="name"), on=["gu", "key"], how="left")
    sp = d["split_from"].notna() if "split_from" in d else pd.Series(False, index=d.index)
    for col, pop in [("welfare_65p", "pop_65_plus"), ("welfare_all", "pop_total")]:
        num = d[["welfare_65p"]].sum(axis=1) if col == "welfare_65p" else d[["welfare_65p", "welfare_18_64", "welfare_u18"]].sum(axis=1, min_count=1)
        den = d[pop].where(~sp, d.groupby("split_from")[pop].transform("sum"))
        d[col + "_rate"] = num / den
    missing = d.loc[d["welfare_65p"].isna(), ["gu", "dong"]].values.tolist()
    for col in ["welfare_65p_rate", "welfare_all_rate"]:
        d[col] = d[col].fillna(d.groupby("gu")[col].transform("median"))
    print(f"  수급자 자료 매칭: {len(d) - len(missing)}/{len(d)}개 동 (구 중앙값으로 채움: {missing})")

    a = pd.read_csv(RAW / "서울시 상권분석서비스(아파트-행정동).csv", encoding="cp949", dtype={"행정동_코드": str})
    a = a[a["기준_년분기_코드"] == a["기준_년분기_코드"].max()]
    a = a.assign(adm_cd10=a["행정동_코드"].str.ljust(10, "0")).set_index("adm_cd10")["아파트_평균_시가"]
    d["apt_price_eok"] = d["adm_cd10"].map(a) / 1e8
    nmiss = d["apt_price_eok"].isna().sum()
    d["apt_price_eok"] = d["apt_price_eok"].fillna(d.groupby("gu")["apt_price_eok"].transform("median"))
    d["log_apt_price"] = np.log(d["apt_price_eok"])
    print(f"  아파트 시가: 최신 분기 {a.index.size}개 동, 없는 {nmiss}개 동은 구 중앙값으로 채움")
    return d.set_index("adm_cd10")[["welfare_65p_rate", "welfare_all_rate", "apt_price_eok", "log_apt_price"]]


def age_structure() -> pd.DataFrame:
    p = pd.read_csv(RAW / "지역별(행정동) 성별 연령별 주민등록 인구수_20260831.csv", encoding="cp949",
                    dtype={"행정기관코드": str})
    p = p[p["시도명"] == "서울특별시"].set_index("행정기관코드")
    cols = [c for c in p.columns if c.endswith(("남자", "여자")) and c[0].isdigit()]
    age = pd.Series({c: int("".join(ch for ch in c.split("세")[0] if ch.isdigit())) for c in cols})
    s = lambda lo, hi=200: p[age[(age >= lo) & (age <= hi)].index].sum(axis=1)
    out = pd.DataFrame({"pop_20_59": s(20, 59), "pop_70_plus": s(70), "share80_in70": s(80) / s(70)})
    out.index.name = "adm_cd10"
    return out


def slope_by_dong(dongs: gpd.GeoDataFrame) -> pd.DataFrame:
    srcs = [rasterio.open(RAW / f"Copernicus_DSM_COG_10_{t}_00_DEM.tif") for t in ["N37_00_E126", "N37_00_E127"]]
    xmin, ymin, xmax, ymax = dongs.to_crs("EPSG:4326").total_bounds
    dem, tr = merge(srcs, bounds=(xmin - 0.01, ymin - 0.01, xmax + 0.01, ymax + 0.01))
    dem = median_filter(dem[0].astype("float32"), size=5)  # 약 150m: 건물 높이로 생기는 가짜 경사를 줄인다
    lat = (ymin + ymax) / 2
    dy = abs(tr.e) * 111_320
    dx = tr.a * 111_320 * np.cos(np.radians(lat))
    gy, gx = np.gradient(dem, dy, dx)
    slope = np.hypot(gx, gy) * 100  # %

    grid = pd.read_parquet(PROC / "grid_mci.parquet")
    grid = grid[grid["habitable"]]
    pts = gpd.GeoSeries(gpd.points_from_xy(grid["x"], grid["y"]), crs="EPSG:5179").to_crs("EPSG:4326")
    rows, cols = rasterio.transform.rowcol(tr, pts.x.values, pts.y.values)
    grid = grid.assign(slope=slope[np.array(rows), np.array(cols)], elev=dem[np.array(rows), np.array(cols)])
    grid[["adm_cd10", "x", "y", "slope", "elev"]].to_parquet(PROC / "grid_slope.parquet")  # 08 역 단위 검증에서 쓴다
    g = grid.groupby("adm_cd10")
    return pd.DataFrame({"slope_mean": g["slope"].mean(), "slope_p75": g["slope"].quantile(0.75),
                         "steep_share": g["slope"].apply(lambda s: (s >= 8).mean()),  # 8% 이상 = 휠체어 경사로 기준 초과
                         "elev_std": g["elev"].std()})


def main():
    dongs = gpd.read_file(PROC / "dong_mci.gpkg")
    trips = load_trips()
    ages = age_structure()
    outings = load_outings(ages[["pop_20_59", "pop_70_plus"]])
    missing = set(dongs["adm_cd10"]) - set(trips.index)
    if missing:
        raise SystemExit(f"생활이동 자료에 없는 행정동: {sorted(missing)}")
    slope = slope_by_dong(dongs)
    income = load_income(dongs)

    hall = gpd.GeoSeries(gpd.points_from_xy([CITY_HALL[0]], [CITY_HALL[1]]), crs="EPSG:4326").to_crs(dongs.crs)[0]
    d = (dongs.merge(trips, left_on="adm_cd10", right_index=True)
         .merge(slope, left_on="adm_cd10", right_index=True)
         .merge(ages[["pop_20_59", "share80_in70"]], left_on="adm_cd10", right_index=True)
         .merge(outings, left_on="adm_cd10", right_index=True)
         .merge(income, left_on="adm_cd10", right_index=True))
    hab_km2 = d["n_points"] * 0.01
    d["pop_density"] = d["pop_total"] / hab_km2  # 거주 가능 면적 기준 (명/km²)
    d["dist_cityhall_km"] = d.centroid.distance(hall) / 1000
    d["motor_trips_70p_per_100"] = d["motor_trips_70p"] / d["pop_70_plus"] * 100
    d.to_file(PROC / "dong_usage.gpkg", driver="GPKG")

    show = ["pt_share_70p", "pt_share_adult", "car_share_70p", "motor_trips_70p", "out_rate_70p", "out_rate_adult",
            "slope_mean", "steep_share", "welfare_65p_rate", "welfare_all_rate", "apt_price_eok"]
    print(d[show].describe().round(3).to_string())
    w = d["motor_trips_70p"]
    print(f"서울 전체 대중교통 분담률: 70대 이상 {np.average(d['pt_share_70p'], weights=w):.3f}, "
          f"20~50대 {np.average(d['pt_share_adult'], weights=d['motor_trips_adult']):.3f}")
    print(f"저장: {PROC / 'dong_usage.gpkg'} ({len(d)}개 동)")


if __name__ == "__main__":
    main()
