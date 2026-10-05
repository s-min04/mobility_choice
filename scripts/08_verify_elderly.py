"""A 검증: B 의 "고령자만 특별히 불리하다"를 휴대전화 추정치가 아닌 다른 데이터로 다시 검정한다.

실행: python scripts/08_verify_elderly.py   (06_build_usage.py, 07_model_usage.py 다음)

검정 1. 걷는 거리·경사가 고령자에게만 더 불리한가 (교통카드 실측, 역 단위)
   - 지하철 역별 유임/무임 승차(2025.9~2026.8, 12개월 합계). 무임 = 65세 이상(장애인·국가유공자 포함).
   - 거주 가능 격자점을 가장 가까운 역에 배정하고(1.5km 이내), 동 인구를 격자점에 고르게 나눠 역세권 인구를 만든다.
   - 고령자 상대 이용률 = (무임 승차 ÷ 유임 승차) ÷ (역세권 65세 이상 ÷ 65세 미만), 로그.
     같은 역의 청장년과 비교하므로 역의 크기·목적지 매력은 상당 부분 상쇄된다.
   - 고령자 보행 부담이 실제로 크다면, 역세권 고령자가 멀리(369m 밖) 또는 언덕(경사 8% 이상)에 사는 역일수록
     고령자 상대 이용률이 낮아야 한다.
   - 여러 표본(도심 제외, 큰 역 제외 등)에서 같은 결론이 나오는지 본다.
   - 참고: 같은 역을 계절별로 비교하는 패널(역 고정효과)도 돌렸지만, 언덕 역세권이 등산로 입구와 겹쳐
     등산철 수요와 섞이므로 해석하지 않는다 (표만 남긴다).
검정 2. '고령자만 덜 탐' 동은 차를 가질 여유 때문인가 (자동차 등록)
   - 행정동별 자가용 승용차 등록 대수(인구 1,000명당). 상위 25% 이상이면 '차를 가질 여유가 있는 동'으로 본다.
"""

import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"

for font in ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
    if any(font == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = font
        break
plt.rcParams["axes.unicode_minus"] = False

R_ELDERLY = 369   # 고령자 10분 보행권 (B 와 같은 값)
R_ADULT = 554     # 일반 성인 10분 보행권
R_MAX = 1500      # 이보다 멀면 역세권으로 보지 않는다
MONTHS = (202509, 202608)
CITY_HALL = (126.9780, 37.5665)
station_key = lambda s: re.sub(r"\(.*?\)", "", str(s)).strip()


# ---------------------------------------------------------------- 검정 1

def station_table() -> pd.DataFrame:
    st = pd.read_csv(RAW / "서울시 역사마스터 정보.csv", encoding="cp949")
    st["key"] = st["역사명"].map(station_key)
    st = st.groupby("key").agg(lat=("위도", "mean"), lon=("경도", "mean"), n_lines=("호선", "nunique")).reset_index()
    p = gpd.GeoSeries(gpd.points_from_xy(st["lon"], st["lat"]), crs="EPSG:4326").to_crs("EPSG:5179")
    st["x"], st["y"] = p.x, p.y

    f = pd.read_csv(RAW / "서울시 지하철 호선별 역별 유무임 승하차 인원 정보.csv", encoding="cp949")
    f = f[f["사용월"].between(*MONTHS)].assign(key=lambda d: d["지하철역"].map(station_key))
    use = f.groupby("key")[["유임승차인원", "무임승차인원"]].sum().set_axis(["paid", "free"], axis=1)

    dongs = gpd.read_file(PROC / "dong_mci.gpkg").set_index("adm_cd10")
    grid = pd.read_parquet(PROC / "grid_slope.parquet")
    n = grid.groupby("adm_cd10").size()
    grid["E"] = grid["adm_cd10"].map(dongs["pop_65_plus"] / n)
    grid["A"] = grid["adm_cd10"].map((dongs["pop_total"] - dongs["pop_65_plus"]) / n)
    d, i = cKDTree(st[["x", "y"]].to_numpy()).query(grid[["x", "y"]].to_numpy())  # 서울 밖 역까지 포함해 가장 가까운 역
    grid = grid.assign(st=st["key"].to_numpy()[i], d=d)
    grid = grid[grid["d"] <= R_MAX]
    grid["mid"] = (grid["d"] > R_ELDERLY) & (grid["d"] <= R_ADULT)
    grid["far"] = grid["d"] > R_ADULT
    grid["steep"] = grid["slope"] >= 8

    def summarise(x):
        e = x["E"].sum()
        return pd.Series({"E": e, "A": x["A"].sum(), "mid_E": (x["E"] * x["mid"]).sum() / e,
                          "far_E": (x["E"] * x["far"]).sum() / e, "steep_E": (x["E"] * x["steep"]).sum() / e})

    agg = grid.groupby("st").apply(summarise)
    seoul = dongs.union_all()
    s = st.set_index("key").join(agg, how="inner").join(use, how="inner")
    s = s[gpd.GeoSeries(gpd.points_from_xy(s["x"], s["y"]), crs="EPSG:5179").within(seoul).to_numpy()]
    s = s[(s["E"] > 0) & (s["free"] > 0) & (s["paid"] > 0)]
    hall = gpd.GeoSeries(gpd.points_from_xy([CITY_HALL[0]], [CITY_HALL[1]]), crs="EPSG:4326").to_crs("EPSG:5179")[0]
    s["dist_cbd"] = np.hypot(s["x"] - hall.x, s["y"] - hall.y) / 1000
    s["logvol"] = np.log(s["free"] + s["paid"])
    s["rel_use"] = np.log((s["free"] / s["paid"]) / (s["E"] / s["A"]))
    return s


def station_tests(s: pd.DataFrame) -> pd.DataFrame:
    f = "rel_use ~ mid_E + far_E + steep_E + dist_cbd + logvol + n_lines"
    q75, q50 = s["logvol"].quantile(0.75), s["logvol"].quantile(0.5)
    samples = [("전체 역", s, None),
               ("도심 3km 밖", s[s["dist_cbd"] > 3], None),
               ("이용량 하위 75%", s[s["logvol"] < q75], None),
               ("주거지 역 (이용량 하위 50%·도심 밖)", s[(s["logvol"] < q50) & (s["dist_cbd"] > 3)], None),
               ("단일 노선역", s[s["n_lines"] == 1], None),
               ("고령인구 가중", s, s["E"])]
    rows = []
    for name, df, w in samples:
        ff = f if df["n_lines"].nunique() > 1 else f.replace(" + n_lines", "")  # 단일 노선역은 노선 수가 상수
        m = (smf.wls(ff, df, weights=w) if w is not None else smf.ols(ff, df)).fit(cov_type="HC1")
        ci = m.conf_int()
        for var, label in [("mid_E", "369~554m에 사는 고령자 비율"), ("far_E", "554m 밖에 사는 고령자 비율"),
                           ("steep_E", "경사 8% 이상에 사는 고령자 비율")]:
            per10 = lambda b: (np.exp(b * 0.1) - 1) * 100  # 비율 10%p 증가당 고령자 상대 이용률 변화(%)
            rows.append({"표본": name, "역 수": len(df), "변수": label, "10%p당 변화(%)": per10(m.params[var]),
                         "95% 하한(%)": per10(ci.loc[var, 0]), "95% 상한(%)": per10(ci.loc[var, 1]),
                         "p": m.pvalues[var]})
    return pd.DataFrame(rows)


def seasonal_panel(s: pd.DataFrame) -> pd.DataFrame:
    """참고용. 같은 역을 36개월 비교: 겨울(12~2월)·여름(7~8월)에 언덕 역세권의 고령자 비중이 달라지는가."""
    f = pd.read_csv(RAW / "서울시 지하철 호선별 역별 유무임 승하차 인원 정보.csv", encoding="cp949")
    f = f[f["사용월"].between(202309, 202608)].assign(key=lambda d: d["지하철역"].map(station_key))
    p = f.groupby(["key", "사용월"])[["유임승차인원", "무임승차인원"]].sum().reset_index()
    p = p[p["key"].isin(s.index) & (p["무임승차인원"] > 0) & (p["유임승차인원"] > 0)]
    p["ly"] = np.log(p["무임승차인원"] / p["유임승차인원"])
    p["winter"] = (p["사용월"] % 100).isin([12, 1, 2]).astype(int)
    p["summer"] = (p["사용월"] % 100).isin([7, 8]).astype(int)
    p = p.join(s[["steep_E"]], on="key")
    m = smf.ols("ly ~ C(key) + C(사용월) + winter:steep_E + summer:steep_E", p).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(p["key"])[0]})
    return pd.DataFrame({k: {"계수": m.params[k], "p": m.pvalues[k]} for k in ["winter:steep_E", "summer:steep_E"]}).T


# ---------------------------------------------------------------- 검정 2

def car_ownership() -> pd.Series:
    c = pd.read_csv(RAW / "서울시 자치구 읍면동별 연료별 자동차 등록현황(행정동)(26년8월).csv", encoding="cp949",
                    dtype={"동_코드": str})
    c = c[c["읍면동(행정동)"].str.strip() != "기타"]
    # 승용·승합·화물·특수가 세 번 반복된다: 관용, 자가용, 영업용 순 (자가용이 전체의 대부분인 두 번째 묶음)
    return c.groupby("동_코드")["승용.1"].sum().rename("private_cars")


def affluence_check(cars: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = gpd.read_file(PROC / "dong_usage.gpkg").set_index("adm_cd10")
    r = pd.read_csv(TAB / "a2_expected_vs_actual.csv", dtype={"adm_cd10": str}).set_index("adm_cd10")
    d = d.join(cars).join(r[["resid_70p", "resid_elderly_only", "type"]])
    d["cars_per_1000"] = d["private_cars"] / d["pop_total"] * 1000
    q75 = d["cars_per_1000"].quantile(0.75)
    z = lambda v: (v - v.mean()) / v.std()
    v = d.assign(cars_z=z(np.log(d["cars_per_1000"])), apt_z=z(d["log_apt_price"]), wel_z=z(d["welfare_65p_rate"]))
    reg = {}
    for y in ["resid_elderly_only", "car_share_70p"]:
        m = smf.ols(f"{y} ~ cars_z + apt_z + wel_z", v).fit(cov_type="HC1")
        reg[y] = {**{f"{k} 계수": m.params[k] for k in ["cars_z", "apt_z", "wel_z"]},
                  **{f"{k} p": m.pvalues[k] for k in ["cars_z", "apt_z", "wel_z"]}, "R2": m.rsquared}
    only = d[d["type"] == "고령자만 덜 탐"].copy()
    only["차 보유 상위 25%"] = only["cars_per_1000"] >= q75
    only = only.sort_values("resid_elderly_only")[["gu", "dong", "resid_70p", "resid_elderly_only", "cars_per_1000",
                                                   "apt_price_eok", "welfare_65p_rate", "차 보유 상위 25%"]]
    return only, pd.DataFrame(reg).T.assign(seoul_q75_cars=q75, seoul_median_cars=d["cars_per_1000"].median())


# ---------------------------------------------------------------- 그림

def fig_station(t: pd.DataFrame, path: Path):
    vars_ = t["변수"].unique()
    fig, axes = plt.subplots(1, len(vars_), figsize=(14, 4.2), sharey=True)
    samples = t["표본"].unique()
    for ax, var in zip(axes, vars_):
        sub = t[t["변수"] == var].set_index("표본").loc[samples]
        y = np.arange(len(sub))[::-1]
        ax.errorbar(sub["10%p당 변화(%)"], y, xerr=[sub["10%p당 변화(%)"] - sub["95% 하한(%)"],
                                                   sub["95% 상한(%)"] - sub["10%p당 변화(%)"]],
                    fmt="o", color="#2b6cb0", capsize=3)
        ax.axvline(0, color="#a0aec0", lw=1)
        ax.set_yticks(y, samples)
        ax.set_title(var, fontsize=10)
        ax.set_xlabel("10%p 늘 때 고령자 상대 이용률 변화(%)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("교통카드 실측(304개 역): 멀리·언덕에 사는 고령자가 많은 역일수록 고령자가 덜 타는가? (점: 추정치, 선: 95% 신뢰구간)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    s = station_table()
    print(f"서울 안 역 {len(s)}개, 고령자 상대 이용률(로그) 평균 {s['rel_use'].mean():.3f}")
    t = station_tests(s)
    print(t.round(3).to_string(index=False))
    panel = seasonal_panel(s)
    print("참고(해석 보류) 계절 패널:\n", panel.round(4).to_string())

    only, reg = affluence_check(car_ownership())
    print(reg.round(4).to_string())
    print(only.round(3).to_string(index=False))

    s.to_csv(TAB / "a7_station_catchments.csv", encoding="utf-8-sig")
    t.to_csv(TAB / "a7_station_tests.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(TAB / "a7_station_seasonal_panel_reference.csv", encoding="utf-8-sig")
    only.to_csv(TAB / "a8_elderly_only_affluence.csv", index=False, encoding="utf-8-sig")
    reg.to_csv(TAB / "a8_affluence_regression.csv", encoding="utf-8-sig")
    fig_station(t, FIG / "a5_station_test.png")
    print("그림·표 저장 완료")


if __name__ == "__main__":
    main()
