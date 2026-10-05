"""D 분석: 폭염·한파·큰비가 고령자의 지하철 이용을 청장년보다 더 줄이는가.

실행: python scripts/11_weather_model.py   (10_build_weather.py, 09_cluster.py 다음)

1. 서울 전체 일별 회귀 (2024.7.1 ~ 2026.3.19, 627일)
   log(승차) = 날씨 구간 + 요일×공휴일 + 연·월 고정효과,  Newey-West 표준오차(7일)
   고령자(우대권)와 일반을 따로 추정하고, 둘의 비(고령자 상대 이용)도 추정한다.
   날씨 구간: 폭염(최고 33℃ 이상), 무더위(30~33℃), 한파(최저 -10℃ 이하), 추위(-10~-5℃),
             큰비(10mm 이상), 비(1~10mm). 기준 = 나머지 날.
2. 시간대별: 폭염일 vs 30℃ 미만 여름 평일, 한파일 vs 최저 -5℃ 초과 겨울 평일의 시간대별 승차 변화
3. 지역 차이: 역 × 날짜 패널(역·날짜 고정효과, 역 군집 표준오차). 날씨 × C 유형, 날씨 × 역세권 급경사 비율
4. 쉼터: 기후동행쉼터까지 고령자 걸음 10분(369m) 안에 사는 65세 이상 비율 (C 유형별)
"""

import re
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
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

BINS = {"heat33": "폭염 (최고 33℃ 이상)", "hot30": "무더위 (최고 30~33℃)", "cold10": "한파 (최저 -10℃ 이하)",
        "cold5": "추위 (최저 -10~-5℃)", "rain10": "큰비 (10mm 이상)", "rain1": "비 (1~10mm)"}
HOURS = ["06시 전"] + [f"{h:02d}시" for h in range(6, 24)] + ["24시 후"]
TYPE_NAMES = {0: "대중교통 양호 서민 주거지", 1: "언덕 위 고령 주거지", 2: "고소득 활동 지역",
              3: "교통 사각 저소득 고령 밀집", 4: "외곽 차 중심 신주거지"}
pct = lambda b: (np.exp(b) - 1) * 100
station_key = lambda s: re.sub(r"\(.*?\)", "", str(s)).strip()


def weather_bins(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    c["prcp0"] = c["prcp"].fillna(0)
    c["prcp_na"] = c["prcp"].isna().astype(int)
    c["heat33"] = (c["tmax"] >= 33).astype(int)
    c["hot30"] = ((c["tmax"] >= 30) & (c["tmax"] < 33)).astype(int)
    c["cold10"] = (c["tmin"] <= -10).astype(int)
    c["cold5"] = ((c["tmin"] <= -5) & (c["tmin"] > -10)).astype(int)
    c["rain10"] = (c["prcp0"] >= 10).astype(int)
    c["rain1"] = ((c["prcp0"] >= 1) & (c["prcp0"] < 10)).astype(int)
    return c


# ---------------------------------------------------------------- 1. 서울 전체

def city_models(c: pd.DataFrame) -> pd.DataFrame:
    c = c.assign(ym=c["date"].dt.to_period("M").astype(str), l_elderly=np.log(c["elderly"]),
                 l_general=np.log(c["general"]))
    c["l_ratio"] = c["l_elderly"] - c["l_general"]
    rhs = " + ".join(BINS) + " + prcp_na + C(dow)*C(holiday) + C(ym)"
    rows = []
    for y, label in [("l_elderly", "고령자(우대권)"), ("l_general", "일반"), ("l_ratio", "고령자 상대 (고령자 ÷ 일반)")]:
        m = smf.ols(f"{y} ~ {rhs}", c).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
        ci = m.conf_int()
        for b, name in BINS.items():
            rows.append({"대상": label, "날씨": name, "일수": int(c[b].sum()), "변화(%)": pct(m.params[b]),
                         "95% 하한": pct(ci.loc[b, 0]), "95% 상한": pct(ci.loc[b, 1]), "p": m.pvalues[b]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. 시간대

def hourly_profiles(c: pd.DataFrame) -> pd.DataFrame:
    h = pd.read_parquet(PROC / "weather_city_hourly.parquet").merge(c[["date", "tmax", "tmin", "holiday", "dow"]],
                                                                     on="date")
    h = h[(h["dow"] < 5) & ~h["holiday"]]
    cols = [x for x in h.columns if re.fullmatch(r"h\d\d", x)]
    out = {}
    for name, months, shock, base in [("폭염", [7, 8], lambda x: x["tmax"] >= 33, lambda x: x["tmax"] < 30),
                                      ("한파", [12, 1, 2], lambda x: x["tmin"] <= -10, lambda x: x["tmin"] > -5)]:
        sub = h[h["date"].dt.month.isin(months)]
        for t, lab in [("elderly", "고령자"), ("general", "일반")]:
            x = sub[sub["type"] == t]
            out[(name, lab)] = (x[shock(x)][cols].mean() / x[base(x)][cols].mean() - 1) * 100
    t = pd.DataFrame(out)
    t.index = HOURS[:len(t)]
    return t


# ---------------------------------------------------------------- 3. 지역 차이

def station_panel(c: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_parquet(PROC / "weather_station_daily.parquet")
    st = pd.read_csv(RAW / "서울시 역사마스터 정보.csv", encoding="cp949")
    st["key"] = st["역사명"].map(station_key)
    st = st.groupby("key")[["위도", "경도"]].mean()
    g = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["경도"], st["위도"]), crs="EPSG:4326").to_crs("EPSG:5179")
    cl = gpd.read_file(PROC / "dong_clusters.gpkg")
    typ = gpd.sjoin(g, cl[["cluster", "geometry"]], predicate="within")["cluster"]
    catch = pd.read_csv(TAB / "a7_station_catchments.csv", index_col=0)

    d = d[d["station"].isin(typ.index) & (d["elderly"] > 0) & (d["general"] > 0)]
    d = d.merge(c[["date", "heat33", "cold10", "rain10"]], on="date")
    d["lr"] = np.log(d["elderly"] / d["general"])
    d["type"] = d["station"].map(typ)
    steep = (catch["steep_E"] - catch["steep_E"].mean()) / catch["steep_E"].std()
    d["steep_z"] = d["station"].map(steep)
    rows = []
    shocks = {"heat33": "폭염", "cold10": "한파", "rain10": "큰비"}
    # (a) C 유형별 (기준: 대중교통 양호 서민 주거지)
    cols = []
    for w in shocks:
        for t in [1, 2, 3, 4]:
            d[f"{w}_t{t}"] = d[w] * (d["type"] == t)
            cols.append(f"{w}_t{t}")
    p = d.set_index(["station", "date"])
    m = PanelOLS(p["lr"], p[cols].astype(float), entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True)
    for k in cols:
        w, t = k.split("_t")
        rows.append({"분석": "C 유형 (기준: 대중교통 양호)", "날씨": shocks[w], "비교": TYPE_NAMES[int(t)],
                     "고령자 상대 이용 차이(%)": pct(m.params[k]), "p": m.pvalues[k]})
    # (b) 역세권 급경사 비율 (1표준편차당)
    e = d[d["steep_z"].notna()].copy()
    cols = []
    for w in shocks:
        e[f"{w}_steep"] = e[w] * e["steep_z"]
        cols.append(f"{w}_steep")
    p = e.set_index(["station", "date"])
    m = PanelOLS(p["lr"], p[cols], entity_effects=True, time_effects=True).fit(cov_type="clustered",
                                                                             cluster_entity=True)
    for k in cols:
        rows.append({"분석": "역세권 급경사 비율 (1표준편차당)", "날씨": shocks[k.split("_")[0]], "비교": "급경사 +1SD",
                     "고령자 상대 이용 차이(%)": pct(m.params[k]), "p": m.pvalues[k]})
    out = pd.DataFrame(rows)
    out.attrs["n_stations"] = d["station"].nunique()
    out.attrs["stations_by_type"] = d.groupby("type")["station"].nunique().rename(TYPE_NAMES).to_dict()
    return out


# ---------------------------------------------------------------- 4. 쉼터

def shelter_coverage() -> pd.DataFrame:
    s = pd.read_csv(RAW / "서울시 기후동행쉼터.csv", encoding="cp949")
    p = gpd.GeoSeries(gpd.points_from_xy(s["X좌표(EPSG:5186)"], s["Y좌표(EPSG:5186)"]), crs="EPSG:5186").to_crs("EPSG:5179")
    grid = pd.read_parquet(PROC / "grid_slope.parquet")
    d, _ = cKDTree(np.c_[p.x, p.y]).query(grid[["x", "y"]].to_numpy())
    grid["shelter10"] = d <= 369
    dong = gpd.read_file(PROC / "dong_mci.gpkg").set_index("adm_cd10")
    cl = pd.read_csv(TAB / "c4_dong_clusters.csv", dtype={"adm_cd10": str}).set_index("adm_cd10")
    share = grid.groupby("adm_cd10")["shelter10"].mean()
    t = pd.DataFrame({"pop65": dong["pop_65_plus"], "share": share, "type": cl["type_name"]})
    t["covered"] = t["pop65"] * t["share"]
    out = t.groupby("type").agg(고령인구=("pop65", "sum"), 쉼터_10분내_고령인구=("covered", "sum"))
    out.loc["서울 전체"] = out.sum()
    out["쉼터 10분 안 고령자 비율"] = out["쉼터_10분내_고령인구"] / out["고령인구"]
    out.attrs["n_shelters"] = len(s)
    out.attrs["kinds"] = s["쉼터_구분"].value_counts().to_dict()
    return out


# ---------------------------------------------------------------- 그림

def fig_city(t: pd.DataFrame, path: Path):
    sub = t[t["대상"] != "고령자 상대 (고령자 ÷ 일반)"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = list(BINS.values())
    y = np.arange(len(names))
    for off, (who, col) in zip([0.18, -0.18], [("고령자(우대권)", "#c53030"), ("일반", "#718096")]):
        r = sub[sub["대상"] == who].set_index("날씨").loc[names]
        ax.barh(y + off, r["변화(%)"], 0.34, color=col, label=who,
                xerr=[r["변화(%)"] - r["95% 하한"], r["95% 상한"] - r["변화(%)"]], error_kw={"lw": 0.8})
        for yi, v, lo in zip(y + off, r["변화(%)"], r["95% 하한"]):
            ax.text(lo - 0.3, yi, f"{v:+.1f}%", va="center", ha="right", fontsize=8.5, color=col)
    ax.set_yticks(y, [f"{n} ({int(sub[sub['날씨'] == n]['일수'].iloc[0])}일)" for n in names])
    ax.invert_yaxis()
    ax.axvline(0, color="#4a5568", lw=0.8)
    ax.set_xlim(-22, 4)
    ax.set_xlabel("평소 대비 지하철 승차 변화 (%)  ※ 요일·공휴일·월 효과 통제, 선 = 95% 신뢰구간")
    ax.set_title("궂은 날씨에 고령자의 지하철 이용이 얼마나 줄어드는가 (서울교통공사 1~8호선, 2024.7~2026.3)", fontsize=11)
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_hourly(h: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=False)
    x = np.arange(1, len(h) - 1)  # 06시 전, 24시 후는 표본이 작아 뺀다
    for ax, shock in zip(axes, ["폭염", "한파"]):
        for lab, col in [("고령자", "#c53030"), ("일반", "#718096")]:
            ax.plot(x, h[(shock, lab)].iloc[1:-1], marker="o", ms=3, color=col, label=lab)
        ax.axhline(0, color="#4a5568", lw=0.8)
        ax.set_xticks(x[::2], h.index[1:-1][::2])
        ax.set_ylabel("평소 대비 승차 변화 (%)")
        base = "30℃ 미만 여름 평일" if shock == "폭염" else "최저 -5℃ 초과 겨울 평일"
        ax.set_title(f"{shock}일 시간대별 변화 (기준: {base})", fontsize=10.5)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    c = weather_bins(pd.read_parquet(PROC / "weather_city_daily.parquet"))
    city = city_models(c)
    print(city.round(3).to_string(index=False))
    hourly = hourly_profiles(c)
    print(hourly.round(1).to_string())
    panel = station_panel(c)
    print(panel.attrs)
    print(panel.round(3).to_string(index=False))
    shel = shelter_coverage()
    print(shel.attrs)
    print(shel.round(3).to_string())

    city.to_csv(TAB / "d1_weather_city.csv", index=False, encoding="utf-8-sig")
    hourly.to_csv(TAB / "d2_weather_hourly.csv", encoding="utf-8-sig")
    panel.to_csv(TAB / "d3_weather_by_area.csv", index=False, encoding="utf-8-sig")
    shel.to_csv(TAB / "d4_shelter_coverage.csv", encoding="utf-8-sig")
    fig_city(city, FIG / "d1_weather_city.png")
    fig_hourly(hourly, FIG / "d2_weather_hourly.png")
    print("그림·표 저장 완료")


if __name__ == "__main__":
    main()
