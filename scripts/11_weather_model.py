"""D 분석: 폭염·한파·큰비가 고령자의 이동을 청장년보다 더 줄이는가.

실행: python scripts/11_weather_model.py   (10·12·13 다음, 약 5분)

1. 지하철 1~8호선 서울 전체 일별 회귀 (2022.7.1 ~ 2026.3.19, 1,358일)
   log(승차) = 날씨 구간 + 요일×공휴일 + 연·월 고정효과,  Newey-West 표준오차(7일)
   고령자(우대권), 일반, 둘의 비(고령자 상대 이용)를 각각 추정한다.
   날씨 구간: 폭염(최고 33℃ 이상), 무더위(30~33℃), 한파(최저 -10℃ 이하), 추위(-10~-5℃), 큰비(10mm 이상), 비(1~10mm)
   연속형: 최저기온이 -5℃보다 1℃ 낮을 때마다, 최고기온이 30℃보다 1℃ 높을 때마다의 효과 (모든 날을 쓴다)
2. 날씨 자료를 바꿔도 같은가: 서울 관측소 / 김포 관측소 / S-DoT 센서 서울 중앙값 / 역별 동네 날씨(S-DoT·강우량계)
3. 다른 노선:
   - 9호선 2·3단계 일별 (2023 ~ 2026.3)
   - 서울 관할 전 노선 월별 무임·유임 (2015 ~ 2026, 코로나 2020.2~2022.6 제외): 그 달의 한파·폭염·큰비 일수
     관측값이 빈 날은 서울 관측소 기준으로 보정한 ERA5 재분석 값을 쓴다
4. 버스·차량·도보와 외출 포기: KT 생활이동 연령 × 수단 일별 (겨울·여름 17개월, 2023.1 ~ 2026.2)
5. 시간대별 변화, C 유형·급경사 차이, 쉼터 접근
"""

import importlib.util
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

spec = importlib.util.spec_from_file_location("b10", ROOT / "scripts" / "10_build_weather.py")
b10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b10)
HOLIDAYS = b10.HOLIDAYS

BINS = {"heat33": "폭염 (최고 33℃ 이상)", "hot30": "무더위 (최고 30~33℃)", "cold10": "한파 (최저 -10℃ 이하)",
        "cold5": "추위 (최저 -10~-5℃)", "rain10": "큰비 (10mm 이상)", "rain1": "비 (1~10mm)"}
KEY3 = {"heat33": "폭염", "cold10": "한파", "rain10": "큰비"}
HOURS = ["06시 전"] + [f"{h:02d}시" for h in range(6, 24)] + ["24시 후"]
TYPE_NAMES = {0: "대중교통 양호 서민 주거지", 1: "언덕 위 고령 주거지", 2: "고소득 활동 지역",
              3: "교통 사각 저소득 고령 밀집", 4: "외곽 차 중심 신주거지"}
MODE_NAMES = {"subway": "지하철", "bus": "버스", "car": "차량", "walk": "도보", "other": "기타", "total": "전체 이동"}
FE = " + C(dow)*C(holiday) + C(ym)"
pct = lambda b: (np.exp(b) - 1) * 100
station_key = lambda s: re.sub(r"\(.*?\)", "", str(s)).strip()


def bins(c: pd.DataFrame, tmax="tmax", tmin="tmin", prcp="prcp") -> pd.DataFrame:
    c = c.copy()
    p = c[prcp].fillna(0)
    c["prcp_na"] = c[prcp].isna().astype(int)
    c["heat33"] = (c[tmax] >= 33).astype(int)
    c["hot30"] = ((c[tmax] >= 30) & (c[tmax] < 33)).astype(int)
    c["cold10"] = (c[tmin] <= -10).astype(int)
    c["cold5"] = ((c[tmin] <= -5) & (c[tmin] > -10)).astype(int)
    c["rain10"] = (p >= 10).astype(int)
    c["rain1"] = ((p >= 1) & (p < 10)).astype(int)
    c["cold_deg"] = np.maximum(0, -5 - c[tmin])   # 최저기온이 -5℃보다 낮은 만큼
    c["heat_deg"] = np.maximum(0, c[tmax] - 30)   # 최고기온이 30℃보다 높은 만큼
    c["prcp_mm"] = np.minimum(p, 100)
    return c


def calendar(c: pd.DataFrame) -> pd.DataFrame:
    return c.assign(dow=c["date"].dt.dayofweek, holiday=c["date"].isin(HOLIDAYS),
                    ym=c["date"].dt.to_period("M").astype(str))


def fit(df, y, rhs, keys):
    m = smf.ols(f"{y} ~ {rhs}", df).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    ci = m.conf_int()
    return {k: (pct(m.params[k]), pct(ci.loc[k, 0]), pct(ci.loc[k, 1]), m.pvalues[k]) for k in keys}


# ---------------------------------------------------------------- 1. 서울 전체 (1~8호선)

def city_models(c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = c.assign(l_elderly=np.log(c["elderly"]), l_general=np.log(c["general"]))
    c["l_ratio"] = c["l_elderly"] - c["l_general"]
    rows, cont = [], []
    for y, label in [("l_elderly", "고령자(우대권)"), ("l_general", "일반"), ("l_ratio", "고령자 상대 (고령자 ÷ 일반)")]:
        r = fit(c, y, " + ".join(BINS) + " + prcp_na" + FE, BINS)
        for k, (b, lo, hi, p) in r.items():
            rows.append({"대상": label, "날씨": BINS[k], "일수": int(c[k].sum()), "변화(%)": b, "95% 하한": lo,
                         "95% 상한": hi, "p": p})
        r = fit(c, y, "cold_deg + heat_deg + prcp_mm + prcp_na" + FE, ["cold_deg", "heat_deg", "prcp_mm"])
        for k, lab in [("cold_deg", "최저 -5℃보다 1℃ 추울 때마다"), ("heat_deg", "최고 30℃보다 1℃ 더울 때마다"),
                       ("prcp_mm", "강수 1mm 늘 때마다")]:
            b, lo, hi, p = r[k]
            cont.append({"대상": label, "변수": lab, "변화(%)": b, "95% 하한": lo, "95% 상한": hi, "p": p})
    return pd.DataFrame(rows), pd.DataFrame(cont)


# ---------------------------------------------------------------- 2. 날씨 자료 바꾸기

def robustness(city: pd.DataFrame, station: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    base = city.assign(l_ratio=np.log(city["elderly"] / city["general"]))

    def add(name, df, note=""):
        r = fit(df, "l_ratio", " + ".join(BINS) + " + prcp_na" + FE, KEY3)
        for k, (b, lo, hi, p) in r.items():
            rows.append({"날씨 자료": name, "날씨": KEY3[k], "변화(%)": b, "95% 하한": lo, "95% 상한": hi, "p": p,
                         "일수": int(df[k].sum()), "분석일": len(df), "비고": note})

    add("서울 관측소 (ASOS 108)", base)
    # S-DoT 센서는 지면 가까이 설치돼 관측소보다 덥게 잡힌다 → 서울 관측소와 겹치는 날로 선형 보정해 같은 기준(33℃, -10℃)을 쓴다
    sd = pd.read_parquet(PROC / "sdot_city_daily.parquet")
    ov = sd.merge(base[["date", "tmax", "tmin"]], on="date").dropna()
    cal = {v: np.polyfit(ov[f"{v}_sdot"], ov[v], 1) for v in ["tmax", "tmin"]}
    sdot_fit = {v: {"기울기": c[0], "절편": c[1], "상관": float(np.corrcoef(ov[f"{v}_sdot"], ov[v])[0, 1])}
                for v, c in cal.items()}
    g = bins(base.drop(columns=list(BINS) + ["prcp_na"]), "tmax_gimpo", "tmin_gimpo", "prcp_gimpo")
    add("김포 관측소", g.dropna(subset=["tmax_gimpo", "tmin_gimpo"]), "강서구 끝, 서울 관측소와 약 16km")
    for v in ["tmax", "tmin"]:
        sd[f"{v}_sdot"] = np.polyval(cal[v], sd[f"{v}_sdot"])
    s = bins(base.drop(columns=list(BINS) + ["prcp_na"]).merge(sd, on="date"), "tmax_sdot", "tmin_sdot", "prcp")
    add("S-DoT 센서 서울 중앙값 (기온)", s, "관측소 기준으로 보정, 강수는 서울 관측소, 2023년부터")

    # 역별 동네 날씨: 역 × 날짜 패널. 역 고정효과 + 요일×공휴일 + 연·월 (서울 전체 분석과 같은 비교)
    loc = pd.read_parquet(PROC / "sdot_station_daily.parquet")
    for v in ["tmax", "tmin"]:
        loc[f"{v}_local"] = np.polyval(cal[v], loc[f"{v}_local"])
    rain = pd.read_parquet(PROC / "rain_station_daily.parquet")
    p = (station[station["line"] == "1~8호선"].merge(loc, on=["station", "date"])
         .merge(rain[["station", "date", "prcp_local"]], on=["station", "date"]))
    p = p[(p["elderly"] > 0) & (p["general"] > 0)]
    p = calendar(bins(p, "tmax_local", "tmin_local", "prcp_local"))
    p["lr"] = np.log(p["elderly"] / p["general"])
    X = pd.get_dummies(p["ym"], prefix="ym", drop_first=True).astype(float)
    X = X.join(pd.get_dummies(p["dow"].astype(str) + "_" + p["holiday"].astype(str), prefix="dh", drop_first=True)
               .astype(float))
    X[list(BINS)] = p[list(BINS)].to_numpy(float)
    pp = pd.concat([p[["station", "date", "lr"]], X], axis=1).set_index(["station", "date"])
    m = PanelOLS(pp["lr"], pp.drop(columns="lr"), entity_effects=True).fit(cov_type="clustered", cluster_entity=True,
                                                                           cluster_time=True)
    ci = m.conf_int()
    for k in KEY3:
        rows.append({"날씨 자료": "역별 동네 날씨 (S-DoT 기온 + 강우량계)", "날씨": KEY3[k], "변화(%)": pct(m.params[k]),
                     "95% 하한": pct(ci.loc[k, "lower"]), "95% 상한": pct(ci.loc[k, "upper"]), "p": m.pvalues[k],
                     "일수": int(p.groupby("date")[k].max().sum()), "분석일": p["date"].nunique(),
                     "비고": f"역 {p['station'].nunique()}개 × 날짜, 2023.1~2025.12, 역·날짜 군집 표준오차"})
    # 같은 날 안에서 동네가 더 추웠는지/더 더웠는지/비가 더 왔는지만으로 (날짜 고정효과)
    q = p[["station", "date", "lr", "cold_deg", "heat_deg", "prcp_mm"]].set_index(["station", "date"])
    m2 = PanelOLS(q["lr"], q[["cold_deg", "heat_deg", "prcp_mm"]], entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True)
    local = {k: (pct(m2.params[k]), m2.pvalues[k]) for k in ["cold_deg", "heat_deg", "prcp_mm"]}
    local["sdot_calibration"] = sdot_fit
    return pd.DataFrame(rows), local


# ---------------------------------------------------------------- 3. 다른 노선

def line9_model(station: pd.DataFrame, city: pd.DataFrame) -> pd.DataFrame:
    l9 = station[station["line"] == "9호선 2·3단계"].groupby("date")[["elderly", "general"]].sum().reset_index()
    l9 = calendar(bins(l9.merge(city[["date", "tmax", "tmin", "prcp"]], on="date")))
    l9["l_ratio"] = np.log(l9["elderly"] / l9["general"])
    r = fit(l9, "l_ratio", " + ".join(BINS) + " + prcp_na" + FE, KEY3)
    return pd.DataFrame([{"노선": "9호선 2·3단계 (12개 역, 일별)", "날씨": KEY3[k], "변화(%)": b, "95% 하한": lo,
                          "95% 상한": hi, "p": p, "일수": int(l9[k].sum())} for k, (b, lo, hi, p) in r.items()])


def monthly_all_lines() -> pd.DataFrame:
    f = pd.read_csv(RAW / "서울시 지하철 호선별 역별 유무임 승하차 인원 정보.csv", encoding="cp949")
    f["key"] = f["지하철역"].map(station_key)
    groups = {"1~8호선 (서울교통공사)": [f"{i}호선" for i in range(1, 9)],
              "9호선 (1~3단계)": ["9호선", "9호선2~3단계", "9호선2단계"],
              "경전철 (우이신설·신림)": ["우이신설선", "신림선"]}
    groups["코레일·공항철도 등 광역 (서울 구간)"] = [l for l in f["호선명"].unique() if l not in sum(groups.values(), [])]
    st = pd.read_csv(RAW / "서울시 역사마스터 정보.csv", encoding="cp949")
    st["key"] = st["역사명"].map(station_key)
    st = st.groupby("key")[["위도", "경도"]].mean()
    seoul = gpd.read_file(PROC / "dong_mci.gpkg").to_crs("EPSG:4326").union_all()
    inside = gpd.GeoSeries(gpd.points_from_xy(st["경도"], st["위도"]), index=st.index, crs="EPSG:4326").within(seoul)
    in_seoul = set(inside[inside].index)

    # 월별 날씨: 서울 관측소 값이 있으면 쓰고, 빈 날은 서울 관측소 기준으로 보정한 ERA5
    asos = b10.meteostat("47108").set_index("date")
    era5 = pd.read_parquet(PROC / "weather_era5_calibrated.parquet").set_index("date")
    w = era5[["tmax", "tmin", "prcp"]].copy()
    for v in ["tmax", "tmin", "prcp"]:
        w[v] = asos[v].reindex(w.index).fillna(w[v])
    w = w[w.index >= "2015-01-01"]
    w["ym"] = w.index.year * 100 + w.index.month
    mw = w.groupby("ym").agg(cold=("tmin", lambda x: (x <= -10).sum()), heat=("tmax", lambda x: (x >= 33).sum()),
                             rain=("prcp", lambda x: (x >= 10).sum()))
    rows = []
    for gname, lines in groups.items():
        g = f[f["호선명"].isin(lines) & f["key"].isin(in_seoul)]
        p = g.groupby(["key", "사용월"])[["유임승차인원", "무임승차인원"]].sum().reset_index()
        p = p[(p["무임승차인원"] > 0) & (p["유임승차인원"] > 0) & ~p["사용월"].between(202002, 202206)]
        p = p.join(mw, on="사용월").dropna()
        p["ly"] = np.log(p["무임승차인원"] / p["유임승차인원"])
        p["date"] = pd.to_datetime(p["사용월"].astype(str) + "01")
        X = pd.get_dummies((p["사용월"] % 100).astype(str), prefix="m", drop_first=True).astype(float)
        X = X.join(pd.get_dummies((p["사용월"] // 100).astype(str), prefix="y", drop_first=True).astype(float))
        X[["cold", "heat", "rain"]] = p[["cold", "heat", "rain"]].to_numpy(float)
        pp = pd.concat([p[["key", "date", "ly"]], X], axis=1).set_index(["key", "date"])
        m = PanelOLS(pp["ly"], pp.drop(columns="ly"), entity_effects=True).fit(cov_type="clustered",
                                                                               cluster_entity=True)
        months = p["사용월"].unique()
        for k, lab in [("cold", "한파"), ("heat", "폭염"), ("rain", "큰비")]:
            b = m.params[k]
            rows.append({"노선": gname, "날씨": lab, "그 달 해당 일수 1일당 월 변화(%)": pct(b),
                         "하루 효과로 환산(%)": pct(b * 30.4), "p": m.pvalues[k], "역 수": p["key"].nunique(),
                         "개월 수": len(months), "해당 일수 합": int(mw.loc[months, k].sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. 버스·차량·도보, 외출 포기

def kt_modes(raw_city: pd.DataFrame) -> pd.DataFrame:
    t = pd.read_parquet(PROC / "kt_daily_type_mode.parquet")
    k = t.groupby(["date", "age", "mode"])["trips"].sum().unstack("mode")
    k["total"] = k.sum(axis=1)
    k = k.reset_index().merge(raw_city[["date", "tmax", "tmin", "prcp"]], on="date")
    k = calendar(bins(k))
    rows = []
    for age, lab in [("70p", "70대 이상"), ("adult", "20~50대")]:
        a = k[k["age"] == age]
        for mode in MODE_NAMES:
            r = fit(a.assign(y=np.log(a[mode])), "y", " + ".join(BINS) + " + prcp_na" + FE, KEY3)
            for w, (b, lo, hi, p) in r.items():
                rows.append({"연령": lab, "수단": MODE_NAMES[mode], "날씨": KEY3[w], "변화(%)": b, "95% 하한": lo,
                             "95% 상한": hi, "p": p})
    out = pd.DataFrame(rows)
    a = k[k["age"] == "70p"]
    out.attrs = {"days": int(a["date"].nunique()), "counts": {KEY3[w]: int(a[w].sum()) for w in KEY3},
                 "mode_share_70p": (a[list(MODE_NAMES)[:-1]].sum() / a["total"].sum()).round(3).to_dict()}
    return out


# ---------------------------------------------------------------- 5. 시간대·지역·쉼터

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


def station_panel(c: pd.DataFrame, station: pd.DataFrame) -> pd.DataFrame:
    d = station[station["line"] == "1~8호선"]
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
    rows, cols = [], []
    for w in KEY3:
        for t in [1, 2, 3, 4]:
            d[f"{w}_t{t}"] = d[w] * (d["type"] == t)
            cols.append(f"{w}_t{t}")
    p = d.set_index(["station", "date"])
    m = PanelOLS(p["lr"], p[cols].astype(float), entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True)
    for k in cols:
        w, t = k.split("_t")
        rows.append({"분석": "C 유형 (기준: 대중교통 양호)", "날씨": KEY3[w], "비교": TYPE_NAMES[int(t)],
                     "고령자 상대 이용 차이(%)": pct(m.params[k]), "p": m.pvalues[k]})
    e = d[d["steep_z"].notna()].copy()
    cols = []
    for w in KEY3:
        e[f"{w}_steep"] = e[w] * e["steep_z"]
        cols.append(f"{w}_steep")
    p = e.set_index(["station", "date"])
    m = PanelOLS(p["lr"], p[cols], entity_effects=True, time_effects=True).fit(cov_type="clustered",
                                                                             cluster_entity=True)
    for k in cols:
        rows.append({"분석": "역세권 급경사 비율 (1표준편차당)", "날씨": KEY3[k.split("_")[0]], "비교": "급경사 +1SD",
                     "고령자 상대 이용 차이(%)": pct(m.params[k]), "p": m.pvalues[k]})
    return pd.DataFrame(rows)


def shelter_coverage() -> pd.DataFrame:
    s = pd.read_csv(RAW / "서울시 기후동행쉼터.csv", encoding="cp949")
    p = gpd.GeoSeries(gpd.points_from_xy(s["X좌표(EPSG:5186)"], s["Y좌표(EPSG:5186)"]), crs="EPSG:5186").to_crs("EPSG:5179")
    grid = pd.read_parquet(PROC / "grid_slope.parquet")
    d, _ = cKDTree(np.c_[p.x, p.y]).query(grid[["x", "y"]].to_numpy())
    grid["shelter10"] = d <= 369
    dong = gpd.read_file(PROC / "dong_mci.gpkg").set_index("adm_cd10")
    cl = pd.read_csv(TAB / "c4_dong_clusters.csv", dtype={"adm_cd10": str}).set_index("adm_cd10")
    t = pd.DataFrame({"pop65": dong["pop_65_plus"], "share": grid.groupby("adm_cd10")["shelter10"].mean(),
                      "type": cl["type_name"]})
    t["covered"] = t["pop65"] * t["share"]
    out = t.groupby("type").agg(고령인구=("pop65", "sum"), 쉼터_10분내_고령인구=("covered", "sum"))
    out.loc["서울 전체"] = out.sum()
    out["쉼터 10분 안 고령자 비율"] = out["쉼터_10분내_고령인구"] / out["고령인구"]
    return out


# ---------------------------------------------------------------- 그림

def fig_city(t: pd.DataFrame, path: Path, title: str):
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
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_hourly(h: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(1, len(h) - 1)
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


def fig_modes(kt: pd.DataFrame, path: Path):
    modes = ["지하철", "버스", "도보", "차량", "전체 이동"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, w in zip(axes, ["한파", "큰비", "폭염"]):
        y = np.arange(len(modes))
        for off, (age, col) in zip([-0.18, 0.18], [("70대 이상", "#c53030"), ("20~50대", "#718096")]):
            r = kt[(kt["연령"] == age) & (kt["날씨"] == w)].set_index("수단").loc[modes]
            ax.barh(y + off, r["변화(%)"], 0.34, color=col, label=age,
                    xerr=[r["변화(%)"] - r["95% 하한"], r["95% 상한"] - r["변화(%)"]], error_kw={"lw": 0.7})
        ax.set_yticks(y, modes)
        ax.invert_yaxis()
        ax.axvline(0, color="#4a5568", lw=0.8)
        ax.set_title(w, fontsize=11)
        ax.set_xlabel("평소 대비 이동 변화 (%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("궂은 날 고령자는 다른 수단으로 옮기는가, 외출을 포기하는가 (KT 생활이동, 서울 도착 이동)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_robust(rob: pd.DataFrame, l9: pd.DataFrame, path: Path):
    rows = [(r["날씨 자료"], r["날씨"], r["변화(%)"], r["95% 하한"], r["95% 상한"]) for _, r in rob.iterrows()]
    rows += [("9호선 2·3단계 (서울 관측소)", r["날씨"], r["변화(%)"], r["95% 하한"], r["95% 상한"]) for _, r in l9.iterrows()]
    t = pd.DataFrame(rows, columns=["src", "w", "b", "lo", "hi"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    srcs = list(dict.fromkeys(t["src"]))
    for ax, w in zip(axes, ["한파", "큰비", "폭염"]):
        s = t[t["w"] == w].set_index("src").loc[srcs]
        y = np.arange(len(srcs))
        ax.errorbar(s["b"], y, xerr=[s["b"] - s["lo"], s["hi"] - s["b"]], fmt="o", color="#2b6cb0", capsize=3)
        ax.axvline(0, color="#a0aec0", lw=1)
        ax.set_yticks(y, srcs)
        ax.invert_yaxis()
        ax.set_title(f"{w}: 고령자 상대 이용 변화(%)", fontsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("날씨 자료와 노선을 바꿔도 같은가 (점: 추정치, 선: 95% 신뢰구간)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    raw_city = pd.read_parquet(PROC / "weather_city_daily.parquet")
    city = calendar(bins(raw_city))
    station = pd.read_parquet(PROC / "weather_station_daily.parquet")

    t1, cont = city_models(city)
    print(t1.round(3).to_string(index=False))
    print(cont.round(3).to_string(index=False))
    rob, local = robustness(city, station)
    print(rob.round(3).to_string(index=False))
    print("같은 날 동네 차이만 (날짜 고정효과):", local)
    l9 = line9_model(station, city)
    print(l9.round(3).to_string(index=False))
    mon = monthly_all_lines()
    print(mon.round(3).to_string(index=False))
    kt = kt_modes(raw_city)
    print(kt.attrs)
    print(kt.round(2).to_string(index=False))
    hourly = hourly_profiles(city)
    print(hourly.round(1).to_string())
    panel = station_panel(city, station)
    print(panel.round(3).to_string(index=False))
    shel = shelter_coverage()

    t1.to_csv(TAB / "d1_weather_city.csv", index=False, encoding="utf-8-sig")
    cont.to_csv(TAB / "d1b_weather_continuous.csv", index=False, encoding="utf-8-sig")
    hourly.to_csv(TAB / "d2_weather_hourly.csv", encoding="utf-8-sig")
    panel.to_csv(TAB / "d3_weather_by_area.csv", index=False, encoding="utf-8-sig")
    shel.to_csv(TAB / "d4_shelter_coverage.csv", encoding="utf-8-sig")
    rob.to_csv(TAB / "d5_weather_sources.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({k: v for k, v in local.items() if k != "sdot_calibration"}, index=["변화(%)", "p"]).T.to_csv(
        TAB / "d5b_local_within_day.csv", encoding="utf-8-sig")
    pd.DataFrame(local["sdot_calibration"]).to_csv(TAB / "d5c_sdot_calibration.csv", encoding="utf-8-sig")
    l9.to_csv(TAB / "d6_line9.csv", index=False, encoding="utf-8-sig")
    mon.to_csv(TAB / "d7_monthly_all_lines.csv", index=False, encoding="utf-8-sig")
    kt.to_csv(TAB / "d8_kt_modes.csv", index=False, encoding="utf-8-sig")
    fig_city(t1, FIG / "d1_weather_city.png",
             "궂은 날씨에 고령자의 지하철 이용이 얼마나 줄어드는가 (서울교통공사 1~8호선, 2022.7~2026.3)")
    fig_hourly(hourly, FIG / "d2_weather_hourly.png")
    fig_modes(kt, FIG / "d3_weather_modes.png")
    fig_robust(rob, l9, FIG / "d4_weather_robustness.png")
    print("그림·표 저장 완료")


if __name__ == "__main__":
    main()
