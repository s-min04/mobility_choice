"""A 분석: 고령자의 실제 이동(수도권 생활이동)으로 B 지수를 검증하고, 기대보다 대중교통을 덜 쓰는 동을 찾는다.

실행: python scripts/07_model_usage.py   (06_build_usage.py 다음)

1. 검증: 고령자 지수(MCI 고령자)가 일반 지수보다 고령자의 실제 대중교통 분담률을 잘 설명하는가
2. 기대 대비 실제: 공급 조건(역·정류장·대여소·큰길 접근, 경사, 밀도, 도심 거리, 연령 구성)만으로
   랜덤포레스트가 예측한 분담률(기대값)과 실제의 차이(잔차)를 구한다 (5겹 교차검증 × 5회, 표본 밖 예측).
   잔차는 '동네 전체가 덜 타는 부분'(청장년 잔차로 설명되는 몫)과 '고령자만 덜 타는 부분'으로 나눈다.
   9월 평일을 홀수일/짝수일로 나눠 같은 결과가 다시 나오는지(반분 신뢰도) 확인한다.
3. 요인: SHAP 으로 각 조건이 고령자 분담률을 얼마나, 어느 방향으로 바꾸는지 본다.
4. 외출: 고령자 1인당 하루 외출 횟수가 교통 조건과 관련 있는지 본다.
"""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
SEED = 42

for font in ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
    if any(font == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = font
        break
plt.rcParams["axes.unicode_minus"] = False

FEATURES = {
    "subway_elderly": "지하철 접근 (고령자 10분)",
    "a_bus_elderly": "버스 접근 (고령자 5분)",
    "a_bike_general": "따릉이 대여소 접근",
    "taxi_street_elderly": "큰길 접근 (고령자 5분)",
    "slope_mean": "평균 경사(%)",
    "steep_share": "경사 8% 이상 비율",
    "pop_density": "인구밀도",
    "dist_cityhall_km": "시청까지 거리(km)",
    "elderly_share": "65세 이상 비율",
    "share80_in70": "70세 이상 중 80세 이상 비율",
}


def rf():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, random_state=SEED, n_jobs=-1)


def oof(model, X, y, repeats=5):
    """5겹 교차검증 표본 밖 예측을 반복해 평균한다 (자기 자신을 보고 맞히지 않은 기대값)."""
    return np.mean([cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=s)) for s in range(repeats)],
                   axis=0)


# ---------------------------------------------------------------- 1. 검증

def validate(d: pd.DataFrame, pri: pd.Series) -> dict:
    z = lambda s: (s - s.mean()) / s.std()
    v = d.assign(e=z(d["mci_elderly"]), g=z(d["mci_general"]))
    r2 = {}
    for y in ["pt_share_70p", "pt_share_adult"]:
        for name, f in [("고령자 지수", "e"), ("일반 지수", "g"), ("둘 다", "e + g")]:
            m = smf.ols(f"{y} ~ {f}", v).fit()
            r2[f"{y} ~ {name}"] = {"R2": m.rsquared, **{k: m.params[k] for k in m.params.index if k != "Intercept"}}

    q = pd.qcut(d["mci_elderly"], 4, labels=["1분위(낮음)", "2분위", "3분위", "4분위(높음)"])
    quart = d.groupby(q, observed=True).agg(
        동수=("dong", "size"), 고령자지수=("mci_elderly", "median"),
        고령자_대중교통=("pt_share_70p", "median"), 고령자_차량=("car_share_70p", "median"),
        청장년_차량=("car_share_adult", "median"), 고령자_외출=("out_rate_70p", "median"))

    comp = {}
    for c in ["pt_share_70p", "car_share_70p", "car_share_adult", "out_rate_70p", "slope_mean"]:
        comp[c] = {"우선지역": d.loc[pri, c].median(), "나머지": d.loc[~pri, c].median(),
                   "p(Mann-Whitney)": mannwhitneyu(d.loc[pri, c], d.loc[~pri, c]).pvalue}
    w = d["motor_trips_70p"]
    comp["car_share_70p_가중"] = {"우선지역": np.average(d.loc[pri, "car_share_70p"], weights=w[pri]),
                                "나머지": np.average(d.loc[~pri, "car_share_70p"], weights=w[~pri])}
    comp["우선지역_고령자_하루_차량이동"] = {"우선지역": float(d.loc[pri, "trips_70p_car"].sum())}
    return {"r2": r2, "quartiles": quart, "priority": pd.DataFrame(comp).T}


# ---------------------------------------------------------------- 2. 기대 대비 실제

def expected_vs_actual(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    X = d[list(FEATURES)]
    cv_rows = []
    for yname in ["pt_share_70p", "pt_share_adult"]:
        y = d[yname].values
        for name, model in [("선형회귀", LinearRegression()),
                            ("그래디언트 부스팅", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04,
                                                                       max_leaf_nodes=12, min_samples_leaf=12,
                                                                       l2_regularization=1.0, random_state=SEED)),
                            ("랜덤포레스트", rf())]:
            p = oof(model, X, y, repeats=3)
            cv_rows.append({"결과변수": yname, "모델": name, "교차검증 R2": r2_score(y, p)})
    cv = pd.DataFrame(cv_rows)

    out = d[["adm_cd10", "gu", "dong", "pop_65_plus", "pt_share_70p", "pt_share_adult", "car_share_70p",
             "trips_70p_car", "mci_elderly"]].copy()
    out["expected_70p"] = oof(rf(), X, d["pt_share_70p"].values)
    out["expected_adult"] = oof(rf(), X, d["pt_share_adult"].values)
    out["resid_70p"] = out["pt_share_70p"] - out["expected_70p"]
    out["resid_adult"] = out["pt_share_adult"] - out["expected_adult"]
    b = np.polyfit(out["resid_adult"], out["resid_70p"], 1)[0]
    out["resid_area"] = b * out["resid_adult"]                     # 동네 전체가 덜 타는 몫
    out["resid_elderly_only"] = out["resid_70p"] - out["resid_area"]  # 고령자만 덜 타는 몫

    # 반분 신뢰도: 홀수일과 짝수일 자료로 각각 잔차를 구해 서로 얼마나 같은지
    rel = {"slope_area_on_adult": b}
    for comp in ["resid_70p", "resid_elderly_only"]:
        halves = {}
        for h in ["odd", "even"]:
            r70 = d[f"pt_share_70p_{h}"] - out["expected_70p"]
            ra = d[f"pt_share_adult_{h}"] - out["expected_adult"]
            halves[h] = r70 if comp == "resid_70p" else r70 - b * ra
        rel[f"{comp}_반분상관"] = float(np.corrcoef(halves["odd"], halves["even"])[0, 1])
    rel["resid_area_share_of_variance"] = float(out["resid_area"].var() / out["resid_70p"].var())

    sd = out["resid_70p"].std()
    low = out["resid_70p"] <= -sd
    out["type"] = np.select(
        [low & (out["resid_elderly_only"] <= -sd / 2), low],
        ["고령자만 덜 탐", "동네 전체가 덜 탐"], default="")
    rel["low_threshold_pp"] = float(-sd * 100)
    return cv, out, rel


# ---------------------------------------------------------------- 3. SHAP

def shap_analysis(d: pd.DataFrame):
    X = d[list(FEATURES)].rename(columns=FEATURES)
    model = rf().fit(X, d["pt_share_70p"])
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = pd.DataFrame({"요인": X.columns, "평균 |SHAP| (%p)": np.abs(sv).mean(axis=0) * 100,
                        "방향(요인과 SHAP 의 순위상관)": [spearmanr(X[c], sv[:, i])[0] for i, c in enumerate(X.columns)]})
    return imp.sort_values("평균 |SHAP| (%p)", ascending=False), sv, X


# ---------------------------------------------------------------- 4. 외출

def outing(d: pd.DataFrame) -> pd.DataFrame:
    z = lambda s: (s - s.mean()) / s.std()
    v = d.assign(**{c + "_z": z(d[c]) for c in ["out_rate_adult", "share80_in70", "elderly_share", "pop_density",
                                                  "dist_cityhall_km", "mci_elderly", "slope_mean"]})
    m = smf.ols("out_rate_70p ~ out_rate_adult_z + share80_in70_z + elderly_share_z + pop_density_z"
                " + dist_cityhall_km_z + mci_elderly_z + slope_mean_z", v).fit(cov_type="HC1")
    t = pd.DataFrame({"계수(1표준편차당 외출 횟수 변화)": m.params, "p": m.pvalues})
    t.loc["R2", "계수(1표준편차당 외출 횟수 변화)"] = m.rsquared
    return t


# ---------------------------------------------------------------- 그림

def fig_validation(val: dict, path: Path):
    q, c = val["quartiles"], val["priority"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    x = np.arange(len(q))
    ax.bar(x - 0.2, q["고령자_대중교통"] * 100, 0.4, label="대중교통", color="#2b6cb0")
    ax.bar(x + 0.2, q["고령자_차량"] * 100, 0.4, label="차량", color="#c05621")
    for i, (a, b) in enumerate(zip(q["고령자_대중교통"], q["고령자_차량"])):
        ax.text(i - 0.2, a * 100, f"{a:.0%}", ha="center", va="bottom", fontsize=9)
        ax.text(i + 0.2, b * 100, f"{b:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, q.index)
    ax.set_xlabel("B 고령자 이동선택권 지수 (행정동 4분위)")
    ax.set_ylabel("70세 이상 이동 중 비율(%)")
    ax.set_title("선택권이 낮은 동일수록 고령자가 차를 더 탄다")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    rows = [("70세 이상", "car_share_70p"), ("20~50대", "car_share_adult")]
    x = np.arange(len(rows))
    pv = [c.loc[k, "우선지역"] for _, k in rows]
    nv = [c.loc[k, "나머지"] for _, k in rows]
    ax.bar(x - 0.2, np.array(pv) * 100, 0.4, label="B 우선지역 25개 동", color="#c53030")
    ax.bar(x + 0.2, np.array(nv) * 100, 0.4, label="나머지 402개 동", color="#a0aec0")
    for i in range(len(rows)):
        ax.text(i - 0.2, pv[i] * 100, f"{pv[i]:.0%}", ha="center", va="bottom", fontsize=9)
        ax.text(i + 0.2, nv[i] * 100, f"{nv[i]:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("차량 분담률(%) (중앙값)")
    ax.set_title("B 우선지역의 차량 의존")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_shap(sv, X, path: Path):
    plt.figure()
    shap.summary_plot(sv * 100, X, show=False, plot_size=(8, 5), color_bar_label="요인 값 (낮음 → 높음)")
    ax = plt.gca()
    ax.set_xlabel("고령자 대중교통 분담률에 대한 영향 (%p, SHAP)")
    ax.set_title("무엇이 고령자의 대중교통 이용을 좌우하는가 (랜덤포레스트 + SHAP)", fontsize=11)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def fig_residual(g: gpd.GeoDataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    lim = np.nanpercentile(np.abs(g["resid_70p"]), 98) * 100
    for ax, col, title in [(axes[0], "resid_70p", "고령자 대중교통 분담률: 실제 - 기대 (%p)"),
                           (axes[1], "resid_elderly_only", "그중 고령자만의 차이 (동네 전체 효과 제외)")]:
        g.assign(v=g[col] * 100).plot(column="v", ax=ax, cmap="RdBu", vmin=-lim, vmax=lim, edgecolor="white",
                                       linewidth=0.2, legend=True, legend_kwds={"shrink": 0.6, "label": "%p"})
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    pri = g[g["priority"]]
    pri.boundary.plot(ax=axes[0], color="black", linewidth=0.8)
    axes[0].text(0.01, 0.01, "검은 테두리: B 우선지역", transform=axes[0].transAxes, fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    g = gpd.read_file(PROC / "dong_usage.gpkg")
    pri_tab = pd.read_csv(TAB / "b2_priority_dongs.csv")
    g["priority"] = (g["gu"] + g["dong"]).isin(pri_tab["gu"] + pri_tab["dong"])
    pri = g["priority"].values

    val = validate(g, pri)
    print(json.dumps(val["r2"], ensure_ascii=False, indent=1, default=float))
    print(val["quartiles"].round(3).to_string())
    print(val["priority"].round(4).to_string())

    cv, res, rel = expected_vs_actual(g)
    print(cv.round(3).to_string(index=False))
    print(json.dumps(rel, ensure_ascii=False, indent=1, default=float))
    res["priority"] = pri
    low = res[res["type"] != ""].sort_values("resid_70p")
    print(f"\n기대보다 {abs(rel['low_threshold_pp']):.1f}%p 이상 덜 타는 동 {len(low)}개 "
          f"(고령자만 덜 탐 {(low['type'] == '고령자만 덜 탐').sum()}개, B 우선지역과 겹침 {low['priority'].sum()}개)")
    print(low[["gu", "dong", "pt_share_70p", "expected_70p", "resid_70p", "resid_elderly_only", "type", "priority"]]
          .round(3).to_string(index=False))

    imp, sv, X = shap_analysis(g)
    print(imp.round(3).to_string(index=False))
    out = outing(g)
    print(out.round(4).to_string())
    print("외출 횟수(70+) 우선지역/나머지 중앙값:", val["priority"].loc["out_rate_70p", ["우선지역", "나머지"]].round(3).tolist())

    val["quartiles"].to_csv(TAB / "a1_mci_quartiles.csv", encoding="utf-8-sig")
    val["priority"].to_csv(TAB / "a1_priority_vs_rest.csv", encoding="utf-8-sig")
    pd.DataFrame(val["r2"]).T.to_csv(TAB / "a1_validation_r2.csv", encoding="utf-8-sig")
    cv.to_csv(TAB / "a2_model_cv.csv", index=False, encoding="utf-8-sig")
    res.to_csv(TAB / "a2_expected_vs_actual.csv", index=False, encoding="utf-8-sig")
    imp.to_csv(TAB / "a3_shap_importance.csv", index=False, encoding="utf-8-sig")
    out.to_csv(TAB / "a4_outing_regression.csv", encoding="utf-8-sig")
    (TAB / "a_summary.json").write_text(json.dumps({"reliability": rel, "r2": val["r2"]}, ensure_ascii=False,
                                                    indent=2, default=float))

    fig_validation(val, FIG / "a1_validation.png")
    fig_shap(sv, X, FIG / "a2_shap.png")
    fig_residual(g.merge(res[["adm_cd10", "resid_70p", "resid_elderly_only"]], on="adm_cd10"),
                 FIG / "a3_residual_map.png")
    print("그림·표 저장 완료")


if __name__ == "__main__":
    main()
