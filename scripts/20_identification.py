"""J. 단면 연관의 식별 강화와 S1 범위, 1인당 차량 CO2 격차.

실행: python scripts/20_identification.py   (15_carbon_scenarios.py 다음, 약 1분)

E의 S1은 427개 동 단면 회귀의 기울기(B 고령자 지수 1 → 70대 이상 차량 분담 −11.5%p)를 '공급을 늘리면 생길 변화'로 썼다.
단면 연관이 인과라는 보장은 없으므로, 기울기가 설명변수·비교 방식에 얼마나 민감한지 보고 S1을 범위로 다시 낸다.

1. 기울기 (70대 이상 / 20~50대 차량 분담 ~ 고령자 지수)
   (a) 통제 없음  (b) 기본: 소득·밀도·도심거리 (E)  (c) + 구 고정효과: 같은 구 안의 동끼리만 비교
   (d) + 인구 구성·경사  (e) + 자가용 등록(차 보유는 결과에 가까운 변수라 과잉 통제, 하한 참고)
   (f) SLX: 이웃 동(Queen)의 지수·통제변수 평균을 함께 넣는다 (공간 파급을 분리)
   (g) 매칭: 지수 하위 1/3 동마다 통제변수(마할라노비스)가 가장 가까운 상위 1/3 동을 짝지어 Δ차량분담 ÷ Δ지수
   (h) Oster(2019) 보정: 관측 통제변수만큼 강한 미관측 교란(δ=1, R²max = 1.3×R²)을 가정한 기울기
2. 각 기울기로 S1 감축량을 다시 계산한다 (E의 다른 가정은 기본값)
3. 1인당 차량 CO2: 동 도착 차량 이동 × 거리 × 배출계수 ÷ 인구, 이동선택권 4분위·4분면별
   (도착 기준이라 방문 차량이 섞인다 → 방문 많은 동 제외, 거주자 몫 보정을 함께 낸다)
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from libpysal.weights import Queen
from esda.moran import Moran
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "tables"

spec = importlib.util.spec_from_file_location("e15", ROOT / "scripts" / "15_carbon_scenarios.py")
e15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e15)

BASE_CTRL = ["welfare_65p_rate", "log_apt_price", "pop_density", "dist_cityhall_km"]
DEMO = ["elderly_share", "share80_in70", "slope_mean"]


def prepare():
    d, ef = e15.load()
    cars = pd.read_csv(TAB / "c4_dong_clusters.csv", dtype={"adm_cd10": str})[["adm_cd10", "cars_log"]]
    d = d.merge(cars, on="adm_cd10")
    z = lambda s: (s - s.mean()) / s.std()
    for c in BASE_CTRL + DEMO + ["cars_log"]:
        d[c + "_z"] = z(d[c])
    w = Queen.from_dataframe(d, use_index=False)
    w.transform = "r"
    for c in ["mci_elderly"] + [x + "_z" for x in BASE_CTRL]:
        d["W_" + c] = w.sparse @ d[c].to_numpy()
    return d, ef, w


def slopes(d, w) -> pd.DataFrame:
    base = " + ".join(c + "_z" for c in BASE_CTRL)
    demo = " + ".join(c + "_z" for c in DEMO)
    slx = " + ".join("W_" + c for c in ["mci_elderly"] + [x + "_z" for x in BASE_CTRL])
    specs = {"(a) 통제 없음": "", "(b) 기본: 소득·밀도·도심거리": base, "(c) + 구 고정효과": base + " + C(gu)",
             "(d) + 인구 구성·경사": base + " + " + demo, "(e) + 자가용 등록 (과잉 통제)": base + " + " + demo + " + cars_log_z",
             "(f) SLX (이웃 동 평균 포함)": base + " + " + slx}
    rows = []
    for age in ["70p", "adult"]:
        y = f"car_share_{age}"
        fits = {}
        for name, rhs in specs.items():
            m = smf.ols(f"{y} ~ mci_elderly" + (" + " + rhs if rhs else ""), d).fit(cov_type="HC1")
            fits[name] = m
            lo, hi = m.conf_int().loc["mci_elderly"]
            mi = Moran(m.resid.to_numpy(), w, permutations=199)
            rows.append({"연령": age, "모형": name, "기울기": m.params["mci_elderly"], "95%하한": lo, "95%상한": hi,
                         "R2": m.rsquared, "잔차_Moran_I": mi.I, "잔차_Moran_p": mi.p_sim})
        # (g) 매칭
        q = d["mci_elderly"].quantile([1 / 3, 2 / 3]).to_numpy()
        lo_g, hi_g = d[d["mci_elderly"] <= q[0]], d[d["mci_elderly"] >= q[1]]
        X = d[[c + "_z" for c in BASE_CTRL]].to_numpy()
        vi = np.linalg.inv(np.cov(X.T))
        dist = cdist(lo_g[[c + "_z" for c in BASE_CTRL]].to_numpy(), hi_g[[c + "_z" for c in BASE_CTRL]].to_numpy(),
                     "mahalanobis", VI=vi)
        j = dist.argmin(axis=1)
        dy = lo_g[y].to_numpy() - hi_g[y].to_numpy()[j]
        dx = lo_g["mci_elderly"].to_numpy() - hi_g["mci_elderly"].to_numpy()[j]
        rng = np.random.default_rng(0)
        boot = [np.mean(dy[k]) / np.mean(dx[k]) for k in (rng.integers(0, len(dy), len(dy)) for _ in range(999))]
        rows.append({"연령": age, "모형": "(g) 매칭 (하위 1/3 ↔ 상위 1/3, 마할라노비스)", "기울기": dy.mean() / dx.mean(),
                     "95%하한": np.percentile(boot, 2.5), "95%상한": np.percentile(boot, 97.5), "R2": np.nan,
                     "잔차_Moran_I": np.nan, "잔차_Moran_p": np.nan, "짝수": len(dy),
                     "짝_통제변수_평균차(표준화)": float(np.abs(lo_g[[c + '_z' for c in BASE_CTRL]].to_numpy()
                                                      - hi_g[[c + '_z' for c in BASE_CTRL]].to_numpy()[j]).mean())})
        # (h) Oster: 통제 없음(a) → 기본(b)
        b0, r0 = fits["(a) 통제 없음"].params["mci_elderly"], fits["(a) 통제 없음"].rsquared
        b1, r1 = fits["(b) 기본: 소득·밀도·도심거리"].params["mci_elderly"], fits["(b) 기본: 소득·밀도·도심거리"].rsquared
        rmax = min(1.0, 1.3 * r1)
        beta_star = b1 - (b0 - b1) * (rmax - r1) / (r1 - r0)
        delta_zero = b1 * (r1 - r0) / ((b0 - b1) * (rmax - r1))
        rows.append({"연령": age, "모형": "(h) Oster 보정 (δ=1, R²max=1.3R²)", "기울기": beta_star, "95%하한": np.nan,
                     "95%상한": np.nan, "R2": rmax, "잔차_Moran_I": np.nan, "잔차_Moran_p": np.nan,
                     "기울기를_0으로_만드는_δ": delta_zero})
    return pd.DataFrame(rows)


def s1_by_slope(d, ef, sl: pd.DataFrame) -> pd.DataFrame:
    p = {**e15.BASE, "ef": ef.loc[(ef["region"] == "서울") & (ef["차종"] == "승용") & (ef["year"] == 2024), "g_per_km"].iloc[0]}
    ratio = pd.read_csv(TAB / "e3_occupancy_check.csv")["60세이상_거리비(서울도착, 모든 수단)"].iloc[0]
    rows = []
    for model in sl["모형"].unique():
        s = sl[sl["모형"] == model].set_index("연령")["기울기"]
        fake = pd.DataFrame([{"age": a, "kind": "x", "기울기": s[a]} for a in ["70p", "adult"]])
        trips = e15.shifted_trips(d, fake, "x")
        t = e15.tco2_per_year(d, trips, p, ratio)
        rows.append({"모형": model, "고령자_기울기": s["70p"], "S1_고령자_차량이동감소_평일": trips["S1_70p"].sum(),
                     "S1_고령자만_t": t["S1_70p"].sum(), "S1_동네전체_t": t[["S1_70p", "S1_adult"]].sum().sum()})
    return pd.DataFrame(rows)


def per_capita_co2(d, ef) -> pd.DataFrame:
    """지금의 차량 이동이 만드는 1인당 연간 CO2 (기본 가정: 서울 출발 이동, 우회 1.3, 재차 1.3, 평일 250일).

    KT는 도착 동 기준이라 도심처럼 방문 차량이 많은 동은 1인당 값이 부풀려진다. 그래서 세 가지로 계산한다.
    - 전체: 도착 차량 이동 그대로
    - 방문 많은 동 제외 (기본): 70대 이상 도착 이동 ÷ 70세 이상 인구 상위 20% 동을 뺀다
    - 거주자 몫 보정: 차량 이동 × (귀가 이동 ÷ 전체 도착 이동), 연령별
    """
    p = {**e15.BASE, "ef": ef.loc[(ef["region"] == "서울") & (ef["차종"] == "승용") & (ef["year"] == 2024), "g_per_km"].iloc[0]}
    ratio = pd.read_csv(TAB / "e3_occupancy_check.csv")["60세이상_거리비(서울도착, 모든 수단)"].iloc[0]
    tot70 = d[[c for c in d.columns if c.startswith("trips_70p_")]].sum(axis=1)
    tota = d[[c for c in d.columns if c.startswith("trips_adult_")]].sum(axis=1)
    visit = tot70 / d["pop_70_plus"]
    quad = pd.read_csv(TAB / "f2_dong_quadrant.csv", dtype={"adm_cd10": str}).set_index("adm_cd10")["quadrant"]
    qlab = ["1분위(낮음)", "2분위", "3분위", "4분위(높음)"]
    base = d.assign(quadrant=d["adm_cd10"].map(quad), q=pd.qcut(d["mci_elderly"], 4, labels=qlab))
    versions = {"방문 많은 동 제외 (기본)": (visit < visit.quantile(0.8), 1.0, 1.0),
                "거주자 몫 보정": (pd.Series(True, index=d.index), d["home_trips_70p"] / tot70, d["home_trips_adult"] / tota),
                "전체 (보정 없음)": (pd.Series(True, index=d.index), 1.0, 1.0)}
    rows = []
    for ver, (keep, w70, wad) in versions.items():
        trips = pd.DataFrame({"X_70p": d["trips_70p_car"] * w70, "X_adult": d["trips_adult_car"] * wad}, index=d.index)
        t = e15.tco2_per_year(d, trips, p, ratio)
        x0 = base.assign(co2_70p=t["X_70p"], co2_adult=t["X_adult"])[keep]
        for col, groups in [("q", qlab), ("quadrant", ["공급 먼저", "이동권 우선", "수요관리 가능", "유지"])]:
            for gname in groups:
                x = x0[x0[col] == gname]
                rows.append({"계산": ver, "구분": f"{'고령자 지수 ' if col == 'q' else ''}{gname}", "동수": len(x),
                             "70대이상_1인당_kg": x["co2_70p"].sum() * 1e3 / x["pop_70_plus"].sum(),
                             "20~50대_1인당_kg": x["co2_adult"].sum() * 1e3 / x["pop_20_59_x"].sum()})
    out = pd.DataFrame(rows)
    for ver in versions:
        v = out[(out["계산"] == ver) & out["구분"].str.startswith("고령자 지수")].set_index("구분")
        for c in ["70대이상_1인당_kg", "20~50대_1인당_kg"]:
            out.loc[(out["계산"] == ver) & (out["구분"] == "고령자 지수 1분위(낮음)"), c.replace("_kg", "_1분위÷4분위")] = \
                v.loc["고령자 지수 1분위(낮음)", c] / v.loc["고령자 지수 4분위(높음)", c]
    return out


def main():
    d, ef, w = prepare()
    sl = slopes(d, w)
    sl.to_csv(TAB / "j1_slope_robustness.csv", index=False, encoding="utf-8-sig")
    print(sl.round(4).to_string())
    s1 = s1_by_slope(d, ef, sl)
    s1.to_csv(TAB / "j2_s1_by_slope.csv", index=False, encoding="utf-8-sig")
    print(s1.round(1).to_string())
    pc = per_capita_co2(d, ef)
    pc.to_csv(TAB / "j3_per_capita_co2.csv", index=False, encoding="utf-8-sig")
    print(pc.round(1).to_string())


if __name__ == "__main__":
    main()
