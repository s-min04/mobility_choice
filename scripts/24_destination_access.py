"""N. 광역 유입 감축 여지(C′)는 어디에, 언제 있나: 도착지 정책으로 연결하기.

실행: python scripts/24_destination_access.py   (16·21·22·23 다음, 약 1분)

M(23번)의 C′는 서울 밖에서 선택권 1 미만 동으로 들어오는 차량 이동을, 같은 거리대에서 선택권 1 이상 동으로
들어오는 유입의 차량 분담까지 낮춘다고 놓은 목표형 감축 여지다(연 30.9만 t). 이 감축은 출발지(경기·인천 등)가
아니라 도착지의 접근에 걸려 있다. 정책으로 잇기 위해 C′를 네 방향으로 나눈다.

1. 어느 동으로: 도착 동별 C′, 집중도(상위 10·30·50·100개 동 몫)
2. 무엇이 끌어들이나: 도착 목적 구성(KT 목적 OA-22298, 2026.9 평일, 모든 출발지), 종합병원 위치(OA-20337),
   거주 인구 대비 도착 이동
3. 언제: 도착 시각별 C′와 고령자(70대 이상) 도착 시각 (같은 동, KT 연령×수단 OA-22655)
4. 어디서: 출발 시군구별 C′
5. 확인: 시간대별로도, 일반 성인 기준 지수로 나눠도 '도착 동 선택권 낮을수록 유입 차량 분담 높음'이 유지되는가

C′는 23번과 같은 방법으로 (도착 동 × 거리대) 셀마다 계산하고, 셀 안에서는 차량 인·km 비례로 시각·출발지에 나눈다.
고령자 몫, 정책 대상 묶음별 몫, 1대당 운영비(22번 단가)를 함께 낸다.
"""

import json
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
for font in ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
    if any(font == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = font
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"

DAYS = ["20260903", "20260907", "20260909", "20260915", "20260918"]   # 23번과 같은 평일 5일
BANDS = [0, 2, 5, 10, 20, 1e9]
BAND_LABEL = ["0~2km", "2~5km", "5~10km", "10~20km", "20km+"]
CAR, TRANSIT = 8, [4, 5, 6]
OCC, WEEKDAYS = 1.3, 250          # 재차인원 1.3 (근거 확인 필요), 평일 250일 (23번과 같음)
LOW = ["공급 먼저", "이동권 우선"]  # 고령자 이동선택권 1 미만
TOP_N = 50
MAEUL_COST_DAY = 510_457          # 원/대·일, 22번과 같은 값 (서울시·마을버스조합 합의, 보도)
TOD = {"출근 7~9시": range(7, 10), "낮 10~16시": range(10, 17), "퇴근 17~19시": range(17, 20)}


def hour_of(code: pd.Series) -> pd.Series:
    """KT 시각 코드: 평시는 0~23(시), 출퇴근 시간은 700·720…1940(20분 단위) → 시."""
    return np.where(code >= 100, code // 100, code).astype(int)


def load_od() -> pd.DataFrame:
    """서울 도착, 내국인, 차량·대중교통 이동을 (도착 동, 출발 시군구, 도착 시각, 거리대, 수단)으로 모은다 (평일 하루 평균)."""
    parts = []
    for day in DAYS:
        z = zipfile.ZipFile(RAW / f"seoul_trans_admdong3_final_{day}.zip")
        d = pd.read_csv(z.open(z.namelist()[0]), usecols=["o_admdong_cd", "d_admdong_cd", "fns_time_cd", "in_forn_div_nm", "move_trans", "move_dist", "cnt"])
        d = d[(d["in_forn_div_nm"] == "내국인") & d["d_admdong_cd"].astype(str).str.startswith("11") & d["move_trans"].isin([CAR, *TRANSIT])]
        o = d["o_admdong_cd"].astype(str)
        d["o_sgg"] = np.where(o.str.startswith("11"), "11", o.str[:5])
        d["hour"] = hour_of(d["fns_time_cd"])
        d["band"] = pd.cut(d["move_dist"] / 1000, BANDS, labels=BAND_LABEL, right=False)
        d["mode"] = np.where(d["move_trans"] == CAR, "car", "pt")
        d["pkm"] = d["cnt"] * d["move_dist"] / 1000
        parts.append(d.groupby(["d_admdong_cd", "o_sgg", "hour", "band", "mode"], observed=True)[["cnt", "pkm"]].sum())
    g = (pd.concat(parts).groupby(level=[0, 1, 2, 3, 4], observed=True).sum() / len(DAYS)).reset_index()
    g = g.pivot_table(index=["d_admdong_cd", "o_sgg", "hour", "band"], columns="mode", values=["cnt", "pkm"], aggfunc="sum", fill_value=0, observed=True)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()
    g["adm_cd10"] = g["d_admdong_cd"].astype(str).str.ljust(10, "0")
    g["origin"] = np.where(g["o_sgg"] == "11", "서울", "서울 밖")
    return g


def c_prime(g: pd.DataFrame, to_t: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """23번 C′와 같은 계산 (도착 동 × 거리대 셀), 그다음 셀 안 차량 인·km 비례로 시각·출발지에 나눈다."""
    cell = g.groupby(["adm_cd10", "origin", "band", "quadrant"], observed=True)[["cnt_car", "cnt_pt", "pkm_car"]].sum().reset_index()
    cell["motor"] = cell["cnt_car"] + cell["cnt_pt"]
    cell["car_share"] = cell["cnt_car"] / cell["motor"].where(cell["motor"] > 0)
    cell["km_car"] = cell["pkm_car"] / cell["cnt_car"].where(cell["cnt_car"] > 0)
    low, inflow = cell["quadrant"].isin(LOW), cell["origin"] == "서울 밖"
    ref = cell[inflow & ~low].groupby("band", observed=True)[["cnt_car", "motor"]].sum()
    tgt = (ref["cnt_car"] / ref["motor"]).rename("target_dest")
    c = cell[inflow & low].merge(tgt, left_on="band", right_index=True)
    c["shift"] = ((c["car_share"] - c["target_dest"]).clip(lower=0) * c["motor"]).fillna(0)
    c["cut_t"] = c["shift"] * c["km_car"].fillna(0) * to_t
    c["co2_t"] = c["pkm_car"] * to_t
    x = g[(g["origin"] == "서울 밖") & g["quadrant"].isin(LOW)].merge(
        c[["adm_cd10", "band", "cut_t", "shift", "pkm_car"]].rename(columns={"pkm_car": "cell_pkm", "shift": "cell_shift"}), on=["adm_cd10", "band"])
    w = (x["pkm_car"] / x["cell_pkm"].where(x["cell_pkm"] > 0)).fillna(0)
    x["cut_a"], x["shift_a"] = x["cut_t"] * w, x["cell_shift"] * w
    return c, x


def main():
    ef = pd.read_csv(TAB / "e1_emission_factors.csv")
    EF = ef.loc[(ef["region"] == "서울") & (ef["차종"] == "승용") & (ef["year"] == 2024), "g_per_km"].iloc[0]
    detour = pd.read_csv(TAB / "i4_rank_stability.csv")["우회계수_실측_중앙값"].iloc[0]
    to_t = detour / OCC * EF * WEEKDAYS / 1e6
    quad = pd.read_csv(TAB / "f2_dong_quadrant.csv", dtype={"adm_cd10": str}).set_index("adm_cd10")
    g = load_od()
    g["quadrant"] = g["adm_cd10"].map(quad["quadrant"])
    c, x = c_prime(g, to_t)
    m3 = pd.read_csv(TAB / "m3_carbon_levers.csv")
    ref23 = m3.loc[m3["레버"].str.startswith("C′"), "감축_t"].iloc[0]
    print(f"C′ = {c['cut_t'].sum():,.0f} t (23번 {ref23:,.0f} t), 배분 확인 {x['cut_a'].sum():,.0f} t")

    # 1·2. 도착 동별 C′와 무엇이 끌어들이는가
    dong = gpd.read_file(PROC / "dong_mci.gpkg")[["adm_cd10", "pop_total", "pop_65_plus", "mci_general", "mci_elderly", "geometry"]]
    z = zipfile.ZipFile(RAW / "seoul_purpose_admdong1_in_202609.zip")
    ps = []
    for n in sorted(z.namelist()):
        if pd.Timestamp(n.split("_")[-1][:8]).weekday() < 5:
            ps.append(pd.read_csv(z.open(n), usecols=["d_admdong_cd", "move_purpose", "total_cnt"]).assign(day=n))
    p = pd.concat(ps)
    p["adm_cd10"] = p["d_admdong_cd"].astype(int).astype(str).str.ljust(10, "0")
    pp = p.groupby(["adm_cd10", "move_purpose"])["total_cnt"].sum().unstack(fill_value=0) / p["day"].nunique()
    arrivals = pp.sum(axis=1)
    mix = pp.div(arrivals, axis=0)[[1, 3, 6]].rename(columns={1: "도착목적_출근", 3: "도착목적_귀가", 6: "도착목적_병원"})
    h = pd.read_csv(RAW / "서울시 병의원 위치 정보.csv", encoding="cp949").dropna(subset=["병원경도", "병원위도"])
    h = h[h["병원분류명"] == "종합병원"]
    hg = gpd.GeoDataFrame(h, geometry=gpd.points_from_xy(h["병원경도"], h["병원위도"]), crs=4326)
    n_hosp = gpd.sjoin(hg, dong.to_crs(4326), predicate="within").groupby("adm_cd10").size()
    clus = pd.read_csv(TAB / "c4_dong_clusters.csv", dtype={"adm_cd10": str}).set_index("adm_cd10")["type_name"]
    med = pd.read_csv(TAB / "k2_dong_medical_access.csv", dtype={"adm_cd10": str}).set_index("adm_cd10")["double_blind_pop"]

    allv = dong.drop(columns="geometry").set_index("adm_cd10").join(mix)
    allv["도착_하루"] = arrivals
    allv["도착_인구비"] = allv["도착_하루"] / allv["pop_total"]
    allv["종합병원"] = n_hosp.reindex(allv.index).fillna(0).astype(int)
    allv["유형"] = clus
    allv["의료이중사각_65세이상"] = med
    allv = allv.join(quad[["gu", "dong", "quadrant", "priority1"]])
    by = c.groupby("adm_cd10")[["cut_t", "shift", "co2_t"]].sum()
    n1 = allv.join(by, how="inner").sort_values("cut_t", ascending=False)
    n1["누적몫"] = n1["cut_t"].cumsum() / n1["cut_t"].sum()
    n1["순위"] = np.arange(1, len(n1) + 1)
    n1.reset_index().to_csv(TAB / "n1_cprime_by_dong.csv", index=False, encoding="utf-8-sig")
    print({k: round(n1["누적몫"].iloc[k - 1], 3) for k in [10, 30, 50, 100]})

    top = n1.index[:TOP_N]
    allv["묶음"] = np.where(allv.index.isin(top), f"C′ 상위 {TOP_N}개 동",
                          np.where(allv["quadrant"].isin(LOW), "그 밖 선택권 1 미만 동", "선택권 1 이상 동"))
    rows = []
    for name, s in allv.groupby("묶음"):
        wts = s["도착_하루"]
        rows.append({"묶음": name, "동수": len(s), "C′_t": n1["cut_t"].reindex(s.index).sum(),
                     **{k: (s[k] * wts).sum() / wts.sum() for k in ["도착목적_출근", "도착목적_병원", "도착목적_귀가"]},
                     "도착_인구비_중앙값": s["도착_인구비"].median(), "종합병원_수": s["종합병원"].sum(),
                     "종합병원_있는_동_비율": (s["종합병원"] > 0).mean(), "65세이상": s["pop_65_plus"].sum(),
                     "의료이중사각_65세이상": s["의료이중사각_65세이상"].sum(), "1순위72개동_포함": s["priority1"].sum(),
                     "일반성인지수_중앙값": s["mci_general"].median(), "고령자지수_중앙값": s["mci_elderly"].median()})
    n2 = pd.DataFrame(rows)
    n2["C′_몫"] = n2["C′_t"] / n1["cut_t"].sum()
    n2.to_csv(TAB / "n2_cprime_groups.csv", index=False, encoding="utf-8-sig")
    print(n2.round(3).T.to_string())

    # 3. 언제: 도착 시각별 C′, 그리고 같은 동의 고령자(70대 이상) 도착 시각
    hh = x.groupby("hour")[["cut_a", "shift_a", "cnt_car", "cnt_pt"]].sum()
    hh["C′_몫"] = hh["cut_a"] / hh["cut_a"].sum()
    hh["유입_차량분담"] = hh["cnt_car"] / (hh["cnt_car"] + hh["cnt_pt"])
    xt = x[x["adm_cd10"].isin(top)].groupby("hour")["cut_a"].sum()
    hh[f"C′_몫_상위{TOP_N}"] = xt / xt.sum()
    z = zipfile.ZipFile(RAW / "seoul_trans_admdong1_in_202609.zip")
    ts = []
    for n in sorted(z.namelist()):
        if pd.Timestamp(n.split("_")[-1][:8]).weekday() < 5:
            ts.append(pd.read_csv(z.open(n), usecols=["d_admdong_cd", "time_cd", "male_70_cnt", "feml_70_cnt", "total_cnt"]))
    t = pd.concat(ts)
    t["adm_cd10"] = t["d_admdong_cd"].astype(int).astype(str).str.ljust(10, "0")
    t["hour"] = hour_of(t["time_cd"])
    t["e70"] = t["male_70_cnt"] + t["feml_70_cnt"]
    tt = t[t["adm_cd10"].isin(top)].groupby("hour")[["e70", "total_cnt"]].sum()
    hh[f"70대이상_도착_몫_상위{TOP_N}"] = tt["e70"] / tt["e70"].sum()
    hh[f"전연령_도착_몫_상위{TOP_N}"] = tt["total_cnt"] / tt["total_cnt"].sum()
    hh.to_csv(TAB / "n3_cprime_by_hour.csv", encoding="utf-8-sig")
    tod = pd.DataFrame({k: hh.loc[list(v)].sum() for k, v in TOD.items()}).T[
        ["C′_몫", f"C′_몫_상위{TOP_N}", f"70대이상_도착_몫_상위{TOP_N}", f"전연령_도착_몫_상위{TOP_N}"]]
    print(tod.round(3).to_string())

    # 4. 어디서: 출발 시군구
    names = gpd.read_file(RAW / "HangJeongDong_ver20260701.geojson")[["sgg", "sidonm", "sggnm"]].drop_duplicates("sgg").set_index("sgg")
    oo = x.groupby("o_sgg")["cut_a"].sum().sort_values(ascending=False).to_frame("C′_t").join(names)
    oo["몫"] = oo["C′_t"] / oo["C′_t"].sum()
    oo.to_csv(TAB / "n5_cprime_by_origin.csv", encoding="utf-8-sig")
    sido = oo.groupby("sidonm")["몫"].sum().sort_values(ascending=False)
    print(oo.head(5).round(3).to_string()); print(sido.head(5).round(3).to_string())

    # 5. 확인: 시간대 × 거리대 × 도착 동 지수 4분위(고령자·일반 성인 기준), 서울 밖 출발, 2km 이상
    s = g[(g["origin"] == "서울 밖") & (g["band"] != "0~2km")].copy()
    s["시간대"] = "밤·새벽"
    for k, v in TOD.items():
        s.loc[s["hour"].isin(list(v)), "시간대"] = k
    out = []
    mci = dong.set_index("adm_cd10")
    for col, label in [("mci_elderly", "고령자 기준"), ("mci_general", "일반 성인 기준")]:
        s["q"] = s["adm_cd10"].map(pd.qcut(mci[col], 4, labels=["1분위(낮음)", "2분위", "3분위", "4분위(높음)"]))
        r = s.groupby(["시간대", "band", "q"], observed=True)[["cnt_car", "cnt_pt"]].sum()
        r = (r["cnt_car"] / (r["cnt_car"] + r["cnt_pt"])).unstack("q")
        r["1분위÷4분위"] = r["1분위(낮음)"] / r["4분위(높음)"]
        out.append(r.assign(지수=label).reset_index())
    n4 = pd.concat(out)
    n4.to_csv(TAB / "n4_inflow_gap_by_time.csv", index=False, encoding="utf-8-sig")
    rng = n4.groupby(["지수", "시간대"])["1분위÷4분위"].agg(["min", "max"])
    print(rng.round(2).to_string())

    # 6. 정책 대상 묶음별 C′ 몫과 규모
    total = n1["cut_t"].sum()
    am = x[x["hour"].isin(list(TOD["출근 7~9시"]))]
    groups = {
        f"C′ 상위 {TOP_N}개 동": n1.index[:TOP_N],
        "종합병원이 있는 선택권 1 미만 동": n1.index[n1["종합병원"] > 0],
        "도착 인구비 상위 25% (끌어들이는 동)": n1.index[n1["도착_인구비"] >= allv["도착_인구비"].quantile(0.75)],
        "'외곽 차 중심 신주거지' 유형 (C 군집)": n1.index[n1["유형"] == "외곽 차 중심 신주거지"],
        "1순위 72개 동과 겹치는 동": n1.index[n1["priority1"].fillna(False).astype(bool)],
    }
    rows = [{"대상": k, "동수": len(v), "C′_t": n1.loc[v, "cut_t"].sum(), "C′_몫": n1.loc[v, "cut_t"].sum() / total,
             "줄일_차량이동_평일": n1.loc[v, "shift"].sum(), "65세이상": n1.loc[v, "pop_65_plus"].sum()} for k, v in groups.items()]
    rows.append({"대상": "출근 시간(7~9시 도착)분만", "동수": np.nan, "C′_t": am["cut_a"].sum(), "C′_몫": am["cut_a"].sum() / total,
                 "줄일_차량이동_평일": am["shift_a"].sum(), "65세이상": np.nan})
    rows.append({"대상": "C′ 전체", "동수": len(n1), "C′_t": total, "C′_몫": 1.0, "줄일_차량이동_평일": n1["shift"].sum(),
                 "65세이상": n1["pop_65_plus"].sum()})
    n6 = pd.DataFrame(rows)
    n6["대체_1건당_연tCO2"] = n6["C′_t"] / n6["줄일_차량이동_평일"]   # 평일 매일 1건이 대중교통으로 옮길 때 연간 감축
    n6["동당_줄일_차량이동_평일"] = n6["줄일_차량이동_평일"] / n6["동수"]
    n6.to_csv(TAB / "n6_policy_targets.csv", index=False, encoding="utf-8-sig")
    print(n6.round(3).to_string())

    veh_year = MAEUL_COST_DAY * 365
    per_trip = n6.loc[n6["대상"] == "C′ 전체", "대체_1건당_연tCO2"].iloc[0]
    e5 = pd.read_csv(TAB / "e5_scenarios.csv").query("시나리오 == 'S1' and 연령 == '70대 이상'").iloc[0]
    per_trip_elderly = e5["연간감축_tCO2eq"] / e5["줄어드는_차량이동_평일"]   # E S1: 1순위 72개 동 고령자 차량 이동 1건(평일 매일)당
    summ = {"cprime_t": total, "cprime_23": ref23, "top10": n1["누적몫"].iloc[9], "top30": n1["누적몫"].iloc[29],
            "top50": n1["누적몫"].iloc[TOP_N - 1], "top100": n1["누적몫"].iloc[99],
            "time_of_day": tod.round(4).to_dict(), "origin_sido_top": sido.head(5).round(4).to_dict(),
            "gap_range": {f"{a}|{b}": [round(v["min"], 3), round(v["max"], 3)] for (a, b), v in rng.iterrows()},
            "vehicle_cost_year_won": veh_year, "t_per_daily_trip": per_trip,
            "t_per_daily_trip_elderly_S1": per_trip_elderly, "ratio_vs_elderly": per_trip / per_trip_elderly,
            # 운영비 1대분을 탄소만으로 정당화하려면(가정 없는 손익분기): 평일 매일 몇 건의 유입 차량 이동이 옮겨야 하나
            "trips_per_vehicle_for_1m_won_per_t": veh_year / (per_trip * 1e6),
            "trips_per_vehicle_for_v5_low_cost_283m": veh_year / (per_trip * 283e4)}
    json.dump(summ, open(TAB / "n_summary.json", "w"), ensure_ascii=False, indent=2, default=float)
    print(json.dumps({k: v for k, v in summ.items() if not isinstance(v, dict)}, ensure_ascii=False, indent=1, default=float))
    fig_dest(n1, hh, n4)


def fig_dest(n1: pd.DataFrame, hh: pd.DataFrame, n4: pd.DataFrame):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.15, 1.1, 0.9]})
    # (가) 상위 15개 동
    k = n1.head(15).iloc[::-1]
    col = np.where(k["종합병원"] > 0, "#eb6834", np.where(k["도착목적_출근"] >= 0.25, "#2a78d6", "#9a9893"))
    a1.barh(np.arange(len(k)), k["cut_t"] / 1e3, color=col)
    a1.set_yticks(np.arange(len(k)), [f"{g} {d}" for g, d in zip(k["gu"], k["dong"])], fontsize=8)
    a1.set_xlabel("C′ 감축 여지 (천 tCO2/년, 목표형)")
    a1.set_title(f"광역 유입 감축 여지 상위 동 (상위 {TOP_N}개 동 = {n1['누적몫'].iloc[TOP_N - 1]:.0%})", fontsize=10)
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color="#eb6834", label="종합병원 있음"), Patch(color="#2a78d6", label="도착 중 출근 25% 이상"),
                       Patch(color="#9a9893", label="그 밖")], frameon=False, fontsize=7, loc="lower right")
    # (나) 시각별: C′ 몫 vs 같은 동 70대 이상 도착 몫
    hrs = np.arange(5, 24)
    a2.plot(hrs, hh.loc[hrs, f"C′_몫_상위{TOP_N}"] * 100, marker="o", ms=3, color="#eb6834", lw=2, label="광역 유입 C′ (상위 50개 동)")
    a2.plot(hrs, hh.loc[hrs, f"70대이상_도착_몫_상위{TOP_N}"] * 100, marker="s", ms=3, color="#2a78d6", lw=2, label="같은 동 70대 이상 도착")
    for v, lab in [(7, "출근"), (17, "퇴근")]:
        a2.axvspan(v - 0.5, v + 2.5, color="#e8e6e1", zorder=0)
        a2.text(v + 1, 0.97, lab, ha="center", va="top", fontsize=8, color="#52514e", transform=a2.get_xaxis_transform())
    a2.set_xticks(range(6, 24, 2))
    a2.set_ylim(0, None)
    a2.set_xlabel("도착 시각 (시)")
    a2.set_ylabel("하루 중 몫 (%)")
    a2.set_title("언제 들어오나: 유입 차량과 고령자 이동의 시간대", fontsize=10)
    a2.legend(frameon=False, fontsize=7)
    # (다) 시간대별 1분위÷4분위 (고령자·일반 성인 지수)
    tods = ["출근 7~9시", "낮 10~16시", "퇴근 17~19시", "밤·새벽"]
    for j, (lab, c) in enumerate([("고령자 기준", "#eb6834"), ("일반 성인 기준", "#2a78d6")]):
        r = n4[n4["지수"] == lab].groupby("시간대")["1분위÷4분위"]
        lo, hi = r.min().reindex(tods), r.max().reindex(tods)
        xs = np.arange(len(tods)) + (j - 0.5) * 0.25
        a3.vlines(xs, lo, hi, color=c, lw=6, label=f"도착 동 지수: {lab}")
    a3.axhline(1, color="#52514e", lw=0.8, ls="--")
    a3.set_xticks(np.arange(len(tods)), tods, fontsize=8)
    a3.set_ylabel("유입 차량 분담 1분위 ÷ 4분위\n(2km 이상 거리대별 범위)")
    a3.set_title("도착 동 선택권 차이는 출근 시간에 가장 크다", fontsize=10)
    a3.legend(frameon=False, fontsize=7)
    for a in (a1, a2, a3):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "n1_destination_access.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
