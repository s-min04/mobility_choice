"""M. 탄소는 어디서 나오고, 어디를 줄일 수 있나: 모든 연령의 차량 이동을 거리대별로 본다.

실행: python scripts/23_carbon_levers.py   (14·16·19 다음, 약 1분)

E는 70대 이상·20~50대의 '도착 이동 분담'으로 감축을 계산해 고령자 몫이 작았다(연 1천 t).
여기서는 출도착 × 수단 자료(OA-22657, 2026.9 평일 5일, 내국인, 모든 연령)로 서울에 도착하는 차량 이동 전체를
거리대(직선 0~2·2~5·5~10·10~20·20km+)와 출발지(서울/서울 밖)로 나눈다.

1. 배출 지도: 지금의 차량 이동이 만드는 연간 CO2가 어느 4분면(F)·거리대·출발지에서 나오는가
2. 거리대 맞춤 격차 해소: 각 동의 거리대별 차량 분담(차량 ÷ 차량+대중교통)이 같은 거리대 서울 평균보다 높으면
   그 차이만큼 대중교통으로 옮긴다고 놓는다(목표형). 거리대를 맞추므로 '먼 곳에 사는 동이라 차가 많은' 효과는 빠진다.
   - 레버 A: '공급 먼저'·'이동권 우선' 동(선택권 1 미만)의 서울 출발 이동 → 공급 확대가 필요한 감축
   - 레버 B: '수요관리 가능'·'유지' 동의 서울 출발 이동 → 대안이 이미 있는 곳의 수요관리 감축
   - 레버 C′: 서울 밖에서 선택권 1 미만 동으로 들어오는 이동 → 같은 거리대에서 선택권 1 이상 동으로 들어오는 유입의
     분담까지 (도착지 쪽 접근 개선, 관측된 차이에 근거)
   - 레버 C: 서울 밖에서 들어오는 이동 → 광역 교통 감축 (서울 출발 이동의 같은 거리대 분담까지, 상한)
     서울 밖 출발지의 대중교통 여건은 이 자료로 알 수 없어 과감한 가정이다 → 격차의 절반만 줄이는 경우(C½)도 낸다
4. 연결고리 확인: 같은 거리대 안에서도 선택권 낮은 동의 차량 분담이 높은가 (서울 출발, 고령자 이동선택권 4분위별)
5. 서울 밖에서 들어오는 이동도 도착 동의 선택권에 따라 차량 분담이 다른가 (도착지 쪽 접근의 효과)
3. 공통 가정: 우회계수 1.39(19번 실측), 재차인원 1.3, 서울 승용 194.6 g/km, 평일 250일
"""

import json
import zipfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
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

DAYS = ["20260903", "20260907", "20260909", "20260915", "20260918"]
BANDS = [0, 2, 5, 10, 20, 1e9]
BAND_LABEL = ["0~2km", "2~5km", "5~10km", "10~20km", "20km+"]
CAR, TRANSIT = 8, [4, 5, 6]
OCC, WEEKDAYS = 1.3, 250


def load() -> pd.DataFrame:
    parts = []
    for day in DAYS:
        z = zipfile.ZipFile(RAW / f"seoul_trans_admdong3_final_{day}.zip")
        d = pd.read_csv(z.open(z.namelist()[0]), usecols=["o_admdong_cd", "d_admdong_cd", "in_forn_div_nm", "move_trans", "move_dist", "cnt"])
        d = d[(d["in_forn_div_nm"] == "내국인") & d["d_admdong_cd"].astype(str).str.startswith("11") & d["move_trans"].isin([CAR, *TRANSIT])]
        d["origin"] = np.where(d["o_admdong_cd"].astype(str).str.startswith("11"), "서울", "서울 밖")
        d["band"] = pd.cut(d["move_dist"] / 1000, BANDS, labels=BAND_LABEL, right=False)
        d["mode"] = np.where(d["move_trans"] == CAR, "car", "pt")
        d["pkm"] = d["cnt"] * d["move_dist"] / 1000
        parts.append(d.groupby(["d_admdong_cd", "origin", "band", "mode"], observed=True)[["cnt", "pkm"]].sum())
    g = pd.concat(parts).groupby(level=[0, 1, 2, 3]).sum() / len(DAYS)
    g = g.unstack("mode", fill_value=0)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()
    g["adm_cd10"] = g["d_admdong_cd"].astype(str).str.ljust(10, "0")
    return g


def main():
    ef = pd.read_csv(TAB / "e1_emission_factors.csv")
    EF = ef.loc[(ef["region"] == "서울") & (ef["차종"] == "승용") & (ef["year"] == 2024), "g_per_km"].iloc[0]
    detour = pd.read_csv(TAB / "i4_rank_stability.csv")["우회계수_실측_중앙값"].iloc[0]
    to_t = detour / OCC * EF * WEEKDAYS / 1e6  # 차량 인·km(직선, 평일 하루) → 연간 tCO2
    g = load()
    quad = pd.read_csv(TAB / "f2_dong_quadrant.csv", dtype={"adm_cd10": str}).set_index("adm_cd10")["quadrant"]
    g["quadrant"] = g["adm_cd10"].map(quad)
    g["co2_t"] = g["pkm_car"] * to_t
    print(f"우회계수 {detour:.2f}, 배출계수 {EF:.1f} g/km → 차량 인·km 1km(평일 하루)당 연 {to_t * 1e6:.1f} g")

    # 1. 배출 지도
    total = g["co2_t"].sum()
    m1 = g.groupby(["origin", "band"], observed=True)["co2_t"].sum().unstack("origin")
    m1["합계"] = m1.sum(axis=1)
    m1["비중"] = m1["합계"] / total
    m1.loc["합계"] = m1.sum()
    m1.to_csv(TAB / "m1_co2_by_band_origin.csv", encoding="utf-8-sig")
    m2 = g.groupby(["quadrant", "origin"], observed=True)["co2_t"].sum().unstack("origin")
    m2["합계"] = m2.sum(axis=1)
    m2["비중"] = m2["합계"] / total
    m2.to_csv(TAB / "m2_co2_by_quadrant.csv", encoding="utf-8-sig")
    print(f"서울 도착 차량 이동의 연간 CO2 (모든 연령, 평일 기준) = {total:,.0f} t"); print(m1.round(3).to_string()); print(m2.round(3).to_string())

    # 2. 거리대 맞춤 격차 해소
    g["motor"] = g["cnt_car"] + g["cnt_pt"]
    g["car_share"] = g["cnt_car"] / g["motor"].where(g["motor"] > 0)
    g["km_car"] = g["pkm_car"] / g["cnt_car"].where(g["cnt_car"] > 0)
    seoul = g[g["origin"] == "서울"].groupby("band", observed=True)[["cnt_car", "motor"]].sum()
    target = (seoul["cnt_car"] / seoul["motor"]).rename("target")
    g = g.merge(target, left_on="band", right_index=True)
    g["shift"] = ((g["car_share"] - g["target"]).clip(lower=0) * g["motor"]).fillna(0)
    g["cut_t"] = g["shift"] * g["km_car"].fillna(0) * to_t
    lever = {"A 공급 확대 (선택권 1 미만 동, 서울 출발)": (g["origin"] == "서울") & g["quadrant"].isin(["공급 먼저", "이동권 우선"]),
             "B 수요관리 (선택권 1 이상 동, 서울 출발)": (g["origin"] == "서울") & g["quadrant"].isin(["수요관리 가능", "유지"]),
             "C 광역 유입 (서울 밖 출발)": g["origin"] == "서울 밖"}
    rows = []
    for name, m in lever.items():
        x = g[m]
        rows.append({"레버": name, "지금_차량CO2_t": x["co2_t"].sum(), "줄어드는_차량이동_평일": x["shift"].sum(),
                     "감축_t": x["cut_t"].sum(), "감축률": x["cut_t"].sum() / x["co2_t"].sum(),
                     **{f"감축_{b}": x.loc[x["band"] == b, "cut_t"].sum() for b in BAND_LABEL}})
    # C′ 도착지 접근 개선: 서울 밖에서 선택권 1 미만 동으로 들어오는 이동의 거리대별 차량 분담을
    #    같은 거리대에서 선택권 1 이상 동으로 들어오는 유입 이동의 분담까지 (관측된 차이에 근거)
    low = g["quadrant"].isin(["공급 먼저", "이동권 우선"])
    inflow = g["origin"] == "서울 밖"
    ref = g[inflow & ~low].groupby("band", observed=True)[["cnt_car", "motor"]].sum()
    tgt2 = (ref["cnt_car"] / ref["motor"]).rename("target_dest")
    gd = g[inflow & low].merge(tgt2, left_on="band", right_index=True)
    gd["shift2"] = ((gd["car_share"] - gd["target_dest"]).clip(lower=0) * gd["motor"]).fillna(0)
    gd["cut2"] = gd["shift2"] * gd["km_car"].fillna(0) * to_t
    rows.append({"레버": "C′ 광역 유입 중 선택권 낮은 동 도착분 (도착지 접근 개선)", "지금_차량CO2_t": gd["co2_t"].sum(),
                 "줄어드는_차량이동_평일": gd["shift2"].sum(), "감축_t": gd["cut2"].sum(), "감축률": gd["cut2"].sum() / gd["co2_t"].sum(),
                 **{f"감축_{b}": gd.loc[gd["band"] == b, "cut2"].sum() for b in BAND_LABEL}})
    xc = g[lever["C 광역 유입 (서울 밖 출발)"]]
    rows.append({"레버": "C½ 광역 유입, 격차 절반만", "지금_차량CO2_t": xc["co2_t"].sum(), "줄어드는_차량이동_평일": xc["shift"].sum() / 2,
                 "감축_t": xc["cut_t"].sum() / 2, "감축률": xc["cut_t"].sum() / 2 / xc["co2_t"].sum(),
                 **{f"감축_{b}": xc.loc[xc["band"] == b, "cut_t"].sum() / 2 for b in BAND_LABEL}})
    lv = pd.DataFrame(rows)
    seoul_kt = ef.loc[(ef["region"] == "서울") & (ef["차종"] == "승용") & (ef["year"] == 2024), "배출량_천tCO2eq"].iloc[0]
    for name, keys in [("서울 안 합계 (A+B)", [0, 1]), ("관측 근거 합계 (A+B+C′)", [0, 1, 3]), ("합계 (A+B+C½)", [0, 1, 4]), ("합계 (A+B+C)", [0, 1, 2])]:
        part = lv.loc[keys].drop(columns="레버").sum()
        part["감축률"] = part["감축_t"] / part["지금_차량CO2_t"]
        lv.loc[len(lv)] = {"레버": name, **part.to_dict()}
    lv["서울승용배출_대비"] = lv["감축_t"] / (seoul_kt * 1e3)
    lv["주행거리4.5%목표_대비"] = lv["서울승용배출_대비"] / 0.045
    lv.to_csv(TAB / "m3_carbon_levers.csv", index=False, encoding="utf-8-sig")
    print(target.round(3).to_string()); print(lv.round(3).T.to_string())
    # 4. 연결고리: 같은 거리대 안에서 고령자 이동선택권 4분위별 차량 분담 (서울 출발).
    #    4분면은 차량 분담으로도 나눈 분류라 쓰면 순환논리가 된다 → 이동선택권만으로 나눈다
    mci = gpd.read_file(PROC / "dong_mci.gpkg")[["adm_cd10", "mci_elderly"]].set_index("adm_cd10")["mci_elderly"]
    qlab = ["1분위(낮음)", "2분위", "3분위", "4분위(높음)"]
    g["mci_q"] = g["adm_cd10"].map(pd.qcut(mci, 4, labels=qlab))
    sh = g[g["origin"] == "서울"].groupby(["band", "mci_q"], observed=True)[["cnt_car", "motor"]].sum()
    sh = (sh["cnt_car"] / sh["motor"]).unstack("mci_q")[qlab]
    out = g[g["origin"] == "서울 밖"].groupby("band", observed=True)[["cnt_car", "motor"]].sum()
    sh["서울 밖 출발 (전체)"] = out["cnt_car"] / out["motor"]
    sh["서울 출발 평균"] = target
    sh["1분위÷4분위"] = sh["1분위(낮음)"] / sh["4분위(높음)"]
    sh.to_csv(TAB / "m4_car_share_by_band.csv", encoding="utf-8-sig")
    print(sh.round(3).to_string())
    # 5. 서울 밖에서 들어오는 이동도 도착 동의 선택권에 따라 차량 분담이 다른가 (도착지 쪽 접근의 효과)
    si = g[g["origin"] == "서울 밖"].groupby(["band", "mci_q"], observed=True)[["cnt_car", "motor", "co2_t"]].sum()
    si_share = (si["cnt_car"] / si["motor"]).unstack("mci_q")[qlab]
    si_share["1분위÷4분위"] = si_share["1분위(낮음)"] / si_share["4분위(높음)"]
    si_share.to_csv(TAB / "m5_inflow_car_share_by_dest_choice.csv", encoding="utf-8-sig")
    print("서울 밖 출발, 도착 동 선택권 분위별 차량 분담"); print(si_share.round(3).to_string())
    json.dump({"detour": detour, "ef": EF, "total_t": total}, open(TAB / "m_summary.json", "w"), ensure_ascii=False, indent=2)
    fig_levers(m1, sh, lv)


def fig_levers(m1: pd.DataFrame, sh: pd.DataFrame, lv: pd.DataFrame):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    x = np.arange(len(BAND_LABEL))
    m = m1.loc[BAND_LABEL]
    a1.bar(x, m["서울"] / 1e4, color="#2a78d6", label="서울에서 출발")
    a1.bar(x, m["서울 밖"] / 1e4, bottom=m["서울"] / 1e4, color="#eb6834", label="서울 밖에서 출발")
    for k, v in enumerate(m["합계"]):
        a1.text(k, v / 1e4, f"{v / m1.loc['합계', '합계']:.0%}", ha="center", va="bottom", fontsize=8)
    a1.set_xticks(x, BAND_LABEL)
    a1.set_ylabel("연간 CO2 (만 t)")
    a1.set_title("서울 도착 차량 이동의 CO2: 어디서 오는가", fontsize=10)
    a1.legend(frameon=False, fontsize=8)
    cols = {"1분위(낮음)": "#eb6834", "2분위": "#f0a07a", "3분위": "#7fb0e8", "4분위(높음)": "#2a78d6"}
    for q, c in cols.items():
        a2.plot(x, sh.loc[BAND_LABEL, q] * 100, marker="o", color=c, lw=2, label=f"선택권 {q}")
    a2.plot(x, sh.loc[BAND_LABEL, "서울 밖 출발 (전체)"] * 100, marker="s", color="#52514e", lw=1.2, ls="--", label="서울 밖 출발")
    a2.set_xticks(x, BAND_LABEL)
    a2.set_ylabel("차량 분담 (%, 차량 ÷ 차량+대중교통)")
    a2.set_title("같은 거리라도 선택권 낮은 동이 차를 더 탄다", fontsize=10)
    a2.legend(frameon=False, fontsize=7)
    sel = lv[lv["레버"].str.match(r"^(A|B|C½|C) ")].set_index("레버")
    names = ["A 공급 확대\n(선택권 낮은 동)", "B 수요관리\n(선택권 있는 동)", "C½ 광역\n(격차 절반)", "C 광역\n(상한)"]
    vals = [sel.iloc[0]["감축_t"], sel.iloc[1]["감축_t"], sel.loc["C½ 광역 유입, 격차 절반만", "감축_t"], sel.loc["C 광역 유입 (서울 밖 출발)", "감축_t"]]
    a3.barh(np.arange(4)[::-1], np.array(vals) / 1e4, color=["#2a78d6", "#1baf7a", "#f0a07a", "#eb6834"])
    for k, v in zip(np.arange(4)[::-1], vals):
        a3.text(v / 1e4, k, f" {v / 1e4:,.1f}만 t", va="center", fontsize=8)
    a3.set_yticks(np.arange(5)[::-1], names, fontsize=8)
    a3.set_xlabel("연간 감축 (만 tCO2eq, 목표형)")
    a3.set_title("거리대를 맞춘 격차 해소의 감축 여지", fontsize=10)
    a3.set_xlim(0, max(vals) / 1e4 * 1.3)
    for a in (a1, a2, a3):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "m1_carbon_levers.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
