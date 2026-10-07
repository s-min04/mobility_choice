"""F. 이동선택권을 정책 판별 도구로 쓴다: 4분면 분류, 현행 정책의 대상 적합성 진단.

실행: python scripts/16_policy_diagnosis.py   (15_carbon_scenarios.py 다음, 약 10초)

1. 판별 도구 (행정동 4분면)
   - 가로: B 고령자 이동선택권 < 1 이면 '선택권 낮음' (B의 사각지대 정의: 제대로 쓸 수 있는 대중교통 수단이 1개 미만)
   - 세로: 동네 전체(70대 이상 + 20~50대) 차량 분담이 서울 평균보다 높으면 '차량 높음'
   - 공급 먼저(낮음·높음) / 이동권 우선(낮음·낮음) / 수요관리 가능(높음·높음) / 유지(높음·낮음)
   - '수요관리 가능' 동에 대해 E와 같은 방법으로 '차량 분담을 서울 평균까지' 목표형 감축량(S3)을 계산한다.
2. 요금 정책의 혜택이 어디로 가는가 (정적 귀착)
   - 65세 이상 지하철 무임(현행), 70세 이상 시내·마을버스 무료(2026.6 조례, 시행 전)
   - 혜택 ∝ 지금의 70대 이상 지하철·버스 이동 (KT 생활이동, 도착 동 기준) ÷ 70세 이상 인구
   - 도착 기준이라 방문객이 많은 동을 빼거나 거주자 몫으로 보정한 값도 함께 낸다
   - 조례 예산(2027년 1,047억 원, 보도 수치·원문 확인 필요)을 지금 버스 이용 비율로 나눴을 때 각 집단 몫
3. 장소 기반 현행 정책이 판별 도구의 어디에 있는가
   - 녹색교통지역(한양도성 15개 동, 5등급 차량 운행 제한), 서울 첫 자치구 수요응답형 버스(동작CALL버스, 사당3·4동)
"""

import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"

for font in ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
    if any(font == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = font
        break
plt.rcParams["axes.unicode_minus"] = False

spec = importlib.util.spec_from_file_location("e15", ROOT / "scripts" / "15_carbon_scenarios.py")
e15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e15)

MCI_LOW = 1.0
QUAD = {(True, True): "공급 먼저", (True, False): "이동권 우선", (False, True): "수요관리 가능", (False, False): "유지"}
QUAD_COLOR = {"공급 먼저": "#2a78d6", "이동권 우선": "#eb6834", "수요관리 가능": "#1baf7a", "유지": "#c9c8c2"}
GREEN_ZONE = {"종로구": ["청운효자동", "사직동", "삼청동", "가회동", "종로1·2·3·4가동", "종로5·6가동", "이화동", "혜화동"],
              "중구": ["소공동", "회현동", "명동", "필동", "장충동", "광희동", "을지로동"]}
DRT_DONGJAK = {"동작구": ["사당3동", "사당4동"]}
SENIOR_BUS_BUDGET_2027 = 1047.2  # 억 원, 서울시 추계(머니투데이 2026.6.16 보도). 조례 원문·추계서 확인 필요


def in_list(d: pd.DataFrame, spec: dict) -> pd.Series:
    return pd.Series(False, index=d.index) | np.logical_or.reduce(
        [(d["gu"] == gu) & d["dong"].isin(ds) for gu, ds in spec.items()])


def load() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    d, ef = e15.load()
    e6 = pd.read_csv(TAB / "e6_dong_reduction.csv", dtype={"adm_cd10": str})[["adm_cd10", "tco2_total"]]
    d = d.merge(e6, on="adm_cd10", how="left").fillna({"tco2_total": 0.0})
    motor = d["motor_trips_70p"] + d["motor_trips_adult"]
    d["car_share_all"] = (d["trips_70p_car"] + d["trips_adult_car"]) / motor
    city = (d["trips_70p_car"] + d["trips_adult_car"]).sum() / motor.sum()
    d["choice_low"] = d["mci_elderly"] < MCI_LOW
    d["car_high"] = d["car_share_all"] > city
    d["quadrant"] = [QUAD[(a, b)] for a, b in zip(d["choice_low"], d["car_high"])]
    d.attrs["city_car_share_all"] = city
    return d, ef


def s3_demand_management(d: pd.DataFrame, ef: pd.DataFrame) -> pd.Series:
    """'수요관리 가능' 동의 차량 분담을 서울 평균까지 (E의 S2와 같은 방식, 기본 가정). 동별 연간 tCO2eq."""
    p = {**e15.BASE, "ef": ef.loc[(ef["region"] == "서울") & (ef["차종"] == "승용") & (ef["year"] == 2024), "g_per_km"].iloc[0]}
    ratio = pd.read_csv(TAB / "e3_occupancy_check.csv")["60세이상_거리비(서울도착, 모든 수단)"].iloc[0]
    mask = d["quadrant"] == "수요관리 가능"
    trips = pd.DataFrame(index=d.index)
    for age in e15.AGES:
        motor, share = d[f"motor_trips_{age}"], d[f"car_share_{age}"]
        city = np.average(share, weights=motor)
        trips[f"S3_{age}"] = ((share - city).clip(lower=0) * motor).where(mask, 0)
    return e15.tco2_per_year(d, trips, p, ratio).sum(axis=1)


def quadrant_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("quadrant")
    t = pd.DataFrame({
        "동수": g.size(), "65세이상": g["pop_65_plus"].sum(),
        "B고령자지수_중앙값": g["mci_elderly"].median(),
        "고령자_차량분담": g["trips_70p_car"].sum() / g["motor_trips_70p"].sum(),
        "동네전체_차량분담": g.apply(lambda x: (x["trips_70p_car"] + x["trips_adult_car"]).sum()
                                 / (x["motor_trips_70p"] + x["motor_trips_adult"]).sum()),
        "고령자_외출(1인당 하루)": g["home_trips_70p"].sum() / g["pop_70_plus"].sum(),
        "C_1순위_동수": g["priority1"].sum(),
        "E_S1S2_감축_t": g["tco2_total"].sum(), "S3_수요관리_감축_t": g["tco2_s3"].sum()})
    t["65세이상_비중"] = t["65세이상"] / t["65세이상"].sum()
    order = ["공급 먼저", "이동권 우선", "수요관리 가능", "유지"]
    return t.loc[order]


def fare_incidence(d: pd.DataFrame) -> pd.DataFrame:
    """지하철 무임(65+, 현행)·버스 무료(70+, 조례)의 혜택이 누구에게 가는가. 혜택 = 지금 이동 수 ÷ 70세 이상 인구.

    KT 이동은 도착 동 기준이라 방문객이 많은 동(도심)은 1인당 값이 부풀려진다. 그래서 세 가지로 계산한다.
    - 전체: 427개 동
    - 방문 많은 동 제외 (기본): 70대 이상 도착 이동 ÷ 70세 이상 인구가 상위 20%인 동을 뺀다
    - 거주자 몫 보정: 수단별 도착 이동 × (귀가 이동 ÷ 전체 도착 이동). 귀가 비율은 모든 수단을 합친 값이다
    """
    total = d[[c for c in d.columns if c.startswith("trips_70p_") and c != "trips_70p_walk"] + ["trips_70p_walk"]].sum(axis=1)
    visit = total / d["pop_70_plus"]
    home = d["home_trips_70p"] / total
    q = pd.qcut(d["mci_elderly"], 4, labels=["1분위(낮음)", "2분위", "3분위", "4분위(높음)"])
    versions = {"방문 많은 동 제외 (기본)": (visit < visit.quantile(0.8), 1.0), "전체 427개 동": (pd.Series(True, index=d.index), 1.0),
                "거주자 몫 보정": (pd.Series(True, index=d.index), home)}
    rows = []
    for ver, (keep, w) in versions.items():
        x = d[keep].assign(sub=(d["trips_70p_subway"] * w)[keep], bus=(d["trips_70p_bus"] * w)[keep])
        sets = [(f"B 고령자 지수 {k}", q[keep] == k) for k in q.cat.categories] + \
               [("C 1순위 72개 동", x["priority1"]), ("나머지 동", ~x["priority1"])]
        for name, m in sets:
            y = x[m]
            pop = y["pop_70_plus"].sum()
            rows.append({"계산": ver, "집단": name, "동수": int(m.sum()), "70세이상_비중": pop / x["pop_70_plus"].sum(),
                         "B고령자지수_중앙값": y["mci_elderly"].median(),
                         "지하철_1인당하루": y["sub"].sum() / pop, "버스_1인당하루": y["bus"].sum() / pop,
                         "지하철혜택_비중": y["sub"].sum() / x["sub"].sum(), "버스혜택_비중": y["bus"].sum() / x["bus"].sum()})
    t = pd.DataFrame(rows)
    # 조례 예산을 '지금 버스 이용' 비율로 나눴을 때 vs 인구 비율로 나눴을 때 (정적 귀착, 참고)
    t["버스무료_2027예산_몫_억원(추정)"] = t["버스혜택_비중"] * SENIOR_BUS_BUDGET_2027
    t["인구비례_몫_억원"] = t["70세이상_비중"] * SENIOR_BUS_BUDGET_2027
    return t


def place_policies(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, spec in [("녹색교통지역 (한양도성 15개 동)", GREEN_ZONE), ("동작CALL버스 (사당3·4동)", DRT_DONGJAK)]:
        m = in_list(d, spec)
        for _, r in d[m].iterrows():
            rows.append({"정책": name, "구": r["gu"], "동": r["dong"], "판별": r["quadrant"],
                         "B고령자지수": r["mci_elderly"], "고령자지수_순위(낮은순, 427개 중)": int(d["mci_elderly"].rank()[_]),
                         "동네전체_차량분담": r["car_share_all"], "C유형": r["type_name"], "C_1순위": r["priority1"]})
        missing = sum(len(v) for v in spec.values()) - m.sum()
        if missing:
            print(f"  [주의] {name}: 동 이름이 맞지 않는 {missing}개 동")
    return pd.DataFrame(rows)


def fig_tool(d: gpd.GeoDataFrame, path: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})
    city = d.attrs["city_car_share_all"]
    for qn, c in QUAD_COLOR.items():
        x = d[d["quadrant"] == qn]
        a1.scatter(x["mci_elderly"], x["car_share_all"] * 100, s=np.sqrt(x["pop_65_plus"]) / 4, color=c,
                   edgecolor="white", lw=0.5, label=f"{qn} ({len(x)}개 동)")
    a1.axvline(MCI_LOW, color="#52514e", lw=1, ls="--")
    a1.axhline(city * 100, color="#52514e", lw=1, ls="--")
    a1.text(MCI_LOW, a1.get_ylim()[1], " 선택권 1개", va="top", fontsize=8, color="#52514e")
    a1.text(a1.get_xlim()[1], city * 100, f"서울 평균 {city:.0%} ", ha="right", va="bottom", fontsize=8, color="#52514e")
    a1.set_xlabel("B 고령자 이동선택권 (걸어서 쓸 수 있는 대중교통 수단 수)")
    a1.set_ylabel("동네 전체 차량 분담 (%, 70대 이상 + 20~50대)")
    a1.spines[["top", "right"]].set_visible(False)
    a1.legend(frameon=False, fontsize=8, loc="upper right")
    a1.set_title("이동선택권 판별 도구: 행정동 4분면 (점 크기 = 65세 이상 인구)", fontsize=10)
    d.plot(ax=a2, color=[QUAD_COLOR[q] for q in d["quadrant"]], edgecolor="white", lw=0.3)
    for spec, ls in [(GREEN_ZONE, "-"), (DRT_DONGJAK, "-")]:
        d[in_list(d, spec)].boundary.plot(ax=a2, color="#0b0b0b", lw=1.0, linestyle=ls)
    a2.set_axis_off()
    a2.set_title("판별 결과 지도 (검은 테두리: 녹색교통지역 15개 동, 동작CALL버스 2개 동)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_incidence(t: pd.DataFrame, path: Path):
    x = t[(t["계산"] == "방문 많은 동 제외 (기본)") & t["집단"].str.startswith("B ")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    i = np.arange(len(x))
    w = 0.38
    ax.bar(i - w / 2, x["지하철_1인당하루"], w, color="#2a78d6", label="지하철 (65세 이상 무임, 현행)")
    ax.bar(i + w / 2, x["버스_1인당하루"], w, color="#eb6834", label="버스 (70세 이상 무료 조례, 시행 전)")
    for k, (s, b) in enumerate(zip(x["지하철_1인당하루"], x["버스_1인당하루"])):
        ax.text(k - w / 2, s, f"{s:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(k + w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(i, [s.replace("B 고령자 지수 ", "") for s in x["집단"]])
    ax.set_xlabel("B 고령자 이동선택권 4분위 (행정동)")
    ax.set_ylabel("70대 이상 1인당 하루 이동")
    ax.set_title("무임 지하철 혜택은 선택권 높은 동에 몰리고, 버스 무료는 선택권 낮은 동에 더 가지 않는다\n"
                 "(KT 2026.9 평일, 방문 많은 동 상위 20% 제외)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    d, ef = load()
    d["tco2_s3"] = s3_demand_management(d, ef)
    print(f"서울 동네 전체 차량 분담(70대 이상+20~50대) = {d.attrs['city_car_share_all']:.3f}")

    qt = quadrant_table(d)
    qt.to_csv(TAB / "f1_quadrants.csv", encoding="utf-8-sig")
    print(qt.round(3).to_string())
    d[["adm_cd10", "gu", "dong", "type_name", "priority1", "mci_elderly", "car_share_70p", "car_share_all",
       "out_rate_70p", "quadrant", "tco2_total", "tco2_s3"]].to_csv(TAB / "f2_dong_quadrant.csv", index=False,
                                                                     encoding="utf-8-sig")

    fi = fare_incidence(d)
    fi.to_csv(TAB / "f3_fare_incidence.csv", index=False, encoding="utf-8-sig")
    print(fi.round(3).to_string())

    pp = place_policies(d)
    pp.to_csv(TAB / "f4_place_policies.csv", index=False, encoding="utf-8-sig")
    print(pp.round(3).to_string())

    # E 감축량을 주행거리로: 2030 총 주행거리 4.5% 감축 목표(제1차 탄소중립·녹색성장 기본계획)와 비교
    s = json.loads((TAB / "e_summary.json").read_text())
    ef_car = s["배출계수_g_per_km"]["승용_서울2024"]
    vkm_seoul = ef.loc[(ef["region"] == "서울") & (ef["차종"] == "승용") & (ef["year"] == 2024), "주행거리_백만km"].iloc[0]
    rows = []
    for name, t in [("E S1+S2 (기본)", s["연간감축_tCO2eq_기본"]["합계_동네전체"]),
                    ("F S3 수요관리 가능 동 (기본)", d["tco2_s3"].sum()),
                    ("S1+S2+S3", s["연간감축_tCO2eq_기본"]["합계_동네전체"] + d["tco2_s3"].sum())]:
        vkm = t * 1e6 / ef_car / 1e6  # 백만 km
        rows.append({"묶음": name, "연간감축_tCO2eq": t, "줄어드는_승용주행_백만km": vkm,
                     "서울_승용주행_대비(%)": vkm / vkm_seoul * 100,
                     "4.5%목표_대비_몫(%)": vkm / vkm_seoul * 100 / 4.5 * 100})
    vk = pd.DataFrame(rows)
    vk.to_csv(TAB / "f5_vkt_share.csv", index=False, encoding="utf-8-sig")
    print(vk.round(3).to_string())

    fig_tool(d, FIG / "f1_choice_tool.png")
    fig_incidence(fi, FIG / "f2_fare_incidence.png")
    print(f"저장: {TAB}/f*.csv, {FIG}/f*.png")


if __name__ == "__main__":
    main()
