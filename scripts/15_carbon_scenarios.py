"""E. 이동 격차를 줄이는 정책의 탄소 감축 시나리오.

실행: python scripts/15_carbon_scenarios.py   (14_build_carbon_inputs.py 다음, 약 10초)

연간 감축량 = 줄어드는 차량 이동(인·통행/평일) × 차량 이동 거리(km) ÷ 재차인원 × 승용차 배출계수(g/km) × 연간 평일 수
- 차량 이동: A의 KT 생활이동 2026년 9월 평일 평균 (dong_usage.gpkg, 70대 이상 / 20~50대, 도착 동 기준)
- 거리: 14번의 OA-22657 차량 이동 직선거리 × 우회계수. 기본은 '서울에서 출발한 이동만 바뀐다'(보수적)
- 고령자 거리: 60세 이상 ÷ 전체 평균 이동거리 비율(모든 수단, 14번)을 70대 이상 이동에 곱한다

시나리오
- S1 1순위 72개 동(C): 마을버스·수요응답형 버스로 B 고령자 지수를 '대중교통 양호 서민 주거지' 유형 평균까지 올린다.
     차량 분담 변화 = (B 고령자 지수 → 차량 분담) 기울기 × 지수 상승분 (목표 = 그 유형의 중앙값, C 표와 같음). 기울기는 427개 동 회귀로 구한다
     (기본: 소득·밀도·도심거리 통제, 비교: 통제 없음).
- S2 외곽 차 중심 신주거지 77개 동(C) 중 S1과 겹치는 암사1동을 뺀 76개 동: 차량 분담을 서울 평균까지 낮춘다고 놓는
     목표형(what-if) 계산.
- 두 시나리오 모두 '고령자만'과 '동네 전체(20~50대 포함)'를 따로 낸다. 10대·60대는 A 자료에 없어 넣지 않는다.
- 추가 버스 운행의 배출은 빼지 않고, 감축량이 상쇄되는 '손익분기 버스 운행거리'로 따로 보인다.
- D(날씨)는 탄소와 섞지 않는다. 한파·큰비 날 사라지는 고령자 이동은 이동권 효과로 별도 표에 둔다.
"""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

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

AGES = {"70p": "70대 이상", "adult": "20~50대"}
GOOD_TYPE = "대중교통 양호 서민 주거지"
CAR_TYPE = "외곽 차 중심 신주거지"
CONTROLS = "welfare_65p_rate_z + log_apt_price_z + pop_density_z + dist_cityhall_km_z"  # A 우선지역 견고성과 같은 통제

# 기본값과 민감도 범위. 출처: e1_emission_factors.csv(배출계수), e3_occupancy_check.csv(재차인원 점검)
BASE = {"ef": None,           # 서울 2024 승용 g/km (e1 에서 읽는다)
        "occupancy": 1.3,     # 재차인원 (근거 확인 필요) — 공식 서울 값 못 찾음
        "detour": 1.3,        # 직선 → 도로 거리 우회계수 (근거 확인 필요) — B 보행거리와 같은 값
        "weekdays": 250,      # 연간 평일 수 (주말·공휴일 제외, 보수적)
        "scope": "seoul_origin",  # 서울 출발 이동만 바뀐다
        "elderly_dist": True,     # 70대 이상 거리에 60세 이상 거리비를 곱한다
        "slope": "controlled"}    # 통제 회귀 기울기


def load() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    d = gpd.read_file(PROC / "dong_usage.gpkg")
    cl = pd.read_csv(TAB / "c4_dong_clusters.csv", dtype={"adm_cd10": str})[["adm_cd10", "type_name"]]
    p1 = pd.read_csv(TAB / "c8_priority1_targets.csv", dtype={"adm_cd10": str})
    p1 = p1.loc[p1["priority1"].str.startswith(("확정", "추가")), "adm_cd10"]
    dist = pd.read_parquet(PROC / "car_dist_dong.parquet")
    d = d.merge(cl, on="adm_cd10").merge(dist, left_on="adm_cd10", right_index=True)
    d["priority1"] = d["adm_cd10"].isin(p1)
    d["seoul_origin_share"] = d["car_trips_seoul_origin"] / d["car_trips_od"]
    z = lambda s: (s - s.mean()) / s.std()
    for c in ["welfare_65p_rate", "log_apt_price", "pop_density", "dist_cityhall_km"]:
        d[c + "_z"] = z(d[c])
    ef = pd.read_csv(TAB / "e1_emission_factors.csv")
    return d, ef


def slopes(d: pd.DataFrame) -> pd.DataFrame:
    """B 고령자 지수 1 상승당 차량 분담 변화 (427개 동, HC1)."""
    rows = []
    for age, name in AGES.items():
        for kind, f in [("simple", "mci_elderly"), ("controlled", "mci_elderly + " + CONTROLS)]:
            m = smf.ols(f"car_share_{age} ~ {f}", d).fit(cov_type="HC1")
            lo, hi = m.conf_int().loc["mci_elderly"]
            rows.append({"age": age, "연령": name, "kind": kind, "기울기": m.params["mci_elderly"],
                         "95%하한": lo, "95%상한": hi, "p": m.pvalues["mci_elderly"], "R2": m.rsquared, "동수": int(m.nobs)})
    return pd.DataFrame(rows)


def s2_mask(d: pd.DataFrame) -> pd.Series:
    """S2 대상: 외곽 차 중심 신주거지 중 S1(1순위)에 이미 들어간 동(암사1동)을 뺀다 → 이중 계산 방지."""
    return (d["type_name"] == CAR_TYPE) & ~d["priority1"]


def shifted_trips(d: pd.DataFrame, sl: pd.DataFrame, slope_kind: str, slope_pick: str = "기울기") -> pd.DataFrame:
    """동 × 연령별로 대중교통으로 옮겨 가는 차량 이동(인·통행/평일)."""
    target_mci = d.loc[d["type_name"] == GOOD_TYPE, "mci_elderly"].median()  # C 유형 프로필(c3)과 같은 중앙값
    out = pd.DataFrame(index=d.index)
    for age in AGES:
        motor, car, share = d[f"motor_trips_{age}"], d[f"trips_{age}_car"], d[f"car_share_{age}"]
        b = sl.set_index(["age", "kind"]).loc[(age, slope_kind), slope_pick]
        d_mci = (target_mci - d["mci_elderly"]).clip(lower=0)
        s1 = (-b * d_mci * motor).clip(lower=0).clip(upper=car).where(d["priority1"], 0)
        city = np.average(share, weights=motor)
        s2 = ((share - city).clip(lower=0) * motor).where(s2_mask(d), 0)
        out[f"S1_{age}"], out[f"S2_{age}"] = s1, s2
    out.attrs["target_mci"] = target_mci
    return out


def tco2_per_year(d: pd.DataFrame, trips: pd.DataFrame, p: dict, elderly_ratio: float) -> pd.DataFrame:
    """동 × (시나리오, 연령) 연간 감축량 (tCO2eq)."""
    if p["scope"] == "seoul_origin":
        n_share, km = d["seoul_origin_share"], d["car_km_mean_seoul_origin"]
    else:
        n_share, km = 1.0, d["car_km_mean"]
    out = pd.DataFrame(index=d.index)
    for col in trips.columns:
        age = col.split("_")[1]
        k = km * (elderly_ratio if (age == "70p" and p["elderly_dist"]) else 1.0)
        vkm = trips[col] * n_share * k * p["detour"] / p["occupancy"]
        out[col] = vkm * p["ef"] * p["weekdays"] / 1e6  # g → t
    return out


def totals(t: pd.DataFrame) -> dict:
    return {"S1_고령자만": t["S1_70p"].sum(), "S1_동네전체": t[["S1_70p", "S1_adult"]].sum().sum(),
            "S2_고령자만": t["S2_70p"].sum(), "S2_동네전체": t[["S2_70p", "S2_adult"]].sum().sum(),
            "합계_동네전체": t.sum().sum()}


def run(d, sl, p, elderly_ratio, slope_pick="기울기"):
    trips = shifted_trips(d, sl, p["slope"], slope_pick)
    return trips, tco2_per_year(d, trips, p, elderly_ratio)


def sensitivity(d, sl, ef, elderly_ratio) -> pd.DataFrame:
    """한 번에 하나씩 바꾼다 + 전부 낮게 / 전부 높게."""
    seoul = ef[ef["region"] == "서울"]
    ef_hi = seoul.loc[(seoul["차종"] == "승용"), "g_per_km"].max()
    ef_lo = ef.loc[(ef["region"] != "서울") & (ef["차종"] == "승용") & (ef["year"] == 2024), "g_per_km"].iloc[0]
    cases = [
        ("기본값", {}, "기울기"),
        ("배출계수 낮게: 전국 2024 승용", {"ef": ef_lo}, "기울기"),
        ("배출계수 높게: 서울 2021 승용", {"ef": ef_hi}, "기울기"),
        ("재차인원 1.0 (운전자 혼자)", {"occupancy": 1.0}, "기울기"),
        ("재차인원 1.6", {"occupancy": 1.6}, "기울기"),
        ("우회계수 1.0 (직선거리 그대로)", {"detour": 1.0}, "기울기"),
        ("우회계수 1.4", {"detour": 1.4}, "기울기"),
        ("기울기 95% 구간 중 효과 작은 끝", {}, "95%상한"),
        ("기울기 95% 구간 중 효과 큰 끝", {}, "95%하한"),
        ("기울기: 통제 없음", {"slope": "simple"}, "기울기"),
        ("서울 밖 출발 이동까지 바뀐다", {"scope": "all"}, "기울기"),
        ("고령자 거리 보정 안 함", {"elderly_dist": False}, "기울기"),
        ("전부 낮게", {"ef": ef_lo, "occupancy": 1.6, "detour": 1.0}, "95%상한"),
        ("전부 높게", {"ef": ef_hi, "occupancy": 1.0, "detour": 1.4, "scope": "all", "elderly_dist": False}, "95%하한"),
    ]
    rows = []
    for name, change, pick in cases:
        p = {**BASE, **change}
        _, t = run(d, sl, p, elderly_ratio, pick)
        rows.append({"경우": name, **totals(t)})
    return pd.DataFrame(rows)


def weather_side(d: pd.DataFrame, p: dict, elderly_ratio: float) -> pd.DataFrame:
    """D와 연결: 한파·큰비 날 사라지는 70대 이상 이동 (이동권 효과, 탄소와 합치지 않음)."""
    kt = pd.read_parquet(PROC / "kt_daily_type_mode.parquet")
    kt = kt[(kt["age"] == "70p") & (kt["date"].dt.weekday < 5) & kt["date"].dt.month.isin([12, 1, 2])]
    base = kt.groupby("date")["trips"].sum().mean()  # 겨울 평일 70대 이상 서울 도착 이동 (모든 수단)
    d8 = pd.read_csv(TAB / "d8_kt_modes.csv")
    d8 = d8[(d8["연령"] == "70대 이상") & (d8["수단"] == "전체 이동") & d8["날씨"].isin(["한파", "큰비"])]
    # 사라진 이동을 차(택시 등)로 되살리면 생기는 배출: 고령자 차량 이동 1건의 평균 배출 (서울 출발 거리, 기본값)
    km = np.average(d["car_km_mean_seoul_origin"], weights=d["trips_70p_car"]) * elderly_ratio * p["detour"]
    g_per_trip = km / p["occupancy"] * p["ef"]
    rows = []
    for _, r in d8.iterrows():
        lost = -r["변화(%)"] / 100 * base
        rows.append({"날씨": r["날씨"], "70대이상_이동_변화(%)": r["변화(%)"], "겨울평일_70대이상_이동": base,
                     "하루_사라지는_이동": lost, "대중교통·수요응답형으로_되살릴때_추가_승용배출_t": 0.0,
                     "전부_차로_되살릴때_추가배출_t(참고)": lost * g_per_trip / 1e6})
    return pd.DataFrame(rows)


def fig_scenarios(sens: pd.DataFrame, n1: int, n2: int, path: Path):
    base = sens.set_index("경우").loc["기본값"]
    lo, hi = sens.set_index("경우").loc["전부 낮게"], sens.set_index("경우").loc["전부 높게"]
    labels = [f"S1 1순위 {n1}개 동\n(접근 개선, 추정)", f"S2 외곽 차 중심 {n2}개 동\n(서울 평균까지, 목표형)"]
    old = [base["S1_고령자만"], base["S2_고령자만"]]
    all_ = [base["S1_동네전체"], base["S2_동네전체"]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(2)[::-1]
    ax.barh(y, old, height=0.5, color="#2a78d6", label="70대 이상")
    ax.barh(y, np.array(all_) - np.array(old), left=old, height=0.5, color="#eb6834", label="20~50대 (같은 동네)")
    for i, k in enumerate(["S1_동네전체", "S2_동네전체"]):
        ax.plot([lo[k], hi[k]], [y[i] - 0.36] * 2, color="#52514e", lw=1.2)
        for x in (lo[k], hi[k]):
            ax.plot([x, x], [y[i] - 0.40, y[i] - 0.32], color="#52514e", lw=1.2)
        ax.text(all_[i] + max(all_) * 0.02, y[i], f"{all_[i]:,.0f} t", va="center", fontsize=10)
        ax.text(lo[k], y[i] - 0.47, f"범위 {lo[k]:,.0f} ~ {hi[k]:,.0f} t", va="top", fontsize=8, color="#52514e")
    ax.set_yticks(y, labels)
    ax.set_xlabel("연간 감축량 (tCO2eq, 평일만)")
    ax.set_title("E. 이동 격차 정책의 연간 탄소 감축 (기본값, 막대 아래 선 = 전부 낮게~전부 높게)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e4df", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.75, 1.4)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_tornado(sens: pd.DataFrame, path: Path):
    s = sens.set_index("경우")["합계_동네전체"]
    base = s["기본값"]
    pairs = [("배출계수", "배출계수 낮게: 전국 2024 승용", "배출계수 높게: 서울 2021 승용"),
             ("재차인원 1.6 ~ 1.0", "재차인원 1.6", "재차인원 1.0 (운전자 혼자)"),
             ("우회계수 1.0 ~ 1.4", "우회계수 1.0 (직선거리 그대로)", "우회계수 1.4"),
             ("S1 기울기 95% 구간", "기울기 95% 구간 중 효과 작은 끝", "기울기 95% 구간 중 효과 큰 끝"),
             ("S1 기울기: 통제 없음", "기본값", "기울기: 통제 없음"),
             ("서울 밖 출발 이동 포함", "기본값", "서울 밖 출발 이동까지 바뀐다"),
             ("고령자 거리 보정 안 함", "기본값", "고령자 거리 보정 안 함")]
    rows = sorted([(n, s[a], s[b]) for n, a, b in pairs], key=lambda r: abs(r[2] - r[1]))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (n, a, b) in enumerate(rows):
        lo, hi = min(a, b), max(a, b)
        ax.barh(i, hi - lo, left=lo, height=0.55, color="#2a78d6")
        ax.text(hi, i, f"  {lo:,.0f}~{hi:,.0f}", va="center", fontsize=8, color="#52514e")
    ax.axvline(base, color="#0b0b0b", lw=1)
    ax.text(base, -0.75, f" 기본값 {base:,.0f} t", ha="left", va="center", fontsize=9)
    ax.set_ylim(-1, len(rows) - 0.5)
    ax.set_xlim(left=0)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("S1+S2 동네 전체 연간 감축량 (tCO2eq)")
    ax.set_title("가정 하나씩 바꿀 때 감축량이 움직이는 폭", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_map(d: gpd.GeoDataFrame, t: pd.DataFrame, path: Path):
    g = d[["geometry"]].assign(t=t.sum(axis=1).values, area_km2=d["area_km2"].values)
    fig, ax = plt.subplots(figsize=(7, 5.6))
    g.plot(ax=ax, color="#f0efec", edgecolor="white", lw=0.3)
    s = g[g["t"] > 0]
    s.plot(ax=ax, column="t", cmap="Blues", edgecolor="white", lw=0.3, legend=True,
           legend_kwds={"label": "연간 감축량 (tCO2eq, 동네 전체, 기본값)", "shrink": 0.6})
    d[d["priority1"]].boundary.plot(ax=ax, color="#0b0b0b", lw=0.5)
    ax.set_axis_off()
    ax.set_title(f"동별 감축량 (검은 테두리 = S1 1순위 {int(d['priority1'].sum())}개 동, "
                 f"나머지 색칠 = S2 외곽 차 중심 {int(s2_mask(d).sum())}개 동)", fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d, ef = load()
    BASE["ef"] = ef.loc[(ef["region"] == "서울") & (ef["차종"] == "승용") & (ef["year"] == 2024), "g_per_km"].iloc[0]
    bus_ef = ef.loc[(ef["region"] == "서울") & (ef["차종"] == "승합") & (ef["year"] == 2024), "g_per_km"].iloc[0]
    elderly_ratio = pd.read_csv(TAB / "e3_occupancy_check.csv")["60세이상_거리비(서울도착, 모든 수단)"].iloc[0]

    sl = slopes(d)
    sl.to_csv(TAB / "e4_slopes.csv", index=False, encoding="utf-8-sig")
    print(sl.round(4).to_string())

    trips, t = run(d, sl, BASE, elderly_ratio)
    print(f"S1 목표 B 고령자 지수 = {trips.attrs['target_mci']:.3f} ('{GOOD_TYPE}' 중앙값), 1순위 72개 동 중앙값 = {d.loc[d['priority1'], 'mci_elderly'].median():.3f}")

    # 시나리오 요약 (기본값)
    rows = []
    for sc, mask in [("S1", d["priority1"]), ("S2", s2_mask(d))]:
        for age, name in AGES.items():
            col = f"{sc}_{age}"
            car = d.loc[mask, f"trips_{age}_car"].sum()
            motor = d.loc[mask, f"motor_trips_{age}"].sum()
            rows.append({"시나리오": sc, "연령": name, "동수": int(mask.sum()),
                         "차량이동_평일": car, "차량분담_현재": car / motor,
                         "차량분담_시나리오": (car - trips.loc[mask, col].sum()) / motor,
                         "줄어드는_차량이동_평일": trips.loc[mask, col].sum(),
                         "그중_서울출발": (trips.loc[mask, col] * d.loc[mask, "seoul_origin_share"]).sum(),
                         "연간감축_tCO2eq": t.loc[mask, col].sum()})
    summ = pd.DataFrame(rows)
    city_kt = ef.loc[(ef["region"] == "서울") & (ef["차종"] == "승용") & (ef["year"] == 2024), "배출량_천tCO2eq"].iloc[0]
    summ["서울_승용배출_대비(%)"] = summ["연간감축_tCO2eq"] / (city_kt * 1e3) * 100
    # 손익분기: 이만큼의 버스 운행(하루 km)을 새로 늘려도 감축이 남는다 (승합차 배출계수 기준)
    summ["손익분기_추가버스_km_평일하루"] = summ["연간감축_tCO2eq"] * 1e6 / BASE["weekdays"] / bus_ef
    summ.to_csv(TAB / "e5_scenarios.csv", index=False, encoding="utf-8-sig")
    print(summ.round(3).to_string())

    per_dong = d[["adm_cd10", "gu", "dong", "type_name", "priority1", "mci_elderly", "car_share_70p", "car_share_adult",
                  "seoul_origin_share", "car_km_mean_seoul_origin"]].join(trips.add_prefix("trips_")).join(t.add_prefix("tco2_"))
    per_dong["tco2_total"] = t.sum(axis=1)
    per_dong["breakeven_bus_km_day"] = per_dong["tco2_total"] * 1e6 / BASE["weekdays"] / bus_ef
    per_dong[per_dong["tco2_total"] > 0].sort_values("tco2_total", ascending=False).to_csv(
        TAB / "e6_dong_reduction.csv", index=False, encoding="utf-8-sig")

    sens = sensitivity(d, sl, ef, elderly_ratio)
    sens.to_csv(TAB / "e7_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(sens.round(0).to_string())

    w = weather_side(d, BASE, elderly_ratio)
    w.to_csv(TAB / "e8_weather_mobility.csv", index=False, encoding="utf-8-sig")
    print(w.round(1).to_string())

    fig_scenarios(sens, int(d["priority1"].sum()), int(s2_mask(d).sum()), FIG / "e1_scenarios.png")
    fig_tornado(sens, FIG / "e2_sensitivity.png")
    fig_map(d, t, FIG / "e3_reduction_map.png")

    base = sens.set_index("경우").loc["기본값"]
    summary = {"배출계수_g_per_km": {"승용_서울2024": BASE["ef"], "승합_서울2024": bus_ef},
               "기본가정": {k: v for k, v in BASE.items() if k != "ef"}, "고령자_거리비": elderly_ratio,
               "S1_목표_B고령자지수": trips.attrs["target_mci"],
               "연간감축_tCO2eq_기본": base.to_dict(),
               "연간감축_범위": {"전부 낮게": sens.set_index("경우").loc["전부 낮게"].to_dict(),
                            "전부 높게": sens.set_index("경우").loc["전부 높게"].to_dict()},
               "서울_승용배출_2024_천t": city_kt}
    (TAB / "e_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print(f"저장: {TAB}/e*.csv, {FIG}/e*.png")


if __name__ == "__main__":
    main()
