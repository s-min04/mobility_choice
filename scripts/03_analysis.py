"""이동선택권 지수로 사각지대를 찾고, 결과 표와 그림을 만든다.

실행: python scripts/03_analysis.py   (02_build_index.py 를 먼저 실행)

1. 격차 분해: 일반 성인 → 고령자 보행속도 → 따릉이 연령 가중치 → 택시 앱 호출 비율
2. 사각지대 고령인구: 실제로 쓸 수 있는 수단이 2개 미만(고를 선택지가 없는 곳)에 사는 65세 이상
3. 공간 군집: 전역 Moran's I, 국지적 Moran(LISA), 이변량 LISA(고령인구 비율 × 이동선택권)
4. 민감도: 보행속도·보행시간·버스 기준 노선 수·택시 가정을 바꿨을 때 사각지대 고령인구 범위
5. 정책 시나리오: 전화 호출 택시(동행 온다 콜택시)가 앱을 못 쓰는 고령자에게 닿는 정도별 효과
"""

import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_BV, Moran_Local, Moran_Local_BV
from libpysal.weights import KNN, Queen, attach_islands

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

# 02 스크립트의 계산 함수를 그대로 다시 쓴다 (민감도 분석용)
spec = importlib.util.spec_from_file_location("build", ROOT / "scripts" / "02_build_index.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

LISA_LABELS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


# ---------------------------------------------------------------- 1. 격차 분해

def decompose(grid: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """일반 성인 기준에서 고령자 기준으로 갈 때 지수가 어디서 줄어드는지 단계별로 본다 (서울 전체, 고령인구 가중)."""
    beta = meta["bike_age_factor"]["beta_65_plus"]
    target = meta["bus_target_routes_used"]
    app = build.PARAMS["taxi_app_share"]["elderly"]
    g = grid[grid["habitable"]]
    bus_g = np.minimum(g["bus_routes_general"] / target, 1)
    bus_e = np.minimum(g["bus_routes_elderly"] / target, 1)
    one = pd.Series(1.0, index=g.index)
    steps = {
        "① 일반 성인": (g["subway_general"], bus_g, g["bike_general"], one),
        "② 고령자 보행속도": (g["subway_elderly"], bus_e, g["bike_elderly"], one),
        "③ + 따릉이 연령별 이용률": (g["subway_elderly"], bus_e, g["bike_elderly"] * beta, one),
        "④ + 택시 앱 호출 비율": (g["subway_elderly"], bus_e, g["bike_elderly"] * beta,
                             app + (1 - app) * g["taxi_street_elderly"]),
    }
    w = g["w65"]
    th = build.PARAMS["choice_threshold"]
    rows = []
    for name, (s, b, k, t) in steps.items():
        m = s + b + k + t
        rows.append({"단계": name, "지하철": np.average(s, weights=w), "버스": np.average(b, weights=w),
                     "따릉이": np.average(k, weights=w), "택시": np.average(t, weights=w),
                     "이동선택권(MCI)": np.average(m, weights=w),
                     "사각지대 고령인구(수단 2개 미만)": float(w[m < th].sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. 공간 군집

def spatial_weights(d: gpd.GeoDataFrame):
    w = Queen.from_dataframe(d, use_index=False)
    if w.islands:
        w = attach_islands(w, KNN.from_dataframe(d, k=1))
    w.transform = "r"
    return w


def lisa(d: gpd.GeoDataFrame, w) -> tuple[gpd.GeoDataFrame, dict]:
    y = d["mci_elderly"].values
    x = d["elderly_share"].values
    np.random.seed(SEED)  # 전역 Moran 은 seed 인자가 없어 난수 상태를 직접 고정한다
    gm = Moran(y, w, permutations=999)
    gbv = Moran_BV(x, y, w, permutations=999)
    lm = Moran_Local(y, w, permutations=999, seed=SEED)
    lbv = Moran_Local_BV(x, y, w, permutations=999, seed=SEED)
    d = d.copy()
    d["lisa_mci"] = np.where(lm.p_sim < 0.05, pd.Series(lm.q).map(LISA_LABELS), "ns")
    d["lisa_mci_p"] = lm.p_sim
    d["lisa_bv"] = np.where(lbv.p_sim < 0.05, pd.Series(lbv.q).map(LISA_LABELS), "ns")
    d["lisa_bv_p"] = lbv.p_sim
    stats = {"moran_I_mci_elderly": gm.I, "moran_p": gm.p_sim,
             "moran_bv_I_elderlyshare_mci": gbv.I, "moran_bv_p": gbv.p_sim,
             "lisa_counts": d["lisa_mci"].value_counts().to_dict(),
             "lisa_bv_counts": d["lisa_bv"].value_counts().to_dict()}
    return d, stats


# ---------------------------------------------------------------- 4. 민감도

class Simulator:
    """가정값을 바꿔 격자점 지수를 다시 계산한다 (민감도 분석과 정책 시나리오에 함께 쓴다)."""

    def __init__(self, grid: pd.DataFrame, meta: dict):
        import copy
        self.copy = copy
        self.stop_xy, self.stop_routes, _ = build.load_bus()
        self.subway_xy, self.bike_xy = build.load_subway(), build.load_bike()
        self.beta = meta["bike_age_factor"]["beta_65_plus"]
        self.g = grid[grid["habitable"]].reset_index(drop=True)
        self.xy = self.g[["x", "y"]].values
        self.base = copy.deepcopy(build.PARAMS)
        self.base_target = meta["bus_target_routes_used"]

    def run(self, override: dict | None = None, target: float | None = None) -> dict:
        build.PARAMS.clear()
        build.PARAMS.update(self.copy.deepcopy(self.base))
        build.PARAMS.update(override or {})
        target = target or self.base_target
        road_xy = build.load_major_roads()
        out = {}
        for prof, b in [("general", 1.0), ("elderly", self.beta)]:
            s = build.within(self.xy, self.subway_xy, build.walk_radius(prof, "subway"))
            r = build.bus_route_counts(self.xy, self.stop_xy, self.stop_routes, build.walk_radius(prof, "bus"))
            k = build.within(self.xy, self.bike_xy, build.walk_radius(prof, "bike")) * b
            app = build.PARAMS["taxi_app_share"][prof]
            t = app + (1 - app) * build.within(self.xy, road_xy, build.walk_radius(prof, "taxi"))
            out[prof] = s + np.minimum(r / target, 1) + k + t
        th = build.PARAMS["choice_threshold"]
        build.PARAMS.clear()
        build.PARAMS.update(self.base)
        w = self.g["w65"]
        return {"사각지대 고령인구(고령자 기준)": float(w[out["elderly"] < th].sum()),
                "사각지대 고령인구(일반 성인 기준)": float(w[out["general"] < th].sum()),
                "평균 MCI 고령자": float(np.average(out["elderly"], weights=w)),
                "평균 MCI 일반": float(np.average(out["general"], weights=w))}


def sensitivity(sim: Simulator) -> pd.DataFrame:
    """가정값을 하나씩 바꿔 사각지대 고령인구(고령자 기준 MCI<2)를 다시 계산한다."""
    base = sim.base
    cases = [("기준값", {}, None)]
    for v in [0.7, 1.0]:
        cases.append((f"고령자 보행속도 {v}m/s", {"walk_speed": {**base["walk_speed"], "elderly": v}}, None))
    for v in [3, 7, 10]:
        cases.append((f"버스·따릉이·택시 보행시간 {v}분",
                      {"walk_minutes": {**base["walk_minutes"], "bus": v, "bike": v, "taxi": v}}, None))
    for v in [7, 15]:
        cases.append((f"지하철 보행시간 {v}분", {"walk_minutes": {**base["walk_minutes"], "subway": v}}, None))
    for v in [4, 10]:
        cases.append((f"버스 기준 노선 수 {v}개", {}, v))
    cases.append(("우회계수 1.2", {"detour": 1.2}, None))
    cases.append(("우회계수 1.4", {"detour": 1.4}, None))
    for v in [0.10, 0.30]:
        cases.append((f"고령자 택시 앱 호출 비율 {v:.0%}",
                      {"taxi_app_share": {**base["taxi_app_share"], "elderly": v}}, None))
    cases.append(("택시 잡는 도로에 tertiary 포함", {"taxi_road_classes": base["taxi_road_classes"] + ["tertiary"]}, None))
    for v in [1.5, 2.5]:
        cases.append((f"사각지대 기준 수단 {v}개 미만", {"choice_threshold": v}, None))
    return pd.DataFrame([{"시나리오": name, **sim.run(o, t)} for name, o, t in cases])


def policy_scenarios(sim: Simulator) -> pd.DataFrame:
    """앱을 못 쓰는 고령자에게 전화 호출 택시(동행 온다 콜택시)가 닿는 정도에 따라 사각지대가 얼마나 줄어드는지."""
    app = sim.base["taxi_app_share"]["elderly"]
    rows = []
    for reach in [0.0, 0.25, 0.5, 1.0]:
        share = app + (1 - app) * reach
        r = sim.run({"taxi_app_share": {**sim.base["taxi_app_share"], "elderly": share}})
        rows.append({"시나리오": f"앱 못 쓰는 고령자 중 {reach:.0%}가 전화 호출 택시를 이용", "택시 호출 가능 비율": share, **r})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 그림

def fig_maps(d: gpd.GeoDataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, col, title in [(axes[0], "mci_general", "일반 성인 기준"), (axes[1], "mci_elderly", "고령자(65+) 기준")]:
        d.plot(column=col, ax=ax, cmap="YlGnBu", vmin=0, vmax=4, edgecolor="white", linewidth=0.2)
        ax.set_title(f"{title}  (서울 평균 {d[col].mean():.2f}개)", fontsize=12)
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=plt.Normalize(0, 4))
    cb = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02)
    cb.set_label("실제로 쓸 수 있는 교통수단 수 (0~4)")
    fig.suptitle("같은 동네, 다른 선택권: 행정동별 이동선택권 지수", fontsize=14)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_decompose(dec: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = {"지하철": "#2b6cb0", "버스": "#38a169", "따릉이": "#dd6b20", "택시": "#805ad5"}
    left = np.zeros(len(dec))
    for mode, c in colors.items():
        ax.barh(dec["단계"], dec[mode], left=left, color=c, label=mode)
        for i, v in enumerate(dec[mode]):
            if v > 0.12:
                ax.text(left[i] + v / 2, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=9)
        left += dec[mode].values
    for i, v in enumerate(left):
        ax.text(v + 0.03, i, f"{v:.2f}개", va="center", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 4)
    ax.set_xlabel("고령인구 가중 평균 이동선택권 (쓸 수 있는 수단 수)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("고령자의 이동선택권은 어디서 줄어드는가")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_lisa(d: gpd.GeoDataFrame, path: Path):
    pal = {"HH": "#d7191c", "LL": "#2c7bb6", "LH": "#abd9e9", "HL": "#fdae61", "ns": "#eeeeee"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    specs = [
        (axes[0], "lisa_mci", "고령자 이동선택권 군집 (LISA)",
         {"LL": "낮음-낮음 (사각지대 군집)", "HH": "높음-높음", "LH": "주변보다 홀로 낮음", "HL": "주변보다 홀로 높음",
          "ns": "유의하지 않음"}),
        (axes[1], "lisa_bv", "고령인구 비율 × 주변 이동선택권 (이변량 LISA)",
         {"HL": "고령 많음-선택권 낮음 (우선지역)", "HH": "고령 많음-선택권 높음", "LL": "고령 적음-선택권 낮음",
          "LH": "고령 적음-선택권 높음", "ns": "유의하지 않음"}),
    ]
    for ax, col, title, labels in specs:
        d.plot(color=d[col].map(pal), ax=ax, edgecolor="white", linewidth=0.2)
        handles = [plt.Rectangle((0, 0), 1, 1, color=pal[k]) for k in labels]
        ax.legend(handles, [f"{v} ({(d[col] == k).sum()})" for k, v in labels.items()],
                  loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=8.5, frameon=False)
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    meta = json.loads((PROC / "mci_meta.json").read_text())
    d = gpd.read_file(PROC / "dong_mci.gpkg")
    grid = pd.read_parquet(PROC / "grid_mci.parquet")

    # 격자점 가중치: 동 안의 65세 이상이 거주 가능 격자점에 고르게 산다고 본다
    hab = grid[grid["habitable"]]
    per_point = d.set_index("adm_cd10")["pop_65_plus"] / hab.groupby("adm_cd10").size()
    grid["w65"] = np.where(grid["habitable"], grid["adm_cd10"].map(per_point), 0.0)

    dec = decompose(grid, meta)
    dec.to_csv(TAB / "b1_gap_decomposition.csv", index=False, encoding="utf-8-sig")
    print(dec.round(3).to_string(index=False))

    w = spatial_weights(d)
    d, stats = lisa(d, w)
    print(json.dumps(stats, ensure_ascii=False, indent=1, default=float))

    # 우선지역: 자기 동의 고령자 이동선택권이 서울 중앙값 미만이면서
    #   (이변량 LISA 에서 고령인구 비율이 높고 주변 이동선택권이 낮은 동(HL)) 또는 (사각지대 군집(LL) 중 고령 비율이 중앙값 이상)
    med = d["elderly_share"].median()
    low_own = d["mci_elderly"] < d["mci_elderly"].median()
    d["priority"] = low_own & ((d["lisa_bv"] == "HL") | ((d["lisa_mci"] == "LL") & (d["elderly_share"] >= med)))
    cols = ["gu", "dong", "pop_65_plus", "elderly_share", "mci_general", "mci_elderly", "mci_gap",
            "subway_elderly", "a_bus_elderly", "a_taxi_elderly", "share_blind_elderly", "blind_elderly_pop", "lisa_mci", "lisa_bv"]
    pri = d[d["priority"]].sort_values("blind_elderly_pop", ascending=False)[cols]
    pri.to_csv(TAB / "b2_priority_dongs.csv", index=False, encoding="utf-8-sig")
    d.drop(columns="geometry").to_csv(TAB / "b0_dong_mci_all.csv", index=False, encoding="utf-8-sig")
    d.to_file(PROC / "dong_mci_lisa.gpkg", driver="GPKG")
    print(f"\n우선지역 {len(pri)}개 동, 65세 이상 {pri['pop_65_plus'].sum():,.0f}명, "
          f"그중 사각지대 {pri['blind_elderly_pop'].sum():,.0f}명")
    print(pri.head(15).round(3).to_string(index=False))

    sim = Simulator(grid, meta)
    sens = sensitivity(sim)
    sens.to_csv(TAB / "b3_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(sens.round(3).to_string(index=False))
    pol = policy_scenarios(sim)
    pol.to_csv(TAB / "b4_policy_taxi_call.csv", index=False, encoding="utf-8-sig")
    print(pol.round(3).to_string(index=False))

    fig_maps(d, FIG / "b1_mci_general_vs_elderly.png")
    fig_decompose(dec, FIG / "b2_gap_decomposition.png")
    fig_lisa(d, FIG / "b3_lisa_clusters.png")

    summary = {
        "pop_65_plus": float(d["pop_65_plus"].sum()),
        "blind_elderly_pop_elderly_profile": float(d["blind_elderly_pop"].sum()),
        "blind_elderly_pop_general_profile": float((d["pop_65_plus"] * d["share_blind_general"]).sum()),
        "priority_dongs": int(len(pri)),
        "priority_pop_65_plus": float(pri["pop_65_plus"].sum()),
        "priority_blind_elderly_pop": float(pri["blind_elderly_pop"].sum()),
        "corr_elderly_share_mci_elderly": float(d[["elderly_share", "mci_elderly"]].corr().iloc[0, 1]),
        **stats,
    }
    (TAB / "b_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print("\n그림·표 저장 완료:", FIG, TAB)


if __name__ == "__main__":
    main()
