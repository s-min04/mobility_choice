"""I. 공식 기준판 이동선택권: 국토부 「대중교통 최소서비스 수준」 기준과 실측 우회계수로 지수를 다시 계산한다.

실행: python scripts/19_official_standard.py   (02_build_index.py 다음, 약 3분)

B의 지수는 보행시간(버스 5분·지하철 10분)과 노선 수(÷7)를 썼다. 여기서는 공식 기준으로 바꿔 결론이 유지되는지 본다.
1. 우회계수 실측: OSM 보행 도로망 표본 20개 동에서 격자점 → 역·정류소의 실제 보행경로 ÷ 직선거리
2. 공간 기준: 국토부(2022) 도시지역 버스정류장 400m, 지하철역 800m (도로 경로 기준) → 직선 반경 = 기준거리 ÷ 우회계수
   고령자는 같은 보행시간으로 환산한다(× 0.8/1.2). 신호 기준 비율(0.8/1.0)도 함께 계산한다.
3. 시간 기준: 국토부 고밀도 지역 운행횟수 기준 시간당 6회 → 버스 점수 = 반경 안 정류소의 평일 07~20시 평균 운행횟수 ÷ 6 (최대 1)
   운행횟수 = 서울시 노선별 정류장별 총 버스 운행횟수(OA-21220, 2026.9.6~10.2 평일, 추석 연휴 제외), 정류소의 모든 노선 합
4. 따릉이: 버스와 같은 반경, 고령자는 β를 곱한다 (B와 같음)
"""

import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TAB = ROOT / "outputs" / "tables"

spec = importlib.util.spec_from_file_location("b02", ROOT / "scripts" / "02_build_index.py")
b02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b02)

MOLIT = {"bus_m": 400, "subway_m": 800, "runs_per_hour": 6}
SPEED = {"general": 1.2, "elderly": 0.8}
HOLIDAYS = {"20260924", "20260925", "20260926"}  # 추석 연휴
HOURS = [f"버스운행횟수_{h:02d}시" for h in range(7, 21)]
TO_5179 = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)


def detour_factor(grid: pd.DataFrame, subway_xy: np.ndarray, stop_xy: np.ndarray) -> pd.DataFrame:
    """표본 동마다 격자점 → 역·정류소 쌍의 (실제 보행경로 + 연결 거리) ÷ 직선거리."""
    data = json.loads((RAW / "osm_walk_samples_seoul.json").read_text())
    rows = []
    hab = grid[grid["habitable"]]
    for s in data:
        nodes, edges = {}, []
        for e in s["elements"]:
            if e.get("type") != "way" or len(e.get("geometry", [])) < 2:
                continue
            xs, ys = TO_5179.transform([p["lon"] for p in e["geometry"]], [p["lat"] for p in e["geometry"]])
            ids = []
            for x, y in zip(xs, ys):
                k = (round(x, 1), round(y, 1))
                ids.append(nodes.setdefault(k, len(nodes)))
            for (a, b), x0, y0, x1, y1 in zip(zip(ids[:-1], ids[1:]), xs[:-1], ys[:-1], xs[1:], ys[1:]):
                edges.append((a, b, float(np.hypot(x1 - x0, y1 - y0))))
        if not edges:
            continue
        xy = np.array(list(nodes.keys()))
        a, b, w = map(np.array, zip(*edges))
        g = coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(len(nodes),) * 2).tocsr()
        tree = cKDTree(xy)
        cx, cy = TO_5179.transform(s["lon"], s["lat"])
        core = hab[(abs(hab["x"] - cx) < 400) & (abs(hab["y"] - cy) < 400)]  # 표본 사각형(약 ±880m) 안쪽
        core = core.sample(min(40, len(core)), random_state=0)
        targets = {"지하철역": subway_xy, "버스정류소": stop_xy}
        dest = {k: v[(abs(v[:, 0] - cx) < 750) & (abs(v[:, 1] - cy) < 750)] for k, v in targets.items()}
        d_o, n_o = tree.query(core[["x", "y"]].to_numpy())
        ok = d_o <= 60
        if not ok.any():
            continue
        dist = dijkstra(g, indices=n_o[ok], limit=2500)
        for kind, pts in dest.items():
            if len(pts) == 0:
                continue
            d_d, n_d = tree.query(pts)
            for i, (ox, oy) in enumerate(core[["x", "y"]].to_numpy()[ok]):
                straight = np.hypot(pts[:, 0] - ox, pts[:, 1] - oy)
                net = dist[i, n_d] + d_o[ok][i] + d_d
                m = (straight >= 200) & (straight <= 1000) & (d_d <= 60) & np.isfinite(net)
                for st, nt in zip(straight[m], net[m]):
                    rows.append({"표본": s["name"], "대상": kind, "직선_m": st, "보행경로_m": nt, "우회계수": nt / st})
    return pd.DataFrame(rows)


def stop_runs() -> pd.DataFrame:
    """정류소별 평일 07~20시 평균 시간당 운행횟수 (그 정류소를 지나는 모든 노선 합)."""
    r = pd.read_csv(RAW / "서울시 노선별 정류장별 총 버스 운행횟수 정보.csv", encoding="cp949",
                    usecols=["기준_날짜", "정류장_ID", *HOURS], dtype={"기준_날짜": str, "정류장_ID": str})
    day = pd.to_datetime(r["기준_날짜"])
    r = r[(day.dt.weekday < 5) & ~r["기준_날짜"].isin(HOLIDAYS)]
    r[HOURS] = r[HOURS].apply(pd.to_numeric, errors="coerce").fillna(0)
    per_day = r.groupby(["정류장_ID", "기준_날짜"])[HOURS].sum().sum(axis=1) / len(HOURS)
    out = per_day.groupby(level=0).mean().rename("runs_per_hour").to_frame()
    out.attrs["n_days"] = r["기준_날짜"].nunique()
    return out


def main():
    grid = pd.read_parquet(PROC / "grid_mci.parquet")
    dongs = gpd.read_file(PROC / "dong_mci.gpkg")
    beta = json.loads((PROC / "mci_meta.json").read_text())["bike_age_factor"]["beta_65_plus"]
    subway_xy = b02.load_subway()
    bike_xy = b02.load_bike()
    stops = pd.read_excel(RAW / "서울시버스정류소위치정보(20260902).xlsx", dtype={"NODE_ID": str})
    stops = stops[stops["정류소타입"] != "한강선착장"].dropna(subset=["X좌표", "Y좌표"])
    runs = stop_runs()
    stops = stops.merge(runs, left_on="NODE_ID", right_index=True, how="left").fillna({"runs_per_hour": 0})
    stop_xy = np.column_stack(TO_5179.transform(stops["X좌표"].astype(float).values, stops["Y좌표"].astype(float).values))
    print(f"정류소 {len(stops):,}개, 운행횟수 자료 평일 {runs.attrs['n_days']}일, 시간당 6회 이상 정류소 "
          f"{(stops['runs_per_hour'] >= MOLIT['runs_per_hour']).mean():.1%}")

    # 1. 우회계수 실측
    det = detour_factor(grid, subway_xy, stop_xy)
    det.to_csv(TAB / "i1_detour_pairs.csv", index=False, encoding="utf-8-sig")
    summ = det.groupby("대상")["우회계수"].describe(percentiles=[0.25, 0.5, 0.75])
    summ.loc["전체"] = det["우회계수"].describe(percentiles=[0.25, 0.5, 0.75])
    summ["표본_동수"] = [det.loc[det["대상"] == k, "표본"].nunique() if k != "전체" else det["표본"].nunique() for k in summ.index]
    summ.to_csv(TAB / "i2_detour_summary.csv", encoding="utf-8-sig")
    print(summ.round(3).to_string())
    detour = float(det["우회계수"].median())

    # 2~4. 공식 기준판 지수
    hab = grid[grid["habitable"]].copy()
    xy = hab[["x", "y"]].to_numpy()
    stop_tree = cKDTree(stop_xy)
    runs_arr = stops["runs_per_hour"].to_numpy()
    rows = {}
    for label, ratio in [("기본 (고령자 0.8 ÷ 일반 1.2)", 0.8 / 1.2), ("신호 기준 비율 (0.8 ÷ 1.0)", 0.8 / 1.0)]:
        for prof in ["general", "elderly"]:
            k = 1.0 if prof == "general" else ratio
            r_bus, r_sub = MOLIT["bus_m"] / detour * k, MOLIT["subway_m"] / detour * k
            sub = (cKDTree(subway_xy).query(xy)[0] <= r_sub).astype(float)
            best = np.array([runs_arr[idx].max() if idx else 0.0 for idx in stop_tree.query_ball_point(xy, r_bus)])
            bus = np.minimum(best / MOLIT["runs_per_hour"], 1.0)
            bike = (cKDTree(bike_xy).query(xy)[0] <= r_bus).astype(float) * (1.0 if prof == "general" else beta)
            hab[f"{label}|{prof}|mci"] = sub + bus + bike
            hab[f"{label}|{prof}|molit_space"] = ((sub > 0) | (best > 0)).astype(float)  # 공간 기준 충족(정류소·역 범위 안)
            hab[f"{label}|{prof}|molit_time"] = (best >= MOLIT["runs_per_hour"]).astype(float)
            rows[(label, prof)] = (r_bus, r_sub)

    pop = dongs.set_index("adm_cd10")["pop_65_plus"]
    out = []
    for label in ["기본 (고령자 0.8 ÷ 일반 1.2)", "신호 기준 비율 (0.8 ÷ 1.0)"]:
        r = {"기준": label}
        for prof, name in [("general", "일반"), ("elderly", "고령자")]:
            g = hab.groupby("adm_cd10")
            blind = g[f"{label}|{prof}|mci"].apply(lambda s: (s < 1).mean())
            r[f"{name}_버스반경_m"], r[f"{name}_지하철반경_m"] = [round(v) for v in rows[(label, prof)]]
            r[f"{name}_평균지수(65+가중)"] = np.average(g[f"{label}|{prof}|mci"].mean().reindex(pop.index), weights=pop)
            r[f"{name}_사각지대_65세이상"] = float((blind.reindex(pop.index) * pop).sum())
            r[f"{name}_공간기준_충족(격자)"] = hab[f"{label}|{prof}|molit_space"].mean()
            r[f"{name}_시간기준_충족(격자)"] = hab[f"{label}|{prof}|molit_time"].mean()
        r["고령자÷일반_사각지대"] = r["고령자_사각지대_65세이상"] / r["일반_사각지대_65세이상"]
        out.append(r)
    b = {"기준": "B 원래 지수 (보행 5·10분, 노선 수 ÷ 7, 우회 1.3)",
         "일반_사각지대_65세이상": float((dongs["share_blind_general"] * dongs["pop_65_plus"]).sum()),
         "고령자_사각지대_65세이상": float((dongs["share_blind_elderly"] * dongs["pop_65_plus"]).sum())}
    b["고령자÷일반_사각지대"] = b["고령자_사각지대_65세이상"] / b["일반_사각지대_65세이상"]
    res = pd.DataFrame([b, *out])
    res.to_csv(TAB / "i3_official_standard.csv", index=False, encoding="utf-8-sig")
    print(res.T.to_string())

    # 동 순위가 유지되는가 (고령자 지수, 원래 vs 공식 기준판)
    lab = "기본 (고령자 0.8 ÷ 일반 1.2)"
    dm = hab.groupby("adm_cd10")[f"{lab}|elderly|mci"].mean().rename("mci_elderly_official")
    d = dongs.set_index("adm_cd10")[["gu", "dong", "mci_elderly", "mci_general"]].join(dm)
    d["mci_general_official"] = hab.groupby("adm_cd10")[f"{lab}|general|mci"].mean()
    p1 = pd.read_csv(TAB / "c8_priority1_targets.csv", dtype={"adm_cd10": str})
    p1 = set(p1.loc[p1["priority1"].str.startswith(("확정", "추가")), "adm_cd10"])
    d["priority1"] = d.index.isin(p1)
    rank = pd.DataFrame([{
        "순위상관(스피어만)_고령자지수": d["mci_elderly"].corr(d["mci_elderly_official"], method="spearman"),
        "1순위72_고령자지수_중앙값(원래)": d.loc[d["priority1"], "mci_elderly"].median(),
        "1순위72_고령자지수_중앙값(공식)": d.loc[d["priority1"], "mci_elderly_official"].median(),
        "나머지_고령자지수_중앙값(공식)": d.loc[~d["priority1"], "mci_elderly_official"].median(),
        "1순위72_중_공식기준_하위25%": (d.loc[d["priority1"], "mci_elderly_official"] <= d["mci_elderly_official"].quantile(0.25)).mean(),
        "우회계수_실측_중앙값": detour}])
    rank.to_csv(TAB / "i4_rank_stability.csv", index=False, encoding="utf-8-sig")
    d.reset_index().to_csv(TAB / "i5_dong_official_mci.csv", index=False, encoding="utf-8-sig")
    print(rank.T.round(3).to_string())


if __name__ == "__main__":
    main()
