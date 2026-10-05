"""이동선택권 지수(Mobility Choice Index, MCI)를 100m 격자와 행정동 단위로 계산한다.

실행: python scripts/02_build_index.py

정의
- 격자점: 서울 행정동 안의 100m 격자 중심점. 산림·공원·수면·묘지·군사시설(OSM)에 들어가는 점은 사람이 살지 않는 곳으로 보고 뺀다.
- 수단별 이용가능도 a(0~1)를 격자점마다 구한다.
    지하철  : 보행 시간 안에 역이 있으면 1
    버스    : 보행 시간 안의 정류소를 지나는 서로 다른 노선 수 / 기준 노선 수 (최대 1)
    따릉이  : 보행 시간 안에 대여소가 있으면 1. 고령자 기준에서는 연령별 실제 이용률 비(β)를 곱한다.
- MCI = a_지하철 + a_버스 + a_따릉이 (0~3). "실제로 쓸 수 있는 교통수단이 몇 개인가"를 뜻한다.
- 일반 성인 기준과 고령자(65세 이상) 기준을 따로 계산한다. 두 기준은 보행속도와 따릉이 가중치만 다르다.
- 행정동 값 = 그 동의 거주 가능 격자점 평균.

모든 가정값은 PARAMS 에 모아 두었다. 근거를 확인해야 하는 값은 주석에 (근거 확인 필요) 로 표시했다.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import polygonize, unary_union

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
CRS = "EPSG:5179"  # 미터 단위 좌표계 (UTM-K)

PARAMS = {
    "grid_m": 100,
    # 보행속도(m/s). 일반 1.2, 고령자 0.8 (교통약자 보행신호 기준값, 근거 확인 필요)
    "walk_speed": {"general": 1.2, "elderly": 0.8},
    # 수단별로 걸어서 가도 된다고 보는 시간(분)
    "walk_minutes": {"subway": 10, "bus": 5, "bike": 5},
    # 실제 보행경로 길이 / 직선거리 (우회계수, 근거 확인 필요)
    "detour": 1.3,
    # 버스 기준 노선 수: None 이면 일반 성인 기준 서울 격자점의 중앙값을 쓴다
    "bus_target_routes": None,
}


def walk_radius(profile: str, mode: str) -> float:
    """보행 시간 안에 닿는 직선거리(m)."""
    p = PARAMS
    return p["walk_speed"][profile] * 60 * p["walk_minutes"][mode] / p["detour"]


# ---------------------------------------------------------------- 불러오기

def load_dongs() -> gpd.GeoDataFrame:
    g = gpd.read_file(RAW / "HangJeongDong_ver20260701.geojson")
    g = g[g["sidonm"] == "서울특별시"].to_crs(CRS)
    g = g.rename(columns={"adm_cd2": "adm_cd10", "sggnm": "gu"})
    g["dong"] = g["adm_nm"].str.split().str[-1]
    return g[["adm_cd10", "gu", "dong", "geometry"]].reset_index(drop=True)


def load_population() -> pd.DataFrame:
    p = pd.read_csv(RAW / "지역별(행정동) 성별 연령별 주민등록 인구수_20260831.csv", encoding="cp949",
                    dtype={"행정기관코드": str})
    p = p[p["시도명"] == "서울특별시"].copy()

    def age_sum(lo, hi=None):
        cols = [c for c in p.columns if c.endswith(("남자", "여자")) and c[0].isdigit()]
        age = pd.Series([int("".join(ch for ch in c if ch.isdigit())) for c in cols], index=cols)
        sel = age[(age >= lo) & ((age <= hi) if hi is not None else True)].index
        return p[sel].sum(axis=1)

    return pd.DataFrame({
        "adm_cd10": p["행정기관코드"],
        "pop_total": p["계"],
        "pop_20_59": age_sum(20, 59),
        "pop_60_64": age_sum(60, 64),
        "pop_65_69": age_sum(65, 69),
        "pop_65_plus": age_sum(65),
        "pop_70_plus": age_sum(70),
    })


def to_points(df, x, y) -> np.ndarray:
    g = gpd.GeoSeries(gpd.points_from_xy(df[x], df[y]), crs="EPSG:4326").to_crs(CRS)
    return np.column_stack([g.x, g.y])


def load_subway() -> np.ndarray:
    s = pd.read_csv(RAW / "서울시 역사마스터 정보.csv", encoding="cp949")
    s = s.dropna(subset=["위도", "경도"])
    return to_points(s, "경도", "위도")


def load_bus():
    """정류소 좌표와 각 정류소를 지나는 노선 ID. 한강버스 선착장은 뺀다."""
    stops = pd.read_excel(RAW / "서울시버스정류소위치정보(20260902).xlsx")
    ferry = set(stops.loc[stops["정류소타입"] == "한강선착장", "NODE_ID"])
    rs = pd.read_excel(RAW / "서울시버스노선별정류소정보(20260902).xlsx")
    rs = rs[~rs["NODE_ID"].isin(ferry)].dropna(subset=["X좌표", "Y좌표"])
    nodes = rs.groupby("NODE_ID").agg(x=("X좌표", "first"), y=("Y좌표", "first"))
    routes = rs.groupby("NODE_ID")["ROUTE_ID"].unique().reindex(nodes.index)
    return to_points(nodes, "x", "y"), list(routes), rs["ROUTE_ID"].nunique()


def load_bike() -> np.ndarray:
    b = pd.read_excel(RAW / "공공자전거 대여소 정보(26.6월 기준).xlsx", header=None, skiprows=5)
    b = b.iloc[:, [0, 4, 5]].set_axis(["station_id", "lat", "lon"], axis=1)
    b = b.apply(pd.to_numeric, errors="coerce").dropna()
    b = b[(b["lat"] > 37) & (b["lat"] < 38) & (b["lon"] > 126) & (b["lon"] < 128)]
    return to_points(b, "lon", "lat")


def bike_age_factor(pop: pd.DataFrame) -> dict:
    """65세 이상의 인구 1인당 따릉이 이용건수 ÷ 20~59세의 인구 1인당 이용건수 (β)."""
    u = pd.read_csv(RAW / "서울특별시 공공자전거 이용정보(월별)_26.1-6.csv", encoding="cp949",
                    usecols=["연령대코드", "이용건수"])
    use = u.groupby("연령대코드")["이용건수"].sum()
    tot = pop[["pop_20_59", "pop_60_64", "pop_65_69", "pop_70_plus"]].sum()
    pop_60s = tot["pop_60_64"] + tot["pop_65_69"]
    rate_ref = use[["20대", "30대", "40대", "50대"]].sum() / tot["pop_20_59"]
    rate_60s = use["60대"] / pop_60s
    rate_70p = use["70대이상"] / tot["pop_70_plus"]
    # 65~69세는 60대 이용률을 쓴다 (원자료가 10세 단위라서)
    rate_65p = (rate_60s * tot["pop_65_69"] + rate_70p * tot["pop_70_plus"]) / (tot["pop_65_69"] + tot["pop_70_plus"])
    return {"rate_ref_per_person_6m": rate_ref, "rate_60s": rate_60s, "rate_70_plus": rate_70p,
            "rate_65_plus": rate_65p, "beta_65_plus": rate_65p / rate_ref}


def load_mask(seoul: Polygon):
    """OSM 산림·공원·수면 등 폴리곤. way 는 닫힌 선만 쓴다.
    relation 은 outer 선을 이어 면을 만들고 inner(구멍)를 뺀다.
    (예: 여의도한강공원은 섬을 둘러싼 고리 모양이라 inner 를 빼지 않으면 섬 전체가 공원이 된다)"""
    data = json.loads((RAW / "osm_nonresidential_seoul.json").read_text())

    def rings(e, role):
        lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                 for m in e.get("members", []) if m.get("role") == role and m.get("geometry")]
        return unary_union(list(polygonize(unary_union(lines)))) if lines else None

    polys = []
    for e in data["elements"]:
        if e["type"] == "way" and "geometry" in e:
            c = [(p["lon"], p["lat"]) for p in e["geometry"]]
            if len(c) >= 4 and c[0] == c[-1]:
                polys.append(Polygon(c))
        elif e["type"] == "relation":
            outer, inner = rings(e, "outer"), rings(e, "inner")
            if outer is not None and not outer.is_empty:
                polys.append(outer.difference(inner) if inner is not None else outer)
    m = gpd.GeoSeries(polys, crs="EPSG:4326").to_crs(CRS)
    m = m[m.is_valid | m.buffer(0).is_valid].buffer(0)
    return unary_union(m.clip(seoul).values)


# ---------------------------------------------------------------- 계산

def make_grid(dongs: gpd.GeoDataFrame, mask) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = dongs.total_bounds
    s = PARAMS["grid_m"]
    xs = np.arange(xmin + s / 2, xmax, s)
    ys = np.arange(ymin + s / 2, ymax, s)
    xx, yy = np.meshgrid(xs, ys)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=CRS)
    pts = gpd.sjoin(pts, dongs[["adm_cd10", "geometry"]], predicate="within").drop(columns="index_right")
    pts["habitable"] = ~pts.within(mask)
    return pts.reset_index(drop=True)


def bus_route_counts(xy, stop_xy, stop_routes, r) -> np.ndarray:
    tree = cKDTree(stop_xy)
    out = np.zeros(len(xy), dtype=int)
    for i, idx in enumerate(tree.query_ball_point(xy, r)):
        if idx:
            out[i] = len(np.unique(np.concatenate([stop_routes[j] for j in idx])))
    return out


def within(xy, target_xy, r) -> np.ndarray:
    d, _ = cKDTree(target_xy).query(xy)
    return (d <= r).astype(float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    dongs = load_dongs()
    pop = load_population()
    missing = set(dongs["adm_cd10"]) ^ set(pop["adm_cd10"])
    if missing:
        raise SystemExit(f"경계와 인구의 행정동 코드가 맞지 않음: {sorted(missing)}")
    dongs = dongs.merge(pop, on="adm_cd10")
    seoul = unary_union(dongs.geometry.values)

    print("격자 만드는 중 ...")
    mask = load_mask(seoul)
    grid = make_grid(dongs, mask)
    print(f"  격자점 {len(grid):,}개 중 거주 가능 {grid['habitable'].sum():,}개")

    subway_xy = load_subway()
    stop_xy, stop_routes, n_routes = load_bus()
    bike_xy = load_bike()
    factor = bike_age_factor(pop)
    print(f"  지하철역 {len(subway_xy)}, 버스정류소 {len(stop_xy)} (노선 {n_routes}), 따릉이 대여소 {len(bike_xy)}")
    print(f"  따릉이 연령 가중치 β(65+) = {factor['beta_65_plus']:.3f}")

    xy = np.column_stack([grid.geometry.x, grid.geometry.y])
    for prof in ["general", "elderly"]:
        grid[f"subway_{prof}"] = within(xy, subway_xy, walk_radius(prof, "subway"))
        grid[f"bus_routes_{prof}"] = bus_route_counts(xy, stop_xy, stop_routes, walk_radius(prof, "bus"))
        grid[f"bike_{prof}"] = within(xy, bike_xy, walk_radius(prof, "bike"))

    hab = grid["habitable"]
    target = PARAMS["bus_target_routes"] or float(np.median(grid.loc[hab, "bus_routes_general"]))
    beta = {"general": 1.0, "elderly": factor["beta_65_plus"]}
    for prof in ["general", "elderly"]:
        grid[f"a_bus_{prof}"] = np.minimum(grid[f"bus_routes_{prof}"] / target, 1.0)
        grid[f"a_bike_{prof}"] = grid[f"bike_{prof}"] * beta[prof]
        grid[f"mci_{prof}"] = grid[f"subway_{prof}"] + grid[f"a_bus_{prof}"] + grid[f"a_bike_{prof}"]
    print(f"  버스 기준 노선 수 = {target:g}")

    # 행정동 집계 (거주 가능 격자점 평균)
    g = grid[hab]
    cols = [c for c in grid.columns if c.startswith(("subway_", "a_bus_", "a_bike_", "mci_", "bus_routes_"))]
    agg = g.groupby("adm_cd10")[cols].mean()
    # 고령자 기준 수단이 1개 미만인 격자점 비율 → 동 안에서 고령자가 고르게 산다고 보고 사각지대 고령인구 추정
    agg["share_blind_elderly"] = g.assign(b=g["mci_elderly"] < 1).groupby("adm_cd10")["b"].mean()
    agg["share_blind_general"] = g.assign(b=g["mci_general"] < 1).groupby("adm_cd10")["b"].mean()
    agg["n_points"] = g.groupby("adm_cd10").size()

    out = dongs.merge(agg, left_on="adm_cd10", right_index=True, how="left")
    out["elderly_share"] = out["pop_65_plus"] / out["pop_total"]
    out["mci_gap"] = out["mci_general"] - out["mci_elderly"]
    out["blind_elderly_pop"] = out["pop_65_plus"] * out["share_blind_elderly"]
    out["area_km2"] = out.area / 1e6

    out.to_file(OUT / "dong_mci.gpkg", driver="GPKG")
    grid.drop(columns="geometry").assign(x=xy[:, 0], y=xy[:, 1]).to_parquet(OUT / "grid_mci.parquet")
    meta = {"params": PARAMS, "bus_target_routes_used": target, "bike_age_factor": factor,
            "walk_radius_m": {p: {m: round(walk_radius(p, m)) for m in PARAMS["walk_minutes"]}
                              for p in PARAMS["walk_speed"]},
            "n_grid": int(len(grid)), "n_habitable": int(hab.sum()),
            "n_subway": int(len(subway_xy)), "n_bus_stops": int(len(stop_xy)), "n_bike": int(len(bike_xy))}
    (OUT / "mci_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float))
    print(f"저장: {OUT / 'dong_mci.gpkg'} ({len(out)}개 동)")


if __name__ == "__main__":
    main()
