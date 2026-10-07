"""K. 의료 목적지 접근: 고령자 걸음으로 동네 의원과 큰 병원에 닿는가.

실행: python scripts/21_medical_access.py   (02_build_index.py, 16_policy_diagnosis.py 다음, 약 1분)

- 병의원 위치(OA-20337): 의원(내과·가정의학 등 구분 없음), 병원급(병원·종합병원), 보건소
- 거주 가능 100m 격자마다
  (1) 걸어서 10분 안에 의원이 있는가 (일반 554m, 고령자 369m 직선, B와 같은 우회계수 1.3)
  (2) 가장 가까운 병원급·보건소까지 직선거리
  (3) 의료 이중 사각: 고령자 이동선택권 < 1 이면서 고령자 걸음 10분 안에 의원이 없는 곳
- 동 안에서 고령자가 고르게 산다고 보고 격자 비율 × 65세 이상 인구로 집계한다 (B와 같은 방법)
- 집계: 서울 전체, 이동선택권 4분면(F), C 1순위 72개 동
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TAB = ROOT / "outputs" / "tables"

R10 = {"general": 1.2 * 600 / 1.3, "elderly": 0.8 * 600 / 1.3}  # 10분 보행권 직선거리 (B와 같음)


def main():
    c = pd.read_csv(RAW / "서울시 병의원 위치 정보.csv", encoding="cp949").dropna(subset=["병원경도", "병원위도"])
    pts = gpd.GeoSeries(gpd.points_from_xy(c["병원경도"], c["병원위도"]), crs="EPSG:4326").to_crs("EPSG:5179")
    c = c.assign(x=pts.x.values, y=pts.y.values)
    clinic = c[c["병원분류명"] == "의원"][["x", "y"]].to_numpy()
    hosp = c[c["병원분류명"].isin(["병원", "종합병원", "보건소"])][["x", "y"]].to_numpy()

    g = pd.read_parquet(PROC / "grid_mci.parquet")
    g = g[g["habitable"]].copy()
    xy = g[["x", "y"]].to_numpy()
    d_clinic = cKDTree(clinic).query(xy)[0]
    g["hosp_km"] = cKDTree(hosp).query(xy)[0] / 1000
    for prof in ["general", "elderly"]:
        g[f"clinic10_{prof}"] = d_clinic <= R10[prof]
    g["double_blind"] = (g["mci_elderly"] < 1) & ~g["clinic10_elderly"]

    dongs = gpd.read_file(PROC / "dong_mci.gpkg")[["adm_cd10", "gu", "dong", "pop_65_plus"]]
    quad = pd.read_csv(TAB / "f2_dong_quadrant.csv", dtype={"adm_cd10": str})[["adm_cd10", "quadrant", "priority1"]]
    agg = g.groupby("adm_cd10").agg(no_clinic_general=("clinic10_general", lambda s: 1 - s.mean()),
                                    no_clinic_elderly=("clinic10_elderly", lambda s: 1 - s.mean()),
                                    double_blind=("double_blind", "mean"), hosp_km=("hosp_km", "median"))
    d = dongs.merge(agg, left_on="adm_cd10", right_index=True).merge(quad, on="adm_cd10")
    for col in ["no_clinic_general", "no_clinic_elderly", "double_blind"]:
        d[col + "_pop"] = d[col] * d["pop_65_plus"]
    d.to_csv(TAB / "k2_dong_medical_access.csv", index=False, encoding="utf-8-sig")

    def summarize(x, name):
        pop = x["pop_65_plus"].sum()
        return {"구분": name, "동수": len(x), "65세이상": pop,
                "의원_10분밖_일반기준_65세이상": x["no_clinic_general_pop"].sum(),
                "의원_10분밖_고령자기준_65세이상": x["no_clinic_elderly_pop"].sum(),
                "의원_10분밖_고령자기준_비율": x["no_clinic_elderly_pop"].sum() / pop,
                "의료_이중사각_65세이상": x["double_blind_pop"].sum(),
                "의료_이중사각_비율": x["double_blind_pop"].sum() / pop,
                "병원급까지_거리_km(동 중앙값의 65세이상 가중평균)": np.average(x["hosp_km"], weights=x["pop_65_plus"])}
    rows = [summarize(d, "서울 전체")]
    rows += [summarize(d[d["quadrant"] == q], q) for q in ["공급 먼저", "이동권 우선", "수요관리 가능", "유지"]]
    rows += [summarize(d[d["priority1"]], "C 1순위 72개 동"), summarize(d[~d["priority1"]], "나머지 355개 동")]
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "k1_medical_access.csv", index=False, encoding="utf-8-sig")
    print(t.round(3).to_string())
    print(d.sort_values("double_blind_pop", ascending=False).head(10)[["gu", "dong", "quadrant", "priority1", "double_blind", "double_blind_pop", "hosp_km"]].round(2).to_string())


if __name__ == "__main__":
    main()
