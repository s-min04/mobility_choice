"""D 보강 입력: KT 생활이동(연령 × 수단, 일별)으로 고령자의 버스·차량·도보 이동과 전체 이동량을 날짜별로 만든다.

실행: python scripts/12_build_kt_daily.py   (09_cluster.py 다음, 약 15~25분)

- 수도권 생활이동 (도착 행정동 기준 시간대별 성연령별 수단), 겨울·여름 17개월 (2023.1 ~ 2026.2)
- 서울 행정동에 도착한 이동만 쓴다. 연령: 70대 이상 / 20~50대. 수단 코드는 06_build_usage.py 와 같다
  (6 지하철, 4·5 버스, 7 도보, 8 차량, 1·2·3·9 기타).
- 결과: 날짜 × 연령 × 수단 이동 건수 (서울 전체, C 유형별)
"""

import glob
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TAB = ROOT / "outputs" / "tables"

MODES = {6: "subway", 4: "bus", 5: "bus", 7: "walk", 8: "car", 1: "other", 2: "other", 3: "other", 9: "other"}
AGES = {"70p": ["male_70_cnt", "feml_70_cnt"],
        "adult": [f"{s}_{a}_cnt" for s in ["male", "feml"] for a in ["20", "30", "40", "50"]]}


def main():
    cl = pd.read_csv(TAB / "c4_dong_clusters.csv", dtype={"adm_cd10": str})
    dong_type = dict(zip(cl["adm_cd10"].str[:8].astype(int), cl["type_name"]))
    cols = ["d_admdong_cd", "move_trans", *AGES["70p"], *AGES["adult"]]
    out = []
    for zf in sorted(glob.glob(str(RAW / "seoul_trans_admdong1_in_*.zip"))):
        z = zipfile.ZipFile(zf)
        for n in sorted(z.namelist()):
            d = pd.read_csv(z.open(n), usecols=cols)
            d = d[d["d_admdong_cd"].isin(dong_type)]
            d["mode"] = d["move_trans"].map(MODES)
            d["type"] = d["d_admdong_cd"].map(dong_type)
            for age, c in AGES.items():
                d[age] = d[c].sum(axis=1)
            g = d.groupby(["type", "mode"])[["70p", "adult"]].sum().reset_index()
            g["date"] = pd.to_datetime(n[-12:-4])
            out.append(g)
        print(f"  {Path(zf).name} 완료", flush=True)
    t = pd.concat(out).melt(id_vars=["date", "type", "mode"], var_name="age", value_name="trips")
    t.to_parquet(PROC / "kt_daily_type_mode.parquet")
    city = t.groupby(["date", "age", "mode"])["trips"].sum().unstack("mode")
    city["total"] = city.sum(axis=1)
    print(city.groupby(level="age").mean().round(0).to_string())
    print(f"날짜 {t['date'].nunique()}일: {t['date'].min().date()} ~ {t['date'].max().date()}")


if __name__ == "__main__":
    main()
