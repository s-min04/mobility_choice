"""H. KT 생활이동 수단·목적 코드 판별 검증.

실행: python scripts/18_verify_mode_codes.py   (10_build_weather.py, 14_build_carbon_inputs.py 다음, 약 5분)

공개 명세에는 수단 9개(항공, 기차, 고속버스, 지하철, 광역버스, 일반버스, 도보, 차량, 기타)와
목적 7개(출근, 등교, 쇼핑, 관광, 병원, 귀가, 기타)의 이름만 있고 코드 번호별 정의가 없다.
06번은 시간대 패턴으로 6=지하철, 4·5=버스, 7=도보, 8=차량, 3=귀가로 판별했다. 이를 외부 실측과 대조한다.

1. 일별 대조: KT 코드별 서울 도착 이동(평일·주말 포함 일별) vs 서울교통공사 교통카드 1~8호선 일별 승차
   (2025.7·8·12, 2026.1·2 - 두 자료가 겹치는 달). 지하철 코드라면 실측과 상관이 가장 높아야 한다.
2. 시간대 대조: 코드별 시간대 분포 vs 교통카드 시간대 분포. 지하철은 01~04시 운행이 없다.
3. 이동거리: 출도착 × 수단(OA-22657, 2026.9 평일 5일)의 코드별 평균 직선거리 (항공·기차는 수백 km, 도보는 1km 미만)
4. 목적 코드: 70대 이상 / 20~50대 구성. 등교라면 70대 이상이 0, 병원이라면 70대 이상 비중이 높아야 한다.
"""

import glob
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TAB = ROOT / "outputs" / "tables"

MONTHS = ["202507", "202508", "202512", "202601", "202602"]
E70 = ["male_70_cnt", "feml_70_cnt"]
ADULT = [f"{s}_{a}_cnt" for s in ["male", "feml"] for a in ["20", "30", "40", "50"]]
hour_of = lambda c: c // 100 if c >= 100 else c  # 출퇴근 시간대는 20분 단위(700, 720, ...)로 나뉘어 있다


def kt_by_code() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily, hourly = [], []
    for m in MONTHS:
        z = zipfile.ZipFile(RAW / f"seoul_trans_admdong1_in_{m}.zip")
        for n in sorted(z.namelist()):
            d = pd.read_csv(z.open(n), usecols=["d_admdong_cd", "time_cd", "move_trans", "total_cnt", *E70])
            d = d[d["d_admdong_cd"].astype(str).str.startswith("11")]
            d["e70"] = d[E70].sum(axis=1)
            d["hour"] = d["time_cd"].map(hour_of)
            date = pd.to_datetime(n[-12:-4])
            daily.append(d.groupby("move_trans")[["total_cnt", "e70"]].sum().assign(date=date))
            hourly.append(d.groupby(["move_trans", "hour"])["total_cnt"].sum())
        print(f"  KT {m} 완료", flush=True)
    daily = pd.concat(daily).reset_index()
    hourly = pd.concat(hourly).groupby(level=[0, 1]).sum().unstack(fill_value=0)
    return daily, hourly


def main():
    card = pd.read_parquet(PROC / "weather_city_daily.parquet")[["date", "elderly", "general"]]
    daily, hourly = kt_by_code()
    rows = []
    for code, g in daily.groupby("move_trans"):
        x = g.merge(card, on="date")
        rows.append({"코드": code, "일수": len(x), "KT_하루평균(전체)": x["total_cnt"].mean(),
                     "상관_전체_vs_카드일반": np.corrcoef(x["total_cnt"], x["general"])[0, 1],
                     "상관_70대_vs_카드우대권": np.corrcoef(x["e70"], x["elderly"])[0, 1]})
    t = pd.DataFrame(rows).set_index("코드")

    # 시간대: 카드 h00 = 06시 이전, h01 = 06~07시, ..., h18 = 23~24시, h19 = 24시 이후
    ch = pd.read_parquet(PROC / "weather_city_hourly.parquet")
    ch = ch[ch["date"].dt.strftime("%Y%m").isin(MONTHS)].groupby("type")[[f"h{i:02d}" for i in range(20)]].sum()
    card_prof = ch.sum()
    card_prof = card_prof / card_prof.sum()
    for code in hourly.index:
        h = hourly.loc[code]
        kt = [h.loc[h.index <= 5].sum()] + [h.get(k, 0) for k in range(6, 24)] + [0]  # 같은 20칸으로 맞춘다
        kt = np.array(kt, dtype=float) / h.sum()
        t.loc[code, "시간대모양_상관_vs_카드"] = np.corrcoef(kt, card_prof.values)[0, 1]
        t.loc[code, "01~04시_비중"] = h.loc[[1, 2, 3, 4]].sum() / h.sum() if h.sum() else np.nan

    # 이동거리: OA-22657 하루(평일)의 코드별 평균 직선거리 (서울 도착, 내국인)
    z = zipfile.ZipFile(RAW / "seoul_trans_admdong3_final_20260903.zip")
    od = pd.read_csv(z.open(z.namelist()[0]), usecols=["d_admdong_cd", "in_forn_div_nm", "move_trans", "move_dist", "cnt"])
    od = od[(od["in_forn_div_nm"] == "내국인") & od["d_admdong_cd"].astype(str).str.startswith("11")]
    dist = od.groupby("move_trans").apply(lambda g: np.average(g["move_dist"], weights=g["cnt"]) / 1000)
    t["평균_직선거리_km"] = dist
    judged = {1: "항공", 2: "기차", 3: "고속버스", 4: "버스(광역 추정)", 5: "버스(일반 추정)", 6: "지하철", 7: "도보", 8: "차량", 9: "기타"}
    t["판별"] = t.index.map(judged)
    t.to_csv(TAB / "h1_mode_code_check.csv", encoding="utf-8-sig")
    print(t.round(3).to_string())

    # 목적 코드: 2026.9 평일, 70대 이상 / 20~50대 구성
    z = zipfile.ZipFile(RAW / "seoul_purpose_admdong1_in_202609.zip")
    parts = []
    for n in sorted(z.namelist()):
        day = pd.to_datetime(n[-12:-4])
        if day.weekday() >= 5:
            continue
        d = pd.read_csv(z.open(n), usecols=["d_admdong_cd", "time_cd", "move_purpose", *E70, *ADULT])
        d = d[d["d_admdong_cd"].astype(str).str.startswith("11")]
        d["e70"], d["adult"] = d[E70].sum(axis=1), d[ADULT].sum(axis=1)
        d["hour"] = d["time_cd"].map(hour_of)
        parts.append(d.groupby(["move_purpose", "hour"])[["e70", "adult"]].sum())
    p = pd.concat(parts).groupby(level=[0, 1]).sum()
    tot = p.groupby(level=0).sum()
    pr = pd.DataFrame({"70대_구성": tot["e70"] / tot["e70"].sum(), "청장년_구성": tot["adult"] / tot["adult"].sum()})
    pr["70대÷청장년"] = pr["70대_구성"] / pr["청장년_구성"]
    peak_hour = p["adult"].unstack().idxmax(axis=1)
    pr["청장년_최다_시간대"] = peak_hour
    pr["판별"] = pr.index.map({1: "출근 (오전 7~8시 최다)", 2: "등교 (70대 이상 0)", 3: "귀가 (저녁 최다)", 4: "쇼핑 또는 관광",
                                5: "쇼핑 또는 관광", 6: "병원 (70대 이상 비중 높음)", 7: "기타"})
    pr.to_csv(TAB / "h2_purpose_code_check.csv", encoding="utf-8-sig")
    print(pr.round(4).to_string())


if __name__ == "__main__":
    main()
