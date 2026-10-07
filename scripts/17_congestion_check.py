"""G. 문제 제기 검증: 고령자가 지하철 혼잡을 만드는가, 무임 승차는 얼마나 되는가.

실행: python scripts/17_congestion_check.py   (01_download.py 다음, 약 1분)

1. 혼잡 시간대의 고령자 비중
   - 서울교통공사 1~8호선 역별·일별·시간대별 승객유형별 승차, 최근 1년(2025.7~2026.6) 평일(공휴일 제외)
   - 우대권(65세 이상, 장애인·국가유공자 포함) ÷ 모든 승객유형 승차
   - 시간대별 비중, 출퇴근 시간(07~09시, 18~19시)과 낮 시간(10~16시)의 비중
   - 가장 붐비는 역·시간대(평일 평균 승차 상위 10%)에서의 고령자 비중
2. 무임 승차 규모: 서울시 지하철 호선별 역별 유·무임 승차(OA-12251), 최근 12개월(2025.9~2026.8) 합계
3. 출퇴근 시간 고령자는 어떻게 움직이나: KT 생활이동 2026.9 평일(추석 제외) 70대 이상 / 20~50대
   - 시간대별 이동량, 수단 구성, 목적 구성 (출근 7~9시, 퇴근 18~19시, 낮 10~16시)
   - 출퇴근 시간대 코드는 20분 단위(700, 720, ...)라 시간 단위로 묶는다. 목적 코드는 18번에서 판별했다
     (1 출근, 2 등교, 3 귀가, 6 병원, 7 기타)
"""

import glob
import importlib.util
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"

for font in ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
    if any(font == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = font
        break
plt.rcParams["axes.unicode_minus"] = False

spec = importlib.util.spec_from_file_location("b10", ROOT / "scripts" / "10_build_weather.py")
b10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b10)

FILES = ["서울교통공사_역별일별시간대별승객유형별승하차_20251231.csv", "서울교통공사_역별일별시간대별승객유형별승하차_20260630.csv"]
PEAK = ["07-08시간대", "08-09시간대", "18-19시간대"]
DAY = ["10-11시간대", "11-12시간대", "12-13시간대", "13-14시간대", "14-15시간대", "15-16시간대"]


def load() -> pd.DataFrame:
    parts = []
    for f in FILES:
        d = pd.read_csv(RAW / f, encoding="cp949")
        d = d[d["승하차구분"] == "승차"]
        d["date"] = pd.to_datetime(d["수송일자"].astype(str))
        d = d[(d["date"].dt.weekday < 5) & ~d["date"].isin(b10.HOLIDAYS)]
        d["elderly"] = d["승객유형"] == "우대권"
        parts.append(d)
    return pd.concat(parts)


def kt_peak() -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("u06", ROOT / "scripts" / "06_build_usage.py")
    u06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(u06)
    hour_of = lambda c: c // 100 if c >= 100 else c
    modes = {6: "지하철", 4: "버스", 5: "버스", 7: "도보", 8: "차량"}
    purposes = {1: "출근", 3: "귀가", 7: "기타"}

    def read(zipname: str, key: str) -> pd.DataFrame:
        import zipfile
        z = zipfile.ZipFile(RAW / zipname)
        names = u06.weekdays(z)
        parts = []
        for n in names:
            d = pd.read_csv(z.open(n), usecols=["d_admdong_cd", "time_cd", key, *u06.AGES["70p"], *u06.AGES["adult"]])
            d = d[d["d_admdong_cd"].astype(str).str.startswith("11")]
            d["70p"], d["adult"] = d[u06.AGES["70p"]].sum(axis=1), d[u06.AGES["adult"]].sum(axis=1)
            d["hour"] = d["time_cd"].map(hour_of)
            parts.append(d.groupby(["hour", key])[["70p", "adult"]].sum())
        return (pd.concat(parts).groupby(level=[0, 1]).sum() / len(names)).reset_index()

    m = read("seoul_trans_admdong1_in_202609.zip", "move_trans")
    m["mode"] = m["move_trans"].map(modes).fillna("기타")
    p = read("seoul_purpose_admdong1_in_202609.zip", "move_purpose")
    p["purpose"] = p["move_purpose"].map(purposes).fillna("그 외")
    rows = []
    for label, hours in [("출근 7~9시", [7, 8]), ("퇴근 18~19시", [18]), ("낮 10~16시", list(range(10, 16))), ("하루", list(range(24)))]:
        x = m[m["hour"].isin(hours)].groupby("mode")[["70p", "adult"]].sum()
        y = p[p["hour"].isin(hours)].groupby("purpose")[["70p", "adult"]].sum()
        r = {"구간": label, "70대_하루중몫": x["70p"].sum() / m["70p"].sum(), "청장년_하루중몫": x["adult"].sum() / m["adult"].sum(),
             "70대_비중(70대+청장년 중)": x["70p"].sum() / x.sum().sum()}
        r.update({f"70대_수단_{k}": v for k, v in (x["70p"] / x["70p"].sum()).items()})
        r.update({f"청장년_수단_{k}": v for k, v in (x["adult"] / x["adult"].sum()).items()})
        r.update({f"70대_목적_{k}": v for k, v in (y["70p"] / y["70p"].sum()).items()})
        r.update({f"청장년_목적_{k}": v for k, v in (y["adult"] / y["adult"].sum()).items()})
        rows.append(r)
    return pd.DataFrame(rows).set_index("구간").T


def main():
    kp = kt_peak()
    kp.to_csv(TAB / "g3_kt_peak_elderly.csv", encoding="utf-8-sig")
    print(kp.round(3).to_string())
    d = load()
    hours = [c for c in d.columns if "시간대" in c]
    ndays = d["date"].nunique()
    print(f"평일 {ndays}일: {d['date'].min().date()} ~ {d['date'].max().date()}")

    tot = d[hours].sum() / ndays
    eld = d.loc[d["elderly"], hours].sum() / ndays
    by_hour = pd.DataFrame({"전체_승차(평일 하루)": tot, "우대권_승차": eld, "우대권_비중": eld / tot,
                            "우대권_하루중_비중": eld / eld.sum(), "전체_하루중_비중": tot / tot.sum()})
    by_hour.to_csv(TAB / "g1_elderly_share_by_hour.csv", encoding="utf-8-sig")
    print(by_hour.round(3).to_string())

    # 역·시간대 단위: 평일 평균 승차 → 상위 10% 가장 붐비는 칸의 고령자 비중
    st = d.groupby(["역명", "elderly"])[hours].sum().div(ndays)
    total = st.groupby(level=0).sum().stack()
    elderly = st.xs(True, level=1).stack().reindex(total.index).fillna(0)
    cells = pd.DataFrame({"total": total, "elderly": elderly})
    top = cells["total"] >= cells["total"].quantile(0.9)
    summary = pd.DataFrame([
        {"구분": "하루 전체", "우대권_비중": eld.sum() / tot.sum()},
        {"구분": "출퇴근 (07~09시, 18~19시)", "우대권_비중": eld[PEAK].sum() / tot[PEAK].sum()},
        {"구분": "낮 (10~16시)", "우대권_비중": eld[DAY].sum() / tot[DAY].sum()},
        {"구분": "가장 붐비는 역·시간대 상위 10%", "우대권_비중": cells.loc[top, "elderly"].sum() / cells.loc[top, "total"].sum()},
        {"구분": "나머지 역·시간대 90%", "우대권_비중": cells.loc[~top, "elderly"].sum() / cells.loc[~top, "total"].sum()},
        {"구분": "우대권 승차 중 출퇴근 시간 몫", "우대권_비중": eld[PEAK].sum() / eld.sum()},
        {"구분": "전체 승차 중 출퇴근 시간 몫", "우대권_비중": tot[PEAK].sum() / tot.sum()},
    ])

    # 무임 승차 규모 (서울 관할 전 노선, 최근 12개월)
    f = pd.read_csv(RAW / "서울시 지하철 호선별 역별 유무임 승하차 인원 정보.csv", encoding="cp949")
    f = f[f["사용월"].between(202509, 202608)]
    free, paid = f["무임승차인원"].sum(), f["유임승차인원"].sum()
    summary = pd.concat([summary, pd.DataFrame([
        {"구분": "무임 승차 (2025.9~2026.8, 건)", "우대권_비중": free},
        {"구분": "무임 승차 비중 (유임+무임 중)", "우대권_비중": free / (free + paid)}])])
    summary.to_csv(TAB / "g2_congestion_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(8, 3.2))
    labels = [h.replace("시간대", "").replace("이전", "↓").replace("이후", "↑") for h in hours]
    x = np.arange(len(hours))
    peak = [h in PEAK for h in hours]
    ax.bar(x, by_hour["우대권_비중"] * 100, color=["#eb6834" if p else "#2a78d6" for p in peak], width=0.75)
    for k in x[peak]:
        ax.text(k, by_hour["우대권_비중"].iloc[k] * 100, f"{by_hour['우대권_비중'].iloc[k]:.0%}", ha="center", va="bottom", fontsize=7)
    ax.axhline(eld.sum() / tot.sum() * 100, color="#52514e", lw=1, ls="--")
    ax.text(len(hours) - 0.5, eld.sum() / tot.sum() * 100, f"하루 평균 {eld.sum() / tot.sum():.0%}", ha="right", va="bottom", fontsize=8, color="#52514e")
    ax.set_xticks(x, labels, rotation=60, fontsize=7)
    ax.set_ylabel("승차 중 65세 이상(우대권) 비중 (%)")
    ax.set_title("가장 붐비는 출퇴근 시간(주황)에 고령자 비중이 가장 낮다 (1~8호선, 2025.7~2026.6 평일)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "g1_elderly_share_by_hour.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {TAB}/g*.csv, {FIG}/g1_elderly_share_by_hour.png")


if __name__ == "__main__":
    main()
