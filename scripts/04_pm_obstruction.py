"""공유 전동킥보드: 고령자의 수단이 아니라 보행을 막는 요인으로 본다.

실행: python scripts/04_pm_obstruction.py   (03_analysis.py 다음)

- 업체별 기기 대수 추이 (2023.10 → 2025.2 → 2025.12)
- 2025년 견인 신고를 유형별로 나눠, 보행 공간(횡단보도·보도·보호구역·점자블록·정류소·역 입구)을 막은 비율을 본다
- 자치구별 견인 건수와 이동선택권 우선지역 수를 나란히 놓는다

주의: 견인 건수는 자치구의 단속·견인 시행 여부에 크게 좌우된다 (예: 종로·성북구는 거의 0건).
따라서 구별 건수는 '방치 정도'가 아니라 '견인으로 확인된 사례'로만 해석한다.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
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

# 보행자가 다니는 공간을 막은 유형 (차도·자전거도로·교통섬·그 외는 제외)
WALKWAY_TYPES = ["횡단보도 주변(3m)", "보호구역(어린이·노인·장애인)", "상기 사항 외 보도",
                 "버스정류소 및 택시승강장 주변(5m 이내)", "점자블록 또는 교통약자 엘리베이터", "지하철역 전면(5m이내)"]


def device_counts() -> pd.DataFrame:
    files = {"2023-10": ("서울시 공유 전동킥보드 운영 현황_20231018.csv", 0, 2),
             "2025-02": ("서울시 민간대여 공유 전동킥보드 기기 현황_25.2월기준.csv", 2, 3),
             "2025-12": ("서울시 민간대여 공유 전동킥보드 기기 현황_25.12월기준.csv", 2, 3)}
    rows = []
    for when, (fname, header, col) in files.items():
        t = pd.read_csv(RAW / fname, encoding="cp949", header=header)
        n = pd.to_numeric(t.iloc[:, col].astype(str).str.replace(",", ""), errors="coerce").sum()
        rows.append({"기준": when, "기기 대수": int(n), "브랜드 수": int(t.iloc[:, col].notna().sum())})
    return pd.DataFrame(rows)


def towing() -> pd.DataFrame:
    t = pd.concat([pd.read_excel(RAW / f, skiprows=1) for f in
                   ["서울특별시_전동킥보드_견인_현황(25.1~6월).xlsx", "서울특별시_전동킥보드_견인_현황(25.7~12월).xlsx"]])
    t = t[t["유형"] != "오신고"]
    t["보행공간"] = t["유형"].isin(WALKWAY_TYPES)
    return t


def main():
    dev = device_counts()
    tow = towing()
    by_type = tow["유형"].value_counts().rename("건수").to_frame()
    by_type["비율"] = by_type["건수"] / len(tow)
    by_type["보행공간"] = by_type.index.isin(WALKWAY_TYPES)

    pri = pd.read_csv(TAB / "b2_priority_dongs.csv")
    by_gu = (tow.groupby("구정보").agg(견인=("유형", "size"), 보행공간_견인=("보행공간", "sum"))
             .join(pri["gu"].value_counts().rename("우선지역_동수"), how="outer").fillna(0).astype(int)
             .sort_values("견인", ascending=False))

    dev.to_csv(TAB / "b5_pm_devices.csv", index=False, encoding="utf-8-sig")
    by_type.to_csv(TAB / "b5_pm_towing_by_type.csv", encoding="utf-8-sig")
    by_gu.to_csv(TAB / "b5_pm_towing_by_gu.csv", encoding="utf-8-sig")
    print(dev.to_string(index=False))
    print(f"\n2025년 견인 {len(tow):,}건 중 보행공간 {tow['보행공간'].sum():,}건 ({tow['보행공간'].mean():.1%})")
    print(by_type.round(3).to_string())
    print(by_gu.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 2]})
    ax = axes[0]
    ax.bar(dev["기준"], dev["기기 대수"], color="#718096")
    for i, v in enumerate(dev["기기 대수"]):
        ax.text(i, v, f"{v:,}대", ha="center", va="bottom", fontsize=10)
    ax.set_title("서울 공유 전동킥보드 기기 대수")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])

    ax = axes[1]
    bt = by_type.iloc[::-1]
    ax.barh(bt.index, bt["건수"], color=["#c53030" if w else "#cbd5e0" for w in bt["보행공간"]])
    for i, (n, r) in enumerate(zip(bt["건수"], bt["비율"])):
        ax.text(n, i, f" {n:,} ({r:.0%})", va="center", fontsize=9)
    ax.set_title(f"2025년 킥보드 견인 {len(tow):,}건 중 보행 공간을 막은 경우 {tow['보행공간'].mean():.0%} (빨강)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, bt["건수"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(FIG / "b5_pm_obstruction.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
