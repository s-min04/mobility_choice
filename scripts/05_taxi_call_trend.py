"""동행 온다 콜택시(전화 호출 택시) 이용 추이와, 이용이 적은 이유를 가리는 비교 수치.

실행: python scripts/05_taxi_call_trend.py

공개 데이터셋이 없어 서울시 발표를 인용한 기사 수치를 그대로 옮겼다 (SOURCES 참고).
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"

for font in ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
    if any(font == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = font
        break
plt.rcParams["axes.unicode_minus"] = False

# 월간 운행 완료 건수
MONTHLY = {"2025-07": 909, "2025-08": 2013, "2025-10": 3038, "2025-12": 4476, "2026-05": 6820, "2026-07": 10968}
EVENTS = {"2026-07-06": "120 다산콜 연결", "2026-08-13": "120 → 6번 바로 연결(ARS)"}
ARS_DAILY = {"before_0807_0812": 125, "after_0813_0818": 160}
SOURCES = {
    "2025-07~2026-05": "경향신문 2026.7.6 https://www.khan.co.kr/article/202607062105025",
    "2026-07, 누적 6.3만": "머니투데이 2026.8.18 https://www.mt.co.kr/policy/2026/08/18/2026081809501533651",
    "ARS 전후 하루 평균": "헤럴드경제 https://biz.heraldcorp.com/article/10870850",
}

ELDERLY_65 = 1_961_251           # 행정안전부 주민등록 2026.8.31 (02_build_index 와 같은 값)
NO_APP_SHARE = 0.81              # 60세 이상 택시 앱 미사용 (서울연구원 2024, 기사 재인용)
BIKE_65_PER_PERSON_6M = None     # data/processed/mci_meta.json 에서 읽는다
TAXI_MAIN_MODE_65 = 0.069        # 2023 노인실태조사: 외출 시 주 이용 교통수단이 택시 (전국)
MONTHLY_PENSION = 695_000        # 2025 고령자 통계: 65세 이상 월평균 연금 수급액 (2023)
FARE = {"base": 4800, "base_m": 1600, "per_100won_m": 131}  # 서울 중형택시 (2023.2~)


def taxi_fare(km: float) -> int:
    """거리요금만 계산 (시간요금 제외)."""
    extra = max(0, km * 1000 - FARE["base_m"])
    return FARE["base"] + int(-(-extra // FARE["per_100won_m"])) * 100


def main():
    meta = json.loads((ROOT / "data" / "processed" / "mci_meta.json").read_text())
    bike_per_person_month = meta["bike_age_factor"]["rate_65_plus"] / 6
    no_app = ELDERLY_65 * NO_APP_SHARE
    s = pd.Series(MONTHLY)
    fare_5km = taxi_fare(5.4)  # 서울 택시 평균 탑승거리 5.4km (2012 서울시 조사, 오래된 값)
    out = {
        "monthly": MONTHLY,
        "growth_2025-12_to_2026-05_per_month": (s["2026-05"] / s["2025-12"]) ** (1 / 5) - 1,
        "growth_2026-05_to_2026-07_per_month": (s["2026-07"] / s["2026-05"]) ** (1 / 2) - 1,
        "ars_daily_change": ARS_DAILY["after_0813_0818"] / ARS_DAILY["before_0807_0812"] - 1,
        "no_app_elderly": no_app,
        "call_taxi_per_1000_no_app_2026-07": s["2026-07"] / no_app * 1000,
        "bike_per_1000_elderly_month": bike_per_person_month * 1000,
        "taxi_main_mode_elderly_est": ELDERLY_65 * TAXI_MAIN_MODE_65,
        "fare_5_4km": fare_5km,
        "round_trip_weekly_share_of_pension": fare_5km * 2 * 4 / MONTHLY_PENSION,
        "sources": SOURCES,
    }
    (TAB / "b6_taxi_call.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "sources"}, ensure_ascii=False, indent=1, default=float))

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = pd.to_datetime(list(MONTHLY) , format="%Y-%m") + pd.Timedelta(days=14)
    ax.plot(x, s.values, marker="o", color="#805ad5")
    for xi, v in zip(x, s.values):
        ax.annotate(f"{v:,}", (xi, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    for d, label in EVENTS.items():
        ax.axvline(pd.Timestamp(d), color="#a0aec0", ls="--", lw=1)
        ax.text(pd.Timestamp(d), s.max() * 0.35, f" {label}", rotation=90, va="bottom", fontsize=8.5, color="#4a5568")
    ax.set_ylim(0, s.max() * 1.25)
    ax.set_title(f"동행 온다 콜택시 월 이용 건수 — 늘고 있지만 앱 못 쓰는 고령자 1,000명당 월 "
                 f"{out['call_taxi_per_1000_no_app_2026-07']:.0f}건", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("월 운행 완료 건수")
    fig.savefig(FIG / "b6_taxi_call_trend.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
