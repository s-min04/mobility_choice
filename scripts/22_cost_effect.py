"""L. 재배분 비용과 효과: 버스 요금 지원 예산 일부를 선택권 낮은 동의 마을버스·DRT 공급에 쓰면.

실행: python scripts/22_cost_effect.py   (20_identification.py 다음, 1초)

공개 데이터가 없는 단가·예산은 보도로 확인한 값이며 아래 상수에 출처를 적었다(원문 확인 필요).
- 어르신 버스 교통비 지원 예산: 조례안 비용추계서 2027년 1,047.2억 원(5년 5,788.6억 원), 서울시 검토안(월 15회 미만 이용자 한정) 연 약 525억 원
- 마을버스 1대 하루 운송원가: 510,457원 (서울시·마을버스조합 합의, 2026년 적용)
- 공급 규모: 1순위 72개 동마다 마을버스·DRT 2대(동작CALL버스 규모) / 4대, 선택권 낮은 258개 동마다 2대
- 효과: E의 S1 (20번의 기울기 범위: Oster 보정 ~ 기본 ~ 통제 없음). S1은 72개 동의 고령자 지수를 목표까지 올렸을 때의 값이다.
  공급 대수가 그 지수 상승을 실제로 만들어 내는지는 이 분석으로 확인할 수 없다 → 대수는 시나리오 가정이다.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "tables"

BUDGET_ORDINANCE = 1047.2e8     # 원, 2027년 (조례안 비용추계서, 보도: 머니투데이 2026.6.16)
BUDGET_LIMITED = 525e8          # 원, 월 15회 미만 이용자 한정안 (보도: 브라보마이라이프·네이트 2026.6.24)
MAEUL_COST_DAY = 510_457        # 원/대·일 (서울시·마을버스조합 합의, 보도: 머니투데이 2025.12.22)
DAYS = 365


def main():
    s1 = pd.read_csv(TAB / "j2_s1_by_slope.csv").set_index("모형")
    eff = {"낮음 (Oster 보정)": s1.loc["(h) Oster 보정 (δ=1, R²max=1.3R²)"],
           "기본": s1.loc["(b) 기본: 소득·밀도·도심거리"],
           "높음 (통제 없음)": s1.loc["(a) 통제 없음"]}
    per_vehicle = MAEUL_COST_DAY * DAYS
    rows = []
    for supply, n_dongs, n_veh in [("1순위 72개 동 × 2대", 72, 2), ("1순위 72개 동 × 4대", 72, 4), ("선택권 낮은 258개 동 × 2대", 258, 2)]:
        cost = n_dongs * n_veh * per_vehicle
        r = {"공급 시나리오": supply, "차량_대수": n_dongs * n_veh, "연간비용_억원": cost / 1e8,
             "조례예산(1,047억)_대비": cost / BUDGET_ORDINANCE,
             "한정안과의_차액(522억)_대비": cost / (BUDGET_ORDINANCE - BUDGET_LIMITED)}
        if n_dongs == 72:
            for k, e in eff.items():
                r[f"고령자_차량이동_1건당_원({k})"] = cost / (e["S1_고령자_차량이동감소_평일"] * 250)
                r[f"tCO2_1톤당_만원({k}, 동네전체)"] = cost / e["S1_동네전체_t"] / 1e4
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "l1_cost_effect.csv", index=False, encoding="utf-8-sig")
    print(f"마을버스 1대 연간 운송원가 = {per_vehicle / 1e8:.2f}억 원")
    print(t.T.round(2).to_string())


if __name__ == "__main__":
    main()
