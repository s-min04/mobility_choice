"""전체 분석을 한 줄로 다시 실행한다.

    python run_all.py            # 원본 내려받기(약 5GB)부터 22번까지 전부 (1~2시간)
    python run_all.py --quick    # 원본 없이, 저장소에 들어 있는 가공 데이터로 돌아가는 단계만 (약 3분)
    python run_all.py --from 14  # 14번부터 끝까지
    python run_all.py --only 19 20

각 단계가 끝나면 걸린 시간을 출력하고, 실패하면 그 자리에서 멈춘다.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (번호, 파일, 원본 필요 여부, 내용)
STEPS = [
    (1, "01_download.py", True, "원본 데이터 내려받기 (manifest.csv 에 출처·SHA-256 기록)"),
    (2, "02_build_index.py", True, "B 이동선택권 지수: 100m 격자 → 행정동"),
    (3, "03_analysis.py", True, "B 격차 분해, LISA, 민감도"),
    (4, "04_pm_obstruction.py", True, "B 공유 킥보드 보행 방해"),
    (5, "05_taxi_call_trend.py", False, "B 동행 콜택시 이용 추이"),
    (6, "06_build_usage.py", True, "A KT 생활이동 동별 수단·외출, 소득, 경사"),
    (7, "07_model_usage.py", False, "A 검증 회귀, 랜덤포레스트·SHAP, 외출 회귀"),
    (8, "08_verify_elderly.py", True, "A 교통카드 역 단위 검정, 자동차 등록"),
    (9, "09_cluster.py", True, "C 동 유형 군집, 합의 군집, 1순위 72개 동"),
    (10, "10_build_weather.py", True, "D 지하철 일별 승차, 기상"),
    (12, "12_build_kt_daily.py", True, "D KT 연령×수단 일별 (약 20분)"),
    (13, "13_build_local_weather.py", True, "D S-DoT·강우량계 동네 날씨 (약 20분)"),
    (11, "11_weather_model.py", True, "D 날씨 효과와 교차 확인"),
    (14, "14_build_carbon_inputs.py", True, "E 배출계수, 차량 이동거리"),
    (15, "15_carbon_scenarios.py", False, "E 탄소 감축 시나리오, 민감도"),
    (16, "16_policy_diagnosis.py", False, "F 판별 4분면, 요금 혜택 귀착"),
    (17, "17_congestion_check.py", True, "G 혼잡 쟁점, 출퇴근 시간 고령자"),
    (18, "18_verify_mode_codes.py", True, "H KT 수단·목적 코드 검증"),
    (19, "19_official_standard.py", True, "I 공식 기준판 지수, 우회계수 실측"),
    (20, "20_identification.py", False, "J 식별 강화, S1 범위, 1인당 CO2"),
    (21, "21_medical_access.py", True, "K 의료 목적지 접근"),
    (22, "22_cost_effect.py", False, "L 재배분 비용·효과"),
    (23, "23_carbon_levers.py", True, "M 탄소 레버: 거리대·출발지별 배출과 감축 여지"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="원본이 필요 없는 단계만")
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--only", type=int, nargs="*")
    a = ap.parse_args()
    steps = [s for s in STEPS if s[0] >= a.start and (not a.quick or not s[2]) and (not a.only or s[0] in a.only)]
    t0 = time.time()
    for num, file, _, desc in steps:
        print(f"\n=== [{num:02d}] {desc} ({file})", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / file)], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"[{num:02d}] {file} 실패 (코드 {r.returncode}). 위 오류를 확인하세요.")
        print(f"--- [{num:02d}] 완료 {time.time() - t:.0f}초", flush=True)
    print(f"\n전체 {len(steps)}단계 완료, {(time.time() - t0) / 60:.1f}분. 결과: outputs/figures, outputs/tables")


if __name__ == "__main__":
    main()
