"""C 분석: 공급(B)·행동(A)·소득·인구·목적지·지형으로 행정동 유형을 나누고 유형별 정책을 정한다.

실행: python scripts/09_cluster.py   (06·07·08 다음)

- 변수 11개 (표준화):
    공급   지하철 접근(고령자 10분), 버스 접근(고령자 5분)
    목적지 고령자 걸음 10분(369m) 안 병·의원 수 (로그)
    행동   70대 이상 차량 분담률, 70대 이상 1인당 하루 외출
    인구   65세 이상 비율, 70세 이상 중 80세 이상 비율
    소득   고령자 기초생계급여 수급률, 아파트 평균 시가(로그), 인구 1,000명당 자가용 승용차(로그)
    지형   경사 8% 이상 비율
- K-평균. 군집 수는 3~9개 중 실루엣 지수, Ward 계층 군집과의 일치도(ARI), 부트스트랩 안정성을 보고 정한다.
- 유형 이름은 군집별 평균 특징(표준점수)을 보고 붙였다. 군집 번호는 고령자 차량 분담률이 낮은 순서로 정렬해 고정한다.
- 동별 확신도(합의 군집): 동의 80%를 뽑아 군집하기를 300번 반복해, 각 동이 자기 유형의 다른 동들과 같은 군집에
  묶인 비율의 평균을 구한다. 0.8 이상 = 핵심 동, 0.7~0.8 = 보통, 0.7 미만 = 경계 동.
  경계 동에는 두 번째로 자주 함께 묶인 유형을 같이 적는다.
"""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
SEED = 0
K = 5
R_ELDERLY = 369

for font in ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
    if any(font == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = font
        break
plt.rcParams["axes.unicode_minus"] = False

FEATURES = {
    "subway_elderly": "지하철 접근",
    "a_bus_elderly": "버스 접근",
    "clinics_log": "병·의원 접근",
    "car_share_70p": "고령자 차량 분담률",
    "out_rate_70p": "고령자 외출 횟수",
    "elderly_share": "65세 이상 비율",
    "share80_in70": "80세 이상 비중",
    "welfare_65p_rate": "고령 수급률",
    "log_apt_price": "아파트 시가",
    "cars_log": "자가용 보유",
    "steep_share": "급경사 비율",
}

# 군집별 평균 표준점수를 보고 붙인 이름과 정책 처방 (번호는 고령자 차량 분담률이 낮은 순)
NAMES = {
    0: "대중교통 양호 서민 주거지",
    1: "언덕 위 고령 주거지",
    2: "고소득 활동 지역",
    3: "교통 사각 저소득 고령 밀집",
    4: "외곽 차 중심 신주거지",
}
POLICY = {
    0: "현 수준 유지. 고령자 대중교통 이용이 이미 높다 → 역·정류장 보행 편의(쉼터·엘리베이터) 위주",
    1: "보행 부담 완화: 언덕 구간 마을버스 증회·순환 노선, 계단 대신 경사로·쉼터, 가까운 병·의원 연계",
    2: "정책 우선순위 낮음. 차량 이용은 소득·차 보유로 설명 → 탄소 측면의 일반 교통수요관리 대상",
    3: "최우선: 수요응답형 버스·마을버스로 지하철까지 연결, 저소득 고령자 교통비(동행 온다 콜택시 요금) 지원, "
       "복지관·병원 등 목적지 연계 외출 지원",
    4: "탄소 감축 우선: 차량 이동을 대중교통으로 돌릴 간선·광역 버스 공급, 고령화 대비 선제적 노선 확보",
}
TIER = {0: 3, 1: 2, 2: 4, 3: 1, 4: 2}  # 정책 우선순위 (1 = 최우선)
COLORS = {0: "#4299e1", 1: "#9f7aea", 2: "#a0aec0", 3: "#e53e3e", 4: "#ed8936"}


def build() -> gpd.GeoDataFrame:
    d = gpd.read_file(PROC / "dong_usage.gpkg").set_index("adm_cd10")
    c = pd.read_csv(RAW / "서울시 병의원 위치 정보.csv", encoding="cp949")
    c = c[c["병원분류명"].isin(["의원", "병원", "종합병원", "보건소", "한의원", "요양병원", "한방병원"])]
    c = c.dropna(subset=["병원경도", "병원위도"])
    p = gpd.GeoSeries(gpd.points_from_xy(c["병원경도"], c["병원위도"]), crs="EPSG:4326").to_crs("EPSG:5179")
    g = pd.read_parquet(PROC / "grid_slope.parquet")
    g["clinics"] = [len(x) for x in cKDTree(np.c_[p.x, p.y]).query_ball_point(g[["x", "y"]].to_numpy(), R_ELDERLY)]
    d["clinics_elderly10"] = g.groupby("adm_cd10")["clinics"].mean()
    cars = pd.read_csv(RAW / "서울시 자치구 읍면동별 연료별 자동차 등록현황(행정동)(26년8월).csv", encoding="cp949",
                       dtype={"동_코드": str})
    cars = cars[cars["읍면동(행정동)"].str.strip() != "기타"].groupby("동_코드")["승용.1"].sum()
    d["cars_per_1000"] = cars / d["pop_total"] * 1000
    d["clinics_log"] = np.log1p(d["clinics_elderly10"])
    d["cars_log"] = np.log(d["cars_per_1000"])
    return d


def choose_k(Z: np.ndarray) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)
    for k in range(3, 10):
        km = KMeans(k, n_init=50, random_state=SEED).fit(Z)
        ward = AgglomerativeClustering(k, linkage="ward").fit(Z)
        boot = []
        for _ in range(50):
            idx = rng.choice(len(Z), len(Z), replace=True)
            b = KMeans(k, n_init=10, random_state=SEED).fit(Z[idx])
            boot.append(adjusted_rand_score(km.labels_, b.predict(Z)))
        rows.append({"군집 수": k, "실루엣": silhouette_score(Z, km.labels_),
                     "Calinski-Harabasz": calinski_harabasz_score(Z, km.labels_),
                     "Ward와 일치도(ARI)": adjusted_rand_score(km.labels_, ward.labels_),
                     "부트스트랩 안정성(ARI 평균)": np.mean(boot), "가장 작은 군집": np.bincount(km.labels_).min()})
    return pd.DataFrame(rows)


def cluster_stability(Z: np.ndarray, labels: np.ndarray, n=200) -> pd.Series:
    """군집별 부트스트랩 Jaccard 평균 (Hennig 2007): 0.75 이상 안정, 0.6~0.75 보통, 0.6 미만 불안정."""
    rng = np.random.default_rng(SEED)
    jac = {c: [] for c in np.unique(labels)}
    for _ in range(n):
        idx = rng.choice(len(Z), len(Z), replace=True)
        b = KMeans(K, n_init=10, random_state=SEED).fit(Z[idx]).predict(Z)
        for c in jac:
            a = labels == c
            jac[c].append(max(((a & (b == j)).sum() / (a | (b == j)).sum()) for j in np.unique(b)))
    return pd.Series({c: np.mean(v) for c, v in jac.items()})


def fig_map(d: gpd.GeoDataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    core = d["certainty"] == "핵심"
    d[core].plot(color=d.loc[core, "cluster"].map(COLORS), ax=ax, edgecolor="white", linewidth=0.2)
    d[~core].plot(color=d.loc[~core, "cluster"].map(COLORS), ax=ax, edgecolor="white", linewidth=0.2, alpha=0.4)
    d[d["b_priority"]].boundary.plot(ax=ax, color="black", linewidth=0.9)
    d[d["a_overlap10"]].plot(ax=ax, facecolor="none", edgecolor="black", hatch="///", linewidth=0)
    gu = d.dissolve("gu")
    gu.boundary.plot(ax=ax, color="#4a5568", linewidth=0.6)
    for name, row in gu.iterrows():
        pt = row.geometry.representative_point()
        ax.text(pt.x, pt.y, name, fontsize=6.5, ha="center", color="#2d3748")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in NAMES]
    labels = [f"{NAMES[c]} ({(d['cluster'] == c).sum()}개 동)" for c in NAMES]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black"),
                plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", hatch="///")]
    labels += ["B 우선지역 (검은 테두리)", "B·A 공통 사각지대 10개 동 (빗금)"]
    handles += [plt.Rectangle((0, 0), 1, 1, color="#a0aec0", alpha=0.4)]
    labels += [f"흐린 색: 유형 경계·보통 동 ({(~core).sum()}개, 합의 군집 확신도 0.8 미만)"]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=8.5, frameon=False)
    ax.set_title("고령자 이동 여건에 따른 서울 행정동 유형 (K-평균, 5개)", fontsize=12)
    ax.set_axis_off()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(zs: pd.DataFrame, prof: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(zs.to_numpy(), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(zs.shape[1]), zs.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(zs)), [f"{NAMES[c]}\n({prof.loc[c, '동수']}개 동, 고령자 {prof.loc[c, '고령인구'] / 1e4:.0f}만)"
                                   for c in zs.index], fontsize=8.5)
    for i in range(zs.shape[0]):
        for j in range(zs.shape[1]):
            ax.text(j, i, f"{zs.iat[i, j]:+.1f}", ha="center", va="center", fontsize=7.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="서울 평균 대비 (표준점수)")
    ax.set_title("유형별 특징 (빨강 = 서울 평균보다 높음, 파랑 = 낮음)", fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def consensus(Z: np.ndarray, labels: np.ndarray, n_rep=300, frac=0.8) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = len(Z)
    together = np.zeros((n, n))
    sampled = np.zeros((n, n))
    for b in range(n_rep):
        idx = rng.choice(n, int(n * frac), replace=False)
        lab = KMeans(K, n_init=5, random_state=b).fit(Z[idx]).labels_
        together[np.ix_(idx, idx)] += lab[:, None] == lab[None, :]
        sampled[np.ix_(idx, idx)] += 1
    C = together / np.maximum(sampled, 1)
    by_type = np.column_stack([C[:, labels == c].mean(axis=1) for c in range(K)])
    own = by_type[np.arange(n), labels]
    other = by_type.copy()
    other[np.arange(n), labels] = -1
    out = pd.DataFrame({"confidence": own, "second_type": other.argmax(axis=1), "second_score": other.max(axis=1)})
    out["certainty"] = np.select([own >= 0.8, own >= 0.7], ["핵심", "보통"], default="경계")
    return out


def type_robustness(Z: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """각 유형이 군집 수(4·6개)나 방법(Ward)을 바꿔도 비슷한 묶음으로 다시 나오는지 (가장 비슷한 군집과의 Jaccard)."""
    alts = {"K-평균 4개": KMeans(4, n_init=50, random_state=SEED).fit(Z).labels_,
            "K-평균 6개": KMeans(6, n_init=50, random_state=SEED).fit(Z).labels_,
            "Ward 5개": AgglomerativeClustering(5, linkage="ward").fit(Z).labels_}
    rows = {}
    for c in range(K):
        a = labels == c
        rows[NAMES[c]] = {k: max(((a & (b == j)).sum() / (a | (b == j)).sum()) for j in np.unique(b))
                          for k, b in alts.items()}
    return pd.DataFrame(rows).T


def main():
    d = build()
    X = d[list(FEATURES)]
    Z = StandardScaler().fit_transform(X)

    sel = choose_k(Z)
    print(sel.round(3).to_string(index=False))

    km = KMeans(K, n_init=100, random_state=SEED).fit(Z)
    order = pd.Series(d["car_share_70p"].to_numpy()).groupby(km.labels_).mean().sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    d["cluster"] = [remap[l] for l in km.labels_]
    zs = pd.DataFrame(Z, columns=list(FEATURES), index=d.index).groupby(d["cluster"]).mean().rename(columns=FEATURES)
    stab = cluster_stability(Z, d["cluster"].to_numpy())

    pri = pd.read_csv(TAB / "b2_priority_dongs.csv")
    d["b_priority"] = (d["gu"] + d["dong"]).isin(pri["gu"] + pri["dong"])
    res = pd.read_csv(TAB / "a2_expected_vs_actual.csv", dtype={"adm_cd10": str}).set_index("adm_cd10")
    d["a_overlap10"] = d["b_priority"] & res["type"].reindex(d.index).notna()
    aff = pd.read_csv(TAB / "a8_elderly_only_affluence.csv")
    d["a_elderly_only_not_rich"] = (d["gu"] + d["dong"]).isin(
        aff.loc[~aff["차 보유 상위 25%"], "gu"] + aff.loc[~aff["차 보유 상위 25%"], "dong"])

    prof = d.groupby("cluster").agg(
        동수=("dong", "size"), 고령인구=("pop_65_plus", "sum"),
        지하철_접근=("subway_elderly", "median"), 버스_접근=("a_bus_elderly", "median"),
        병의원_10분내=("clinics_elderly10", "median"), 고령자_차량분담=("car_share_70p", "median"),
        고령자_외출=("out_rate_70p", "median"), 고령비율=("elderly_share", "median"),
        고령수급률=("welfare_65p_rate", "median"), 아파트시가_억=("apt_price_eok", "median"),
        자가용_천명당=("cars_per_1000", "median"), 급경사비율=("steep_share", "median"),
        B고령자지수=("mci_elderly", "median"), B우선지역=("b_priority", "sum"), BA겹침10=("a_overlap10", "sum"),
        고령자만덜탐_비부촌=("a_elderly_only_not_rich", "sum"))
    prof["고령인구_비중"] = prof["고령인구"] / prof["고령인구"].sum()
    prof["부트스트랩_Jaccard"] = stab
    print(zs.round(2).to_string())
    print(prof.round(3).T.to_string())
    for c in range(K):
        sub = d[d["cluster"] == c].sort_values("pop_65_plus", ascending=False)
        print(c, "예:", (sub["gu"] + " " + sub["dong"]).head(12).tolist())

    prof.insert(0, "유형", [NAMES[c] for c in prof.index])
    prof["정책 처방"] = [POLICY[c] for c in prof.index]
    d["type_name"] = d["cluster"].map(NAMES)
    cons = consensus(Z, d["cluster"].to_numpy())
    d[["confidence", "second_type", "second_score", "certainty"]] = cons.to_numpy()
    d["confidence"] = d["confidence"].astype(float)
    d["second_type_name"] = d["second_type"].astype(int).map(NAMES)
    core = (d["certainty"] == "핵심").to_numpy()
    cons_summary = {"핵심 동 비율": float(core.mean()), "보통 이상 비율": float((d["confidence"] >= 0.7).mean()),
                    "실루엣(전체)": float(silhouette_score(Z, d["cluster"])),
                    "실루엣(핵심 동만)": float(silhouette_score(Z[core], d["cluster"].to_numpy()[core]))}
    print(json.dumps(cons_summary, ensure_ascii=False, indent=1))
    prof["핵심 동"] = d[d["certainty"] == "핵심"].groupby("cluster").size()
    prof["경계 동"] = d[d["certainty"] == "경계"].groupby("cluster").size()
    prof[["핵심 동", "경계 동"]] = prof[["핵심 동", "경계 동"]].fillna(0).astype(int)
    print(prof[["유형", "동수", "핵심 동", "경계 동"]].to_string())
    focus = d[d["a_overlap10"] | d["a_elderly_only_not_rich"]][["gu", "dong", "type_name", "certainty", "confidence",
                                                                 "second_type_name", "b_priority", "a_overlap10",
                                                                 "a_elderly_only_not_rich"]]
    focus.to_csv(TAB / "c6_focus_dongs.csv", encoding="utf-8-sig")
    print(focus.sort_values("type_name").round(2).to_string())
    # 경계 동 중 두 번째 후보 유형의 정책 우선순위가 달라 실제로 판단이 갈리는 동
    bd = d[d["certainty"] == "경계"].copy()
    bd["own_tier"] = bd["cluster"].map(TIER)
    bd["second_tier"] = bd["second_type"].astype(int).map(TIER)
    bd["tier_differs"] = bd["own_tier"] != bd["second_tier"]
    bd["to_priority1"] = (bd["second_tier"] == 1) & (bd["own_tier"] != 1)
    bd[["gu", "dong", "type_name", "confidence", "second_type_name", "own_tier", "second_tier", "tier_differs",
        "to_priority1"]].sort_values("confidence").to_csv(TAB / "c7_boundary_dongs.csv", encoding="utf-8-sig")
    cons_summary.update({"경계 동": int(len(bd)), "경계 동 중 정책 순위가 갈리는 동": int(bd["tier_differs"].sum()),
                         "그중 두 번째 후보가 1순위 유형인 동": int(bd["to_priority1"].sum())})
    print({k: cons_summary[k] for k in ["경계 동", "경계 동 중 정책 순위가 갈리는 동", "그중 두 번째 후보가 1순위 유형인 동"]})
    print(bd.loc[bd["to_priority1"], ["gu", "dong", "type_name", "confidence"]].to_string())
    # 최종 1순위 대상: 1순위 유형의 핵심 동은 확정, 경계 후보는 B·A 지표로 판정
    d["a_low"] = res["type"].reindex(d.index).notna()  # A: 기대보다 6%p 이상 대중교통을 덜 탐
    p1 = d["cluster"] == 3
    cand = (p1 & (d["certainty"] != "핵심")) | ((d["certainty"] == "경계") & (d["second_type"].astype(int) == 3))
    d["priority1"] = np.select([p1 & (d["certainty"] == "핵심"), cand & (d["b_priority"] | d["a_low"]), cand],
                               ["확정 (핵심 동)", "추가 (경계·B/A 지표 충족)", "보류 (경계·지표 미충족)"], default="")
    pr1 = d[d["priority1"] != ""].sort_values(["priority1", "pop_65_plus"], ascending=[True, False])
    pr1[["gu", "dong", "type_name", "certainty", "confidence", "b_priority", "a_low", "pop_65_plus", "priority1"]] \
        .to_csv(TAB / "c8_priority1_targets.csv", encoding="utf-8-sig")
    cnt = pr1.groupby("priority1").agg(동수=("dong", "size"), 고령인구=("pop_65_plus", "sum"))
    print(cnt.to_string())
    cons_summary["1순위 대상"] = {k: {"동수": int(v["동수"]), "고령인구": int(v["고령인구"])} for k, v in cnt.iterrows()}
    fig_map(d, FIG / "c1_cluster_map.png")
    fig_heatmap(zs, prof, FIG / "c2_cluster_profile.png")
    rob = type_robustness(Z, d["cluster"].to_numpy())
    print(rob.round(2).to_string())
    rob.to_csv(TAB / "c5_type_robustness.csv", encoding="utf-8-sig")
    sel.to_csv(TAB / "c1_choose_k.csv", index=False, encoding="utf-8-sig")
    zs.to_csv(TAB / "c2_cluster_zscores.csv", encoding="utf-8-sig")
    prof.to_csv(TAB / "c3_cluster_profiles.csv", encoding="utf-8-sig")
    d.drop(columns="geometry").reset_index()[["adm_cd10", "gu", "dong", "cluster", "type_name", "certainty",
                                              "confidence", "second_type_name", "pop_65_plus", "b_priority",
                                              "a_overlap10", "a_elderly_only_not_rich", *FEATURES]] \
        .to_csv(TAB / "c4_dong_clusters.csv", index=False, encoding="utf-8-sig")
    d[["gu", "dong", "cluster", "geometry"]].to_file(PROC / "dong_clusters.gpkg", driver="GPKG")
    (TAB / "c_summary.json").write_text(json.dumps({"k": K, "stability": stab.to_dict(), "consensus": cons_summary},
                                                   ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
