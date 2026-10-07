# Cooper 게이트 — BubbleML 거친 입력 검증 파이프라인 (초안)

BubbleML 원본을 **거친 입력**으로 뭉갠 뒤, 거기서 나온 벽 열유속 `q″` 만으로
**벽 과열도 `ΔT_sup`** 를 되찾을 수 있는지를 Cooper 상관식으로 검증한다.

근거 문서: Notion `거친 입력 검증` (9/21), `정답 기포장` (9/20) 및 각 페이지의 댓글 스레드.

---

## 1. 이 게이트가 확인하려는 것

연구 전체 흐름에서 이 코드는 **②단계**만 담당한다.

```
①  BubbleML 원본 ──뭉개기──▶ 거친 입력 (q″, 증기분율)      ← 그냥 평균. Cooper 아님
②  거친 q″ ──── Cooper 역산 ────▶ 예측 ΔT_sup              ← ★ 이 저장소
③  ΔT_sup ──▶ 핵생성밀도·이탈직경·이탈빈도 ──▶ 기포 명세서 ──▶ 합성 기포장
```

③단계의 기포 상관식들은 **전부 입력으로 ΔT_sup 를 요구**한다. 그런데 거친 격자 CFD는
`q″` 만 주고 `ΔT_sup` 은 안 준다 — 격자가 굵어서 벽 근처 온도 변화를 못 담기 때문이다.
그 다리 역할이 Cooper이고, 그 다리가 실제로 버티는지를 BubbleML **한 데이터 안에서**
미리 재는 것이 이 게이트다.

- **예측** : BubbleML을 뭉갠 거친 `q″` → Cooper → 예측 `ΔT_sup`
- **정답** : BubbleML 원본이 가지고 있는 진짜 벽 과열도 (`T_wall − T_sat`, Dirichlet 입력값)

우리 액침탱크 CFD는 아직 들어오지 않는다. 정답이 없어서 채점 자체가 불가능하기 때문이다.

### 왜 `q″` 가 아니라 `ΔT` 로 채점하나
`ΔT = q″/h ∝ q″^0.33` 이라 **`q″` 오차는 세제곱근으로 감쇠한다** (20% → 약 6%).
반대로 **프리팩터 오차는 `ΔT` 에 그대로 전달**된다. 그래서 통과 기준은 우리 `q″` 정밀도가
아니라 Cooper 자체의 정밀도에 맞춰야 한다.

---

## 2. 통과 기준 두 가지

| 기준 | 값 | 이유 |
|---|---|---|
| **MAPE** | ≤ 30% | Cooper의 문헌 산포가 `h` 기준 ±30~40%. ±10%를 걸면 **Cooper가 정상 작동해도 떨어진다.** 30%는 봐주는 게 아니라 도구의 실제 정밀도에 맞춘 값이다. |
| **log-log 기울기** | 0.33 ± 0.05 | `ΔT = (상수) × q″^0.33` 이므로 log-log에서 기울기 0.33 직선이어야 한다. 프리팩터는 직선을 평행이동만 시키고 **기울기는 못 바꾼다.** |

**단일점 일치는 의미가 없다.** 점이 하나면 상수를 적당히 고르는 것만으로 무조건 맞출 수
있기 때문이다. 그래서 벽 과열도가 다른 케이스를 5개 이상 모아 **비등곡선 기울기**를 본다
(`gate.min_cases`, 기본 5). 케이스가 모자라면 코드가 `판정 불가` 를 낸다.

### 기울기 기준의 전제 — 사이트 수가 처방된 자료에서는 참고만

Cooper의 `q″^0.67` 은 실제 표면에서 과열도가 오르면 사이트가 저절로 늘어나는 효과를 품고 있다.
BubbleML은 사이트 수를 조건별 **입력**으로 넣는다(Twall 80→100 °C 에 10→30개). 그러면
비등곡선 기울기는 누가 사이트를 몇 개 넣었는지로 정해진다(`q ∝ N^0.84`, R² 0.9997).
그래서 케이스별 처방 사이트 수가 주어지고 조건마다 다르면 기울기는 **판정에서 빼고 참고로만**
보고한다(`gate.slope_role: auto`). 판정은 조건별 오차(MAPE)로 한다. 자세한 점검은
[`docs/GROUND_TRUTH_CHECK.md`](docs/GROUND_TRUTH_CHECK.md) 에 있다.

| 판정 (사이트 처방 자료) | 조건 |
|---|---|
| `통과 (조건별 크기 일치)` | MAPE O — 기울기는 사이트 수가 물리로 정해지는 자료에서 따로 확인 |
| `불통과 (조건별 크기 이탈)` | MAPE X — 기준·R_p·q 정의를 바꿔 맞추지 않는다 |

### 판정은 중단/진행이 아니라 경로 선택

멘토 조언(9/9)대로 자폭 분기를 만들지 않는다.

| 판정 | 조건 | 다음 경로 |
|---|---|---|
| `통과` | 기울기 O, MAPE O | Cooper 경로로 ③단계 진행 |
| `조건부 통과` | 기울기 O, MAPE X | 형태는 맞고 상수만 어긋남 → Cooper 유지, 10월에 벽함수와 프리팩터 비교 |
| `불일치` | 기울기 X | 상수로 못 속이는 진짜 불일치 → 벽함수(Kader) 경로를 붙여 **비교 실험**으로 전환 |
| `판정 불가` | 케이스 부족 | 케이스를 더 모은 뒤 재실행 |

`불일치` 도 결론이다 — "Cooper는 안 되고 벽함수는 된다" 도, "두 표준 경로가 이 조건에서
모두 안 된다" 도 논문거리가 된다.

---

## 3. 입력 (Input)

### 3.1 데이터
BubbleML HDF5 **여러 개** (= 벽 과열도가 서로 다른 케이스들). 파일당 필요한 필드:

| 키 | 모양 | 쓰임 |
|---|---|---|
| `temperature` | `(n_t, n_y, n_x)` | 벽 온도 기울기 → `q″` |
| `dfun` (level-set φ) | `(n_t, n_y, n_x)` | 섞인 셀의 혼합 열전도도, 증기분율 |
| `x`, `y` | 좌표 | 격자 간격 `dy` (없으면 설정값 사용) |

`velx`, `vely`, `pressure` 는 현재 경로에서 쓰지 않는다 (벽함수 경로에서 필요해짐).

### 3.2 케이스별 메타데이터 — **가장 중요한 입력**
`T_wall`, `T_sat`, `T_bulk`. 이 중 `T_wall − T_sat` 가 게이트의 **정답 `ΔT_sup`** 이므로
추측하면 안 된다. 다음 순서로 해결하고, 어느 경로로 정해졌는지 `run_manifest.json` 에 남는다.

1. 설정의 `nondim.case_table`  ← **권장**
2. HDF5 안의 runtime params
3. 파일명 정규식 (`Twall-90.hdf5` → 90 °C)
4. 위 셋 다 실패하면 **에러로 중단** (조용히 추측하지 않음)

### 3.3 설정 파일
`configs/default.yaml` (실데이터) / `configs/synthetic.yaml` (합성 데이터) /
`configs/ground_truth.yaml` (정답 기포장 인계 표).
유체 물성은 `configs/fluids.yaml` 로 덮어쓴다.

### 3.4 정답 기포장 인계 표 (HDF5 대신)
정답 기포장 레포가 열유속 정의를 확정했다 — Flash-X 경계와 같은 1차 벽 기울기 ×
phase-averaged k (산술 기본, 조화 동등 후보). `ground_truth.csv` 를 주면 이 저장소는
q″ 를 다시 뽑지 않고 그 값을 그대로 Cooper 에 넣는다.

| 컬럼 | 쓰임 |
|---|---|
| `input_dT_sup_K` | 정답 ΔT (Twall − 58 °C) |
| `input_n_sites_prescribed` | 처방 사이트 수 — 기울기를 판정에서 뺄지 정하는 근거 |
| `confirmed_q_mix_arith_2d_W_m2` | 판정에 쓰는 q″ (기본) |
| `confirmed_q_mix_harm_2d_W_m2` | 동등 후보 — 판정이 갈리는지 매번 같이 계산 |
| `confirmed_q_kl_all_2d_W_m2`, `confirmed_q_legacy_2d_W_m2` | 참고 — 표에만 남김 |

원본 레포가 비공개라 지금은 보고서 PDF 값으로 만든 스냅숏
`data/ground_truth/ground_truth_final_snapshot.csv` 를 쓴다.

---

## 4. 출력 (Output)

`outputs/<run_id>/` 아래에 생성된다.

| 파일 | 내용 |
|---|---|
| `summary.md` | 사람이 읽는 판정 요약 — 판정·근거·케이스별 표·수렴 점검 |
| `gate_result.json` | 판정 + 지표(MAPE, 기울기, 신뢰구간, R²) + 케이스별 점 |
| `case_summary.csv` | 케이스당 한 줄: `q″`, 정답 ΔT, Cooper ΔT, 상대오차, 증기분율, (진단)역산 R_p |
| `coarse_inputs.csv` | **거친 입력 자체** — 거친 셀별 `(x, q″, 증기분율)`. ③단계에서 그대로 쓴다 |
| `run_manifest.json` | 재현 정보 — 설정 전체, git SHA, 파일 해시, 쓰인 물성값, **온도·길이 스케일의 출처** |
| `boiling_curve.png` | log-log 비등곡선 + 기울기 0.33 기준선 |
| `parity.png` | 정답 ΔT vs 예측 ΔT + ±30% 띠 |
| `order_audit.png` | 차분 차수 수렴 점검 (1차/2차/3차) |

핵심 숫자 세 개: **MAPE**, **log-log 기울기**, **판정/경로**.

---

## 5. 두 가지 방법론 결정 (9/20 결론의 코드화)

### 5.1 섞인 셀을 빼지 않는다
기포가 붙어 있는 자리가 바로 열이 가장 많이 빠져나가는 곳이다. 거길 결측 처리하면
**우연히가 아니라 항상 낮은 쪽으로** 틀린다. 대신 열전도도를 `dfun` 비율대로 섞어 쓴다.

```
α_l = H_ε(φ)                      # smoothed Heaviside, ε = 1.5·Δ
k    = k_v + (k_l − k_v)·α_l      # 노트의 "비율로 섞은 값"
q″   = −k · dT/dy|_wall
```

`--exclude-mixed-cells` 로 옛 방식을 재현해 편향 크기를 직접 잴 수 있다.

### 5.2 차분 차수를 2차로 못 박는다
1차/2차가 20% 차이 나는 자로는 남의 예측을 채점할 수 없다. Flash-X가 공간 2차
정확도 스킴이므로 기준은 2차다. 동시에 **3차까지 계산해서 수렴을 확인**한다.

- `|2차−3차| ≪ |1차−2차|` → 2차는 수렴 → 기준으로 사용 가능
- 둘 다 크면 → 격자가 경계층을 못 잡고 있다는 뜻 → **차분 차수 문제가 아니라 더 큰 문제**

셀 중심 격자라 벽에서 첫 셀까지가 `Δ/2` 인 **비균등 간격**이다. 교과서의 균등 간격
계수를 쓰면 틀리므로 Fornberg 알고리즘으로 가중치를 생성한다 (2차 해석해 `(−8/3, 3, −1/3)/Δ`).

### 5.3 R_p 는 1 µm 고정 — 코드가 강제한다
Cooper의 표면조도 `R_p` 는 시뮬레이션에 정의가 없어 사실상 자유 파라미터다.
맞추려고 튜닝하면 "유체별 상수 없음" 주장이 그 자리에서 깨진다.
`CooperModel` 은 `allow_tuning=True` 없이 1 µm 이외의 값을 주면 **예외를 던진다.**
대신 "이 점을 맞추려면 R_p 가 얼마였어야 하나"를 역산해 **리포트에만** 남긴다.

---

## 6. 사용법

```bash
pip install -r requirements.txt

# (a) 실데이터 없이 먼저 돌려보기 — 합성 픽스처 생성 후 실행
python scripts/make_synthetic.py --out data/synthetic --nx 512
python scripts/run_gate.py --config configs/synthetic.yaml

# (b) 실제 BubbleML
#   1) data/bubbleml/ 에 HDF5 를 넣고
#   2) configs/default.yaml 의 nondim.case_table 에 케이스별 T_wall/T_sat/T_bulk 를 적고
#   3) 실행
python scripts/run_gate.py --config configs/default.yaml

# (c) 정답 기포장 인계 표 → Cooper → 정답 ΔT
python scripts/run_gate.py --config configs/ground_truth.yaml
python scripts/run_gate.py --config configs/ground_truth.yaml --ground-truth path/to/ground_truth_final.csv
python scripts/run_gate.py --config configs/ground_truth.yaml --slope-role criterion   # 옛 판정 재현

# 진단용
python scripts/run_gate.py --config configs/default.yaml --order 1        # 1차 기준으로 재계산
python scripts/run_gate.py --config configs/default.yaml --exclude-mixed-cells  # 옛 방식 편향 측정

pytest            # 35개 단위/통합 테스트
```

### 게이트 분기 재현 (합성 데이터)

```bash
python scripts/make_synthetic.py --out /tmp/a --prefactor-scale 2.5   # → 조건부 통과
python scripts/make_synthetic.py --out /tmp/b --exponent 2.0          # → 불일치 (기울기 이탈)
```

---

## 7. 파일 구조

```
src/cooperval/
  fd.py            비균등 간격 유한차분 가중치 (Fornberg)
  heaviside.py     level-set → 액체 분율, 혼합 열전도도
  fluids.py        유체 물성 (문헌 대표값, YAML 로 덮어쓰기)
  bubbleml.py      HDF5 로더 + 무차원→유차원 환산 + 스케일 출처 기록
  wallflux.py      q″ 추출 (섞인 셀 포함, 1/2/3차 + 수렴 감사)
  coarsen.py       뭉개기 → 거친 입력
  cooper.py        Cooper 상관식 정방향/역산, R_p 잠금
  gate.py          MAPE + log-log 기울기 → 판정/경로 (사이트 처방 자료면 기울기는 참고)
  ground_truth.py  정답 기포장 인계 표 로더 (확정 q″, 처방 사이트 수)
  wallfunction.py  Kader 벽함수 (10월 게이트용, 기본 비활성)
  report.py        CSV/JSON/그림/요약
  pipeline.py      전체 오케스트레이션
scripts/
  run_gate.py      실행기 (CLI)
  make_synthetic.py  BubbleML 모양 합성 픽스처 생성기
docs/ASSUMPTIONS.md  초안에서 내가 판단으로 정한 것들 — 실데이터 전에 확인할 목록
docs/GROUND_TRUTH_CHECK.md  정답 q″ → Cooper 비교의 '불일치' 점검과 수정 내용
docs/FC72_POOL_BOILING_DATA.md  Cooper 검증용 FC-72 풀비등 실측 데이터 조사 (BubbleML 대신)
data/ground_truth/   정답 기포장 인계 표 스냅숏
data/experiments/    실측 자료 목록 (fc72_pool_boiling_sources.csv)
```

---

## 8. 아직 안 한 것

- **벽함수(Kader) 경로**: 공식은 `wallfunction.py` 에 있지만 데이터 배선은 안 했다
  (`u_τ` 가 필요하고 속도장을 아직 안 읽는다). 9/21 일정에서는 Cooper 한 경로만 돌린다.
  10월 정식 게이트에서 **정답 ΔT / Cooper ΔT / 벽함수 ΔT** 셋을 나란히 비교한다.
- **δT(coarsening 기준축)**: 9/21 노트대로 이 게이트에는 쓰이지 않아 구현하지 않았다.
- **실제 BubbleML 스펙 확인**: `docs/ASSUMPTIONS.md` 의 ★ 항목들.
- **HDF5 경로를 정답 정의에 맞추기**: dfun 부호(증기가 양수), 1차 벽 기울기, Heaviside 폭,
  히터 범위, 좌표 키, T_sat 58 °C 가 정답 보고서와 다르다. 목록은
  `docs/GROUND_TRUTH_CHECK.md` 6절. 정답 표 경로(3.4)는 영향을 받지 않는다.
