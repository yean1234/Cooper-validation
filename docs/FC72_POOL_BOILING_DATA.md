# Cooper 검증용 FC-72 풀비등 실측 데이터 (10/7 조사)

Cooper 상관식이 "실제 표면에서 열유속 → 벽 과열도"를 맞히는지 보려면, 실제 표면에서 잰
(열유속, 벽 과열도) 쌍이 필요하다. BubbleML은 자리 개수를 사람이 정해서 이 질문의 채점지가
될 수 없다(`docs/GROUND_TRUTH_CHECK.md`). 그래서 FC-72 실측 자료를 찾아 정리했다.

- 같은 유체: FC-72 = PF-5060 = n-perfluorohexane (C6F14)
- 전체 목록(57건, 출처·검증 근거 포함): [`data/experiments/fc72_pool_boiling_sources.csv`](../data/experiments/fc72_pool_boiling_sources.csv)
- 조사 방법: 다섯 방향으로 나눠 찾은 후보 58건을 독립 검증 에이전트가 원문·초록·DOI로 다시 확인했다.
  핵심 수치(아래 ★ 표시)는 원문 PDF나 원자료를 직접 열어 한 번 더 확인했다. 중복 1건은 합쳤다.

---

## 1. 한눈에

| 등급 | 건수 | 뜻 |
|---|---|---|
| A — 바로 쓸 만함 | 11 (자료 묶음으로는 8개) | FC-72, 풀비등, 정상상태, 포화, 매끈한 표면, 압력 확인, 72 kW/m²가 임계열유속 아래 |
| B — 조건부 | 29 | 위 조건 중 하나가 빠짐 (과냉, 마이크로 히터, 유료 원문, 비슷한 다른 유체 등) |
| C — 참고 | 16 | 데이터는 아니지만 Cooper 정확도나 임계열유속 같은 맥락을 줌 |
| 제외 | 1 | 미세중력 단일 기포 실험 |

**알아둘 것 네 가지**

1. **숫자로 바로 받을 수 있는 공개 데이터는 하나뿐이다.** Zimmermann 외 (2020)의 B2SHARE 데이터셋이다.
   나머지는 논문 속 맞춤식·표, 또는 그림을 디지타이즈해야 한다.
2. **수직 표면 자료는 있지만 히터가 작다(10~12.7 mm).** 40 mm 수직 CPU와 같은 크기의 평판 자료는 찾지 못했다.
3. **같은 열유속에서도 실험끼리 과열도가 3배 넘게 다르다.** 표면 상태 차이가 크다는 뜻이고,
   Cooper에서는 표면 거칠기(R_p) 항에 해당한다. 아래 4절 참고.
4. **Cooper를 FC-72와 직접 비교한 선행 결과는 저열유속(≤ 40~45 kW/m²) 것뿐이다.**
   위로 향한 구리 표면에서 평균 편차 10~12%, 아래로 향한 표면에서 23% 또는 "큰 차이"였다.

---

## 2. A등급 — Cooper 검증에 우선 쓸 자료

### 2-1. 숫자로 바로 쓸 수 있는 것

| 자료 | 표면·크기·방향 | 압력 | 열유속 범위 | 데이터 형태 |
|---|---|---|---|---|
| ★ **Zimmermann, Heinz, Sielaff, Gambaryan-Roisman, Stephan (2020)**, Exp. Heat Transfer 33(4) · [논문 DOI](https://doi.org/10.1080/08916152.2019.1635228) · [데이터 B2SHARE](https://b2share.eudat.eu/records/4znr8-k5h80) | 매끈한 구리 원기둥 윗면, Ø35 mm. 방향은 명시 없음(아래서 가열 → 위로 향한 면으로 추정) | **포화, 0.49 / 0.62 / 0.68 / 0.95 / 1.26 / 1.71 bar** | 0.5 ~ 123–203 kW/m², 압력당 13~20점 | **공개 엑셀 (압력, 전압, 전류, q, T_sat, 벽 온도, ΔT, HTC)**, CC BY-NC 4.0 |
| ★ **Mudawar & Anderson (1993)**, J. Electron. Packag. 115 · [DOI](https://doi.org/10.1115/1.2909306) · [PDF](https://engineering.purdue.edu/mudawar/files/articles-all/1993/1993_7.pdf) — *B등급이지만 수직 기준 곡선이 숫자로 있어 여기에 둠* | 구리 12.7 mm, 증기 블라스트 평판, **수직** | 포화, 1 atm | 0 ~ 임계열유속(약 20 W/cm²) | 표 1(a): 구간별 맞춤식 q[W/cm²] = a·ΔT^n (5구간) |
| ★ **Parker & El-Genk (2008)** UNM-ISNPS-1-2008 보고서 · [PDF](https://isnps.unm.edu/reports/ISNPS_Tech_Report_85.pdf) (같은 실험의 학술지판: JHT 2006, [DOI](https://doi.org/10.1115/1.2352783)) | 구리 10 mm, #1500 연마, **0~180° (90° 포함, 그림 4.14)** | 포화, 약 0.085 MPa (앨버커키) | 임계열유속 16.9 W/cm² (21.3 K) | 표 4.1: 11 / 14 / 17 K에서 6.3 / 11.9 / 14.8 W/cm². 나머지는 그림 |
| **Ghiu (2007)** 조지아공대 박사논문 · [저장소](https://repository.gatech.edu/entities/publication/1c953dba-aa1f-4b66-92f0-d9e0c8eb4f74) | PF-5060, 구리 10 mm, 위로 향함(추정) | 포화, 1 atm. 시작 시 용존기체 있음 | 0.9 ~ 15.4 W/cm² | 맞춤식 q[W/cm²] = 0.0293·ΔT^1.9366 (+ 3구간 식) |

### 2-2. 그림을 디지타이즈해야 하지만 조건이 좋은 것

| 자료 | 왜 좋은가 | 주의 |
|---|---|---|
| **Anderson & Mudawar (1989)**, JHT 111 · [DOI](https://doi.org/10.1115/1.3250747) · [PDF](https://engineering.purdue.edu/mudawar/files/articles-all/1989/1989_4.pdf) | **수직** 12.7 mm 구리, 표면 마감 3종(거울면 / 600방 / 증기 블라스트), 1 atm 포화. 곡선당 10~14점 | 마감 차이만으로 과열도가 크게 달라짐. Ra 미기재 |
| **Mudawar & Anderson (1990)**, J. Electron. Packag. 112 · [DOI](https://doi.org/10.1115/1.2904392) · [PDF](https://engineering.purdue.edu/mudawar/files/articles-all/1990/1990_3.pdf) | **수직** 12.7 mm, **1 / 2 / 3 atm 포화** → Cooper의 압력(p_r) 항을 수직면에서 시험 가능. 정상상태 단계 측정 | 그림 7에서 읽어야 함 |
| **El-Genk & Suszko (2014)**, JHT 136 · [DOI](https://doi.org/10.1115/1.4027365) / **Suszko (2015)** UNM 박사논문 · [저장소](https://digitalrepository.unm.edu/me_etds/25/) | PF-5060, 구리 10 mm, **거칠기 Ra 0.039~1.79 µm**, 기울기 0~180° → Cooper의 R_p 항 시험에 가장 적합. h = A·q^B에서 B = 0.81→0.69 | 각도 목록 미확인(초록은 범위만). 0.085 MPa. 논문은 유료, 학위논문은 공개(여기선 접속 차단) |
| **Ujereh, Fisher & Mudawar (2007)**, IJHMT 50 · [DOI](https://doi.org/10.1016/j.ijheatmasstransfer.2007.01.030) · [PDF](https://engineering.purdue.edu/mudawar/files/articles-all/2007/2007-02.pdf) | 맨 구리(400방) 12.7 mm, 1 atm 포화, 0.5 W/cm² 단계, 임계열유속 약 18 W/cm² | 수평. 매끈한 Si 곡선은 이상하게 큰 과열도라 구리 쪽만 권장 |
| **Ali & El-Genk (2013)** UNM-ISNPS-3-2013 보고서 · [PDF](https://isnps.unm.edu/reports/ISNPS_Tech_Report_83.pdf) | PF-5060, 매끈한 구리 Ra 0.04 µm, 위로 향함, 0.085 MPa | 그림만 |

---

## 3. B등급 중 눈여겨볼 것

| 자료 | 쓸모 | 막힌 점 |
|---|---|---|
| **Rainey & You (2001)**, IJHMT 44 · [DOI](https://doi.org/10.1016/S0017-9310(00)00318-5) | **20 mm·50 mm 구리 평판, 수직 포함 여러 방향** — CPU 크기에 가장 가까움 | 유료. 내용은 2차 출처로만 확인 |
| **Rainey, You & Lee (2003)**, JHT 125 · [DOI](https://doi.org/10.1115/1.1527890) | 평판 구리 1 cm², **30~150 kPa** 압력 시리즈 | 유료, 수평 |
| **Chang & You (1996)**, JHT 118 · [DOI](https://doi.org/10.1115/1.2822592) | 코팅 없는 구리 10 mm, 0/45/90/135/180° | 유료 |
| **Reed & Mudawar (1997)**, IJHMT 40 · [PDF](https://engineering.purdue.edu/mudawar/files/articles-all/1997/1997_5.pdf) | 맨 구리 12.7 mm, 0~180° 15° 간격 | 1.5 K 과냉, 과열도를 액체 온도 기준으로 잼 |
| **Priarone (2005)**, Int. J. Therm. Sci. 44 · [DOI](https://doi.org/10.1016/j.ijthermalsci.2005.02.014) | 구리 원판 30 mm, 0~175° | 유료, 내용은 2차 출처로만 확인 |
| **Wei & Honda (2003)**, JSME 논문집 B 69 (일본어) · [J-STAGE](https://www.jstage.jst.go.jp/article/kikaib1979/69/679/69_679_682/_pdf) | 매끈한 Si 칩 10 mm, 1 atm. 드물게 수직 대 수평 비교가 있음(과냉 25 K) | 스캔 그림만 |
| **Kaniowski & Pastuszko (2021)**, Energies 14 · [DOI](https://doi.org/10.3390/en14217283) | 매끈한 구리 27 mm, 1 atm. 데이터는 저자에게 요청 가능 | 수평, 그림만 |
| **Schlindwein 외 (2009)**, Heat Mass Transfer 45 · [PDF](https://lepten.ufsc.br/publicacoes/boiling/periodicos/2009/HMT/schlindwein_martin-jr.pdf) | **Cooper 직접 비교: 순수 FC-72 평균 편차 11.8%** (Ra 1.1 µm, 구리 12 mm) | 40 kW/m²까지만 |
| **McHale & Garimella (2013)**, Exp. Therm. Fluid Sci. 44 · [DOI](https://doi.org/10.1016/j.expthermflusci.2012.08.005) | 거칠기를 조절한 표면 7종 (R_p 항 시험) | 유리/ITO 기판이라 금속 뚜껑과 다름 |
| 다른 불소계 유체: **Fan 외 (2020)** HFE-7100 · [PDF](https://bura.brunel.ac.uk/bitstream/2438/21718/1/FullText.pdf), **Yang 외 (2025)** HFE-7100·Opteon 2P50 · [PDF](https://www.osti.gov/servlets/purl/2586914) | Fan: Ø40 mm, 0.7~2 bar, Cooper 포함 18개 식 평가. Yang: Cooper가 HFE-7100에서 1.6~18% 차이. **Yang 부록 표 A1에 FC-72 매끈한 표면 연구 26건 목록**이 있어 찾기 색인으로 좋음 | FC-72가 아님 |

나머지 B·C 항목(마이크로히터, 과냉 전용, 미세중력, 리뷰 등)은 CSV에 이유와 함께 있다.

**참고(C) 중 쓸 만한 것**

- **Cardoso & Passos (2005)**, COBEM: Ra 1.1 µm 구리에서 위로 향한 면 편차 10.2%, 아래로 향한 면 23.3% (≤ 40 kW/m²)
- **Passos 외 (2004)**: 아래로 향한 면에서 Cooper와 "큰 차이"
- **Howard & Mudawar (1999)**: 수직 FC-72의 임계열유속 약 14~19 W/cm². 72 kW/m²가 수직면 임계열유속의 40~50% 수준이라는 여유 확인용

---

## 4. 첫 점검 (예비) — 72 kW/m²에서 Cooper와 실측 비교

정식 검증 전에, 숫자를 바로 얻을 수 있는 자료들로 탱크 CPU 열유속(72 kW/m²) 한 점만 비교했다.
Cooper는 이 저장소 코드를 그대로 썼다(R_p = 1 µm 고정, 압력은 각 실험 값).

| 자료 | 조건 | 실측 ΔT | Cooper ΔT | Cooper 오차 | 맞추려면 필요한 R_p (진단용) |
|---|---|---|---|---|---|
| ★ Zimmermann 2020 | Ø35 mm 구리, 0.95 bar | 9.6 K | 21.9 K | +129% | 25 µm |
| ★ (같은 자료) | 0.49 → 1.71 bar | 12.9 → 7.2 K | 26.5 → 18.1 K | +106 → +153% | 10 → 91 µm |
| ★ Parker & El-Genk 2008 | 10 mm 구리, 0.085 MPa | 11.6 K (표 보간) | 22.7 K | +96% | 12 µm |
| ★ Mudawar & Anderson 1993 | 12.7 mm 구리, **수직**, 1 atm | 13.5 K (맞춤식) | 21.5 K | +59% | 6.4 µm |
| Ghiu 2007 | 10 mm 구리, PF-5060, 1 atm | 17.1 K (맞춤식) | 21.5 K | +25% | 2.5 µm |
| Wei & Honda 2003 | 매끈한 Si 10 mm, 1 atm | 약 26 K (그림 눈대중) | 21.5 K | −17% | 0.47 µm |

**보이는 것**

- **실험끼리 차이가 Cooper 오차보다 크다.** 같은 72 kW/m²에서 실측 과열도가 7~26 K로 3배 넘게 벌어진다.
  표면 재질·마감·준비 방법 차이 때문이고, ±30% 기준보다 훨씬 넓다.
- **구리 표면에서는 Cooper(R_p = 1 µm)가 과열도를 더 높게 예측한다(+25~+153%).**
  BubbleML에서 Cooper가 과열도를 낮게 냈던 것과 반대 방향이다.
- **R_p로 맞추는 건 답이 아니다.** 맞추려면 R_p가 2.5~91 µm여야 하는데, 이 표면들은 매끈하거나 연마한 표면이다(Ra ≲ 1 µm).
  물리적으로 말이 안 되고, 코드도 R_p 조정을 막아 둔다.
- **기울기도 자료마다 다르다.** Zimmermann은 ΔT ∝ q^0.19~0.26(30 kW/m² 이상, 상승 구간, 직접 계산),
  El-Genk & Suszko는 h ∝ q^0.81(ΔT ∝ q^0.19)이다. Mudawar·Ghiu는 72 kW/m² 근처 국소 기울기가 0.43~0.51이다. Cooper는 0.33이다.
- **저열유속 Cooper 비교(10~12%)와 여기 결과가 다르다.** 실험 장치와 표면이 다르기 때문이고,
  그만큼 "Cooper가 FC-72에서 얼마나 정확한가"는 어떤 표면을 기준으로 하느냐에 달려 있다.

> 한 점 비교라 결론이 아니다. 정식 검증은 곡선 전체를 게이트(평균 오차 + 기울기)에 넣어야 한다.
> 다만 **표면 하나당 따로 채점해야 하고**, 실제 탱크 CPU 표면(뚜껑 재질과 마감)을 알아야
> 어느 실험을 기준으로 삼을지 정할 수 있다는 점은 분명하다.

---

## 5. 빈 곳

- **40 mm급 수직 평판 FC-72 자료가 없다.** 수직 자료는 10~12.7 mm 히터뿐이다.
  큰 수직면에서는 아래쪽 기포가 위로 쓸고 올라가는 효과가 있어 크기 효과를 무시하기 어렵다.
  가장 가까운 Rainey & You (2001, 50 mm, 수직 포함)는 유료라 내용을 직접 확인하지 못했다.
- **대부분 그림뿐이다.** WebPlotDigitizer 같은 도구로 읽어야 한다.
- **유료 원문(ScienceDirect·ASME)은 열지 못했다.** 학교 도서관 접속으로 확인할 수 있다.
- **탱크 유체는 FC-72가 아니다(기화점 47 °C).** 최종적으로는 그 유체의 비등곡선과 물성(분자량, 임계압력)이 필요하다.

---

## 6. 다음 단계 제안

1. **Zimmermann 데이터로 정식 게이트 돌리기.** 실험 CSV 입력 경로를 추가하면, 6개 압력의 곡선 전체로
   평균 오차·기울기·압력 항을 한 번에 볼 수 있다. 사이트 수 열이 없으니 기울기 기준이 자동으로 켜진다.
2. **수직 자료 디지타이즈.** Mudawar & Anderson (1990) 그림 7(1/2/3 atm), Parker & El-Genk (2008) 그림 4.14(90°).
3. **유료 원문 확보.** Rainey & You (2001, 50 mm·방향), Rainey·You·Lee (2003, 압력), Chang & You (1996, 방향).
4. **데이터 요청.** Kaniowski & Pastuszko (2021), Yang 외 (2025) 모두 "요청 시 제공"이라고 적혀 있다.
