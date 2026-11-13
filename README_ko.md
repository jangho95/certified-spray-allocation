# 고정 커널 정지 분사 배분: 코드와 데이터

이 저장소는 JangHo Seo와 Joonwoo Lee의 논문 "Separating operator, index, and
certificate structure in fixed-kernel stop-and-spray allocation"의 코드, 입력, 저장된 결과를 담고
있습니다.

문제는 전체 타수가 정해졌을 때 최종 폼 두께가 가장 고르게 되도록 각 waypoint의
분사 횟수를 정수로 정하는 것입니다. 시뮬레이터, 완화·반올림 코드, 재구현한 PSO,
논문에 보고한 모든 배분, 그리고 보고한 하한을 정확 산술로 다시 확인하는 데 필요한
자료가 들어 있습니다.

영문 안내는 [README.md](README.md)입니다.

## 저장된 결과 확인

Linux(WSL)와 Python 3.12.14에서 확인했습니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python tools/verify_integrity.py
python verification/verify_certificates.py --workers 4
```

첫 번째 스크립트는 `MANIFEST.json`에 적힌 파일을 기록된 해시와 비교합니다. 두 번째
스크립트는 저장된 배분의 분산과 하한을 정수·유리수 산술로 다시 계산합니다. NumPy와
Python 표준 라이브러리만 쓰고 최적화 프로그램은 돌리지 않습니다. 둘 다 JSON 요약을
출력하고, 확인에 실패하면 오류로 멈춥니다. 확인이 assert 문이므로 `python -O`로
실행하면 안 됩니다.

`--suite 이름`으로 한 묶음만 확인할 수 있습니다.

| 묶음 | 다시 확인하는 것 |
|---|---|
| `pilot` | 목표 예산 네 사례와, 모든 셀에 waypoint를 둔 두 사례 |
| `stage5` | 첫 PSO 비교: 필드–예산 40쌍, 배분 104개 |
| `stage5_long25` | 논문의 PSO 비교: 배분 200개(필드당 seed 25개, 500회와 5000회) |
| `benchmark` | 예산 스윕: 네 필드, 필드당 예산 21개 |
| `stage9` | 무작위 필드 59개, 필드당 예산 21개(1,239건) |
| `sensitivity` | 목표 두께·필드 분포·격자 계열 156건 |

무작위 필드 59개에서 검증된 폭의 최댓값은 올림해서 0.0515%입니다.

2차 수정에서 확인 두 가지를 추가했습니다. 둘 다 값이 다르면 오류로 멈춥니다.

```bash
python verification/kernel_width/replay_kernel_witnesses.py
python verification/second_revision_numbers.py
```

첫 번째는 보충자료 표 S22의 커널 폭 11건을 다시 확인하며 NumPy가 필요합니다. 두 번째는 수정 원고 본문에 인용한
횟수와 시간을 저장된 파일에서 다시 계산하며 표준 라이브러리만 씁니다.

## 실험 다시 실행

배포된 결과가 바뀌지 않도록 먼저 작업 사본을 만듭니다.

```bash
python tools/new_run.py ../stop-spray-stage3 --build --stage 3
```

이 명령은 C++ 솔버를 빌드하고 저타수 실험을 새 폴더에서 다시 실행합니다. 빌드는
C++17과 requirements에 고정된 Zig/Clang 컴파일러를 쓰며, `-ffp-contract=off`를
주고 `-march=native`는 쓰지 않습니다.

표와 그림도 같은 방식으로 저장된 결과에서 다시 만들 수 있습니다.

```bash
python tools/new_run.py ../stop-spray-tables --stage tables
python tools/new_run.py ../stop-spray-figures --stage figures
```

다른 단계는 [docs/REPRODUCING.md](docs/REPRODUCING.md)에 있습니다. 실행 시간은
기기에 따라 다릅니다.

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `src/` | C++ 시뮬레이터와 교환 탐색, Python 모델·QP·PSO·비교 코드 |
| `pipeline/` | 실험을 실행하고 표와 그림을 만든 스크립트 |
| `inputs/` | 초기 두께 필드(무작위 필드 59개 포함) |
| `results/` | 저장된 결과와 정확 검증에 쓰는 자료 |
| `work/` | 개별 실행의 배분, 최종 필드, PSO 이력 |
| `reference/` | 첫 제출본의 CSV와 표 텍스트(비교용) |
| `verification/` | 정확 산술 검증기 |
| `tools/` | 빌드, 작업 사본 생성, 무결성 확인 스크립트 |
| `provenance/` | 각 파일의 출처와 패키지 시험 기록 |
| `docs/` | 데이터, 재실행, 논문 표에 대한 안내 |

실험은 번호가 붙은 단계로 진행했고 폴더 이름에 그 번호가 남아 있습니다
(`results/stage3`, `work/stage7`). 논문의 표마다 어느 폴더가 해당하는지는
[docs/PAPER_MAP.md](docs/PAPER_MAP.md)에, 파일 형식은 [docs/DATA.md](docs/DATA.md)에
있습니다.

이름에 `pre_review`, `_v1`, `_before_`, `backup`이 들어간 파일과 폴더는 기록용으로
남긴 이전 버전입니다. `work/`와 `reference/`는 확인에 필요하므로 지우지 마십시오.

## 참고

- PSO 코드(`src/paper_pso.py`)는 저자의 이전 논문에 실린 설명을 보고 다시 구현한
  것이며, 그 논문에서 쓴 코드가 아닙니다.
- 정확 검증의 대상은 저장된 배정밀도 연산자와 초기 필드입니다. 모델이 실제 코팅
  공정과 얼마나 맞는지는 다루지 않습니다.
- 공개 소스 파일에는 주석이 없습니다. 진입점과 데이터 형식은 `docs/`의 안내에
  적었습니다.
- 인용과 라이선스 상태는 [NOTICE.md](NOTICE.md)에 있습니다.
