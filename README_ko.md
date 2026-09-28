# 공개용 코드·데이터 안내

이 폴더에는 논문 재현에 필요한 코드, 입력, 실행별 원시 결과, 정수 배분, 엄밀 검증 자료를 모았습니다. 제출본과 이전 실험 기록을 보존하며 수정 원고의 추가 자료도 포함합니다.

## 먼저 볼 파일

- [README.md](README.md): 공개용 영문 소개와 빠른 실행 방법
- [docs/REPRODUCING.md](docs/REPRODUCING.md): 빌드·검산·단계별 재실행
- [docs/DATA.md](docs/DATA.md): 단계별 데이터 위치와 주요 변수
- [docs/PROVENANCE.md](docs/PROVENANCE.md): 과거 기록, 공개 코드 변경, 해석 범위
- [provenance/validation.json](provenance/validation.json): 이번 공개 패키지 검증 결과

## 정리 기준

- Python 주석·docstring, C++ 주석, CMake 주석을 제거했습니다. 실행에 필요한 `#include`와 셸의 `#!`는 유지했습니다.
- 코드 설명은 별도 Markdown 문서에 적었습니다.
- 입력, 저장된 할당, 수치 결과는 원래 바이트 그대로 보존했습니다. 로그와 결과 기록에 남은 절대 경로는 앞부분만 공개용으로 줄였고, 원본 해시는 `provenance/source_mapping.json`에 남아 있습니다. `MANIFEST.json`으로 파일 손상·변경을 확인할 수 있습니다.
- 가상환경, 컴파일 결과, 캐시, 논문 PDF, 리뷰 원문, 내부 원고 초안은 포함하지 않았습니다.
- `work/`에는 실제 배분·이력이 있으므로 삭제하면 안 됩니다.
- `pre_review*`, `*_v1*`, `*backup*` 등은 과거 결과입니다. 최종 결과와 구분해 두었습니다.

## 실행 순서

Linux 또는 WSL에서 Python 3.12를 준비하고 이 폴더에서 실행합니다. 검증에 사용한 버전은 3.12.14입니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python tools/verify_integrity.py
python verification/verify_certificates.py --workers 4
```

실험을 다시 실행할 때는 배포 결과를 보존하도록 별도 사본을 만듭니다. 다음 명령은 사본 생성, C++ 빌드, Stage 3 실행을 수행합니다.

```bash
python tools/new_run.py ../stop-spray-stage3 --build --stage 3
```

## 보고할 때의 구분

PSO는 선행 논문의 설명에 따른 **재구현**입니다. 원 구현의 재실행이라고 쓰면 안 됩니다. Stage 9의 사전 고정 기록과 사후 엄밀 검증은 각각 보존돼 있습니다. 공개 코드의 주석 제거·경로 수정으로 코드 해시가 달라졌으므로, 과거 사전 고정 해시가 공개 코드와 같다고 주장하지 않습니다.

저장소는 https://github.com/jangho95/certified-spray-allocation 입니다. DOI는 발급하지 않았고 라이선스 상태는 NOTICE.md를 따릅니다.

## 이번 수정 원고의 추가 자료

84개 benchmark sweep, 후보 배분 252개, 검증 witness와 그림 재생성 스크립트를 추가했습니다. 다음 명령은 최적화 없이 저장된 결과를 검산하고 별도 폴더에 그림을 만듭니다.

```bash
python verification/verify_certificates.py --suite benchmark
python pipeline/figures_revised.py --output-dir /tmp/stop-spray-figures
```

Stage 9의 제출본 두 후보 규칙도 정확히 평가해 검증기에 연결했습니다. 수정 그림은 witness의 배분을 읽으므로 이후 레지스트리 갱신에 영향을 받지 않습니다. 추가 코드에도 주석과 docstring은 없습니다.

### 최신 ZIP 추가 내용

- `results/stage5_long25/`: 5,000회 반복 PSO 100건과 비교용 500회 반복 100건의 배분·이력·정확 평가·검증 자료
- 최신 원고의 그림 9개를 만드는 코드와 입력, 생성된 PNG·PDF
- `results/revision_tables/`: 본문·보충자료 표의 표시값 CSV. 반올림 전 원시 값은 각 실험 결과 폴더에 있습니다.
- 장기 PSO까지 포함하도록 확장한 통합 검증기와 재실행 안내

```bash
python tools/verify_integrity.py
python verification/verify_certificates.py --workers 4
python tools/new_run.py ../stop-spray-all-figures --stage figures
```

추가 자료와 그림별 생성 코드의 위치는 [docs/REVISION_UPDATE.md](docs/REVISION_UPDATE.md)에 있습니다. ZIP은 압축을 푼 내용물을 저장소 루트에 추가·덮어쓰는 용도이며, ZIP 자체와 가상환경·인증 정보는 포함하지 않습니다.

### 민감도 실험 추가 자료

- 목표 두께 24개, 초기장 분포 120개, 격자 해상도 12개: 총 156개 사례
- `inputs/sensitivity_extension/`: 네 분포에서 생성한 초기장 120개
- `results/sensitivity_extension_v2/`: 최종 결과, 연산자, 하한 기준점, 실행가능 연속점, 정수 후보 세 개와 정확 유리수
- `results/sensitivity_extension/`: 보존된 첫 검증 결과
- `pipeline/make_sensitivity_tables.py`: 본문 Table 15와 SI S13–S15 재생성
- `src/matrix_exchange.cpp`: 격자 실험의 행렬 기반 정수 탐색 코드; 기존 빌드 명령에 연결

```bash
python verification/verify_certificates.py --suite sensitivity --workers 4
python pipeline/make_sensitivity_tables.py --output-dir /tmp/spray-sensitivity-tables
```

통합 검산 명령은 이제 여섯 묶음을 검사합니다. 민감도 검사는 최적화를 실행하지 않고 v1·v2 각각 156개 인증을 독립 산술로 다시 평가합니다. 격자 인증은 저장된 이산 모델에 대한 것이며 연속 모델의 이산화 오차는 별도 수치 비교입니다. 재실행 방법은 [docs/SENSITIVITY.md](docs/SENSITIVITY.md)에 있습니다.
