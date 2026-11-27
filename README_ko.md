# 고정 커널 정지 분사 배분

JangHo Seo와 Joonwoo Lee의 논문 **Separating operator, index, and certificate
structure in fixed-kernel stop-and-spray allocation**의 코드와 데이터입니다.
[논문 인용과 이용 안내](NOTICE.md)를 참고하세요.

## 결과 검증

Linux 또는 WSL에서 Python 3.12를 사용해 저장소 루트에서 다음 명령을 실행합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python tools/verify_integrity.py
python verification/verify_certificates.py --workers 4
python verification/kernel_width/replay_kernel_witnesses.py
python verification/verify_pso_random_fields.py
```

무결성 검사는 배포 파일을 기록된 SHA-256 해시와 대조합니다. 검증 스크립트는
선별된 사례의 인증서 하한과 배분 목적값을 저장된 입력에서 정확산술로 다시 계산합니다. 이 검증에는
최적화 솔버가 필요하지 않습니다. 개별 검증과 실험 재실행 방법은
[검증·재현 안내](docs/REPRODUCING.md)에 있습니다.

## 데이터와 코드 찾기

- [논문 표·그림 대응표](docs/PAPER_MAP.md)에서 각 결과의 데이터, 검증 상태,
  확인 명령을 찾을 수 있습니다.
- [데이터 안내](docs/DATA.md)에 초기 두께장, 배분, 검증 자료, 결과 파일의 형식이
  설명돼 있습니다.
- [입력 데이터](inputs/), [저장된 결과](results/), [실행 기록](work/)에서
  논문에 사용한 자료를 볼 수 있습니다.
- [시뮬레이션·최적화 코드](src/), [실험 스크립트](pipeline/),
  [검증 코드](verification/)를 제공합니다.

[English guide](README.md)
