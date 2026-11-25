# deepthinkcar — 라즈베리파이 4 차선 인식 자율주행 RC카

가상환경은 `venv/` 입니다. 이 폴더 안에서 터미널을 열면 자동으로 활성화되므로 `python` 으로 바로 실행합니다.

## 폴더 구조
```
main.py              20Hz 메인 루프 (모드: manual / opencv / cnn) ← 실행 진입점
config.yaml          모든 설정값 (HSV, PID, 서보/ESC 펄스, 속도, 웹 등)
config.py            설정 로드/저장

hardware/            하드웨어
  camera.py            picamera2 캡처 스레드 (--mock: 동영상 파일)
  actuator.py          PCA9685 ch0 서보 / ch1 ESC (+ 워치독)
  calibrate.py         조향 서보 보정 → config.yaml 저장
  straight_test.py     바닥 직진 보정 (2초)
  test_all.py          하드웨어 전체 점검
inputs/              조작 장치
  keyboard_input.py    SSH 키보드
  joystick.py          게임패드 (+ 테스트용 가상 입력)
autopilot/           자율주행
  lane_detect.py       HSV→블러→Canny→ROI→Hough→좌우 차선→오차 e
  controller.py        PID (±30°, 변화율 제한) + 오차 기반 감속
  cnn_pilot.py         TFLite 추론
training/            CNN 학습
  recorder.py          수동 주행 기록 (data/<세션>/images + log.csv)
  make_video.py        녹화 → 동영상 (videos/)
  train.py             AI 모델 학습: 키트 모델 추가 학습(기본) / --from-scratch
web/                 웹 대시보드 (web_dashboard.py 서버, index.html 화면)
sim/simulator.py     트랙 시뮬레이터 (샘플 영상·학습 데이터 생성)
tests/test_safety.py 안전 동작 테스트
assets/sample.mp4    mock 카메라용 샘플 영상 (빨간 테이프 트랙)
models/              lane_kit.h5 (키트 원본, 추가 학습용) / lane_kit.tflite (cnn 모드 기본)
data/                녹화 데이터 (git 제외)
```

## 실차 사용 순서
1. 보정 (바퀴를 띄우고): `python hardware/calibrate.py`
2. 주행: `python main.py` → 로그의 `http://<pi>:8080` 로 접속
   - 영상 보며 HSV/PID 튜닝 → [config.yaml 에 저장]
   - 움직이면 모든 모드에서 자동 녹화 → `data/` (사진마다 모드 기록)
   - [OpenCV] / [AI 주행] 버튼으로 자율주행
3. AI: `python training/train.py` → 결과 파일을 config.yaml 의 cnn.model 에 지정

다른 조작 장치: `python main.py --input keyboard` (SSH) / `--input gamepad`

## 하드웨어 없이
```
python main.py --mock --mode opencv            # 샘플 영상 + 가상 입력
python main.py --sim --input web               # 시뮬레이터를 웹으로 조종
python main.py --sim --mock --mode opencv --laps 5   # 5바퀴 폐루프 검증
python tests/test_safety.py
```

## 안전 동작
- 비상정지(웹 버튼/Space) → 스로틀 0, 조향 중립
- 차선 0.5초 미검출 / 조작 장치 1초 끊김 / 카메라 끊김 → 정지 후 수동 모드
- Ctrl+C, 예외, SIGTERM, SSH 종료 → 모터 정지 후 종료
- 메인 루프가 0.5초 멈추면 액추에이터 워치독이 중립

## 배선 (deepThinkCar-mini 키트, JD-edu: github.com/JD-edu/deepThinkCar_mini)
| 부품 | 연결 |
|---|---|
| 18650 ×2 | HAT 의 BAT, HAT 아래쪽 전원 스위치 |
| 오른쪽 모터 / 왼쪽 모터 | HAT M1 / M2 (GPIO PWM 19·DIR 13 / PWM 12·DIR 16, 100Hz) |
| 조향 서보 | HAT JP1:S1, 주황색 선이 S (PCA9685 0번 채널, 90° = 직진) |

HAT 은 GPIO 40핀에 어긋남 없이 꽂는다.
