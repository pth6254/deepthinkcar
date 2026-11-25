"""직진 보정: 조향 중앙으로 2초간 직진. 바닥에 놓고 앞을 비운 뒤 실행.

    python hardware/straight_test.py
왼쪽으로 휘면 servo_center_deg 를 올리고(+1), 오른쪽으로 휘면 내린다(-1).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # 프로젝트 루트
import time

from config import load_config
from hardware.actuator import Actuator

if __name__ == "__main__":
    if input("앞쪽 2m 를 비웠습니까? 차가 출발합니다 (y) > ").strip().lower() != "y":
        raise SystemExit("취소")
    act = Actuator(load_config())
    try:
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            act.set(0.0, 0.2)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        act.close()
