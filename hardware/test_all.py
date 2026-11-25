"""하드웨어 전체 점검: 전원 → 서보 드라이버 → 카메라 → 조향 서보 → 뒷바퀴 모터(왼쪽/오른쪽/양쪽/후진).

    python hardware/test_all.py            # 바퀴를 띄웠는지 물어본 뒤 진행
    python hardware/test_all.py --yes      # 묻지 않고 진행 (바퀴를 띄운 상태에서만!)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # 프로젝트 루트
import argparse
import subprocess
import time

from config import load_config


def check(name, fn):
    try:
        ok, msg = fn()
    except Exception as e:
        ok, msg = False, f"{type(e).__name__}: {e}"
    print(f"[{'정상' if ok else '이상'}] {name}: {msg}", flush=True)
    return ok


def power():
    out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
    return out.endswith("=0x0"), out


def camera(cfg):
    from hardware.camera import Camera
    cam = Camera(cfg)
    try:
        cam.wait_ready()
        time.sleep(1.5)
        frame, _ = cam.read()
        return frame.mean() > 10, f"{frame.shape[1]}x{frame.shape[0]}, 밝기 {frame.mean():.0f}/255"
    finally:
        cam.close()


def main():
    ap = argparse.ArgumentParser(description="하드웨어 전체 점검")
    ap.add_argument("--yes", action="store_true", help="바퀴 띄움 확인 질문 생략")
    args = ap.parse_args()
    cfg = load_config()
    if cfg["actuator"]["drive_motor"] != "l9110":
        raise SystemExit("drive_motor: l9110 설정에서만 사용")
    from hardware.actuator import Actuator

    ok = check("Pi 전원", power)
    ok &= check("카메라", lambda: camera(cfg))
    try:
        act = Actuator(cfg, watchdog=False)      # 아래에서 단계마다 직접 정지시킨다
    except Exception as e:
        raise SystemExit(f"[이상] 서보 드라이버/모터 핀: {type(e).__name__}: {e}")
    print("[정상] 서보 드라이버(PCA9685) 연결", flush=True)
    neutral = cfg["actuator"]["esc_neutral_us"]
    motors, duty = act.motors, 0.40
    saved = {s: cfg["motor"][f"{s}_reverse"] for s in ("left", "right")}

    def drive(label, left, right):
        print(f"[눈으로 확인] {label} (듀티 {duty:.0%}, 1.5초)", flush=True)
        for side, v in (("left", left), ("right", right)):
            v = -v if saved[side] else v
            pwm, dirp = motors._dev[side]
            dirp.value = v < 0
            pwm.value = v if v >= 0 else 1.0 + v         # 역회전은 PWM 반전
        time.sleep(1.5)
        motors.stop()
        time.sleep(1.0)

    try:
        for label, s in (("정면", 0), ("왼쪽", -1), ("정면", 0), ("오른쪽", 1), ("정면", 0)):
            deg = act.steer_to_deg(s)
            print(f"[눈으로 확인] 조향 {label} {deg:.0f}°", flush=True)
            act.write_us(act.deg_to_us(deg), neutral)
            time.sleep(1.5)
        if not args.yes and input("바퀴를 바닥에서 띄웠습니까? (y) > ").strip().lower() != "y":
            print("구동 모터 점검 건너뜀")
        else:
            drive("왼쪽 바퀴만 전진", duty, 0)
            drive("오른쪽 바퀴만 전진", 0, duty)
            drive("양쪽 전진", duty, duty)
            drive("양쪽 후진", -duty, -duty)
    except (KeyboardInterrupt, EOFError):
        print("\n중단")
    finally:
        act.close()
    print("자동 점검:", "모두 정상" if ok else "이상 있음 (위 목록 확인)")


if __name__ == "__main__":
    main()
