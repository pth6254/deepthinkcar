"""조향 서보(PCA9685 ch0) + 뒷바퀴 구동 (L9110 DC 모터 2개, GPIO PWM 100Hz) — 또는 ESC(ch1).

drive_motor: l9110 (키트 기본) | esc

    python hardware/actuator.py --mock                 # 하드웨어 없이 동작 로그 확인
    python hardware/actuator.py --throttle 0.2         # 실제 구동 테스트 (바퀴를 띄우고!)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # 프로젝트 루트
import argparse
import atexit
import logging
import math
import threading
import time

from config import load_config

log = logging.getLogger("actuator")

# ServoKit 채널을 이 범위의 '펄스폭(us) 출력기'로 쓴다 (fraction 0~1 ↔ 500~2500us)
PULSE_MIN, PULSE_MAX = 500, 2500


class L9110Motors:
    """L9110 드라이버 + DC 기어드 모터 2개 (BCM 핀). 핀 하나는 PWM(속도), 하나는 DIR(방향).

    L9110 진리표: IA=1,IB=0 → 정회전 / IA=0,IB=1 → 역회전 / 둘 다 0 → 정지.
    DIR=1(역회전)일 때는 PWM 이 LOW 인 동안 모터가 돌므로 듀티를 반전해서 낸다.
    """

    def __init__(self, cfg, mock=False):
        m = cfg["motor"]
        self.m = m
        self.mock = mock
        self._dev = {}
        if not mock:
            from gpiozero import DigitalOutputDevice, PWMOutputDevice
            for side in ("left", "right"):
                self._dev[side] = (PWMOutputDevice(m[f"{side}_pwm_pin"], frequency=m["pwm_hz"]),
                                   DigitalOutputDevice(m[f"{side}_dir_pin"]))
        self.duty = {"left": 0.0, "right": 0.0}

    def throttle_to_duty(self, throttle):
        """-1~1 → 부호 있는 듀티. 0 이 아니면 최소 듀티(모터가 실제로 도는 값) 이상."""
        t = max(-1.0, min(1.0, throttle))
        if abs(t) < 1e-3:
            return 0.0
        d = self.m["duty_min"] + abs(t) * (self.m["duty_max"] - self.m["duty_min"])
        return d if t > 0 else -d

    def set(self, throttle):
        d = self.throttle_to_duty(throttle)
        for side in ("left", "right"):
            ds = -d if self.m[f"{side}_reverse"] else d
            self.duty[side] = ds
            if side in self._dev:
                pwm, dirp = self._dev[side]
                if ds >= 0:
                    dirp.off(); pwm.value = ds
                else:
                    dirp.on(); pwm.value = 1.0 - abs(ds)      # 역회전: PWM 반전

    def stop(self):
        self.set(0.0)
        for pwm, dirp in self._dev.values():                   # 둘 다 LOW = 확실한 정지
            pwm.value = 0.0; dirp.off()

    def close(self):
        self.stop()
        for pwm, dirp in self._dev.values():
            pwm.close(); dirp.close()


class Actuator:
    def __init__(self, cfg, mock=False, watchdog=True):
        self.a = a = cfg["actuator"]
        self.mock = mock
        self._lock = threading.Lock()
        self._servo = self._esc = None
        self.motors = L9110Motors(cfg, mock) if a["drive_motor"] == "l9110" else None
        if not mock:
            from adafruit_servokit import ServoKit
            kit = ServoKit(channels=16, address=a["i2c_address"], frequency=a["pwm_freq"])
            self._servo = kit.servo[a["servo_channel"]]
            self._servo.set_pulse_width_range(PULSE_MIN, PULSE_MAX)
            if self.motors is None:
                self._esc = kit.servo[a["esc_channel"]]
                self._esc.set_pulse_width_range(PULSE_MIN, PULSE_MAX)
        self.servo_us = self.esc_us = None
        self._neutral = True
        self._closed = False
        self._last_cmd = time.monotonic()
        self.stop()
        if not mock and self.motors is None and a["arm_seconds"] > 0:
            log.info("ESC 아밍: 중립 신호 %.1f초", a["arm_seconds"])
            time.sleep(a["arm_seconds"])
        atexit.register(self.close)
        if watchdog and a["watchdog_s"] > 0:
            threading.Thread(target=self._watchdog, daemon=True).start()

    # ---- 정규화 값 → 펄스폭 ----
    def deg_to_us(self, deg):
        """서보 각도(0~180°) → 펄스폭(us)."""
        a = self.a
        deg = max(0.0, min(180.0, deg))
        return a["servo_us_at_0deg"] + deg / 180.0 * (a["servo_us_at_180deg"] - a["servo_us_at_0deg"])

    def steer_to_deg(self, steer):
        """정규화 조향(-1~1) → 서보 각도. 0 → servo_center_deg(90° = 직진)."""
        a, s = self.a, max(-1.0, min(1.0, steer))
        end = a["servo_right_deg"] if s >= 0 else a["servo_left_deg"]
        return a["servo_center_deg"] + abs(s) * (end - a["servo_center_deg"])

    def steer_to_us(self, steer):
        return self.deg_to_us(self.steer_to_deg(steer))

    def throttle_to_us(self, throttle):
        a, t = self.a, max(-1.0, min(1.0, throttle))
        if abs(t) < 1e-3:
            return a["esc_neutral_us"]
        lo, hi = (a["esc_fwd_min_us"], a["esc_fwd_max_us"]) if t > 0 else \
                 (a["esc_rev_min_us"], a["esc_rev_max_us"])
        return lo + abs(t) * (hi - lo)

    # ---- 출력 ----
    def write_us(self, servo_us, esc_us):
        """펄스폭(us)을 직접 출력 (calibrate.py 용 저수준 인터페이스)."""
        servo_us = max(PULSE_MIN, min(PULSE_MAX, servo_us))
        esc_us = max(PULSE_MIN, min(PULSE_MAX, esc_us))
        with self._lock:
            if self._servo is not None:
                span = PULSE_MAX - PULSE_MIN
                if self._esc is not None:
                    self._esc.fraction = (esc_us - PULSE_MIN) / span  # 스로틀을 먼저
                self._servo.fraction = (servo_us - PULSE_MIN) / span
            self.servo_us, self.esc_us = servo_us, esc_us
        if self.mock:
            if self.motors is not None:
                log.debug("[mock] servo(ch%d)=%4.0fus  motor L=%+.2f R=%+.2f", self.a["servo_channel"],
                          servo_us, self.motors.duty["left"], self.motors.duty["right"])
            else:
                log.debug("[mock] servo(ch%d)=%4.0fus  esc(ch%d)=%4.0fus",
                          self.a["servo_channel"], servo_us, self.a["esc_channel"], esc_us)

    def set(self, steer, throttle):
        if not (math.isfinite(steer) and math.isfinite(throttle)):
            log.error("유효하지 않은 명령(steer=%s, throttle=%s) → 정지", steer, throttle)
            return self.stop()
        self._last_cmd = time.monotonic()
        self._neutral = False
        if self.motors is not None:
            with self._lock:
                self.motors.set(throttle)
        self.write_us(self.steer_to_us(steer), self.throttle_to_us(throttle))

    def stop(self):
        """스로틀 0, 조향 중립."""
        self._neutral = True
        if self.motors is not None:
            with self._lock:
                self.motors.stop()
        self.write_us(self.deg_to_us(self.a["servo_center_deg"]), self.a["esc_neutral_us"])

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.stop()
            if self.motors is not None:
                self.motors.close()
            log.info("액추에이터 정지 (스로틀 중립, 조향 중립)")
        except Exception:
            log.exception("액추에이터 정지 실패!")

    def _watchdog(self):
        """메인 루프가 멈춰 명령이 끊기면 스스로 중립으로 되돌린다."""
        while not self._closed:
            time.sleep(0.05)
            if not self._neutral and time.monotonic() - self._last_cmd > self.a["watchdog_s"]:
                log.warning("워치독: %.1f초간 명령 없음 → 정지", self.a["watchdog_s"])
                try:
                    self.stop()
                except Exception:
                    log.exception("워치독 정지 실패!")


def main():
    ap = argparse.ArgumentParser(description="액추에이터 동작 테스트: 조향 좌/중/우 스윕")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--throttle", type=float, default=0.0,
                    help="스윕 중 줄 스로틀 (기본 0 = 조향만). 반드시 바퀴를 띄우고 사용")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("actuator").setLevel(logging.DEBUG)

    act = Actuator(load_config(args.config), mock=args.mock)
    try:
        for steer in (0.0, -0.5, -1.0, 0.0, 0.5, 1.0, 0.0):
            log.info("steer=%+.1f (서보 %.0f°) throttle=%+.2f", steer, act.steer_to_deg(steer), args.throttle)
            t_end = time.monotonic() + 0.8
            while time.monotonic() < t_end:
                act.set(steer, args.throttle)
                time.sleep(0.05)
    finally:
        act.close()


if __name__ == "__main__":
    main()
