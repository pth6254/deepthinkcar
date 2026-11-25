"""게임패드 입력: 수동 조향/스로틀, 모드 버튼, 비상정지 버튼, 녹화 버튼.

    python inputs/joystick.py           # 축/버튼 번호 확인 (config.yaml 의 joystick 항목에 반영)
    python inputs/joystick.py --mock    # 가상 패드 동작 확인

main 루프가 매 주기 poll() 을 호출한다. 버튼은 '이번 poll 사이에 눌렸는지'(edge)로 전달된다.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # 프로젝트 루트
import argparse
import logging
import os
import time
from dataclasses import dataclass

from config import load_config

log = logging.getLogger("joystick")


@dataclass
class JoyState:
    connected: bool = False
    steer: float = 0.0        # -1(좌) ~ +1(우)
    throttle: float = 0.0     # -1(후진) ~ +1(전진)
    estop: bool = False       # 비상정지 버튼 눌림
    mode: bool = False        # 모드 버튼 눌림
    record: bool = False      # 녹화 버튼 눌림
    lost_for: float = 0.0     # 연결이 끊긴 뒤 지난 시간(초). 연결 중이면 0
    set_mode: str | None = None   # 특정 모드로 바로 전환 요청 (웹 대시보드)


class Joystick:
    def __init__(self, cfg):
        self.c = cfg["joystick"]
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")                 # 창 없이 이벤트만 사용
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        self.pg = pygame
        pygame.display.init()
        pygame.joystick.init()
        self._js = None
        self._last_ok = float("-inf")      # 한 번도 연결된 적 없으면 lost_for = inf → 즉시 '끊김'
        self._buttons = {self.c["button_estop"]: "estop", self.c["button_mode"]: "mode",
                         self.c["button_record"]: "record"}
        self._open()
        if self._js is None:
            log.warning("게임패드가 연결되어 있지 않음 — 연결을 기다립니다")

    def _open(self):
        pg = self.pg
        if pg.joystick.get_count() > self.c["index"]:
            self._js = pg.joystick.Joystick(self.c["index"])
            log.info("게임패드 연결: %s (축 %d, 버튼 %d)", self._js.get_name(),
                     self._js.get_numaxes(), self._js.get_numbuttons())

    def _axis(self, idx, invert):
        v = self._js.get_axis(idx)
        dz = self.c["deadzone"]
        if abs(v) < dz:
            return 0.0
        v = (abs(v) - dz) / (1 - dz) * (1 if v > 0 else -1)       # 데드존 밖을 0~1 로 재배치
        return -v if invert else v

    def poll(self, now=None):
        pg, c = self.pg, self.c
        now = time.monotonic() if now is None else now
        st = JoyState()
        for ev in pg.event.get():
            if ev.type == pg.JOYDEVICEREMOVED:
                if self._js is not None and ev.instance_id == self._js.get_instance_id():
                    log.warning("게임패드 연결 끊김")
                    self._js = None
            elif ev.type == pg.JOYDEVICEADDED:
                if self._js is None:
                    self._open()
            elif ev.type == pg.JOYBUTTONDOWN and self._js is not None:
                name = self._buttons.get(ev.button)
                if name and ev.instance_id == self._js.get_instance_id():
                    setattr(st, name, True)
        if self._js is not None:
            try:
                st.steer = self._axis(c["axis_steer"], c["invert_steer"])
                st.throttle = self._axis(c["axis_throttle"], c["invert_throttle"])
                st.connected = True
                self._last_ok = now
            except pg.error:
                log.warning("게임패드 읽기 실패 → 연결 끊김으로 처리")
                self._js = None
        if not st.connected:
            st = JoyState(lost_for=max(0.0, now - self._last_ok))   # 끊긴 동안의 버튼/축은 무시
        return st

    def raw(self):
        """(축 값 목록, 눌린 버튼 번호 목록) — 매핑 확인용."""
        if self._js is None:
            return [], []
        return ([round(self._js.get_axis(i), 2) for i in range(self._js.get_numaxes())],
                [i for i in range(self._js.get_numbuttons()) if self._js.get_button(i)])

    def close(self):
        self.pg.quit()


class MockJoystick:
    """하드웨어 없는 가상 패드. 코드(press/set_axes/set_connected)나 키보드(key)로 조작."""

    KEYS = "a/d: 조향  w/s: 스로틀  x: 축 중립  space: 비상정지  m: 모드  r: 녹화"

    def __init__(self, cfg=None):
        self.steer = self.throttle = 0.0
        self.connected = True
        self._pressed = set()
        self._lost_since = None

    def press(self, name):
        self._pressed.add(name)

    def set_axes(self, steer, throttle):
        self.steer = max(-1.0, min(1.0, steer))
        self.throttle = max(-1.0, min(1.0, throttle))

    def set_connected(self, connected):
        self.connected = connected
        if connected:
            self._lost_since = None

    def key(self, k):
        """cv2.waitKey() 값으로 조작 (main.py --show)."""
        ch = chr(k & 0xFF) if k >= 0 else ""
        if ch in "ad":
            self.set_axes(self.steer + (0.25 if ch == "d" else -0.25), self.throttle)
        elif ch in "ws":
            self.set_axes(self.steer, self.throttle + (0.25 if ch == "w" else -0.25))
        elif ch == "x":
            self.set_axes(0, 0)
        elif ch in " mr":
            self.press({" ": "estop", "m": "mode", "r": "record"}[ch])

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        pressed, self._pressed = self._pressed, set()
        if not self.connected:
            if self._lost_since is None:
                self._lost_since = now
            return JoyState(lost_for=now - self._lost_since)
        return JoyState(True, self.steer, self.throttle, "estop" in pressed, "mode" in pressed,
                        "record" in pressed)

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser(description="게임패드 축/버튼 확인")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seconds", type=float, default=0, help="0 이면 Ctrl+C 까지")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    cfg = load_config(args.config)

    if args.mock:
        js = MockJoystick(cfg)
        script = [("축 입력", lambda: js.set_axes(-0.5, 0.3)), ("모드 버튼", lambda: js.press("mode")),
                  ("비상정지 버튼", lambda: js.press("estop")), ("연결 끊김", lambda: js.set_connected(False)),
                  ("재연결", lambda: js.set_connected(True))]
        for i, (label, action) in enumerate(script):
            action()
            print(f"{label:10s} → {js.poll(now=float(i))}")
            if label == "연결 끊김":
                print(f"{'  1.5초 뒤':10s} → {js.poll(now=i + 1.5)}")
        return

    js = Joystick(cfg)
    print("스틱을 움직이고 버튼을 눌러 번호를 확인하세요 (Ctrl+C 종료)")
    t0 = time.monotonic()
    try:
        while not args.seconds or time.monotonic() - t0 < args.seconds:
            st = js.poll()
            axes, buttons = js.raw()
            events = [n for n in ("estop", "mode", "record") if getattr(st, n)]
            print(f"\r축 {axes} 버튼 {buttons} | steer={st.steer:+.2f} thr={st.throttle:+.2f} "
                  f"연결={st.connected} 끊김={st.lost_for:.1f}s {events}      ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        js.close()


if __name__ == "__main__":
    main()
