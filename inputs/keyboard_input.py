"""SSH 터미널 키보드로 수동 조향/스로틀, 모드 전환, 비상정지, 녹화.

    python inputs/keyboard_input.py      # 키 입력과 연결 상태 확인

조작 (한글 입력 상태여도 동작)
    w / ↑ : 스로틀 +      s / ↓ : 스로틀 -      a / ← : 왼쪽      d / → : 오른쪽
    x     : 스로틀 0 · 조향 중앙
    space : 비상정지 (스로틀 0에서 다시 누르면 해제)
    m     : 모드 전환 (manual → opencv → cnn)
    r     : 녹화 시작/종료 (manual 모드)
    h     : 도움말          Ctrl+C : 종료 (모터 정지)

연결 끊김 감지
    키보드에는 '연결됨' 신호가 없으므로, 0.2초마다 터미널에 커서 위치 질의(ESC[6n)를 보내고
    SSH 너머의 터미널이 자동으로 돌려주는 응답(ESC[행;열R)을 heartbeat 로 쓴다.
    응답이 끊기면 main.py 가 차를 세운다 (input.timeout_s).
    tmux/screen 은 SSH 가 끊겨도 스스로 응답해 버려 끊김을 감지할 수 없으므로 그 안에서는 실행을 거부한다.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # 프로젝트 루트
import atexit
import logging
import os
import re
import select
import termios
import time
import tty

from inputs.joystick import JoyState

log = logging.getLogger("keyboard")

HELP = ("[키보드] w/s(↑↓): 스로틀  a/d(←→): 조향  x: 정지·중앙  space: 비상정지  "
        "m: 모드  r: 녹화  h: 도움말  Ctrl+C: 종료")
_ESC_SEQ = re.compile(rb"\x1b(?:\[([\d;]*)|O)([A-Za-z~])")      # CSI / SS3 시퀀스
_ESC_PARTIAL = re.compile(rb"\x1b(?:\[[\d;]*|O)?")              # 아직 다 도착하지 않은 시퀀스
_ARROWS = {b"A": "w", b"B": "s", b"C": "d", b"D": "a"}
_HANGUL = {"ㅈ": "w", "ㄴ": "s", "ㅁ": "a", "ㅇ": "d", "ㅌ": "x", "ㅡ": "m", "ㄱ": "r", "ㅗ": "h"}


class InputError(RuntimeError):
    pass


def _clip(v):
    return max(-1.0, min(1.0, v))


class KeyboardPad:
    """joystick.Joystick 과 같은 인터페이스: poll() → JoyState."""

    def __init__(self, cfg, tty_path="/dev/tty"):
        c = cfg["keyboard"]
        self.steer_step = c["steer_step"]
        self.throttle_step = c["throttle_step"]
        self.ping_interval = c["ping_interval_s"]
        self.reply_timeout = c["reply_timeout_s"]
        if os.environ.get("TMUX") or os.environ.get("STY"):
            raise InputError("tmux/screen 안에서는 SSH 연결 끊김을 감지할 수 없어 키보드 조작을 쓸 수 없습니다. "
                             "SSH 셸에서 직접 실행하세요")
        try:
            self.fd = os.open(tty_path, os.O_RDWR | os.O_NOCTTY)
        except OSError as e:
            raise InputError(f"터미널을 열 수 없음({e}) — SSH 터미널에서 직접 실행하세요") from e
        self._old_attr = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)          # 한 글자씩 즉시, 에코 없음. Ctrl+C(SIGINT)는 그대로 동작
        self._closed = False
        atexit.register(self.close)

        self.steer = self.throttle = 0.0
        self._buf = b""
        self._events = set()
        self._eof = False
        self._last_ping = self._last_reply = float("-inf")

        # 터미널이 커서 위치 질의에 응답하는지 확인 (응답이 없으면 끊김 감지가 불가능 → 사용 거부)
        end = time.monotonic() + 1.5
        while self._last_reply == float("-inf"):
            now = time.monotonic()
            if now > end or self._eof:
                self.close()
                raise InputError("터미널이 커서 위치 질의(ESC[6n)에 응답하지 않아 연결 끊김을 감지할 수 없습니다")
            self._pump(now, wait=0.05)
        self._write(("\r\n" + HELP + "\r\n").encode())

    # ---- 터미널 입출력 ----
    def _write(self, data):
        try:
            os.write(self.fd, data)
        except OSError:
            self._eof = True

    def _pump(self, now, wait=0.0):
        if now - self._last_ping >= self.ping_interval:
            self._last_ping = now
            self._write(b"\x1b[6n")
        while not self._eof:
            ready, _, _ = select.select([self.fd], [], [], wait)
            wait = 0.0
            if not ready:
                break
            try:
                data = os.read(self.fd, 1024)
            except OSError:
                data = b""
            if not data:                 # SSH 세션 종료
                self._eof = True
                break
            self._buf += data
        self._parse(now)

    def _parse(self, now):
        buf = self._buf
        while buf:
            b = buf[0]
            if b == 0x1B:
                m = _ESC_SEQ.match(buf)
                if m:
                    buf = buf[m.end():]
                    params, final = m.group(1), m.group(2)
                    if final == b"R" and params is not None and params.count(b";") == 1:
                        self._last_reply = now           # heartbeat 응답
                    elif final in _ARROWS:
                        self._key(_ARROWS[final])
                    continue
                if _ESC_PARTIAL.fullmatch(buf):
                    break                                # 나머지 바이트를 기다린다
                buf = buf[1:]                            # ESC 단독 / 모르는 시퀀스
            elif b >= 0x80:                              # UTF-8 (한글 자모)
                n = 3 if 0xE0 <= b < 0xF0 else 2 if 0xC0 <= b < 0xE0 else 4 if b >= 0xF0 else 1
                if len(buf) < n:
                    break
                self._key(_HANGUL.get(buf[:n].decode("utf-8", "ignore"), ""))
                buf = buf[n:]
            else:
                self._key(chr(b).lower())
                buf = buf[1:]
        self._buf = buf

    def _key(self, k):
        if not k:
            return
        if k == "w":
            self.throttle = _clip(self.throttle + self.throttle_step)
        elif k == "s":
            self.throttle = _clip(self.throttle - self.throttle_step)
        elif k == "a":
            self.steer = _clip(self.steer - self.steer_step)
        elif k == "d":
            self.steer = _clip(self.steer + self.steer_step)
        elif k == "x":
            self.reset_axes()
        elif k == " ":
            self.reset_axes()
            self._events.add("estop")
        elif k == "m":
            self._events.add("mode")
        elif k == "r":
            self._events.add("record")
        elif k == "h":
            self._write(("\r\n" + HELP + "\r\n").encode())

    # ---- 공개 인터페이스 ----
    def reset_axes(self):
        """조향 중앙, 스로틀 0. 모드 전환/안전 정지 때 main 이 호출 (이전 스로틀로 다시 출발하지 않도록)."""
        self.steer = self.throttle = 0.0

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        self._pump(now)
        events, self._events = self._events, set()
        silent = now - self._last_reply
        if self._eof or silent > self.reply_timeout:
            self.reset_axes()            # 끊긴 동안의 입력은 버리고, 재연결돼도 0에서 시작
            return JoyState(lost_for=float("inf") if self._eof else silent)
        return JoyState(True, self.steer, self.throttle,
                        "estop" in events, "mode" in events, "record" in events)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old_attr)
        except (OSError, termios.error):
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass


def main():
    from config import load_config
    logging.basicConfig(level=logging.INFO)
    pad = KeyboardPad(load_config())
    try:
        while True:
            st = pad.poll()
            ev = [n for n in ("estop", "mode", "record") if getattr(st, n)]
            print(f"\rsteer={st.steer:+.2f} throttle={st.throttle:+.2f} 연결={st.connected} "
                  f"끊김={min(st.lost_for, 99):.1f}s {ev}        ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        pad.close()
        print()


if __name__ == "__main__":
    main()
