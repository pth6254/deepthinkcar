"""안전 동작 검증 (하드웨어 불필요).   python tests/test_safety.py

가짜 시계/카메라/액추에이터로 Car.step() 을 돌려 정지 조건의 타이밍을 확인하고,
main.py 를 실제로 실행해 Ctrl+C / SIGTERM / 예외 시 모터가 중립으로 끝나는지 확인한다.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # 프로젝트 루트
import signal
import subprocess
import sys
import time

import numpy as np

import main as main_mod
from config import load_config
from inputs.joystick import MockJoystick
from main import Car
from sim.simulator import SimWorld

CFG = load_config()
LANE = SimWorld().render()                       # 차선이 보이는 프레임
BLANK = np.full_like(LANE, 150)                  # 차선이 없는 프레임
DT = 0.05


class FakeCamera:
    def __init__(self):
        self.frame, self.t = LANE, 0.0
        self.stalled = False

    def tick(self, now):
        if not self.stalled:
            self.t = now

    def read(self):
        return self.frame, self.t


class FakeActuator:
    def __init__(self):
        self.cmd = None

    def set(self, steer, throttle):
        self.cmd = (steer, throttle)


class FakeRecorder:
    active = False


class Rig:
    def __init__(self, mode):
        self.cam, self.act, self.js = FakeCamera(), FakeActuator(), MockJoystick()
        self.car = Car(CFG, self.cam, self.act, self.js, FakeRecorder(), mode=mode)
        self.now = 100.0

    def step(self, n=1):
        for _ in range(n):
            self.now += DT
            self.cam.tick(self.now)
            self.car.step(self.now)
        return self.act.cmd

    def run_until_stopped(self, limit=3.0):
        """정지할 때까지 걸린 시간(초). 정지 전까지는 스로틀이 유지되는지도 확인."""
        t0 = self.now
        while self.now - t0 < limit:
            if self.step() == (0.0, 0.0):
                return self.now - t0
        raise AssertionError("정지하지 않음")


def test_opencv_drives():
    r = Rig("opencv")
    steer, throttle = r.step(10)
    assert r.car.mode == "opencv" and throttle > 0


def test_estop():
    r = Rig("opencv")
    assert r.step(10)[1] > 0
    r.js.press("estop")
    assert r.step() == (0.0, 0.0), "비상정지 즉시 스로틀 0 · 조향 중립"
    assert r.car.estop and r.car.mode == "manual"
    r.js.set_axes(0.5, 1.0)                      # 스틱을 밀어도, 모드 버튼을 눌러도 움직이지 않아야 함
    r.js.press("mode")
    assert r.step(5) == (0.0, 0.0) and r.car.mode == "manual"
    r.js.press("estop")                          # 스로틀을 민 채로는 해제 불가
    assert r.step() == (0.0, 0.0) and r.car.estop
    r.js.set_axes(0.0, 0.0)
    r.js.press("estop")
    r.step()
    assert not r.car.estop
    r.js.set_axes(0.0, 1.0)
    assert r.step()[1] > 0, "해제 후 수동 주행 가능"


def test_lane_lost():
    r = Rig("opencv")
    assert r.step(10)[1] > 0
    r.cam.frame = BLANK
    assert r.step(4)[1] > 0, "0.2초 미검출: 아직 주행(저속)"
    took = 0.2 + r.run_until_stopped()
    assert 0.5 <= took <= 0.5 + DT + 1e-6, f"차선 미검출 정지까지 {took:.2f}s"
    assert r.car.mode == "manual" and "차선" in r.car.stop_reason
    r.cam.frame = LANE                           # 차선이 다시 보여도 스스로 재출발하지 않는다
    assert r.step(10) == (0.0, 0.0)


def _ai_rig():
    class FakePilot:
        def predict(self, frame):
            return 0.1, 0.2
    r = Rig("manual")
    r.car.pilot = FakePilot()
    r.step()
    r.car._enter("cnn", r.now)
    return r


def test_lane_lost_ai():
    r = _ai_rig()
    assert r.step(10)[1] > 0 and r.car.mode == "cnn"
    r.cam.frame = BLANK
    assert r.step(4)[1] > 0
    took = 0.2 + r.run_until_stopped()
    assert 0.5 <= took <= 0.5 + DT + 1e-6, f"AI 차선 미검출 정지까지 {took:.2f}s"
    assert r.car.mode == "manual" and "차선" in r.car.stop_reason
    r.cam.frame = LANE
    assert r.step(10) == (0.0, 0.0), "차선이 다시 보여도 스스로 재출발하지 않음"


def test_lane_lost_ai_recovers_within_timeout():
    r = _ai_rig()
    r.step(10)
    r.cam.frame = BLANK
    r.step(6)
    r.cam.frame = LANE
    r.step(20)
    assert max(r.step()[1] for _ in range(3)) > 0 and r.car.mode == "cnn"   # 끊어 전진의 쉬는 순간 감안


def test_lane_lost_recovers_within_timeout():
    r = Rig("opencv")
    r.step(10)
    r.cam.frame = BLANK
    r.step(6)                                    # 0.3초
    r.cam.frame = LANE
    r.step(20)
    assert max(r.step()[1] for _ in range(3)) > 0 and r.car.mode == "opencv"   # 끊어 전진의 쉬는 순간 감안


def test_camera_stall():
    r = Rig("opencv")
    r.step(10)
    r.cam.stalled = True
    took = r.run_until_stopped()
    assert took <= 0.5 + DT + 1e-6, f"카메라 정지 후 {took:.2f}s"
    assert r.car.mode == "manual"


def test_gamepad_lost_in_auto():
    r = Rig("opencv")
    r.step(10)
    r.js.set_connected(False)
    assert r.step(10)[1] > 0, "0.5초 끊김: 아직 주행"
    took = 0.5 + r.run_until_stopped()
    assert 1.0 <= took <= 1.0 + 2 * DT + 1e-6, f"게임패드 끊김 정지까지 {took:.2f}s"
    assert r.car.mode == "manual" and "조작 장치" in r.car.stop_reason
    r.js.set_connected(True)                     # 재연결되어도 자율주행으로 돌아가지 않는다
    assert r.step(5) == (0.0, 0.0) and r.car.mode == "manual"


def test_gamepad_lost_in_manual():
    r = Rig("manual")
    r.js.set_axes(0.3, 1.0)
    assert r.step(3)[1] > 0
    r.js.set_connected(False)
    assert r.step() == (0.0, 0.0), "수동 모드에서는 끊기는 즉시 정지"


def test_no_gamepad_blocks_auto_start():
    """패드가 한 번도 연결되지 않았으면(lost_for=inf) 자율 모드로 시작해도 한 발짝도 움직이지 않는다."""
    from inputs.joystick import JoyState
    r = Rig("opencv")
    r.js.poll = lambda now=None: JoyState(lost_for=float("inf"))
    for _ in range(30):
        assert r.step() == (0.0, 0.0)
    assert r.car.mode == "manual"


def test_exception_stops_motor(monkeypatch=None):
    """루프 도중 예외가 나도 액추에이터가 중립으로 끝난다."""
    made = []

    class SpyActuator(main_mod.Actuator):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    calls = {"n": 0}
    orig_step = Car.step

    def bad_step(self, now):
        calls["n"] += 1
        getattr(self.joystick, "primary", self.joystick).set_axes(0.0, 1.0)   # 전진 중에
        if calls["n"] > 5:
            raise RuntimeError("테스트용 예외")
        return orig_step(self, now)

    saved = main_mod.Actuator, Car.step, sys.argv
    main_mod.Actuator, Car.step = SpyActuator, bad_step
    sys.argv = ["main.py", "--mock"]
    try:
        main_mod.main()
    except SystemExit as e:
        assert e.code == 1
    finally:
        main_mod.Actuator, Car.step, sys.argv = saved
    a = made[0]
    assert calls["n"] > 5
    assert (a.servo_us, a.esc_us) == (a.deg_to_us(a.a["servo_center_deg"]), a.a["esc_neutral_us"])


def _run_and_signal(sig):
    p = subprocess.Popen([sys.executable, str(_Path(__file__).resolve().parents[1] / "main.py"), "--mock", "--mode", "opencv", "-v"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(4)
    p.send_signal(sig)
    out, _ = p.communicate(timeout=10)
    cmds = [l for l in out.splitlines() if "[mock] servo" in l]
    assert any("L=+0.00 R=+0.00" not in l for l in cmds[:-1]), "주행 중이었어야 함"
    assert "servo(ch0)=1500us  motor L=+0.00 R=+0.00" in cmds[-1], cmds[-1]
    assert "액추에이터 정지" in out


def test_ctrl_c_stops_motor():
    _run_and_signal(signal.SIGINT)


def test_sigterm_stops_motor():
    _run_and_signal(signal.SIGTERM)


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {name}: {e}")
    sys.exit(1 if failed else 0)
