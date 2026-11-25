"""메인 제어 루프 (20Hz). 모드: manual / opencv / cnn

    python main.py                               # 실차, 웹 대시보드로 조작 (로그에 접속 주소 출력)
    python main.py --input keyboard              # SSH 터미널 키보드로 조작
    python main.py --sim --input web             # 시뮬레이터 + 웹 대시보드로 연습
    python main.py --mock --mode opencv          # 하드웨어 없이: assets/sample.mp4 + 가상 입력 + 로그 출력
    python main.py --sim --mock --mode opencv --laps 5    # 시뮬레이터 폐루프 주행 검증

안전 동작
    비상정지(space/버튼)          → 스로틀 0, 조향 중립 (다시 누르면 해제)
    차선 0.5초 이상 미검출        → 정지 후 manual 모드로 복귀
    조작 장치(웹/SSH/패드) 1초 끊김 → 정지 후 manual 모드로 복귀
    Ctrl+C / 예외 / SIGTERM / SIGHUP(SSH 종료) → 모터 정지 후 종료
"""
import argparse
import logging
import signal
import sys
import time

import cv2

from hardware.actuator import Actuator
from hardware.camera import Camera
from config import DEFAULT_PATH, load_config
from autopilot.controller import LaneController, RateLimiter
from inputs.joystick import Joystick, MockJoystick
from inputs.keyboard_input import InputError, KeyboardPad
from autopilot.difficulty import Difficulty
from autopilot.lane_detect import LaneDetector
from training.recorder import Recorder
from web.web_dashboard import WebDashboard, WebPad, WithDashboard

log = logging.getLogger("main")
MODES = ("manual", "opencv", "cnn")


class Car:
    """한 주기(step)마다 입력을 읽고 → 안전 조건 확인 → 모드별 명령 계산 → 액추에이터 출력."""

    def __init__(self, cfg, camera, actuator, joystick, recorder, pilot=None, mode="manual"):
        self.cfg = cfg
        self.camera, self.actuator, self.joystick = camera, actuator, joystick
        self.recorder, self.pilot = recorder, pilot
        self.detector = LaneDetector(cfg)
        self.controller = LaneController(cfg)
        self._cnn_limiter = RateLimiter(0.0)
        self.difficulty = Difficulty()
        self.pad_timeout = cfg["input"]["timeout_s"]
        self.configure(cfg)
        self.mode = "manual"
        self.estop = False
        self.stop_reason = None         # 마지막 안전 정지 사유
        self.stop_time = None
        self._initial_mode = mode
        self._assist_n = self._assist_side = 0
        self._sharp = self._sharp_n = self._sharp_side = 0
        self._extreme = self._extreme_n = self._extreme_side = 0
        self._full_lock = False
        self._single_n = 0
        self._auto_rec = False          # 지금 녹화가 자동으로 시작된 것인지
        self._rec_paused = False
        self._moved_at = float("-inf")
        self._pad_lost = False
        self._frame_t = None            # 마지막으로 처리한 프레임의 캡처 시각
        self._frame_seen = None         # 새 프레임을 마지막으로 본 루프 시각
        self._lane_seen = 0.0
        self._auto_cmd = (0.0, 0.0)
        self._cnn_t = None

    def configure(self, cfg):
        """튜닝 값 반영 (웹 대시보드에서 실시간으로 호출)."""
        self.detector.configure(cfg)
        self.controller.configure(cfg)
        c = cfg["control"]
        self.max_steer_deg = c["max_steer_deg"]
        self.normal_steer = self.controller.normal_steer_deg / self.max_steer_deg
        self.lane_timeout = c["lane_lost_timeout_s"]
        self.creep_throttle = c["throttle_min"]
        self.manual_scale = cfg["manual"]["throttle_scale"]
        self.cnn_throttle_max = cfg["cnn"]["throttle_max"]
        self.single_lane_boost = bool(c.get("single_lane_boost", False))
        self.difficulty_speed = bool(c.get("difficulty_speed", False))
        self.extreme_error = c.get("extreme_error", 0.0)
        self.sharp_brake = int(c.get("sharp_brake_frames", 4))
        self.sharp_pulse = max(1, int(c.get("sharp_on_frames", 2))), max(0, int(c.get("sharp_off_frames", 1)))
        self.assist_error = cfg["cnn"].get("assist_error", 0.0)
        self._cnn_limiter.rate = c["max_steer_rate_deg_s"] / c["max_steer_deg"]

    # ---- 모드 전환 ----
    def _enter(self, mode, now):
        if mode == "cnn" and self.pilot is None:
            log.warning("AI 모델이 없어 AI 주행 모드를 쓸 수 없음 → manual")
            mode = "manual"
        self.mode = mode
        self.detector.reset()
        self._single_n = 0
        self._assist_n = self._sharp = self._sharp_n = self._extreme = self._extreme_n = 0
        reset_axes = getattr(self.joystick, "reset_axes", None)
        if reset_axes:                    # 키보드: 이전 스로틀 값으로 다시 출발하지 않도록
            reset_axes()
        self.controller.reset()
        self._cnn_limiter.reset()
        self._full_lock = False
        self.difficulty.reset()
        self._cnn_t = None
        self._lane_seen = now
        self._auto_cmd = (0.0, 0.0)
        log.info("모드: %s", mode)

    def _next_mode(self, now):
        modes = [m for m in MODES if m != "cnn" or self.pilot is not None]
        self._enter(modes[(modes.index(self.mode) + 1) % len(modes)], now)

    def _safe_stop(self, reason, now):
        log.warning("안전 정지: %s → manual 모드로 복귀", reason)
        self.stop_reason, self.stop_time = reason, now
        self._enter("manual", now)

    def _set_sharp(self, lane):
        if lane.sharp and not self._sharp:
            self._sharp_n = 0                             # 급커브 진입: 멈칫부터 다시
        self._sharp, self._sharp_side = lane.sharp, lane.sharp_side
        # 크게 벗어남: 오차 0.9 이상이 3장 연속이면 감속, 12장(0.6초) 연속이면 최대 조향, 0.5 아래로 돌아오면 해제
        e = lane.error
        if self.extreme_error <= 0 or lane.sharp:
            self._extreme = self._extreme_n = 0
        elif e is not None:
            side = 1 if e > 0 else -1
            if self._extreme and (abs(e) < 0.5 or side != self._extreme):
                self._extreme = self._extreme_n = 0
            self._extreme_n = (self._extreme_n + 1 if side == self._extreme_side else 1) if abs(e) >= self.extreme_error else 0
            self._extreme_side = side
            if self._extreme_n >= 3:
                self._extreme = side
                self._full_lock = self._full_lock or self._extreme_n >= 12
        if not self._extreme:
            self._full_lock = False

    def _boosted(self, lane):
        """한쪽 차선만 2장 이상 보이며 크게 벗어남 → 오차에 비례해 최대 조향(60°)까지 허용."""
        single = lane.error is not None and (lane.left is None) != (lane.right is None)
        self._single_n = self._single_n + 1 if single else 0
        return self.single_lane_boost and self._single_n >= 2 and abs(lane.error) >= 0.6

    def _assist(self, steer, error):
        """급커브 보정: 차선이 크게 벗어났는데(연속 3장, 같은 방향) AI 가 덜 꺾을 때만 OpenCV 만큼 꺾는다."""
        if error is None or self.assist_error <= 0:
            self._assist_n = 0
            return steer
        side = 1 if error > 0 else -1
        on = self._assist_n >= 3 and side == self._assist_side
        # 한번 켜지면 오차가 절반(기본 0.4) 아래로 돌아올 때까지 유지 (기준 근처에서 켜졌다 꺼지지 않도록)
        if abs(error) < self.assist_error * (0.5 if on else 1.0):
            self._assist_n = 0
            return steer
        self._assist_n = self._assist_n + 1 if side == self._assist_side else 1
        self._assist_side = side
        if self._assist_n < 3:
            return steer
        want = min(self.normal_steer, self.controller.pid.kp * abs(error) / self.max_steer_deg)
        if abs(error) < self.assist_error and want - side * steer < 11.25 / self.max_steer_deg:
            return steer                  # 돌아오는 중이고 AI 도 충분히 꺾고 있으면 맡긴다
        return side * max(want, side * steer)

    # ---- 한 주기 ----
    def step(self, now):
        js = self.joystick.poll(now)
        frame, frame_t = self.camera.read()
        if self._frame_seen is None:                      # 첫 주기
            self._frame_seen = now
            self._enter(self._initial_mode, now)
        new_frame = frame is not None and frame_t != self._frame_t
        if new_frame:
            self._frame_t, self._frame_seen = frame_t, now

        # 1) 비상정지 (래치)
        if js.estop:
            if not self.estop:
                self.estop = True
                self.stop_reason, self.stop_time = "비상정지", now
                log.warning("비상정지!")
                self._enter("manual", now)
            elif abs(js.throttle) < 0.05:
                self.estop = False
                log.info("비상정지 해제")
            else:
                log.warning("스로틀을 중립에 놓아야 비상정지를 해제할 수 있음")

        # 2) 게임패드 연결 끊김
        pad_lost = js.lost_for >= self.pad_timeout
        if pad_lost and not self._pad_lost:
            if self.mode != "manual":
                self._safe_stop(f"조작 장치 연결 {self.pad_timeout:.1f}초 이상 끊김", now)
            elif js.lost_for == float("inf"):
                log.info("조작 장치 연결 대기 중 (출력 0)")
            else:
                log.warning("조작 장치 연결 끊김 (수동 모드: 출력 0)")
        elif self._pad_lost and not pad_lost:
            log.info("조작 장치 재연결")
        self._pad_lost = pad_lost

        # 3) 버튼
        if js.mode and not self.estop:
            self._next_mode(now)
        if js.set_mode and not self.estop and js.set_mode != self.mode:
            self._enter(js.set_mode, now)
        if js.record and not self.estop:
            if self.recorder.active:
                self.recorder.stop()
                self._auto_rec, self._rec_paused = False, True    # 직접 껐으면 멈출 때까지 자동 녹화도 쉰다
            else:
                self.recorder.start()
                self._auto_rec = False

        # 4) 모드별 명령
        steer = throttle = 0.0
        lane = None
        if self.estop or pad_lost:
            pass
        elif self.mode == "manual":
            if js.connected:                              # 끊긴 순간부터는 즉시 0
                steer, throttle = js.steer, js.throttle * self.manual_scale
        elif now - self._frame_seen >= self.lane_timeout:
            self._safe_stop("카메라 프레임 끊김", now)
        elif self.mode == "opencv":
            if new_frame:
                lane = self.detector.process(frame)
                self._set_sharp(lane)
                if lane.detected:
                    self._lane_seen = now
                    steer_deg, thr = self.controller.update(lane.error, now, lane.sharp == 2 or self._full_lock, self._boosted(lane))
                    self._auto_cmd = (steer_deg / self.max_steer_deg, thr)
                else:                                     # 잠깐 놓쳤을 때: 조향 유지, 저속
                    self._auto_cmd = (self._auto_cmd[0], min(self._auto_cmd[1], self.creep_throttle))
            if new_frame:
                self.difficulty.update(lane, self._auto_cmd[0] * self.max_steer_deg, self.controller.normal_steer_deg)
            if now - self._lane_seen >= self.lane_timeout:
                self._safe_stop(f"차선 {self.lane_timeout:.1f}초 이상 미검출", now)
            else:
                steer, throttle = self._auto_cmd
        elif self.mode == "cnn":
            if new_frame:
                seen = self.detector.process(frame)           # 조향은 AI, 차선 감시와 급커브 보정은 OpenCV
                if seen.detected:
                    self._lane_seen = now
                s, t = self.pilot.predict(frame)
                ai_deg = s * self.max_steer_deg
                boost = self._boosted(seen)
                s = max(-self.normal_steer, min(self.normal_steer, self._assist(s, seen.error)))
                if boost:
                    side = 1 if seen.error > 0 else -1
                    s = side * max(side * s, min(1.0, self.controller.pid.kp * abs(seen.error) / self.max_steer_deg))
                if seen.sharp == 1:
                    s = seen.sharp_side * max(0.0, seen.sharp_side * s)
                elif seen.sharp == 2:
                    s = float(seen.sharp_side)
                self._set_sharp(seen)
                if self._full_lock:
                    s = float(self._extreme)
                dt = 0.05 if self._cnn_t is None else max(0.01, min(0.2, now - self._cnn_t))
                self._cnn_t = now
                self._auto_cmd = (self._cnn_limiter.update(max(-1.0, min(1.0, s)), dt),
                                  max(0.0, min(self.cnn_throttle_max, t, self.controller.speed.update(s, dt))))
                lim = self.controller.normal_steer_deg
                want = None if seen.error is None else max(-lim, min(lim, self.controller.pid.kp * seen.error))
                self.difficulty.update(seen, self._auto_cmd[0] * self.max_steer_deg, lim, ai_deg, want)
            if now - self._lane_seen >= self.lane_timeout:
                self._safe_stop(f"차선 {self.lane_timeout:.1f}초 이상 미검출", now)
            else:
                steer, throttle = self._auto_cmd

        hard = self.difficulty_speed and self.difficulty.hard
        if self.difficulty_speed and self.mode != "manual" and throttle > 0:
            throttle = self.difficulty.limit(throttle, self.controller.speed.t_max, self.controller.speed.t_min)
        if self.mode == "manual" or not (self._sharp or self._extreme or hard):
            self._sharp_n = 0
        else:                                             # 급커브: 잠깐 멈칫(감속) 후 짧게 끊어 천천히 전진
            if new_frame:
                self._sharp_n += 1
                if not self._sharp and self._sharp_n <= self.sharp_brake:
                    self._sharp_n = self.sharp_brake         # 이미 벗어나는 중: 멈칫 없이 끊어 전진
            n, (on, off) = self._sharp_n - self.sharp_brake, self.sharp_pulse
            swing = self._sharp == 2 and abs(self._sharp_side - steer) > 0.25    # 핸들이 다 돌 때까지 서서 기다린다
            throttle = min(throttle, self.creep_throttle) if n >= 0 and not swing and n % (on + off) < on else 0.0

        # 5) 출력 / 기록
        self.actuator.set(steer, throttle)
        moving = (throttle != 0 or (self._sharp_n > 0 and self.mode != "manual")) and not self.estop
        if moving:
            self._moved_at = now
            if getattr(self.recorder, "auto", False) and not self.recorder.active and not self._rec_paused:
                self._auto_rec = self.recorder.start() is not None
        elif now - self._moved_at >= getattr(self.recorder, "auto_stop_s", 2.0):
            self._rec_paused = False
            if self._auto_rec and self.recorder.active:
                self.recorder.stop()
            self._auto_rec = False
        if self.recorder.active and new_frame and not self.estop and (moving or not self._auto_rec):
            self.recorder.add(frame, steer, throttle, now, self.mode)
        return {"mode": self.mode, "steer": steer, "throttle": throttle, "frame": frame,
                "lane": lane, "error": lane.error if lane else None,
                "difficulty": 0.0 if self.mode == "manual" else self.difficulty.value, "reason": self.difficulty.reason}

    def render(self, tel):
        """미리보기/저장용 화면."""
        frame = tel["frame"]
        if frame is None:
            return None
        view = self.detector.draw(frame, tel["lane"]) if tel["lane"] else frame.copy()
        state = "E-STOP" if self.estop else tel["mode"] + (" REC" if self.recorder.active else "")
        text = f"{state} st={tel['steer']:+.2f} th={tel['throttle']:+.2f}"
        y = view.shape[0] - 8
        cv2.putText(view, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(view, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        return view


def run(car, args, hz, world=None, dash=None):
    period = 1.0 / hz
    writer = None
    fps_ema = None
    fps_all, proc_all = [], []
    prev = None
    start = next_t = time.monotonic()
    last_log = 0.0
    try:
        while True:
            t0 = time.monotonic()
            if dash is not None:
                dash.apply_tuning(car)
            tel = car.step(t0)
            proc = time.monotonic() - t0
            if prev is not None:
                fps = 1.0 / (t0 - prev)
                fps_ema = fps if fps_ema is None else 0.9 * fps_ema + 0.1 * fps
                fps_all.append(fps)
            prev = t0
            proc_all.append(proc)
            if dash is not None:
                dash.publish(car, tel, fps_ema or 0.0, proc, t0)

            if not args.quiet or t0 - last_log >= 1.0:
                last_log = t0
                e = "  --  " if tel["error"] is None else f"{tel['error']:+.3f}"
                log.info("fps=%5.1f proc=%5.1fms mode=%-6s e=%s steer=%+.2f thr=%+.2f%s%s",
                         fps_ema or 0.0, 1000 * proc, tel["mode"], e, tel["steer"], tel["throttle"],
                         " [E-STOP]" if car.estop else "", " [REC]" if car.recorder.active else "")

            if args.show or args.save_video:
                view = car.render(tel)
                if view is not None and args.save_video:
                    if writer is None:
                        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), hz,
                                                 (view.shape[1], view.shape[0]))
                    writer.write(view)
                if view is not None and args.show:
                    cv2.imshow("deepthinkcar", view)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord("q"):
                        break
                    pad = getattr(car.joystick, "primary", car.joystick)
                    if isinstance(pad, MockJoystick):
                        pad.key(key)

            if args.duration and t0 - start >= args.duration:
                break
            if args.laps:
                if world.laps >= args.laps:
                    break
                if car.stop_reason:
                    log.error("안전 정지로 주행 중단: %s", car.stop_reason)
                    break

            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()         # 주기 초과: 밀린 만큼 따라잡으려 하지 않는다
    finally:
        if writer is not None:
            writer.release()
        if fps_all:
            log.info("루프 %d회 | FPS 평균 %.1f, 최저 %.1f | 처리시간 평균 %.1fms, 최대 %.1fms",
                     len(proc_all), sum(fps_all) / len(fps_all), min(fps_all),
                     1000 * sum(proc_all) / len(proc_all), 1000 * max(proc_all))


def main():
    ap = argparse.ArgumentParser(description="차선 인식 자율주행 RC카 메인 루프")
    ap.add_argument("--mode", choices=MODES, default="manual", help="시작 모드")
    ap.add_argument("--input", choices=("web", "keyboard", "gamepad"), help="조작 장치 (기본: config 의 input.device)")
    ap.add_argument("--mock", nargs="?", const="all", default="",
                    help="하드웨어 없이 실행. 일부만: --mock camera,actuator,input")
    ap.add_argument("--video", default=str(DEFAULT_PATH.parent / "assets" / "sample.mp4"), help="mock 카메라 입력 동영상")
    ap.add_argument("--sim", action="store_true", help="시뮬레이터를 카메라/액추에이터로 사용")
    ap.add_argument("--laps", type=float, default=0, help="(--sim) 이만큼 돌면 종료")
    ap.add_argument("--trace", help="(--sim) 주행 궤적 이미지 저장 경로")
    ap.add_argument("--duration", type=float, default=0, help="지정한 초가 지나면 종료")
    ap.add_argument("--record", action="store_true", help="시작하자마자 녹화 (manual 모드)")
    ap.add_argument("--show", action="store_true", help="미리보기 창 (q: 종료)")
    ap.add_argument("--save-video", help="미리보기 화면을 동영상으로 저장")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", help="cnn 모드 TFLite 모델 경로 (기본: config 의 cnn.model)")
    ap.add_argument("--quiet", action="store_true", help="FPS 로그를 1초에 한 번만")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.laps and not args.sim:
        ap.error("--laps 는 --sim 과 함께 사용")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s.%(msecs)03d %(name)-8s %(message)s", datefmt="%H:%M:%S")
    mock = {"camera", "actuator", "input"} if args.mock == "all" else set(filter(None, args.mock.split(",")))
    mock = {"input" if m == "joystick" else m for m in mock}
    if mock - {"camera", "actuator", "input"}:
        ap.error(f"알 수 없는 --mock 항목: {args.mock}")
    cfg = load_config(args.config)

    def on_sigterm(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_sigterm)
    signal.signal(signal.SIGHUP, on_sigterm)          # SSH 세션이 닫힘

    actuator = camera = joystick = recorder = world = dash = None
    code = 0
    try:
        if args.sim:
            from sim.simulator import SimActuator, SimCamera, SimWorld
            world = SimWorld()
            camera, actuator = SimCamera(world), SimActuator(world)
        else:
            actuator = Actuator(cfg, mock="actuator" in mock)
            camera = Camera(cfg, mock="camera" in mock, source=args.video)
            camera.wait_ready()
        device = args.input or cfg["input"]["device"]
        web_drive = device == "web" and "input" not in mock
        if web_drive or cfg["web"]["enabled"]:
            try:
                dash = WebDashboard(cfg, args.config or DEFAULT_PATH, drive_enabled=web_drive)
            except OSError as e:
                if web_drive:
                    raise InputError(f"웹 대시보드를 시작할 수 없음: {e}") from e
                log.warning("웹 대시보드를 시작할 수 없음: %s", e)
        if "input" in mock:
            joystick = MockJoystick(cfg)
        elif device == "web":
            joystick = WebPad(dash)
        elif device == "keyboard":
            joystick = KeyboardPad(cfg)
        else:
            joystick = Joystick(cfg)
        if dash is not None and not isinstance(joystick, WebPad):
            joystick = WithDashboard(joystick, dash)      # 다른 장치로 조종해도 웹의 비상정지/모드/녹화 버튼은 동작
        recorder = Recorder(cfg)
        if args.sim or mock:              # 가짜 영상/입력이 학습 데이터에 섞이지 않도록 자동 녹화 끔
            recorder.auto = False
        pilot = None
        try:
            from autopilot.cnn_pilot import CnnPilot
            pilot = CnnPilot(cfg, model_path=args.model)
            log.info("AI 모델: %s", pilot.path)
        except Exception as e:
            (log.error if args.mode == "cnn" else log.info)("AI 모델 사용 불가: %s", e)
            if args.mode == "cnn":
                raise SystemExit(1)

        car = Car(cfg, camera, actuator, joystick, recorder, pilot, mode=args.mode)
        if args.record:
            recorder.start()
        if args.show and "input" in mock:
            log.info("키보드 조작 — %s  q: 종료", MockJoystick.KEYS)
        run(car, args, cfg["loop_hz"], world, dash)
    except KeyboardInterrupt:
        log.info("Ctrl+C / 종료 신호 → 종료")
    except InputError as e:
        log.error("%s", e)
        code = 1
    except Exception:
        log.exception("예외 발생 → 모터 정지 후 종료")
        code = 1
    finally:
        # 어떤 경우에도 모터부터 멈춘다
        for part in (actuator, recorder, camera, joystick, dash):
            if part is not None:
                try:
                    part.close()
                except Exception:
                    log.exception("종료 처리 실패: %s", type(part).__name__)
        if args.show:
            cv2.destroyAllWindows()
    if world is not None:
        print(world.summary())
        if args.trace:
            world.save_trace(args.trace)
        if args.laps and (world.laps < args.laps or world.departures):
            code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
