"""웹 대시보드: 카메라·차선 오버레이 영상, 조종(가상 조이스틱 / WASD), 비상정지, 모드·녹화, 실시간 튜닝.

    python main.py                         # input.device: web → 로그에 찍힌 주소로 접속
    python main.py --sim --input web       # 시뮬레이터로 연습

- 표준 라이브러리 http.server 만 사용 (추가 설치 없음)
- 영상: MJPEG (/stream.mjpg). 인코딩은 서버 스레드에서 하므로 제어 루프를 막지 않는다
- 조종: 브라우저가 100ms 마다 POST /api/control → 이것이 heartbeat.
  web.reply_timeout_s 동안 없으면 수동 입력 0, input.timeout_s 동안 없으면 main 이 안전 정지
- 조종은 한 화면만 (조종권). 비상정지는 접속한 어느 화면에서나 가능
- web.token 을 정하면 토큰이 있어야 접속 가능 (비우면 같은 Wi-Fi 의 누구나 접속·조종 가능)
"""
import copy
import json
import logging
import math
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from config import update_yaml
from inputs.joystick import JoyState
from autopilot.lane_detect import LaneDetector

log = logging.getLogger("web")
PAGE = Path(__file__).with_name("index.html")
VIEWS = ("overlay", "mask", "edges", "raw")

# (키, 이름, 최소, 최대, 간격, 그룹)
TUNABLES = [
    ("lane.hsv_lower.0", "H 최소", 0, 179, 1, "차선 색 (HSV)"),
    ("lane.hsv_upper.0", "H 최대", 0, 179, 1, "차선 색 (HSV)"),
    ("lane.hsv_lower.1", "S 최소", 0, 255, 1, "차선 색 (HSV)"),
    ("lane.hsv_upper.1", "S 최대", 0, 255, 1, "차선 색 (HSV)"),
    ("lane.hsv_lower.2", "V 최소", 0, 255, 1, "차선 색 (HSV)"),
    ("lane.hsv_upper.2", "V 최대", 0, 255, 1, "차선 색 (HSV)"),
    ("lane.blur_ksize", "블러 크기", 1, 15, 2, "에지 · 직선"),
    ("lane.canny_low", "Canny 하한", 0, 255, 1, "에지 · 직선"),
    ("lane.canny_high", "Canny 상한", 0, 255, 1, "에지 · 직선"),
    ("lane.hough_threshold", "Hough 임계값", 1, 100, 1, "에지 · 직선"),
    ("lane.hough_min_line_length", "최소 선 길이", 1, 100, 1, "에지 · 직선"),
    ("lane.hough_max_line_gap", "최대 선 간격", 1, 50, 1, "에지 · 직선"),
    ("lane.min_angle_deg", "최소 기울기 °", 0, 60, 1, "에지 · 직선"),
    ("lane.roi_top", "ROI 상단 (화면 비율)", 0.0, 0.95, 0.01, "영역"),
    ("lane.lookahead", "오차 측정 행 (화면 비율)", 0.3, 1.0, 0.01, "영역"),
    ("control.kp", "Kp", 0.0, 150.0, 0.5, "PID"),
    ("control.ki", "Ki", 0.0, 50.0, 0.1, "PID"),
    ("control.kd", "Kd", 0.0, 30.0, 0.1, "PID"),
    ("control.max_steer_rate_deg_s", "조향 변화율 제한 °/s", 30, 1000, 10, "PID"),
    ("control.curve_slowdown", "커브 감속 강도", 0.0, 4.0, 0.1, "속도"),
    ("control.throttle_max", "직선 속도", 0.0, 0.6, 0.01, "속도"),
    ("control.throttle_min", "커브 속도", 0.0, 0.6, 0.01, "속도"),
    ("control.sharp_brake_frames", "급커브 진입 멈칫 (장)", 0, 15, 1, "속도"),
    ("control.sharp_on_frames", "급커브 전진 (장)", 1, 6, 1, "속도"),
    ("control.sharp_off_frames", "급커브 쉼 (장, 클수록 느림)", 0, 6, 1, "속도"),
    ("control.throttle_accel", "커브 뒤 가속 (작을수록 천천히)", 0.05, 2.0, 0.05, "속도"),
    ("manual.throttle_scale", "수동 스로틀 배율", 0.0, 1.0, 0.01, "속도"),
    ("cnn.throttle_max", "AI 최대 스로틀", 0.0, 0.6, 0.01, "속도"),
]


def _get(cfg, key):
    if key == "lane.roi_top":
        return cfg["lane"]["roi"][1][1]
    s, k, *i = key.split(".")
    v = cfg[s][k]
    return v[int(i[0])] if i else v


def _set(cfg, key, value):
    """값을 넣고, config.yaml 에 저장할 (섹션, 키) 를 돌려준다."""
    if key == "lane.roi_top":                      # 기본 사각형 ROI 의 위쪽 두 꼭짓점
        roi = cfg["lane"]["roi"]
        roi[1][1] = roi[2][1] = value
        return "lane", "roi"
    s, k, *i = key.split(".")
    if i:
        cfg[s][k][int(i[0])] = value
    else:
        cfg[s][k] = value
    return s, k


def _local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))        # UDP connect: 패킷은 보내지 않고 경로만 정한다
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


class WebDashboard:
    def __init__(self, cfg, config_path, drive_enabled):
        w = cfg["web"]
        self.cfg = copy.deepcopy(cfg)              # 튜닝 값의 원본 (HTTP 스레드에서 수정)
        self.config_path = Path(config_path)
        self.drive_enabled = drive_enabled         # False 면 보기/비상정지/모드/튜닝만
        self.token = str(w.get("token") or "")          # 비어 있으면 토큰 없이 접속
        self.stream_period = 1.0 / w["stream_fps"]
        self.jpeg_quality = int(w["jpeg_quality"])
        self.reply_timeout = w["reply_timeout_s"]
        roi = cfg["lane"]["roi"]
        rect_roi = len(roi) == 4 and roi[1][1] == roi[2][1]
        self.specs = {t[0]: t for t in TUNABLES if t[0] != "lane.roi_top" or rect_roi}

        self._lock = threading.Lock()
        self._owner, self._owner_t = None, float("-inf")
        self._axes = (0.0, 0.0)
        self._latched = True                       # 스로틀이 한 번 0 으로 돌아와야 움직인다
        self._events = self._no_events()
        self._cfg_dirty = False
        self._status = {}
        self._frame_cv = threading.Condition()
        self._snap, self._seq = None, 0
        self._preview = LaneDetector(cfg)          # 영상용 (제어 루프의 검출기와 분리)
        self._preview_lock = threading.Lock()
        self._closed = False

        handler = type("Handler", (_Handler,), {"dash": self})
        self.server = ThreadingHTTPServer((w["host"], w["port"]), handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        port = self.server.server_address[1]
        query = f"/?token={self.token}" if self.token else ""
        self.url = f"http://{_local_ip()}:{port}{query}"
        log.info("웹 대시보드: %s   (또는 http://%s.local:%d%s)%s", self.url,
                 socket.gethostname(), port, query, "" if drive_enabled else "  [보기 전용: 조종은 다른 장치]")

    @staticmethod
    def _no_events():
        return {"estop": False, "record": False, "set_mode": None}

    # ================= 메인 루프 스레드에서 호출 =================
    def axes(self, now):
        with self._lock:
            silent = now - self._owner_t
            if not self.drive_enabled or silent > self.reply_timeout:
                return JoyState(lost_for=silent)
            steer, throttle = self._axes
            return JoyState(True, steer, 0.0 if self._latched else throttle)

    def pop_events(self):
        with self._lock:
            ev, self._events = self._events, self._no_events()
            return ev

    def reset_axes(self):
        """모드 전환/안전 정지 후에는 화면의 스로틀이 한 번 중립으로 돌아와야 다시 움직인다."""
        with self._lock:
            self._latched = True

    def apply_tuning(self, car):
        with self._lock:
            if not self._cfg_dirty:
                return
            self._cfg_dirty = False
            cfg = copy.deepcopy(self.cfg)
        car.configure(cfg)

    def publish(self, car, tel, fps, proc_s, now):
        err = tel["error"]
        status = {
            "mode": tel["mode"], "estop": car.estop, "fps": round(fps, 1), "proc_ms": round(1000 * proc_s, 1),
            "error": None if err is None else round(err, 3),
            "steer": round(tel["steer"], 3), "throttle": round(tel["throttle"], 3),
            "rec": car.recorder.active, "rec_count": getattr(car.recorder, "count", 0),
            "stop_reason": car.stop_reason,
            "stop_age": None if car.stop_time is None else round(now - car.stop_time, 1),
            "difficulty": round(tel.get("difficulty", 0.0), 2), "difficulty_reason": tel.get("reason", ""),
            "cnn": car.pilot is not None,
        }
        with self._frame_cv:
            self._status = status
            if tel["frame"] is not None:
                self._snap = (tel["frame"], tel["lane"] if tel["mode"] == "opencv" else None)
                self._seq += 1
                self._frame_cv.notify_all()

    def close(self):
        self._closed = True
        with self._frame_cv:
            self._frame_cv.notify_all()
        self.server.shutdown()
        self.server.server_close()

    # ================= HTTP 스레드에서 호출 =================
    def control(self, data):
        sid = str(data["sid"])[:64]
        steer, throttle = float(data.get("steer", 0)), float(data.get("throttle", 0))
        if not (math.isfinite(steer) and math.isfinite(throttle)):
            steer = throttle = 0.0
        steer, throttle = max(-1.0, min(1.0, steer)), max(-1.0, min(1.0, throttle))
        now = time.monotonic()
        with self._lock:
            active = now - self._owner_t <= self.reply_timeout
            if self._owner == sid or not active:       # 조종 중인 화면이 없으면 이 화면이 조종권을 갖는다
                if self._owner != sid:
                    self._take(sid)
                elif now - self._owner_t > 0.3:
                    log.warning("조종 신호 지연 %.2fs (Wi-Fi 상태 확인)", now - self._owner_t)
                self._owner_t = now
                self._axes = (steer, throttle)
                if self._latched and abs(throttle) < 0.05:
                    self._latched = False
            is_owner = self._owner == sid
            latched = self._latched
        with self._frame_cv:
            status = dict(self._status)
        status.update(owner=is_owner, latched=latched and is_owner, drive_enabled=self.drive_enabled)
        return status

    def _take(self, sid):
        log.info("조종권: 화면 %s", sid[:6])
        self._owner = sid
        self._owner_t = time.monotonic()
        self._latched = True

    def action(self, data):
        name, sid = data.get("action"), str(data.get("sid", ""))[:64]
        with self._lock:
            if name == "estop":
                self._events["estop"] = True
            elif name == "record":
                self._events["record"] = True
            elif name == "mode":
                if data.get("value") not in ("manual", "opencv", "cnn"):
                    raise ValueError("mode")
                self._events["set_mode"] = data["value"]
            elif name == "take":
                self._take(sid)
            else:
                raise ValueError(f"알 수 없는 동작: {name}")
        return {"ok": True}

    def config_info(self):
        with self._lock:
            return {"tunables": [{"key": k, "label": lab, "min": lo, "max": hi, "step": st, "group": g,
                                  "value": _get(self.cfg, k)} for k, lab, lo, hi, st, g in self.specs.values()],
                    "drive_enabled": self.drive_enabled}

    def tune(self, data):
        key = data["key"]
        _, _, lo, hi, step, _ = self.specs[key]
        v = float(data["value"])
        if not math.isfinite(v):
            raise ValueError("value")
        v = max(lo, min(hi, v))
        if isinstance(step, int):
            v = int(round(v))
            if key == "lane.blur_ksize":
                v |= 1                                     # 홀수만
        else:
            v = round(v, 4)
        with self._lock:
            _set(self.cfg, key, v)
            self._cfg_dirty = True
            cfg = copy.deepcopy(self.cfg)
        with self._preview_lock:
            self._preview.configure(cfg)
        return {"key": key, "value": v}

    def save(self):
        with self._lock:
            values = {}
            for key in self.specs:
                s, k = _set(copy.deepcopy(self.cfg), key, 0)  # 저장 위치만 알아낸다
                values[(s, k)] = copy.deepcopy(self.cfg[s][k])
        update_yaml(self.config_path, values)
        log.info("튜닝 값 저장: %s (%d개 항목)", self.config_path, len(values))
        return {"saved": [f"{s}.{k}" for s, k in values], "path": str(self.config_path)}

    def stream(self, handler, view):
        if view not in VIEWS:
            view = "overlay"
        handler.close_connection = True
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Connection", "close")
        handler.end_headers()
        seq = -1
        next_t = time.monotonic()
        while not self._closed:
            with self._frame_cv:
                self._frame_cv.wait_for(lambda: self._seq != seq or self._closed, timeout=1.0)
                snap, seq = self._snap, self._seq
            if snap is None or self._closed:
                continue
            ok, jpg = cv2.imencode(".jpg", self._render(snap, view), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                continue
            try:
                handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                handler.wfile.write(jpg.tobytes() + b"\r\n")
                handler.wfile.flush()
            except OSError:                                # 브라우저가 닫았거나 화면을 바꿈
                return
            next_t += self.stream_period
            time.sleep(max(0.0, next_t - time.monotonic()))
            next_t = max(next_t, time.monotonic() - self.stream_period)

    def _render(self, snap, view):
        frame, lane = snap
        if view == "raw":
            return frame
        with self._preview_lock:
            if view == "overlay" and lane is not None:     # opencv 모드: 실제 조향에 쓰인 결과
                return self._preview.draw(frame, lane)
            res = self._preview.process(frame)             # 그 외: opencv 모드라면 이렇게 보인다
            if view == "mask":
                return cv2.cvtColor(self._preview.mask, cv2.COLOR_GRAY2BGR)
            if view == "edges":
                return cv2.cvtColor(self._preview.edges, cv2.COLOR_GRAY2BGR)
            return self._preview.draw(frame, res)


class _Handler(BaseHTTPRequestHandler):
    dash = None
    protocol_version = "HTTP/1.1"
    server_version = "deepthinkcar"

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self, query):
        origin = self.headers.get("Origin")        # 다른 웹사이트가 몰래 보내는 조종 요청 차단
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            self._send(403, {"error": "다른 사이트의 요청"})
            return False
        if not self.dash.token:
            return True
        given = query.get("token", [""])[0].encode()
        if secrets.compare_digest(given, self.dash.token.encode()):
            return True
        self._send(403, {"error": "토큰이 맞지 않습니다"})
        return False

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if url.path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if not self._authorized(q):
            return
        if url.path == "/api/config":
            return self._send(200, self.dash.config_info())
        if url.path == "/stream.mjpg":
            return self.dash.stream(self, q.get("view", ["overlay"])[0])
        self._send(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        if n > 65536:
            self.close_connection = True
            return self._send(413, {"error": "too large"})
        raw = self.rfile.read(n)
        if not self._authorized(parse_qs(url.query)):
            return
        routes = {"/api/control": self.dash.control, "/api/action": self.dash.action,
                  "/api/tune": self.dash.tune, "/api/save": lambda _: self.dash.save()}
        fn = routes.get(url.path)
        if fn is None:
            return self._send(404, {"error": "not found"})
        try:
            data = json.loads(raw or b"{}")
            if not isinstance(data, dict):
                raise ValueError("JSON object 필요")
            self._send(200, fn(data))
        except (KeyError, ValueError, TypeError) as e:
            self._send(400, {"error": f"잘못된 요청: {e}"})
        except OSError as e:
            self._send(500, {"error": f"저장 실패: {e}"})


class WebPad:
    """브라우저로 조종할 때의 입력 장치 (joystick.Joystick 과 같은 인터페이스)."""

    def __init__(self, dash):
        self.dash = dash

    def poll(self, now=None):
        st = self.dash.axes(time.monotonic() if now is None else now)
        ev = self.dash.pop_events()
        st.estop, st.record, st.set_mode = ev["estop"], ev["record"], ev["set_mode"]
        return st

    def reset_axes(self):
        self.dash.reset_axes()

    def close(self):
        pass


class WithDashboard:
    """다른 장치(키보드/게임패드)로 조종할 때도 웹의 비상정지·모드·녹화 버튼을 함께 받는다."""

    def __init__(self, primary, dash):
        self.primary, self.dash = primary, dash

    def poll(self, now=None):
        st = self.primary.poll(now)
        ev = self.dash.pop_events()
        st.estop = st.estop or ev["estop"]
        st.record = st.record or ev["record"]
        st.set_mode = ev["set_mode"] or st.set_mode
        return st

    def reset_axes(self):
        reset = getattr(self.primary, "reset_axes", None)
        if reset:
            reset()

    def close(self):
        self.primary.close()
