"""하드웨어 없이 전체 파이프라인을 검증하기 위한 간이 트랙 시뮬레이터.

바닥에 파란 테이프 두 줄로 만든 폐곡선 트랙을 위에서 본 지도로 그려 두고,
차량 자세에 맞춰 카메라 시점으로 원근 변환한 320x240 영상을 만든다.
차량은 자전거 모델(조향 서보 지연, 속도 지연 포함)로 움직인다.

    python sim/simulator.py video assets/sample.mp4 --seconds 40     # 샘플 주행 영상 생성
    python sim/simulator.py data  data --laps 4               # 학습용 주행 데이터 생성 (recorder 형식)
    python main.py --sim --mode opencv --laps 5     # 폐루프 자율주행 검증
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # 프로젝트 루트
import argparse
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SimParams:
    width: int = 320
    height: int = 240
    hfov_deg: float = 62.0        # Pi Camera v2 수평 화각
    cam_height: float = 0.18      # m
    cam_pitch_deg: float = 15.0   # 아래로 숙인 각도
    cam_forward: float = 0.10     # 뒤축 기준 카메라 전방 거리 (m)
    lane_width: float = 0.35      # m
    tape_width: float = 0.02      # m
    tape_bgr: tuple = (40, 35, 200)     # 빨간 테이프 (실제 트랙과 같게)
    ppm: float = 250.0            # 지도 해상도 (px/m)
    wheelbase: float = 0.20       # m
    max_steer_deg: float = 30.0
    servo_rate_deg_s: float = 400.0
    v_full: float = 3.0           # throttle=1.0 일 때 속도 (m/s)
    tau_v: float = 0.3            # 속도 응답 시정수 (s)


class Track:
    """중심선이 1cm 간격 점열인 폐곡선 트랙 (좌/우 커브가 모두 있는 삼엽형)."""

    def __init__(self, reverse=False, ds=0.01):
        t = np.linspace(0, 2 * np.pi, 8000, endpoint=False)
        r = 1.4 * (1 + 0.2 * np.cos(3 * t))
        pts = np.c_[r * np.cos(t), r * np.sin(t)]
        if reverse:
            pts = pts[::-1]
        seg = np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T)
        s = np.concatenate([[0], np.cumsum(seg)])
        self.length = float(s[-1])
        n = int(self.length / ds)
        self.ds = self.length / n
        si = np.arange(n) * self.ds
        closed = np.vstack([pts, pts[:1]])
        self.center = np.c_[np.interp(si, s, closed[:, 0]), np.interp(si, s, closed[:, 1])]
        tan = np.roll(self.center, -1, axis=0) - np.roll(self.center, 1, axis=0)
        self.tangent = tan / np.linalg.norm(tan, axis=1, keepdims=True)
        self.normal = np.c_[-self.tangent[:, 1], self.tangent[:, 0]]     # 진행 방향의 왼쪽
        self.n = n

    def nearest(self, x, y):
        """(가장 가까운 중심선 인덱스, 횡방향 오프셋[m] — 양수면 중심선 왼쪽)."""
        d = self.center - (x, y)
        i = int(np.argmin(np.einsum("ij,ij->i", d, d)))
        return i, float(-d[i] @ self.normal[i])


class SimWorld:
    def __init__(self, params=None, reverse=False, seed=0):
        self.p = p = params or SimParams()
        self.track = Track(reverse)
        self._build_map(seed)
        self._build_camera()
        c, t = self.track.center[0], self.track.tangent[0]
        self.x, self.y, self.yaw = float(c[0]), float(c[1]), math.atan2(t[1], t[0])
        self.v = 0.0
        self.delta = 0.0            # 실제 조향각 (rad, 우회전 +)
        self.cmd_steer = 0.0
        self.cmd_throttle = 0.0
        # 통계
        self.time = 0.0
        self.distance = 0.0         # 중심선을 따라 진행한 거리
        self.max_offset = 0.0
        self.departures = 0
        self.off_track = False
        self._last_idx = 0
        self.trace = []

    # ---- 지도 / 카메라 ----
    def _build_map(self, seed):
        p, tr = self.p, self.track
        margin = 1.5
        self.x0 = tr.center[:, 0].min() - margin
        self.y1 = tr.center[:, 1].max() + margin
        w = int((np.ptp(tr.center[:, 0]) + 2 * margin) * p.ppm)
        h = int((np.ptp(tr.center[:, 1]) + 2 * margin) * p.ppm)
        rng = np.random.default_rng(seed)
        # 바닥: 회갈색 + 저주파 얼룩 + 미세 노이즈
        blot = cv2.resize(rng.normal(0, 1, (h // 40, w // 40)).astype(np.float32), (w, h),
                          interpolation=cv2.INTER_CUBIC)
        fine = rng.normal(0, 1, (h, w)).astype(np.float32)
        lum = 150 + 14 * blot + 5 * fine
        self.floor_bgr = (140, 150, 158)
        m = np.stack([lum * 0.93, lum, lum * 1.05], axis=2)
        self.map = np.clip(m, 0, 255).astype(np.uint8)
        thick = max(1, round(p.tape_width * p.ppm))
        for side in (+1, -1):
            line = tr.center + side * tr.normal * p.lane_width / 2
            px = np.c_[(line[:, 0] - self.x0) * p.ppm, (self.y1 - line[:, 1]) * p.ppm]
            cv2.polylines(self.map, [np.round(px * 16).astype(np.int32)], True, p.tape_bgr,
                          thick, cv2.LINE_AA, shift=4)
        # 지도 픽셀 → 월드 좌표(m)
        self._M = np.array([[1 / p.ppm, 0, self.x0], [0, -1 / p.ppm, self.y1], [0, 0, 1]])

    def _build_camera(self):
        p = self.p
        f = (p.width / 2) / math.tan(math.radians(p.hfov_deg / 2))
        K = np.array([[f, 0, p.width / 2], [0, f, p.height / 2], [0, 0, 1]])
        a = math.radians(p.cam_pitch_deg)
        sa, ca = math.sin(a), math.cos(a)
        d, h = p.cam_forward, p.cam_height
        # 차량 좌표(전방, 좌측, 1)의 지면 점 → 카메라 좌표(우, 하, 전방)
        C = np.array([[0, -1, 0], [-sa, 0, sa * d + ca * h], [ca, 0, -ca * d + sa * h]])
        self._KC = K @ C
        self._horizon = int(p.height / 2 - f * math.tan(a)) + 3

    def render(self):
        """현재 자세에서 본 카메라 영상 (BGR)."""
        p = self.p
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        T = np.array([[c, s, -c * self.x - s * self.y], [-s, c, s * self.x - c * self.y], [0, 0, 1]])
        H = self._KC @ T @ self._M
        img = cv2.warpPerspective(self.map, H, (p.width, p.height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=self.floor_bgr)
        if self._horizon > 0:
            img[:self._horizon] = (205, 210, 212)       # 지평선 위: 벽
        return img

    # ---- 차량 동역학 ----
    def command(self, steer, throttle):
        self.cmd_steer = float(np.clip(steer, -1, 1))
        self.cmd_throttle = float(np.clip(throttle, -1, 1))

    def advance(self, dt):
        p = self.p
        while dt > 1e-9:
            h = min(dt, 0.005)
            dt -= h
            target = self.cmd_steer * math.radians(p.max_steer_deg)
            rate = math.radians(p.servo_rate_deg_s) * h
            self.delta += float(np.clip(target - self.delta, -rate, rate))
            self.v += (self.cmd_throttle * p.v_full - self.v) * min(1.0, h / p.tau_v)
            self.x += self.v * math.cos(self.yaw) * h
            self.y += self.v * math.sin(self.yaw) * h
            self.yaw -= self.v / p.wheelbase * math.tan(self.delta) * h    # 우회전(+) = 시계 방향
            self.time += h
        self._update_stats()

    def _update_stats(self):
        tr = self.track
        i, off = tr.nearest(self.x, self.y)
        di = (i - self._last_idx + tr.n // 2) % tr.n - tr.n // 2
        self.distance += di * tr.ds
        self._last_idx = i
        self.offset = off
        self.max_offset = max(self.max_offset, abs(off))
        off_now = abs(off) > self.p.lane_width / 2      # 차량 중심이 테이프를 넘음
        if off_now and not self.off_track:
            self.departures += 1
        self.off_track = off_now
        self.trace.append((self.x, self.y))

    @property
    def laps(self):
        return self.distance / self.track.length

    def summary(self):
        return (f"[sim] {self.time:.1f}s 주행, {self.laps:.2f}바퀴, 이탈 {self.departures}회, "
                f"최대 횡오차 {100 * self.max_offset:.1f}cm (차선 반폭 {100 * self.p.lane_width / 2:.1f}cm)")

    def save_trace(self, path):
        """주행 궤적을 지도 위에 그려 저장."""
        img = self.map.copy()
        if self.trace:
            t = np.array(self.trace)
            px = np.c_[(t[:, 0] - self.x0) * self.p.ppm, (self.y1 - t[:, 1]) * self.p.ppm]
            cv2.polylines(img, [px.astype(np.int32)], False, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(path), img)

    # ---- 기준 운전자 (샘플 영상/학습 데이터 생성용 pure pursuit) ----
    def oracle(self, lookahead=0.45, wobble=0.06):
        tr, p = self.track, self.p
        i, _ = tr.nearest(self.x, self.y)
        j = (i + int(lookahead / tr.ds)) % tr.n
        # 중심선에서 살짝 좌우로 흔들리게 해 다양한 위치/복귀 장면을 만든다
        offset = wobble * math.sin(2 * math.pi * self.distance / 2.3)
        tx, ty = tr.center[j] + tr.normal[j] * offset - (self.x, self.y)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        xf, yl = c * tx + s * ty, -s * tx + c * ty
        delta_left = math.atan(p.wheelbase * 2 * yl / (xf * xf + yl * yl))
        steer = float(np.clip(-delta_left / math.radians(p.max_steer_deg), -1, 1))
        return steer, 0.25 - 0.08 * abs(steer)


class SimCamera:
    """camera.Camera 와 같은 인터페이스. read() 때마다 실제 경과 시간만큼 시뮬레이션을 진행."""

    def __init__(self, world):
        self.world = world
        self._last = None

    def read(self):
        now = time.monotonic()
        if self._last is not None:
            self.world.advance(min(now - self._last, 0.2))
        self._last = now
        return self.world.render(), now

    def close(self):
        pass


class SimActuator:
    """actuator.Actuator 와 같은 인터페이스."""

    def __init__(self, world):
        self.world = world

    def set(self, steer, throttle):
        self.world.command(steer, throttle)

    def stop(self):
        self.world.command(0.0, 0.0)

    def close(self):
        self.stop()


def _drive(world, hz, stop_fn, on_frame):
    """기준 운전자로 고정 시간 간격 주행 (실시간 아님)."""
    dt = 1.0 / hz
    while not stop_fn():
        steer, throttle = world.oracle()
        world.command(steer, throttle)
        on_frame(world.render(), steer, throttle)
        world.advance(dt)


def main():
    ap = argparse.ArgumentParser(description="트랙 시뮬레이터: 샘플 영상 / 학습 데이터 생성")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("video", help="기준 운전자의 카메라 영상을 동영상으로 저장")
    v.add_argument("out")
    v.add_argument("--seconds", type=float, default=40)
    d = sub.add_parser("data", help="recorder 형식(images/ + log.csv)의 학습 데이터 생성")
    d.add_argument("dir", help="데이터 루트 폴더 (예: data)")
    d.add_argument("--laps", type=float, default=4, help="방향별 주행 바퀴 수")
    for s in (v, d):
        s.add_argument("--hz", type=int, default=20)
        s.add_argument("--trace", help="주행 궤적 이미지 저장 경로")
    args = ap.parse_args()

    if args.cmd == "video":
        world = SimWorld()
        wr = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.hz,
                             (world.p.width, world.p.height))
        if not wr.isOpened():
            raise SystemExit(f"동영상을 쓸 수 없음: {args.out}")
        _drive(world, args.hz, lambda: world.time >= args.seconds, lambda f, s, t: wr.write(f))
        wr.release()
        print(world.summary(), "→", args.out)
        if args.trace:
            world.save_trace(args.trace)
    else:
        from training.recorder import Recorder
        for reverse in (False, True):                 # 양방향으로 주행해 좌/우 커브 균형
            world = SimWorld(reverse=reverse)
            rec = Recorder({"recorder": {"dir": args.dir, "hz": args.hz, "jpeg_quality": 90}})
            path = rec.start(suffix="_sim_cw" if reverse else "_sim_ccw")
            _drive(world, args.hz, lambda: world.laps >= args.laps,
                   lambda f, s, t: rec.add(f, s, t, world.time))
            rec.stop()
            print(world.summary(), "→", path)
            if args.trace:
                world.save_trace(args.trace)


if __name__ == "__main__":
    main()
