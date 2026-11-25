"""차선 인식: HSV 필터 → 블러 → Canny → ROI → HoughLinesP → 좌/우 차선 → 오차 e(-1~1).

단독 실행 (하드웨어 불필요):
    python autopilot/lane_detect.py assets/sample.mp4 --out overlay.mp4 --debug
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # 프로젝트 루트
import argparse
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from autopilot.sharp_corner import SharpCorner, sweeps_across
from config import load_config


@dataclass
class LaneResult:
    error: float | None = None      # e>0: 차선 중심이 화면 오른쪽. 미검출이면 None
    left: tuple | None = None       # (a, b) : x = a*y + b
    right: tuple | None = None
    center_x: float | None = None   # lookahead 행에서의 차선 중심 x(px)
    left_segments: list = field(default_factory=list)
    right_segments: list = field(default_factory=list)

    sharp: int = 0                  # 급커브 단계: 0 없음, 1 접근(감속), 2 회전
    sharp_side: int = 0             # -1 좌, +1 우
    held: bool = False              # 급커브 보조: 가로 테이프만 보여 직전 방향을 유지한 결과

    @property
    def detected(self):
        return self.error is not None


class LaneDetector:
    def __init__(self, cfg):
        self.lane_width = float(cfg["lane"]["lane_width_px"])   # 주행 중 자동 보정되는 상태값
        self._track = None   # 직전 (왼쪽 x, 오른쪽 x): 한쪽만 보일 때 좌우 구분용
        self._missed = 0
        self._corner = SharpCorner()
        self._sweep_side, self._sweep_age = 0, 99
        self._recent = []    # 최근 확실한 오차 (급커브 보조용)
        self._held = 0
        self._side, self._side_age = 0, 99    # 마지막으로 크게 꺾던 방향과 지난 프레임 수
        self._cross = []     # 최근 프레임의 가로 테이프 방향 (-1 좌, +1 우, 0 없음)
        self.mask = None     # 디버그용: 마지막 프레임의 HSV 마스크 / 에지
        self.edges = None
        self.configure(cfg)

    def configure(self, cfg):
        """파라미터만 다시 읽는다 (실시간 튜닝용, 학습된 차선 폭은 유지)."""
        c = cfg["lane"]
        self.hsv_lower = np.array(c["hsv_lower"], np.uint8)
        self.hsv_upper = np.array(c["hsv_upper"], np.uint8)
        self.blur_ksize = int(c["blur_ksize"]) | 1          # 홀수 보장
        self.canny_low = c["canny_low"]
        self.canny_high = c["canny_high"]
        self.roi = np.array(c["roi"], np.float32)
        self.hough_threshold = int(c["hough_threshold"])
        self.hough_min_len = c["hough_min_line_length"]
        self.hough_max_gap = c["hough_max_line_gap"]
        self.min_tan = np.tan(np.radians(c["min_angle_deg"]))
        self.min_total_length = c["min_total_length"]
        self.lookahead = c["lookahead"]
        self.sharp_turn = bool(c.get("sharp_turn", False))         # 가로 테이프로 급커브 미리 판단
        self.hold_frames = int(c.get("sharp_hold_frames", 0))      # 0 이면 급커브 보조 끔
        self._roi_mask = None

    # ---- 파이프라인 단계 ----
    def _color_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo, hi = self.hsv_lower, self.hsv_upper
        if lo[0] <= hi[0]:
            return cv2.inRange(hsv, lo, hi)
        # H 범위가 0을 넘나드는 색(빨강 등): 두 구간의 합
        m1 = cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]], np.uint8))
        m2 = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8), hi)
        return cv2.bitwise_or(m1, m2)

    def _roi_polygon(self, w, h):
        return (self.roi * [w - 1, h - 1]).astype(np.int32)

    def _get_roi_mask(self, w, h):
        if self._roi_mask is None or self._roi_mask.shape != (h, w):
            self._roi_mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(self._roi_mask, [self._roi_polygon(w, h)], 255)
        return self._roi_mask

    def _fit(self, segments):
        """선분들을 길이 가중 최소제곱으로 x = a*y + b 에 맞춘다."""
        if not segments:
            return None
        s = np.asarray(segments, np.float64)
        length = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
        if length.sum() < self.min_total_length:
            return None
        ys = np.concatenate([s[:, 1], s[:, 3]])
        xs = np.concatenate([s[:, 0], s[:, 2]])
        wt = np.concatenate([length, length])
        a, b = np.polyfit(ys, xs, 1, w=wt)
        return float(a), float(b)

    # ---- 메인 ----
    def process(self, frame):
        res = self._process(frame)
        if not self.sharp_turn:
            return res
        # 한쪽 차선만 보이며 꺾던 중, 그 테이프가 앞을 가로지르면 오차가 줄어든 것처럼 보여도 풀지 않는다
        self._sweep_age += 1
        single = res.error is not None and not res.held and bool(res.left) != bool(res.right)
        if single:
            side = -1 if res.right else 1
            if res.error * side >= 0.6:
                self._sweep_side, self._sweep_age = side, 0
            if self._sweep_side == side and self._sweep_age <= 15 and 0 <= res.error * side < 0.9 and sweeps_across(self.mask, side):
                res.error, res.held = 0.9 * side, True
                res.center_x = frame.shape[1] / 2 * (1 + res.error)
        res.sharp = self._corner.update(self.mask, res.left, res.right, res.error)
        if res.sharp:
            res.sharp_side = side = self._corner.side
            w = frame.shape[1]
            if res.sharp == 1 and res.detected and res.error * side < 0:
                res.error, res.held = 0.0, True               # 모서리 반대쪽으로 꺾지 않는다
            elif res.sharp == 2 and cv2.countNonZero(cv2.bitwise_and(self.mask, self._get_roi_mask(w, frame.shape[0]))) > 150:
                res.error, res.held = 0.95 * side, True       # 테이프가 보이는 동안만 회전 유지
            if res.held and res.error is not None:
                res.center_x = w / 2 + res.error * (w / 2 - 1)
        return res

    def _process(self, frame):
        """기존 인식 결과를 그대로 쓰고, 실패했을 때만 급커브 보조를 시도한다."""
        res = self._detect(frame)
        turn = self._crossing_turn() if self.sharp_turn else 0
        self._side_age += 1
        if turn and ((res.left and res.right) or (self._side_age <= 30 and turn != self._side)):
            turn = 0       # 양쪽 차선이 보이면 기존 결과를 따르고, 최근 1.5초 안에 꺾던 방향과 반대면 믿지 않는다
        self._cross = (self._cross + [turn])[-3:]
        if turn and self._cross == [turn] * 3 and not (res.detected and res.error * turn >= 0.9):
            self._side, self._side_age = turn, 0
            res.error, res.held = 0.9 * turn, True        # 3장 연속 같은 방향일 때만 개입
            w = frame.shape[1]
            res.center_x = w / 2 + res.error * w / 2
            return res
        if res.detected:
            if abs(res.error) >= 0.5:
                self._side, self._side_age = (1 if res.error > 0 else -1), 0
            self._recent = (self._recent + [res.error])[-5:]
            self._held = 0
        elif self._held < self.hold_frames and self._sharp_ahead():
            self._held += 1
            res.held = True
            res.error = 1.0 if self._recent[-1] > 0 else -1.0
            w = frame.shape[1]
            res.center_x = w / 2 + res.error * (w / 2 - 1)
        return res

    def _crossing_turn(self):
        """앞을 가로막은 누운 테이프(급커브의 바깥 차선)가 멀어지는 방향: -1 좌, +1 우, 0 없음."""
        h, w = self.mask.shape
        roi = cv2.bitwise_and(self.mask, self._get_roi_mask(w, h))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(roi)
        for i in sorted(range(1, n), key=lambda k: -stats[k, cv2.CC_STAT_AREA])[:2]:
            x, _, cw, _, _ = stats[i]
            if cw < 0.55 * w:
                continue
            ys, xs = np.nonzero(labels == i)
            a, b = np.polyfit(xs, ys, 1)
            if 0.08 < abs(a) < 0.36 and a * w / 2 + b > 0.5 * h:     # 20° 이하로 누웠고 화면 중앙 아래까지 다가옴
                turn = -1 if a > 0 else 1                # 왼쪽으로 갈수록 위(멀리) → 좌회전
                far_end_inside = x > 0.04 * w if turn < 0 else x + cw < 0.96 * w
                if far_end_inside:                       # 완만한 커브의 테이프는 화면 밖까지 이어진다
                    return turn
        return 0

    def _sharp_ahead(self):
        """직전까지 한쪽으로 크게 꺾던 중이고(최근 5장 모두 같은 방향), 테이프가 화면을 가로지르고 있는가."""
        e = self._recent
        if len(e) < 5 or min(abs(v) for v in e) < 0.5 or abs(sum(1 if v > 0 else -1 for v in e)) < 5:
            return False
        roi = cv2.bitwise_and(self.mask, self._get_roi_mask(*self.mask.shape[::-1]))
        return float((roi.max(axis=0) > 0).mean()) > 0.4

    def _detect(self, frame):
        h, w = frame.shape[:2]
        self.mask = self._color_mask(frame)
        blur = cv2.GaussianBlur(self.mask, (self.blur_ksize, self.blur_ksize), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        self.edges = cv2.bitwise_and(edges, self._get_roi_mask(w, h))
        lines = cv2.HoughLinesP(self.edges, 1, np.pi / 180, self.hough_threshold,
                                minLineLength=self.hough_min_len,
                                maxLineGap=self.hough_max_gap)
        res = LaneResult()
        if lines is None:
            self._forget()
            return res

        # 기울기 부호로 좌/우 분류. 왼쪽 차선은 위로 갈수록 오른쪽(dx/dy<0), 오른쪽 차선은 반대.
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = float(x2 - x1), float(y2 - y1)
            if abs(dy) < self.min_tan * abs(dx):
                continue                      # 수평에 가까운 선분 제거
            (res.left_segments if dx / dy < 0 else res.right_segments).append((x1, y1, x2, y2))

        res.left = self._fit(res.left_segments)
        res.right = self._fit(res.right_segments)

        y = self.lookahead * (h - 1)
        xl = res.left[0] * y + res.left[1] if res.left else None
        xr = res.right[0] * y + res.right[1] if res.right else None

        if xl is not None and xr is not None and xr - xl < 0.15 * w:
            # 좌우가 뒤집혔거나 붙어 있으면 오분류 → 근거(선분 길이 합)가 많은 쪽만 신뢰
            if _total_length(res.left_segments) >= _total_length(res.right_segments):
                xr, res.right = None, None
            else:
                xl, res.left = None, None

        if self._track and (xl is None) != (xr is None):
            # 한쪽만 보임: 기울기 대신 직전 위치에서 더 가까운 차선으로 본다 (급커브에서 넘어온 왼쪽 테이프 등)
            x = xl if xl is not None else xr
            is_left = abs(x - self._track[0]) <= abs(x - self._track[1])
            if is_left != (xl is not None):
                res.left, res.right = res.right, res.left
                res.left_segments, res.right_segments = res.right_segments, res.left_segments
                xl, xr = xr, xl

        if xl is not None and xr is not None:
            center = (xl + xr) / 2
            self.lane_width += 0.1 * ((xr - xl) - self.lane_width)   # 차선 폭 학습
        elif xl is not None:
            center = xl + self.lane_width / 2
        elif xr is not None:
            center = xr - self.lane_width / 2
        else:
            self._forget()
            return res

        self._missed = 0
        self._track = (xl if xl is not None else xr - self.lane_width,
                       xr if xr is not None else xl + self.lane_width)
        res.center_x = float(center)
        res.error = float(np.clip((center - w / 2) / (w / 2), -1.0, 1.0))
        return res

    def reset(self):
        """직전 차선 위치를 잊는다 (모드 전환, 차를 옮긴 뒤)."""
        self._track, self._missed = None, 0
        self._corner.reset()
        self._sweep_side, self._sweep_age = 0, 99
        self._recent, self._held, self._cross = [], 0, []
        self._side, self._side_age = 0, 99

    def _forget(self):
        self._missed += 1
        if self._missed >= 10:
            self._track, self._missed = None, 0

    # ---- 디버그 오버레이 ----
    def draw(self, frame, res):
        out = frame.copy()
        h, w = out.shape[:2]
        poly = self._roi_polygon(w, h)
        cv2.polylines(out, [poly], True, (0, 255, 255), 1)
        y_top, y_bot = int(poly[:, 1].min()), h - 1
        for segs, color in ((res.left_segments, (0, 200, 0)), (res.right_segments, (0, 0, 220))):
            for x1, y1, x2, y2 in segs:
                cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        for line, color in ((res.left, (0, 255, 0)), (res.right, (0, 0, 255))):
            if line:
                a, b = line
                cv2.line(out, _pt(a * y_top + b, y_top), _pt(a * y_bot + b, y_bot), color, 2)
        y = int(self.lookahead * (h - 1))
        cv2.line(out, (0, y), (w - 1, y), (160, 160, 160), 1)
        cv2.line(out, (w // 2, y - 8), (w // 2, y + 8), (255, 255, 255), 1)     # 화면 중심
        if res.detected:
            cv2.circle(out, _pt(res.center_x, y), 5, (255, 0, 255), -1)        # 차선 중심
            text = f"e={res.error:+.2f}"
        else:
            text = "LANE LOST"
        cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return out

    def draw_debug(self, frame, res):
        """[오버레이 | HSV 마스크 | 에지] 3분할 화면 (튜닝용)."""
        gray = lambda m: cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        return np.hstack([self.draw(frame, res), gray(self.mask), gray(self.edges)])


def _total_length(segments):
    return sum(np.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in segments)


def _pt(x, y):
    return int(np.clip(x, -10000, 10000)), int(y)


def main():
    ap = argparse.ArgumentParser(description="영상/이미지 파일에 차선 인식을 돌려 오버레이를 확인한다")
    ap.add_argument("source", help="동영상 또는 이미지 파일")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", help="오버레이 동영상(.mp4) 또는 이미지 저장 경로")
    ap.add_argument("--debug", action="store_true", help="오버레이|마스크|에지 3분할로 저장/표시")
    ap.add_argument("--show", action="store_true", help="창으로 표시 (q: 종료)")
    args = ap.parse_args()

    det = LaneDetector(load_config(args.config))
    render = det.draw_debug if args.debug else det.draw

    still = cv2.imread(args.source)
    if still is not None:
        res = det.process(still)
        print(f"error={res.error} left={res.left} right={res.right}")
        if args.out:
            cv2.imwrite(args.out, render(still, res))
        if args.show:
            cv2.imshow("lane", render(still, res))
            cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"열 수 없음: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    writer = None
    n = found = both = 0
    proc = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = det.process(frame)
        proc += time.perf_counter() - t0
        n += 1
        found += res.detected
        both += bool(res.left and res.right)
        if args.out or args.show:
            view = render(frame, res)
            if args.out:
                if writer is None:
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (view.shape[1], view.shape[0]))
                writer.write(view)
            if args.show:
                cv2.imshow("lane", view)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                    break
    if writer:
        writer.release()
    if n:
        print(f"프레임 {n}개 | 검출 {100 * found / n:.1f}% (양쪽 {100 * both / n:.1f}%) | "
              f"처리 {1000 * proc / n:.2f} ms/frame ({n / proc:.0f} FPS) | "
              f"학습된 차선 폭 {det.lane_width:.0f}px")


if __name__ == "__main__":
    main()
