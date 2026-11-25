"""카메라 캡처 스레드. 실제: picamera2 / mock: 동영상 파일.

    python hardware/camera.py                         # Pi Camera FPS 측정
    python hardware/camera.py --mock assets/sample.mp4       # 동영상 파일 입력
    python hardware/camera.py --snapshot shot.jpg     # 한 장 저장 (HSV 튜닝용)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # 프로젝트 루트
import argparse
import logging
import threading
import time

import cv2

from config import load_config

log = logging.getLogger("camera")


class Camera:
    """read() 는 (최신 BGR 프레임, 캡처 시각[monotonic]) 을 즉시 돌려준다. 첫 프레임 전에는 (None, 0.0)."""

    def __init__(self, cfg, mock=False, source=None, loop=True):
        c = cfg["camera"]
        self.size = (c["width"], c["height"])
        self._frame, self._t = None, 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cam = self._cap = None
        self.eof = False
        if mock:
            if not source:
                raise ValueError("mock 카메라에는 동영상 파일 경로가 필요합니다")
            self._cap = cv2.VideoCapture(str(source))
            if not self._cap.isOpened():
                raise FileNotFoundError(f"동영상을 열 수 없음: {source}")
            self._interval = 1.0 / (self._cap.get(cv2.CAP_PROP_FPS) or c["fps"])
            self._loop = loop
            target = self._run_video
        else:
            from libcamera import Transform
            from picamera2 import Picamera2
            self._cam = Picamera2()
            # picamera2 의 "RGB888" 은 numpy 배열에서 BGR 순서 → OpenCV 와 바로 호환
            self._cam.configure(self._cam.create_video_configuration(
                main={"size": self.size, "format": "RGB888"},
                controls={"FrameRate": c["fps"]},
                transform=Transform(hflip=c["hflip"], vflip=c["vflip"])))
            self._cam.start()
            target = self._run_picamera
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def _put(self, frame):
        with self._lock:
            self._frame, self._t = frame, time.monotonic()

    def _run_picamera(self):
        try:
            while not self._stop.is_set():
                self._put(self._cam.capture_array())
        except Exception:
            log.exception("카메라 캡처 스레드 오류")     # 이후 프레임이 끊겨 main 의 안전 정지로 이어진다

    def _run_video(self):
        next_t = time.monotonic()
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                if not self._loop:
                    self.eof = True
                    return
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            self._put(frame)
            next_t += self._interval
            time.sleep(max(0.0, next_t - time.monotonic()))

    def read(self):
        with self._lock:
            return self._frame, self._t

    def wait_ready(self, timeout=5.0):
        end = time.monotonic() + timeout
        while self.read()[0] is None:
            if time.monotonic() > end:
                raise TimeoutError("카메라 프레임이 들어오지 않음")
            time.sleep(0.01)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        if self._cam is not None:
            self._cam.stop()
            self._cam.close()
        if self._cap is not None:
            self._cap.release()


def main():
    ap = argparse.ArgumentParser(description="카메라 캡처 테스트 (FPS 측정)")
    ap.add_argument("--mock", metavar="VIDEO", help="동영상 파일을 카메라 대신 사용")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--snapshot", help="마지막 프레임을 이미지로 저장")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    cam = Camera(load_config(args.config), mock=bool(args.mock), source=args.mock)
    try:
        cam.wait_ready()
        n, last, frame = 0, 0.0, None
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            frame, t = cam.read()
            if t != last:
                n, last = n + 1, t
            time.sleep(0.002)
        print(f"{frame.shape[1]}x{frame.shape[0]}  {n / args.seconds:.1f} FPS")
        if args.snapshot:
            cv2.imwrite(args.snapshot, frame)
            print("저장:", args.snapshot)
    finally:
        cam.close()


if __name__ == "__main__":
    main()
