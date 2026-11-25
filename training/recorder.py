"""수동 주행 데이터 기록: 20Hz 로 이미지 + 조향 + 스로틀 저장.

    <dir>/<YYYYmmdd_HHMMSS>/images/000000.jpg ...
    <dir>/<YYYYmmdd_HHMMSS>/log.csv      (timestamp, image, steer, throttle, mode)

JPEG 인코딩과 디스크 쓰기는 별도 스레드에서 처리해 제어 루프를 막지 않는다.
"""
import csv
import logging
import os
import queue
import shutil
import threading
import time
from pathlib import Path

import cv2

log = logging.getLogger("recorder")


class Recorder:
    def __init__(self, cfg):
        c = cfg["recorder"]
        root = Path(c["dir"])
        if not root.is_absolute():                 # 실행 위치와 상관없이 항상 프로젝트 폴더 기준
            root = Path(__file__).resolve().parents[1] / root
        self.root = root
        self.min_interval = 0.9 / c["hz"]          # 루프 지터를 감안해 10% 여유
        self.max_steer_deg = cfg.get("control", {}).get("max_steer_deg", 30.0)   # steer=1 이 몇 도인지 함께 기록
        self.quality = int(c["jpeg_quality"])
        self.auto = bool(c.get("auto", False))             # 움직이면 자동 녹화
        self.auto_stop_s = c.get("auto_stop_s", 2.0)
        self.min_free_mb = c.get("min_free_mb", 1000)
        self.failed = False
        self.active = False
        self.count = self.dropped = 0
        self._q = None
        self._thread = None

    def start(self, suffix=""):
        if self.active:
            return self.session
        self.root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.root).free < self.min_free_mb * 1e6:
            log.error("저장 공간 부족(%dMB 미만) → 녹화하지 않음", self.min_free_mb)
            self.auto = False
            return None
        self.session = self.root / (time.strftime("%Y%m%d_%H%M%S") + suffix)
        (self.session / "images").mkdir(parents=True, exist_ok=True)
        self.count = self.dropped = 0
        self._last_t = None
        self.failed = False
        self._q = queue.Queue(maxsize=200)
        self._thread = threading.Thread(target=self._writer, args=(self.session, self._q), daemon=True)
        self._thread.start()
        self.active = True
        log.info("녹화 시작: %s", self.session)
        return self.session

    def add(self, frame, steer, throttle, t, mode="manual"):
        """프레임 한 장 기록 요청. t(초)는 기록 주기 제한과 timestamp 열에 쓰인다."""
        if not self.active or self.failed or (self._last_t is not None and t - self._last_t < self.min_interval):
            return
        self._last_t = t
        if self.count % 200 == 199 and shutil.disk_usage(self.root).free < self.min_free_mb * 1e6:
            log.error("저장 공간 부족 → 녹화 중단")
            self.failed, self.auto = True, False
            return
        try:
            self._q.put_nowait((self.count, frame, t, steer, throttle, mode))
            self.count += 1
        except queue.Full:                 # 저장 장치가 밀리면 주행을 막지 않고 버린다
            self.dropped += 1

    def _writer(self, session, q):
        try:
            self._write_loop(session, q)
        except Exception:
            log.exception("녹화 저장 실패 → 녹화 중단 (주행에는 영향 없음)")
            self.failed, self.auto = True, False
            while q.get() is not None:                     # stop() 이 막히지 않도록 큐를 비운다
                pass

    def _write_loop(self, session, q):
        with open(session / "log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "image", "steer", "throttle", "mode", "max_steer_deg"])
            while True:
                item = q.get()
                if item is None:
                    f.flush()
                    threading.Thread(target=os.sync, daemon=True).start()   # 제어 루프를 막지 않고 디스크에 확정
                    return
                idx, frame, t, steer, throttle, mode = item
                name = f"images/{idx:06d}.jpg"
                cv2.imwrite(str(session / name), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                w.writerow([f"{t:.3f}", name, f"{steer:.4f}", f"{throttle:.4f}", mode, self.max_steer_deg])
                if idx % 40 == 39:                       # 2초마다 확정: 갑자기 전원이 꺼져도 그때까지는 남는다
                    f.flush()
                    os.sync()

    def stop(self):
        if not self.active:
            return
        self.active = False
        self._q.put(None)
        self._thread.join()
        log.info("녹화 종료: %s (%d장%s)", self.session, self.count,
                 f", {self.dropped}장 누락" if self.dropped else "")

    close = stop
