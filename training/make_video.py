"""녹화(data/<이름>)를 차선 표시가 들어간 동영상으로 변환 → videos/<이름>_모드.mp4

    python training/make_video.py                    # 가장 최신 녹화
    python training/make_video.py 20261125_142339    # 지정한 녹화 (여러 개 가능)
    python training/make_video.py --all              # 모든 녹화
    python training/make_video.py --raw              # 차선 표시 없이
"""
import sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT))   # 프로젝트 루트
import argparse
import csv
import shutil
import subprocess

import cv2

from autopilot.lane_detect import LaneDetector
from config import load_config


def convert(session, raw):
    with open(session / "log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        return print(f"{session.name}: 사진이 부족해 건너뜀")
    det = LaneDetector(load_config())
    modes = [r.get("mode") or "manual" for r in rows]
    out = ROOT / "videos" / f"{session.name}_{max(set(modes), key=modes.count).replace('cnn', 'ai')}{'_raw' if raw else ''}.mp4"
    out.parent.mkdir(exist_ok=True)
    tmp, writer = out.with_suffix(".tmp.mp4"), None
    for i, (r, mode) in enumerate(zip(rows, modes)):
        img = cv2.imread(str(session / r["image"]))
        if img is None:
            continue
        if not raw:
            img = cv2.resize(det.draw(img, det.process(img)), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            deg = float(r["steer"]) * float(r.get("max_steer_deg") or 30)
            t = float(r["timestamp"]) - float(rows[0]["timestamp"])
            text = f"{mode.replace('cnn', 'AI').upper()}  steer {deg:+.0f}deg  throttle {float(r['throttle']):+.2f}  {t:.1f}s"
            cv2.rectangle(img, (0, img.shape[0] - 30), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
            cv2.putText(img, text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        # 멈춰 있던 동안은 사진이 없으므로 사진 간격만큼 같은 장면을 유지해 실제 시간과 맞춘다 (최대 1초)
        gap = float(rows[i + 1]["timestamp"]) - float(r["timestamp"]) if i + 1 < len(rows) else 0.05
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 20, (img.shape[1], img.shape[0]))
        for _ in range(max(1, min(20, round(gap * 20)))):
            writer.write(img)
    if writer is None:
        return print(f"{session.name}: 읽을 수 있는 사진이 없음")
    writer.release()
    if shutil.which("ffmpeg") and subprocess.run(          # 브라우저/휴대폰에서도 재생되는 H.264 로
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]).returncode == 0:
        tmp.unlink()
    else:
        tmp.replace(out)
    print(f"저장: {out.relative_to(ROOT)} ({len(rows)}장, {out.stat().st_size / 1e6:.1f}MB)")


def main():
    ap = argparse.ArgumentParser(description="녹화 → 동영상")
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--raw", action="store_true", help="차선 표시 없이")
    args = ap.parse_args()
    sessions = sorted(p.parent for p in (ROOT / "data").glob("*/log.csv"))
    if args.names:
        sessions = [ROOT / "data" / n for n in args.names]
        bad = [s.name for s in sessions if not (s / "log.csv").exists()]
        if bad:
            raise SystemExit(f"녹화를 찾을 수 없음: {', '.join(bad)}")
    elif not args.all:
        sessions = sessions[-1:]
    if not sessions:
        raise SystemExit("data/ 에 녹화가 없음")
    for s in sessions:
        convert(s, args.raw)


if __name__ == "__main__":
    main()
