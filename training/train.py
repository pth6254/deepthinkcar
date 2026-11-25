"""AI 주행 모델 학습 (결과는 매번 새 파일 models/…_날짜_시간.tflite).

    python training/train.py                       # 키트 모델 추가 학습 (추천, 정답 = OpenCV 조향)
    python training/train.py --labels human        # 키트 모델을 사람 조향으로 추가 학습
    python training/train.py --from-scratch        # 새 모델 학습 (정답 = 녹화된 조향·속도, 데이터 많이 필요)
    python training/train.py --test 20261125_104045   # 이 녹화는 학습에 쓰지 않고 시험용으로만
    python training/train.py --convert-only        # 학습 없이 키트 모델을 TFLite 로 변환만

녹화마다 마지막 15% 를 검증에 쓴다. 조향 정답으로 AI 자신의 조향(mode=cnn)은 쓰지 않는다.
"""
import sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT))   # 프로젝트 루트
import argparse
import csv
import time

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import layers

from autopilot.cnn_pilot import INPUT_H, INPUT_W, CnnPilot, preprocess, preprocess_kit
from autopilot.lane_detect import LaneDetector
from config import load_config


# ---------------- 데이터 ----------------
def load_session(log, cfg, labels):
    """→ [(BGR 이미지, 조향 -1~1, 스로틀)]"""
    det, out = LaneDetector(cfg), []
    c, both = cfg["control"], 99
    lim = c.get("normal_steer_deg", c["max_steer_deg"])
    with open(log, newline="") as f:
        for row in csv.DictReader(f):
            throttle = float(row["throttle"])
            if throttle < 0.01 or (labels == "human" and row.get("mode") == "cnn"):
                continue                                   # 정지/후진, AI 자신의 조향은 제외
            img = cv2.imread(str(log.parent / row["image"]))
            if img is None:
                continue
            steer = float(np.clip(float(row["steer"]) * float(row.get("max_steer_deg") or 30.0) / c["max_steer_deg"], -1, 1))
            if labels == "opencv":
                r = det.process(img)
                if not r.detected or r.held:
                    continue
                if r.left and r.right:
                    both = 0
                else:                                      # 한쪽만 보이면, 양쪽이 보인 직후(0.5초)까지만 믿는다
                    both += 1
                    if both > 10:
                        continue
                steer = float(np.clip(c["kp"] * r.error, -lim, lim) / c["max_steer_deg"])
            out.append((img, steer, throttle))
    return out


def load_data(root, cfg, labels, test_names):
    tr, va, te, missing = [], [], [], set(test_names)
    for log in sorted(_Path(root).rglob("log.csv")):
        rows, name = load_session(log, cfg, labels), log.parent.name
        missing.discard(name)
        if name in test_names:
            te += rows
        else:
            cut = len(rows) - int(len(rows) * 0.15)        # 인접 프레임이 섞이지 않게 시간순 분할
            tr += rows[:cut]
            va += rows[cut:]
        print(f"  {name}: {len(rows)}장{' (시험용)' if name in test_names else ''}")
    if missing:
        raise SystemExit(f"시험용 녹화를 찾을 수 없음: {', '.join(sorted(missing))}")
    return tr, va, te


def arrays(rows, kit, max_deg):
    """좌우반전을 더해 (x, y) 로. kit: 각도 1개 / 자체 모델: [조향, 스로틀]"""
    xs, ys = [], []
    for img, s, t in rows:
        for im, st in ((img, s), (np.ascontiguousarray(img[:, ::-1]), -s)):
            xs.append(preprocess_kit(im) if kit else preprocess(im))
            ys.append((90.0 + max_deg * st,) if kit else (st, t))
    return np.asarray(xs), np.asarray(ys, np.float32)


# ---------------- 모델 ----------------
def build_model():
    """PilotNet 구조를 160x120 입력에 맞춘 것. 입력은 0~255 값 그대로."""
    return keras.Sequential([
        keras.Input((INPUT_H, INPUT_W, 3)),
        layers.Rescaling(1 / 127.5, offset=-1.0),
        layers.Conv2D(24, 5, strides=2, activation="relu"),
        layers.Conv2D(36, 5, strides=2, activation="relu"),
        layers.Conv2D(48, 5, strides=2, activation="relu"),
        layers.Conv2D(64, 3, activation="relu"),
        layers.Conv2D(64, 3, activation="relu"),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(100, activation="relu"),
        layers.Dense(50, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(2, activation="tanh"),               # [steer, throttle]
    ], name="pilotnet")


def save_tflite(model, out, x_rep=None):
    """x_rep 을 주면 int8 양자화, 없으면 float32."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if x_rep is not None:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ([x[None].astype(np.float32)] for x in x_rep)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = conv.inference_output_type = tf.int8
    out = _Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(conv.convert())
    print("저장:", out)


def steer_error(path, rows, cfg):
    """실제 주행과 같은 추론 경로로 잰 조향 오차(°). 좌우반전도 포함."""
    pilot, err = CnnPilot(cfg, model_path=str(path)), []
    for img, s, _ in rows:
        for im, st in ((img, s), (np.ascontiguousarray(img[:, ::-1]), -s)):
            err.append(abs(float(np.clip(pilot.predict(im)[0], -1, 1)) - st))
    return float(np.mean(err)) * cfg["control"]["max_steer_deg"]


def main():
    ap = argparse.ArgumentParser(description="AI 주행 모델 학습")
    ap.add_argument("--from-scratch", action="store_true", help="키트 모델 대신 새 모델을 처음부터 학습")
    ap.add_argument("--labels", choices=("opencv", "human"), help="정답 조향 (기본: 추가 학습 opencv, 새 모델 human)")
    ap.add_argument("--test", nargs="*", default=[], metavar="녹화이름", help="시험용으로만 쓸 녹화 폴더 이름")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--base", default=str(ROOT / "models/lane_kit.h5"), help="추가 학습의 시작 모델(.h5)")
    ap.add_argument("--out", help="결과 파일 (기본: models/ 아래 날짜별)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--min-images", type=int, default=300)
    ap.add_argument("--convert-only", action="store_true")
    args = ap.parse_args()
    keras.utils.set_random_seed(0)
    cfg, kit = load_config(), not args.from_scratch

    if args.convert_only:
        if args.from_scratch:
            ap.error("--convert-only 는 키트 모델용")
        return save_tflite(keras.models.load_model(args.base, compile=False),
                           args.out or ROOT / "models/lane_kit.tflite")

    labels = args.labels or ("opencv" if kit else "human")
    print(f"{'키트 모델 추가 학습' if kit else '새 모델 학습'} | 정답: {labels}")
    tr, va, te = load_data(args.data, cfg, labels, set(args.test))
    if len(tr) < args.min_images or not va:
        raise SystemExit(f"학습에 쓸 사진이 {len(tr)}장뿐입니다 (최소 {args.min_images}장)")
    max_deg = cfg["control"]["max_steer_deg"]
    (xt, yt), (xv, yv) = arrays(tr, kit, max_deg), arrays(va, kit, max_deg)
    print(f"학습 {len(tr)}장, 검증 {len(va)}장, 시험 {len(te)}장 (좌우반전으로 2배)")

    if kit:
        model = keras.models.load_model(args.base, compile=False)
        for layer in model.layers:
            layer.trainable = not layer.name.startswith("conv")
    else:
        model = build_model()
    model.compile(keras.optimizers.Adam(1e-4 if kit else 1e-3), loss="mse")
    model.fit(xt, yt, validation_data=(xv, yv), batch_size=32, epochs=args.epochs or (30 if kit else 40), verbose=2,
              callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)])

    out = _Path(args.out) if args.out else ROOT / f"models/{'lane' if kit else 'scratch'}_{time.strftime('%Y%m%d_%H%M%S')}.tflite"
    rep = None if kit else xt[np.random.default_rng(0).choice(len(xt), min(300, len(xt)), replace=False)]
    save_tflite(model, out, rep)

    rows, kind = (te, "시험용 녹화") if te else (va, "검증 구간")
    current = _Path(cfg["cnn"]["model"])
    current = current if current.is_absolute() else ROOT / current
    new = steer_error(out, rows, cfg)
    print(f"\n[{kind} {len(rows)}장] 조향 오차 (작을수록 좋음)\n  새 모델   {new:.1f}°  {out.name}")
    if current.exists():
        old = steer_error(current, rows, cfg)
        print(f"  현재 모델 {old:.1f}°  {current.name}")
        print("→ 좋아졌습니다. config.yaml 의 cnn.model 을 바꾸고 주행으로 확인하세요:" if new < old else
              "→ 좋아지지 않았습니다. 현재 모델을 그대로 쓰는 것을 권장합니다. 새 모델:",
              out.relative_to(ROOT) if out.is_relative_to(ROOT) else out)
    if not te:
        print("※ 현재 모델이 이 녹화로 학습된 적이 있으면 공정하지 않습니다. --test 로 시험용 녹화를 지정하세요")


if __name__ == "__main__":
    main()
