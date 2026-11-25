"""TFLite(LiteRT) 모델로 카메라 프레임 → (조향, 스로틀) 추론.

preprocess() 는 학습(train.py)과 추론이 반드시 같은 것을 쓰도록 여기 한 곳에만 둔다.
"""
from pathlib import Path

import cv2
import numpy as np

INPUT_W, INPUT_H = 160, 120
KIT_W, KIT_H = 200, 66                # 키트 차선 모델(lane_kit) 입력 크기


def preprocess(frame):
    """BGR 프레임 → 모델 입력 (120, 160, 3) uint8. 정규화는 모델 안에서 한다."""
    return cv2.resize(frame, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)


def _interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter     # PC 에서 TensorFlow 만 있을 때
    return Interpreter


def preprocess_kit(frame):
    """키트 차선 모델 전처리: 아래 절반 → YUV → 블러 → 200x66 → 0~1. 학습(train.py)과 공용."""
    img = cv2.cvtColor(frame[frame.shape[0] // 2:], cv2.COLOR_BGR2YUV)
    img = cv2.resize(cv2.GaussianBlur(img, (3, 3), 0), (KIT_W, KIT_H))
    return img.astype(np.float32) / 255.0


class CnnPilot:
    """입력 크기로 모델 종류를 구분한다.
    (120,160,3) → 자체 모델: [조향, 스로틀]   /   (66,200,3) → 키트 차선 모델: 서보 각도(90°=직진)
    """

    def __init__(self, cfg=None, model_path=None, num_threads=2):
        if cfg is not None:
            model_path = model_path or cfg["cnn"]["model"]
            num_threads = cfg["cnn"].get("num_threads", num_threads)
        path = Path(model_path)
        if not path.is_absolute() and not path.exists():
            path = Path(__file__).resolve().parents[1] / model_path
        if not path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {model_path}")
        self.path = path
        self._interp = _interpreter_class()(model_path=str(path), num_threads=num_threads)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        shape = tuple(self._in["shape"])
        self.kit = shape == (1, KIT_H, KIT_W, 3)
        if not self.kit and shape != (1, INPUT_H, INPUT_W, 3):
            raise ValueError(f"지원하지 않는 모델 입력 크기 {shape}")
        c = cfg["control"] if cfg is not None else {}
        self.max_steer_deg = c.get("max_steer_deg", 30.0)

    def predict(self, frame):
        """BGR 프레임 → (steer -1~1, throttle -1~1)."""
        x = (preprocess_kit if self.kit else preprocess)(frame).astype(np.float32)
        dtype = self._in["dtype"]
        if dtype != np.float32:                           # 양자화 모델: 실수 → 정수
            scale, zero = self._in["quantization"]
            info = np.iinfo(dtype)
            x = np.clip(np.round(x / scale + zero), info.min, info.max)
        self._interp.set_tensor(self._in["index"], x.astype(dtype)[None])
        self._interp.invoke()
        y = self._interp.get_tensor(self._out["index"])[0].astype(np.float32)
        if self._out["dtype"] != np.float32:              # 정수 → 실수
            scale, zero = self._out["quantization"]
            y = (y - zero) * scale
        if self.kit:              # 각도만 나온다. 속도는 main 의 커브 감속이 정한다
            return max(-1.0, min(1.0, (float(y[0]) - 90.0) / self.max_steer_deg)), 1.0
        return float(y[0]), float(y[1])
