"""차선 유지 난이도(0 쉬움 ~ 1 어려움). 속도에만 쓰고 조향은 바꾸지 않는다."""


def _ramp(v, lo, hi):
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


class Difficulty:
    """오를 때는 바로, 내릴 때는 쉬운 상태가 0.5초 이어진 뒤 천천히 (빨라졌다가 다시 이탈하는 것 방지)."""

    EASY_FRAMES = 10            # 20Hz 기준 0.5초
    DOWN_PER_FRAME = 0.05       # 초당 1.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.value, self.reason = 0.0, ""
        self.hard = False       # 끊어 전진할 만큼 어려운가 (0.7 에서 켜지고 0.55 에서 꺼짐)
        self._easy = self._single = 0
        self._hist = []

    def update(self, lane, steer_deg, normal_deg, ai_deg=None, want_deg=None):
        e = lane.error
        scores = {}
        if e is None:
            scores["차선 없음"] = 0.8
            self._single = 0
        else:
            self._single = 0 if lane.left and lane.right else self._single + 1
            scores["오차"] = _ramp(abs(e), 0.3, 0.9)
            if self._single >= 2:
                scores["한쪽 차선"] = 0.55
            if len(self._hist) >= 5 and self._hist[-5] is not None and abs(e) > 0.25:
                scores["오차 증가"] = 0.8 * _ramp((abs(e) - abs(self._hist[-5])) / 0.25, 0.8, 2.4)
            if ai_deg is not None and want_deg is not None:
                scores["AI 불일치"] = 0.8 * _ramp(abs(ai_deg - want_deg), 12.0, 35.0)
        scores["조향 한계"] = 0.7 * _ramp(abs(steer_deg) / normal_deg, 0.65, 1.0)
        self._hist = (self._hist + [e])[-5:]
        reason = max(scores, key=scores.get)
        now = scores[reason]
        if now >= self.value:
            self.value, self.reason = now, reason
        self._easy = self._easy + 1 if now < 0.4 else 0
        if self._easy > self.EASY_FRAMES:
            self.value = max(now, self.value - self.DOWN_PER_FRAME)
        self.hard = self.value >= (0.55 if self.hard else 0.7)
        return self.value

    def limit(self, throttle, t_max, t_min):
        """0.4 까지 그대로, 0.7 에서 커브 속도까지 줄인 상한."""
        return min(throttle, t_max - (t_max - t_min) * _ramp(self.value, 0.4, 0.7)) if t_max > t_min else throttle
