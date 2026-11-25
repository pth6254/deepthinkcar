"""차선 오차 e → 조향각(PID, ±30° 제한, 변화율 제한) + 스로틀(|e| 클수록 감속)."""


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


class PID:
    def __init__(self, kp, ki, kd, limit, d_filter=0.5):
        self.kp, self.ki, self.kd, self.limit = kp, ki, kd, limit
        self.d_filter = d_filter          # 미분항 저역통과 (0=필터 없음)
        self.reset()

    def reset(self):
        self._i = 0.0
        self._d = 0.0
        self._prev = None

    def update(self, e, dt):
        if self._prev is not None:
            self._d = self.d_filter * self._d + (1 - self.d_filter) * (e - self._prev) / dt
        self._prev = e
        i_new = self._i + e * dt
        out = self.kp * e + self.ki * i_new + self.kd * self._d
        if abs(out) <= self.limit or out * e < 0:
            self._i = i_new               # 포화 중에는 적분하지 않는다 (anti-windup)
        else:
            out = self.kp * e + self.ki * self._i + self.kd * self._d
        return _clip(out, -self.limit, self.limit)


class RateLimiter:
    """출력 변화율 제한 (단위/초)."""

    def __init__(self, rate):
        self.rate = rate
        self.value = 0.0

    def reset(self, value=0.0):
        self.value = value

    def update(self, target, dt):
        step = self.rate * dt
        self.value += _clip(target - self.value, -step, step)
        return self.value


class SpeedPlanner:
    """커브 정도(0=직선 ~ 1=급커브) → 스로틀. 커브에서는 바로 줄이고, 빠져나올 때는 천천히 올린다."""

    def __init__(self, cfg):
        self.value = None
        self.configure(cfg)

    def configure(self, cfg):
        c = cfg["control"]
        self.t_max, self.t_min = c["throttle_max"], c["throttle_min"]
        self.gain, self.accel = c["curve_slowdown"], c["throttle_accel"]

    def reset(self):
        self.value = None

    def update(self, curve, dt):
        target = self.t_max - (self.t_max - self.t_min) * min(1.0, self.gain * abs(curve))
        if self.value is None:
            self.value = min(self.t_min, target)          # 저속으로 출발
        self.value = min(target, self.value + self.accel * dt)
        return self.value


class LaneController:
    def __init__(self, cfg):
        self.pid = PID(0.0, 0.0, 0.0, 0.0)
        self.limiter = RateLimiter(0.0)
        self.speed = SpeedPlanner(cfg)
        self._last_t = None
        self.configure(cfg)

    def configure(self, cfg):
        """게인/제한값만 다시 읽는다 (실시간 튜닝용, 적분/미분 상태는 유지)."""
        c = cfg["control"]
        self.max_steer_deg = c["max_steer_deg"]
        self.pid.kp, self.pid.ki, self.pid.kd = c["kp"], c["ki"], c["kd"]
        self.normal_steer_deg = min(self.max_steer_deg, c.get("normal_steer_deg", self.max_steer_deg))
        self.pid.limit = self.normal_steer_deg
        self.limiter.rate = c["max_steer_rate_deg_s"]
        self.speed.configure(cfg)
        self._default_dt = 1.0 / cfg["loop_hz"]

    def reset(self):
        self.pid.reset()
        self.limiter.reset()
        self.speed.reset()
        self._last_t = None

    def update(self, error, now, sharp=False, boost=False):
        """오차 e(-1~1) → (조향각[deg, 우회전 +], 스로틀)."""
        dt = self._default_dt if self._last_t is None else _clip(now - self._last_t, 0.01, 0.2)
        self._last_t = now
        self.pid.limit = self.max_steer_deg if boost else self.normal_steer_deg
        target = self.pid.update(error, dt)
        if sharp:                          # 급커브 회전 단계: 최대 조향
            target = self.max_steer_deg if error > 0 else -self.max_steer_deg
        steer_deg = self.limiter.update(target, dt)
        throttle = self.speed.update(max(abs(error), abs(steer_deg) / self.max_steer_deg), dt)
        return steer_deg, throttle
