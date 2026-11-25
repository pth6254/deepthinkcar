"""급커브(꺾인 모서리) 감지: 안쪽 테이프가 '>' (좌) 또는 '<' (우) 로 꺾인 지점을 찾아 미리 감속하고 제때 꺾는다."""
import cv2
import numpy as np


MIN_SHARPNESS = 25      # 꺾인 모서리 44~99, 둥근 커브 12 이하 (녹화·시뮬레이터 측정값)


def _sharpness(c, side):
    """꼭짓점 바로 위 12줄에서 테이프 가장자리가 옆으로 얼마나 물러나는가(px). 모서리일수록 크다."""
    rows = np.where(c.any(1))[0]
    edge = c.shape[1] - 1 - c[:, ::-1].argmax(1) if side < 0 else -c.argmax(1)
    y0 = rows[np.argmax(edge[rows])]
    up = [edge[y0] - edge[y] for y in rows if y0 - 12 <= y < y0]
    return int(max(up)) if up else 0


def find_corner(mask):
    """→ (방향 -1 좌/+1 우, 꼭짓점 y 비율, 뾰족한 정도 px) 또는 None."""
    h, w = mask.shape
    m = mask.copy()
    m[:int(0.3 * h)] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        if area < 250 or ch < 0.15 * h or cw < 0.12 * w:
            continue
        c = lab[y:y + ch, x:x + cw] == i
        cnt, top, bot = c.sum(0), c.argmax(0), ch - 1 - c[::-1].argmax(0)
        gap = np.where(cnt > 0, bot - top + 1 - cnt, 0)      # 한 열에서 테이프가 두 번 지나가는 사이의 빈칸
        d = np.where(gap > 6)[0]
        if len(d) < 0.08 * w or d[-1] - d[0] + 1 > 1.3 * len(d):
            continue
        k = max(3, len(d) // 4)
        left, right = gap[d[:k]].mean(), gap[d[-k:]].mean()
        if left > 1.5 * right:
            side, t = -1, min(d[-1] + 2, cw - 1)
        elif right > 1.5 * left:
            side, t = 1, max(d[0] - 2, 0)
        else:
            continue
        tip_x, tip_y = (x + t) / w, (y + (top[t] + bot[t]) / 2) / h
        if (side < 0 and tip_x > 0.6) or (side > 0 and tip_x < 0.4) or tip_y < 0.45:
            continue                                          # 바깥 테이프이거나 아직 멀다
        if best is None or tip_y > best[1]:
            best = (side, tip_y, _sharpness(c, side))
    return best


def free_end(mask, side):
    """side 쪽 화면 끝에서 들어온 테이프가 화면 안에서 끝나는 지점(U턴 안쪽 꼭짓점)이
    화면 반대편에서부터 얼마나 side 쪽으로 왔는가 (0~1). 없으면 None."""
    h, w = mask.shape
    m = mask.copy()
    m[:int(0.33 * h)] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    for i in sorted(range(1, n), key=lambda k: -st[k, cv2.CC_STAT_AREA])[:4]:
        x, y, cw, ch, area = st[i]
        touches = x + cw > 0.97 * w if side > 0 else x < 0.03 * w
        if area < 120 or not touches or cw < 0.1 * w:
            continue
        end = x / w if side > 0 else 1 - (x + cw) / w
        col = lab[y:y + ch, x if side > 0 else x + cw - 1] == i
        end_y = (y + np.nonzero(col)[0].mean()) / h
        if end > 0.04 and 0.35 < end_y < 0.6 and y + ch < 0.85 * h:
            return end
    return None


def tape_across(mask):
    """옆으로 누운 테이프가 앞을 가로막고 있는가 (아직 다 돌지 못함)."""
    h, w = mask.shape
    m = mask.copy()
    m[:int(0.33 * h)] = 0
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return any(st[i, 2] > 0.4 * w and st[i, 3] < 0.6 * st[i, 2] and st[i, 4] > 120 for i in range(1, n))


def sweeps_across(mask, side):
    """바깥 테이프 한 줄이 화면 반대편에서 시작해 side 쪽 끝까지 휘어 가로지르는가 (아직 더 꺾어야 함)."""
    h, w = mask.shape
    m = mask.copy()
    m[:int(0.38 * h)] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        if cw < 0.75 * w or area < 300 or ch < 0.2 * h:
            continue
        if (x > 0.03 * w) if side < 0 else (x + cw < 0.97 * w):
            continue
        edge = np.nonzero(lab[y:y + ch, x if side < 0 else x + cw - 1] == i)[0]
        if len(edge) and y + edge.mean() > 0.4 * h:
            return True
    return False


class SharpCorner:
    """phase 0: 평소 / 1: 모서리가 보임 → 감속, 반대로 꺾지 않음 / 2: 다 왔음 → 꺾어서 유지"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = self.side = self.seen = self.age = self.ok = 0
        self.tip_seen = self.after = self.after_side = 0
        self.chained = False
        self.tip_end = 0.0
        self.tip = 0.0

    def update(self, mask, left, right, error):
        corner = find_corner(mask)
        self.age += 1
        self.after = max(0, self.after - 1)
        if self.phase == 0 and self.after and error is not None and (left is None) != (right is None) \
                and error * self.after_side > 0.3:
            return self._turn_back(self.after_side)
        if self.phase == 0:
            if corner and corner[2] < MIN_SHARPNESS:
                corner = None                                 # 둥근 커브는 평소 방식으로 돈다
            self.seen = self.seen + 1 if corner and corner[0] == self.side else (1 if corner else 0)
            self.side = corner[0] if corner else 0
            if self.seen >= 2:
                self.phase, self.age, self.tip = 1, 0, corner[1]
        elif self.phase == 1:
            if corner and corner[0] == self.side:
                self.age, self.tip = 0, corner[1]
            if self.tip >= 0.65 or (self.tip >= 0.55 and self.age >= 2):
                self.phase, self.age, self.ok = 2, 0, 0
            elif self.age > 15:
                self.reset()
        else:
            if not self.chained:                              # 모서리에 바로 이어지는 반대 방향 U턴
                end = free_end(mask, -self.side)
                if end is not None:
                    self.tip_seen, self.tip_end = self.tip_seen + 1, end
                if self.tip_seen >= 3 and (self.tip_end > 0.55 or (end is None and self.tip_end > 0.4)):
                    return self._turn_back(-self.side)
            good = left and right and error is not None and abs(error) < 0.5
            if self.chained:                                  # U턴: 1.2초 이상 돌고, 앞을 막는 테이프가 없어야 끝낸다
                good = good and self.age > 24 and not tape_across(mask)
            self.ok = self.ok + 1 if good and self.age > 8 else 0
            if self.ok >= (5 if self.chained else 3) or self.age > (80 if self.chained else 40):
                side, chained = self.side, self.chained
                self.reset()
                if not chained:                               # 빠져나온 직후 1.5초는 반대 방향 U턴에 대비
                    self.after, self.after_side = 30, -side
        return self.phase

    def _turn_back(self, side):
        self.reset()
        self.phase, self.side, self.chained = 2, side, True
        return self.phase
