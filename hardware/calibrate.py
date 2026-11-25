"""서보/ESC 보정: 펄스폭(us)을 조금씩 바꿔 보며 값을 찾아 config.yaml 에 저장한다.

    python hardware/calibrate.py            # 실제 하드웨어 (ESC 단계에서는 반드시 바퀴를 띄울 것)
    python hardware/calibrate.py --mock     # 하드웨어 없이 절차만 확인

각 단계 입력:  Enter=확정 | +2 / -2 = 상대 조정 | 92 (각도) 또는 1520 (us) = 직접 입력 | s=건너뛰기 | q=저장 없이 종료
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # 프로젝트 루트
import argparse
import logging

from hardware.actuator import Actuator
from config import DEFAULT_PATH, load_config, update_yaml

ESC_RANGE = (1000, 2000)


class Abort(Exception):
    pass


def adjust(label, start, lo, hi, apply, unit="us"):
    """사용자가 값을 조정하고 확정할 때까지 반복. 확정값(us)을 돌려준다."""
    us = int(start)
    while True:
        apply(us)
        try:
            s = input(f"  [{label}] 현재 {us}{unit} > ").strip().lower()
        except EOFError:
            raise Abort
        if s == "":
            return us
        if s == "q":
            raise Abort
        if s == "s":
            return int(start)
        try:
            new = us + int(s) if s[0] in "+-" else int(s)
        except ValueError:
            print("    입력: Enter / +N / -N / 숫자 / s / q")
            continue
        us = max(lo, min(hi, new))
        if us != new:
            print(f"    허용 범위 {lo}~{hi}{unit} 로 제한됨")


def save(path, values):
    """config.yaml 의 actuator 항목 값만 바꾼다 (주석/형식 유지)."""
    try:
        update_yaml(path, {("actuator", k): v for k, v in values.items()})
    except KeyError as e:
        raise SystemExit(f"저장 중단: {e}")


def main():
    ap = argparse.ArgumentParser(description="서보/ESC 펄스폭 보정")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--config", default=str(DEFAULT_PATH))
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.mock else logging.INFO,
                        format="    %(name)s %(message)s")

    cfg = load_config(args.config)
    a = cfg["actuator"]
    act = Actuator(cfg, mock=args.mock, watchdog=False)   # 입력 대기 중에도 값을 유지해야 하므로 워치독 끔
    v = {}
    try:
        print(__doc__.split("각 단계 입력:")[1].strip(), "\n")
        neutral = a["esc_neutral_us"]

        print("1) 조향 서보 (단위: 도) — 90° 근처에서 바퀴가 정면을 향하는 각도, 좌/우 끝 각도를 찾는다")
        servo = lambda deg: act.write_us(act.deg_to_us(deg), neutral)
        v["servo_center_deg"] = cdeg = adjust("직진(정면)", a["servo_center_deg"], 45, 135, servo, unit="°")
        v["servo_left_deg"] = adjust("왼쪽 끝", a["servo_left_deg"], 0, 180, servo, unit="°")
        v["servo_right_deg"] = adjust("오른쪽 끝", a["servo_right_deg"], 0, 180, servo, unit="°")
        servo(cdeg)
        c = act.deg_to_us(cdeg)

        print("\n2) ESC (drive_motor: esc 일 때만) — ★ 차를 받침대에 올려 바퀴를 띄운 뒤 진행 ★")
        try:
            if a["drive_motor"] != "esc":
                print("  L9110 모터 사용 중 → ESC 보정 불필요")
                raise EOFError
            if input("  바퀴를 띄웠습니까? (y 입력 시 진행, 그 외 ESC 단계 건너뜀) > ").strip().lower() != "y":
                raise EOFError
            esc = lambda us: act.write_us(c, us)
            v["esc_neutral_us"] = neutral = adjust("중립(정지)", neutral, *ESC_RANGE, esc)
            print("  전진: 값을 조금씩 올려(+5) 바퀴가 막 돌기 시작하는 값을 찾는다")
            v["esc_fwd_min_us"] = fmin = adjust("전진 시작", neutral, *ESC_RANGE, esc)
            print("  전진 상한: 자율주행에 허용할 최대 속도까지 올린다 (throttle=1.0 에 해당)")
            v["esc_fwd_max_us"] = adjust("전진 상한", max(fmin, min(a["esc_fwd_max_us"], fmin + 50)),
                                         *ESC_RANGE, esc)
            esc(neutral)
            print("  후진: 값을 조금씩 내린다(-5). (ESC에 따라 브레이크→중립→후진 순서가 필요할 수 있음)")
            v["esc_rev_min_us"] = rmin = adjust("후진 시작", neutral, *ESC_RANGE, esc)
            v["esc_rev_max_us"] = adjust("후진 상한", min(rmin, max(a["esc_rev_max_us"], rmin - 50)),
                                         *ESC_RANGE, esc)
            esc(neutral)
        except EOFError:
            print("  ESC 보정 건너뜀")
        act.stop()

        print("\n보정 결과:")
        for k, val in v.items():
            print(f"  {k}: {val}" + ("" if val == a[k] else f"   (기존 {a[k]})"))
        try:
            if input(f"\n{args.config} 에 저장할까요? (y/n) > ").strip().lower() == "y":
                save(args.config, v)
                print("저장 완료")
            else:
                print("저장하지 않음")
        except EOFError:
            print("저장하지 않음")
    except Abort:
        print("\n중단 — 저장하지 않음")
    except KeyboardInterrupt:
        print("\nCtrl+C — 저장하지 않음")
    finally:
        act.close()


if __name__ == "__main__":
    main()
