"""config.yaml 로더 / 부분 저장."""
import re
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).with_name("config.yaml")


def load_config(path=None):
    with open(path or DEFAULT_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_value(v):
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(format_value(x) for x in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(round(v, 6))
    return str(v)


def update_yaml(path, values):
    """{(섹션, 키): 값} 을 config.yaml 에 반영한다. 해당 줄의 값만 바꾸고 주석/형식은 그대로 둔다."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for (section, key), value in values.items():
        start = next((i for i, l in enumerate(lines)
                      if re.match(rf"{re.escape(section)}:\s*(#.*)?$", l)), None)
        if start is None:
            raise KeyError(f"{path}: 섹션 '{section}' 없음")
        found = False
        for j in range(start + 1, len(lines)):
            line = lines[j]
            if line.strip() and not line[0].isspace():
                break                                    # 다음 섹션
            m = re.match(rf"(\s+{re.escape(key)}:\s*)(.*?)(\s+#.*)?(\r?\n)?$", line)
            if m:
                lines[j] = m.group(1) + format_value(value) + (m.group(3) or "") + (m.group(4) or "")
                found = True
                break
        if not found:
            raise KeyError(f"{path}: '{section}.{key}' 없음")
    text = "".join(lines)
    yaml.safe_load(text)                                 # 깨진 YAML 은 저장하지 않는다
    path.write_text(text, encoding="utf-8")
