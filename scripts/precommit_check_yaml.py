#!/usr/bin/env python

from pathlib import Path
import sys

import yaml


def check_file(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            yaml.safe_load(handle)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        print(f"YAML validation failed for {path}: {exc}")
        return False
    return True


def main(paths: list[str]) -> int:
    ok = True
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            ok &= check_file(path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
