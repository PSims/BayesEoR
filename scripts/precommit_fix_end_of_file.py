#!/usr/bin/env python

from pathlib import Path
import sys


def fix_file(path: Path) -> bool:
    try:
        original = path.read_bytes()
    except OSError:
        return False

    if b"\0" in original:
        return False

    stripped = original.rstrip(b"\r\n")
    updated = stripped + b"\n"
    if updated == original:
        return False

    path.write_bytes(updated)
    print(f"Fixed end of file in {path}")
    return True


def main(paths: list[str]) -> int:
    changed = False
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            changed |= fix_file(path)
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
