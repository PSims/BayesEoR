#!/usr/bin/env python

from pathlib import Path
import sys


def trim_file(path: Path) -> bool:
    try:
        original = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    lines = original.splitlines(keepends=True)
    trimmed_lines: list[str] = []
    for line in lines:
        if line.endswith("\r\n"):
            body = line[:-2]
            ending = "\r\n"
        elif line.endswith("\n"):
            body = line[:-1]
            ending = "\n"
        else:
            body = line
            ending = ""
        trimmed_lines.append(body.rstrip(" \t") + ending)

    updated = "".join(trimmed_lines)
    if updated == original:
        return False

    path.write_text(updated, encoding="utf-8")
    print(f"Trimmed trailing whitespace in {path}")
    return True


def main(paths: list[str]) -> int:
    changed = False
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            changed |= trim_file(path)
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
