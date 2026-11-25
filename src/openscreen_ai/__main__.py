"""Module entry point for safe OpenScreen AI maintenance commands."""

from __future__ import annotations

import sys

from openscreen_ai.doctor import run_doctor


def main() -> int:
    if sys.argv[1:] != ["doctor"]:
        print("Usage: python -m openscreen_ai doctor", file=sys.stderr)
        return 2
    return run_doctor()


if __name__ == "__main__":
    raise SystemExit(main())
