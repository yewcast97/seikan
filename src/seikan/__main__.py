"""``python -m seikan`` — the same entry point as the ``seikan`` console script."""

from seikan.cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
