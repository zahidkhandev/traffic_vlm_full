from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from qwen_auto_qc.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
