from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Databricks wrapper for Qwen AutoQC")
    parser.add_argument(
        "--config",
        default="production/qwen_auto_qc/configs/local.yaml",
        help="Path to package config",
    )
    args = parser.parse_args()
    command = [
        sys.executable,
        "production/qwen_auto_qc/run_cli.py",
        "run",
        "--config",
        args.config,
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
