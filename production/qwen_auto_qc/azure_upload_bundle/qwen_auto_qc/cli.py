from __future__ import annotations

import argparse
import json
import time

from .azureml_job import azureml_job_as_dict
from .config import RunConfig, load_run_config, validate_run_config
from .experiments.runner import run_experiment_matrix
from .pipeline.processor import AutoQCPipeline


def _load_config(args: argparse.Namespace) -> RunConfig:
    config = load_run_config(args.config)
    overrides: dict[str, str] = {}
    if getattr(args, "model_path", None):
        overrides["model_path"] = args.model_path
    if getattr(args, "images_path", None):
        overrides["images_path"] = args.images_path
    if getattr(args, "labels_path", None):
        overrides["labels_path"] = args.labels_path
    if overrides:
        config = config.copy_with(**overrides)
    return validate_run_config(config)


def run_command(args: argparse.Namespace) -> int:
    config = _load_config(args)
    summary, run_dir = AutoQCPipeline(config).run()
    print(json.dumps({"run_dir": str(run_dir), "summary": summary.metrics}, indent=2))
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    config = _load_config(args)
    summary, run_dir = AutoQCPipeline(config).run()
    print(f"Evaluation complete for {run_dir}: error_rate={summary.error_rate:.4f}")
    return 0


def benchmark_command(args: argparse.Namespace) -> int:
    config = _load_config(args)
    pipeline = AutoQCPipeline(config)
    samples = pipeline.parser.iter_samples()
    target_samples = samples[: config.benchmark_steps]
    for sample in target_samples[: config.benchmark_warmup]:
        pipeline.inferencer.predict(sample)
    started = time.perf_counter()
    for sample in target_samples:
        pipeline.inferencer.predict(sample)
    elapsed = time.perf_counter() - started
    mean_ms = (elapsed / max(len(target_samples), 1)) * 1000.0
    print(
        json.dumps(
            {"samples": len(target_samples), "mean_latency_ms": mean_ms}, indent=2
        )
    )
    return 0


def azureml_spec_command(args: argparse.Namespace) -> int:
    config = _load_config(args)
    print(json.dumps(azureml_job_as_dict(config), indent=2))
    return 0


def experiment_command(args: argparse.Namespace) -> int:
    config = _load_config(args)
    summary = run_experiment_matrix(config)
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen AutoQC production CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, handler in {
        "run": run_command,
        "evaluate": evaluate_command,
        "benchmark": benchmark_command,
        "azureml-spec": azureml_spec_command,
        "experiment": experiment_command,
    }.items():
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", default="configs/local.yaml")
        sub.add_argument("--model-path")
        sub.add_argument("--images-path")
        sub.add_argument("--labels-path")
        sub.set_defaults(func=handler)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
