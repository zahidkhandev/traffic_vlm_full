import shutil
from pathlib import Path

from qwen_auto_qc.config import RunConfig
from qwen_auto_qc.experiments.runner import run_experiment_matrix


def test_run_experiment_matrix_writes_summary(monkeypatch):
    tmp_path = Path("test_output/experiments_case")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    calls = []

    class _FakePipeline:
        def __init__(self, config):
            self.config = config

        def run(self):
            calls.append(self.config.inference_mode)
            run_dir = tmp_path / self.config.inference_mode / "run_1"
            run_dir.mkdir(parents=True, exist_ok=True)
            summary = type(
                "Summary",
                (),
                {
                    "error_rate": 0.1,
                    "flagged_samples": 1,
                    "total_samples": 10,
                    "mean_latency_ms": 2.0,
                },
            )()
            return summary, run_dir

    monkeypatch.setattr("qwen_auto_qc.experiments.runner.AutoQCPipeline", _FakePipeline)
    config = RunConfig(
        output_root=str(tmp_path / "runs"),
        checkpoint_root=str(tmp_path / "checkpoints"),
        experiment_modes=["without_red_rectangle", "with_red_rectangle"],
    )
    summary = run_experiment_matrix(config)
    assert calls == ["without_red_rectangle", "with_red_rectangle"]
    assert summary["modes"] == ["without_red_rectangle", "with_red_rectangle"]
