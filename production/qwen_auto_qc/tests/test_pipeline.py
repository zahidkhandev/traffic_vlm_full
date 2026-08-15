import shutil
from pathlib import Path

from qwen_auto_qc.config import RunConfig
from qwen_auto_qc.pipeline.processor import AutoQCPipeline
from qwen_auto_qc.types import DetectionSample, InferenceResult


class _FakeParser:
    def iter_samples(self):
        return [
            DetectionSample(
                sample_idx=0,
                image_id="sample",
                image_path=Path("sample.jpg"),
                label_path=Path("sample.json"),
                obj_id=1,
                given_label="car",
                given_label_idx=2,
                box=[0, 0, 20, 20],
            )
        ]


class _FakeInferencer:
    def predict(self, sample):
        return InferenceResult(
            predicted_label="truck",
            predicted_label_idx=3,
            predicted_confidence=0.9,
            pred_probs={"car": 0.1, "truck": 0.9},
            embedding=[0.1, 0.2],
            latency_ms=5.0,
        )


def test_pipeline_run_persists_outputs():
    tmp_root = Path("test_output")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        output_root=str(tmp_root / "runs"),
        checkpoint_root=str(tmp_root / "checkpoints"),
        thresholds={"car": 0.2},
    )
    pipeline = AutoQCPipeline(config, inferencer=_FakeInferencer())
    pipeline.parser = _FakeParser()
    summary, run_dir = pipeline.run()
    assert summary.flagged_samples == 1
    assert (run_dir / "run_summary.json").exists()
    assert any(
        p.exists()
        for p in [run_dir / "all_samples.parquet", run_dir / "all_samples.csv"]
    )
