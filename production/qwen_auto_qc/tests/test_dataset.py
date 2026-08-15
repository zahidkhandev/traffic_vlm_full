from qwen_auto_qc.config import RunConfig
from qwen_auto_qc.dataset import BDDDatasetParser


def test_dataset_parser_filters_small_boxes():
    config = RunConfig(min_box_size=10)
    parser = BDDDatasetParser(config)
    assert parser._normalize_box({"x1": 0, "y1": 0, "x2": 5, "y2": 5}) is None
    assert parser._normalize_box({"x1": 0, "y1": 0, "x2": 20, "y2": 20}) == [
        0,
        0,
        20,
        20,
    ]
