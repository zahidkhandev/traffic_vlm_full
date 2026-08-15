from __future__ import annotations

import pandas as pd


def build_confident_joint_frame(records: list[dict]) -> pd.DataFrame:
    """Prepare a tabular view for future confident-joint analysis."""
    return pd.DataFrame(records)
