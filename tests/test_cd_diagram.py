import warnings
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from cddiagram import draw_cd_diagram


def _make_significant_samples() -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(1)
    labels = [f"model{i}" for i in range(1, 9)]
    arr = np.column_stack(
        [
            rng.normal(loc=0.2, scale=0.1, size=30),
            rng.normal(loc=0.2, scale=0.1, size=30),
            rng.normal(loc=0.4, scale=0.1, size=30),
            rng.normal(loc=0.5, scale=0.1, size=30),
            rng.normal(loc=0.7, scale=0.1, size=30),
            rng.normal(loc=0.7, scale=0.1, size=30),
            rng.normal(loc=0.8, scale=0.1, size=30),
            rng.normal(loc=0.9, scale=0.1, size=30),
        ]
    )
    return arr, labels


def test_draw_cd_diagram_array(tmp_path):
    samples, labels = _make_significant_samples()
    out_file = tmp_path / "df.svg"

    result = draw_cd_diagram(samples, labels=labels, out_file=str(out_file), title="TEST")

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "CD=" in content
    assert isinstance(result, ET.Element)


def test_draw_cd_diagram_in_memory():
    samples, labels = _make_significant_samples()

    result = draw_cd_diagram(samples, labels=labels)

    assert isinstance(result, ET.Element)
    svg_str = ET.tostring(result, encoding="unicode")
    assert "<svg" in svg_str
    assert "CD=" in svg_str


def test_draw_cd_diagram_non_significant(tmp_path):
    rng = np.random.default_rng(7)
    samples = np.column_stack(
        [
            rng.normal(loc=0.5, scale=0.1, size=30),
            rng.normal(loc=0.5, scale=0.1, size=30),
            rng.normal(loc=0.5, scale=0.1, size=30),
        ]
    )
    out_file = tmp_path / "none.svg"
    labels = ["model1", "model2", "model3"]

    with pytest.warns(UserWarning, match="cannot be rejected"):
        result = draw_cd_diagram(samples, labels=labels, out_file=str(out_file))

    assert result is None
    assert not out_file.exists()
