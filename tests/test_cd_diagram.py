import importlib.util
import warnings
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "cddiagram.py"
_SPEC = importlib.util.spec_from_file_location("cddiagram_src", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load local cddiagram module for tests")
_CDDIAGRAM = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CDDIAGRAM)

draw_cd_diagram = _CDDIAGRAM.draw_cd_diagram


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


def test_compute_nonsignificant_groups_overlap_allowed():
    ranks = [1.2, 2.0, 2.4, 3.8]

    groups = _CDDIAGRAM._compute_nonsignificant_groups(ranks, 1.0)

    assert groups == [(1.2, 2.0), (2.0, 2.4)]


def test_rank_to_x_matches_cd_axis_scale():
    k = 5
    start_x = 100.0
    end_x = 500.0
    cd = 1.05

    # Rank 1 is on the right (Demsar convention), rank k on the left.
    assert _CDDIAGRAM._rank_to_x(1.0, k, start_x, end_x) == pytest.approx(end_x)
    assert _CDDIAGRAM._rank_to_x(float(k), k, start_x, end_x) == pytest.approx(start_x)

    x_start = _CDDIAGRAM._rank_to_x(1.0, k, start_x, end_x)
    x_end = _CDDIAGRAM._rank_to_x(1.0 + cd, k, start_x, end_x)

    expected_len = (end_x - start_x) * cd / (k - 1)
    # Rank increases leftward, so x_start is to the right of x_end by expected_len.
    assert (x_start - x_end) == pytest.approx(expected_len)


def test_draw_cd_diagram_side_labels():
    samples, labels = _make_significant_samples()

    result = draw_cd_diagram(samples, labels=labels)

    assert result is not None
    texts = result.findall(".//text")
    start_anchored = [t for t in texts if t.get("text-anchor") == "start"]
    end_anchored = [t for t in texts if t.get("text-anchor") == "end"]
    assert len(start_anchored) >= 1
    assert len(end_anchored) >= 1

    label_set = set(labels)
    classifier_texts = [t for t in texts if (t.text or "") in label_set]
    assert len(classifier_texts) == len(labels)

    # Labels on the same side must not collide; opposite sides may share a y.
    right_ys = [round(float(t.get("y")), 3) for t in classifier_texts if t.get("text-anchor") == "start"]
    left_ys = [round(float(t.get("y")), 3) for t in classifier_texts if t.get("text-anchor") == "end"]
    assert len(set(right_ys)) == len(right_ys)
    assert len(set(left_ys)) == len(left_ys)
