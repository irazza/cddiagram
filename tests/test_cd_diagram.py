import numpy as np
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

    draw_cd_diagram(samples, labels=labels, out_file=str(out_file), title="TEST")

    content = out_file.read_text(encoding="utf-8")
    assert out_file.exists()
    assert "<svg" in content
    assert "CD=" in content


def test_draw_cd_diagram_non_significant(capsys, tmp_path):
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

    draw_cd_diagram(samples, labels=labels, out_file=str(out_file))
    captured = capsys.readouterr()

    assert not out_file.exists()
    assert "cannot be rejected" in captured.out