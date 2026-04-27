import csv
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


def _load_test_csv() -> tuple[np.ndarray, list[str]]:
    csv_path = Path(__file__).resolve().parents[1] / "examples" / "test_data.csv"
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = header[1:]
        rows = [[float(x) for x in row[1:]] for row in reader]
    return np.array(rows, dtype=float), list(labels)


def _cliques_from_groups(
    groups: list[tuple[float, float]],
    sorted_ranks: list[float],
    sorted_labels: list[str],
) -> set[frozenset[str]]:
    out: set[frozenset[str]] = set()
    for rank_start, rank_end in groups:
        i = sorted_ranks.index(rank_start)
        r = sorted_ranks.index(rank_end)
        out.add(frozenset(sorted_labels[i : r + 1]))
    return out


def test_draw_cd_diagram_array_nemenyi(tmp_path):
    samples, labels = _make_significant_samples()
    out_file = tmp_path / "df.svg"

    result = draw_cd_diagram(
        samples, labels=labels, out_file=str(out_file), title="TEST", posthoc="nemenyi"
    )

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "CD=" in content
    assert isinstance(result, ET.Element)


def test_draw_cd_diagram_default_omits_cd_bar(tmp_path):
    samples, labels = _make_significant_samples()
    out_file = tmp_path / "default.svg"

    result = draw_cd_diagram(samples, labels=labels, out_file=str(out_file))

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "<svg" in content
    # Default is Wilcoxon-Holm, which has no single critical distance.
    assert "CD=" not in content
    assert isinstance(result, ET.Element)


def test_draw_cd_diagram_in_memory():
    samples, labels = _make_significant_samples()

    result = draw_cd_diagram(samples, labels=labels)

    assert isinstance(result, ET.Element)
    svg_str = ET.tostring(result, encoding="unicode")
    assert "<svg" in svg_str


def test_draw_cd_diagram_invalid_posthoc():
    samples, labels = _make_significant_samples()

    with pytest.raises(ValueError, match="posthoc"):
        draw_cd_diagram(samples, labels=labels, posthoc="bonferroni")


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
    nonsig = _CDDIAGRAM._nonsig_matrix_from_cd(ranks, 1.0)

    groups = _CDDIAGRAM._compute_nonsignificant_groups(ranks, nonsig)

    assert groups == [(1.2, 2.0), (2.0, 2.4)]


def test_compute_nonsignificant_groups_arbitrary_matrix():
    # A range is a clique only when *every* pair within is non-significant,
    # so a single significant interior pair should split the surrounding range.
    ranks = [1.0, 2.0, 3.0, 4.0]
    nonsig = np.array(
        [
            [True, True, False, False],
            [True, True, True, False],
            [False, True, True, True],
            [False, False, True, True],
        ]
    )

    groups = _CDDIAGRAM._compute_nonsignificant_groups(ranks, nonsig)

    # {1,2}, {2,3}, {3,4} — but not {1,2,3} (1↔3 is significant) nor {2,3,4} (2↔4 is significant).
    assert groups == [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]


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


def test_nemenyi_q_alpha_lookup_matches_scipy_sampled():
    from scipy.stats import studentized_range

    for k in (3, 8, 25, 100):
        expected = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
        actual = _CDDIAGRAM._nemenyi_q_alpha(k, 0.05)
        assert actual == pytest.approx(expected, abs=1e-12)

def test_nemenyi_q_alpha_fallback_out_of_range_and_alpha():
    from scipy.stats import studentized_range

    expected_k = studentized_range.ppf(0.95, 101, np.inf) / np.sqrt(2)
    actual_k = _CDDIAGRAM._nemenyi_q_alpha(101, 0.05)
    assert actual_k == pytest.approx(expected_k)

    expected_alpha = studentized_range.ppf(0.90, 8, np.inf) / np.sqrt(2)
    actual_alpha = _CDDIAGRAM._nemenyi_q_alpha(8, 0.10)
    assert actual_alpha == pytest.approx(expected_alpha)


def test_wilcoxon_holm_nonsig_matrix_is_symmetric_with_true_diagonal():
    samples, _ = _load_test_csv()

    nonsig = _CDDIAGRAM._nonsig_matrix_from_wilcoxon_holm(samples, alpha=0.05)

    assert nonsig.dtype == bool
    assert nonsig.shape == (samples.shape[1], samples.shape[1])
    assert np.array_equal(nonsig, nonsig.T)
    assert np.all(np.diag(nonsig))


def test_wilcoxon_holm_cliques_match_hfawaz_on_test_data():
    """Wilcoxon-Holm on examples/test_data.csv matches what hfawaz/cd-diagram
    and scikit_posthocs (with Conover-Friedman) report: {clf1,clf2,clf4} and {clf3,clf5}.
    Nemenyi, by contrast, would also draw an extra {clf4,clf5} clique because
    their rank diff (1.50) sits just below CD≈1.575."""
    from scipy.stats import rankdata

    samples, labels = _load_test_csv()
    avg_ranks = rankdata(-samples, axis=1, method="average").mean(axis=0)
    sorted_idx = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sorted_idx].tolist()
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_samples = samples[:, sorted_idx]

    nonsig = _CDDIAGRAM._nonsig_matrix_from_wilcoxon_holm(sorted_samples, alpha=0.05)
    groups = _CDDIAGRAM._compute_nonsignificant_groups(sorted_ranks, nonsig)
    cliques = _cliques_from_groups(groups, sorted_ranks, sorted_labels)

    assert cliques == {
        frozenset({"clf3", "clf5"}),
        frozenset({"clf1", "clf2", "clf4"}),
    }


def test_nemenyi_cliques_on_test_data_show_extra_overlapping_clique():
    """Same data, Nemenyi: produces three overlapping cliques per Demsar 2006."""
    from scipy.stats import rankdata

    samples, labels = _load_test_csv()
    N, k = samples.shape
    avg_ranks = rankdata(-samples, axis=1, method="average").mean(axis=0)
    sorted_idx = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sorted_idx].tolist()
    sorted_labels = [labels[i] for i in sorted_idx]

    q = _CDDIAGRAM._nemenyi_q_alpha(k, 0.05)
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    nonsig = _CDDIAGRAM._nonsig_matrix_from_cd(sorted_ranks, cd)
    groups = _CDDIAGRAM._compute_nonsignificant_groups(sorted_ranks, nonsig)
    cliques = _cliques_from_groups(groups, sorted_ranks, sorted_labels)

    assert cliques == {
        frozenset({"clf3", "clf5"}),
        frozenset({"clf4", "clf5"}),
        frozenset({"clf1", "clf2", "clf4"}),
    }


def test_wilcoxon_holm_alpha_monotonicity():
    """Non-significant set shrinks as alpha grows: a pair non-sig at alpha=0.5
    must also be non-sig at any stricter alpha (0.05, 0.001, 0)."""
    rng = np.random.default_rng(42)
    samples = np.column_stack(
        [rng.normal(loc=mu, scale=0.3, size=40) for mu in (0.0, 0.1, 0.2, 0.4, 0.6, 0.8)]
    )

    nonsig_05 = _CDDIAGRAM._nonsig_matrix_from_wilcoxon_holm(samples, alpha=0.5)
    nonsig_005 = _CDDIAGRAM._nonsig_matrix_from_wilcoxon_holm(samples, alpha=0.05)
    nonsig_zero = _CDDIAGRAM._nonsig_matrix_from_wilcoxon_holm(samples, alpha=0.0)

    # Monotone in alpha: nonsig(0.5) ⊆ nonsig(0.05) ⊆ nonsig(0.0).
    assert np.all(nonsig_05 <= nonsig_005)
    assert np.all(nonsig_005 <= nonsig_zero)
    # alpha=0 trivially declares every pair non-significant (p_adj >= 0 always).
    assert np.all(nonsig_zero)
