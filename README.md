# cddiagram

A pure Python library for generating Critical Difference (CD) diagrams as SVG.

CD diagrams visualize the statistical comparison of multiple classifiers (or models) over multiple datasets, as introduced by Demsar (2006). They show the average rank of each model and connect groups of models whose performance differences are **not** statistically significant.

> J. Demsar, "Statistical Comparisons of Classifiers over Multiple Data Sets",
> *Journal of Machine Learning Research*, vol. 7, pp. 1-30, 2006.
> https://jmlr.org/papers/v7/demsar06a.html

## How it works

1. A **Friedman test** checks whether at least one model differs significantly from the others (at alpha = 0.05).
2. If significant, a pairwise post-hoc test determines which pairs of models differ. Two options are supported:
   - `posthoc="wilcoxon-holm"` (default): pairwise two-sided **Wilcoxon signed-rank** tests with **Holm** step-down correction. Each pairwise decision uses only those two models' per-dataset scores, so it is robust to which other models are in the experiment (Benavoli, Corani & Mangili, 2016).
   - `posthoc="nemenyi"`: original **Nemenyi** test from Demšar (2006). A single critical distance `CD = q_α · √(k(k+1)/6N)` is computed; pairs with `|R̄_i − R̄_j| ≤ CD` are deemed non-significant. The diagram includes the iconic CD bar.
3. Maximal contiguous cliques of mutually non-significant models are connected by horizontal bars below the rank axis.
4. The result is rendered as an SVG diagram showing ranked models and significance groups.

## Install

```bash
pip install cddiagram
```

Requires Python 3.12+ and depends on `numpy` and `scipy`.

## Usage

### Write to file

```python
import numpy as np
from cddiagram import draw_cd_diagram

rng = np.random.default_rng(1)

models = {
    "model1": rng.normal(loc=0.2, scale=0.1, size=30),
    "model2": rng.normal(loc=0.2, scale=0.1, size=30),
    "model3": rng.normal(loc=0.4, scale=0.1, size=30),
    "model4": rng.normal(loc=0.5, scale=0.1, size=30),
    "model5": rng.normal(loc=0.7, scale=0.1, size=30),
    "model6": rng.normal(loc=0.7, scale=0.1, size=30),
    "model7": rng.normal(loc=0.8, scale=0.1, size=30),
    "model8": rng.normal(loc=0.9, scale=0.1, size=30),
}

samples = np.column_stack(list(models.values()))
draw_cd_diagram(samples, labels=list(models.keys()), out_file="out.svg", title="Model comparison")
```

<img src="./out.svg">

### Non-significant results

If the Friedman test is not significant, the function issues a warning and returns `None` — no diagram is produced because the data does not support ranking the models.

## API

```python
draw_cd_diagram(
    samples,                   # 2D array-like (rows=datasets, columns=models)
    labels,                    # Sequence of model names (one per column)
    title=None,                # Optional diagram title
    out_file=None,             # Optional path to write SVG file
    fig_size=None,             # Optional (width, height) tuple in pixels
    posthoc="wilcoxon-holm",   # "wilcoxon-holm" (default) or "nemenyi"
) -> Element | None
```

**Input formats**: NumPy arrays, pandas DataFrames, or any object with a `.to_numpy()` / `.values` attribute.

## Release Notes

### 0.0.8

- Added `posthoc` parameter; default switched to **Wilcoxon signed-rank with Holm correction** to avoid Nemenyi's pool-dependency defect (Benavoli, Corani & Mangili, *JMLR* 2016). The original Demšar/Nemenyi behavior is available with `posthoc="nemenyi"`.
- The CD bar is rendered only when `posthoc="nemenyi"` (no single critical distance exists for Wilcoxon-Holm).

### 0.0.7

- Optimized Nemenyi critical-value computation with a precomputed `q_alpha` lookup table for `k=3..100` at `alpha=0.05`.
- Kept a SciPy `studentized_range` fallback for values outside the lookup range (or different alpha), preserving behavior for all valid inputs.

### 0.0.6

- Red clique bars now render on top of all other SVG elements (connectors, markers, axis).

### 0.0.5

- Text labels no longer carry an SVG stroke, removing the bold/smudged appearance.
- `font-family="sans-serif"` set globally for consistent cross-platform rendering.
- Empty title no longer emits a stray `<text/>` node.
- Non-maximal cliques filtered out — only strictly maximal non-significance groups are drawn, eliminating near-duplicate bars.
- Classifier connectors use a single `<polyline>` with `stroke-linejoin="miter"`, removing the corner notch produced by two separate lines.
- Axis markers are taller than the axis stroke so they are visible.
- Clique end-caps enlarged for legibility.
- CD-bar tick length unified with ruler tick length.
- Label-width heuristic bumped to better fit sans-serif glyphs.

### 0.0.4

- Non-significant groups packed into the minimum number of rows (greedy interval scheduling).
- Fixed vertical layout: title → CD bar → ruler → groups → labels (groups now drawn below the axis).

### 0.0.2

- Replaced hardcoded Nemenyi critical-value lookup table with SciPy's `studentized_range` computation.
- Updated CD diagram drawing to follow continuous rank-axis placement and first-anchor non-significant grouping.
- Improved readability for larger numbers of algorithms with adaptive multi-row label layout.

## License

MIT
