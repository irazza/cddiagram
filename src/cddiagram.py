from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree as ET

import numpy as np


STROKE_WIDTH = 3.0
FONT_SIZE = 10
FONT_FAMILY = "sans-serif"
START_Y_PERC = 0.4

__all__ = ["draw_cd_diagram"]


_NEMENYI_ALPHA = 0.05
_NEMENYI_Q_ALPHA_LOOKUP_START_K = 3
_NEMENYI_Q_ALPHA_LOOKUP = (
    2.343700586378,
    2.569031772546,
    2.727774370870,
    2.849705419610,
    2.948320017530,
    3.030878449614,
    3.101730341303,
    3.163683577053,
    3.218653607329,
    3.268003924466,
    3.312738593351,
    3.353617751852,
    3.391230283765,
    3.426041379371,
    3.458424707347,
    3.488684799379,
    3.517073008692,
    3.543799131518,
    3.569040029951,
    3.592946136985,
    3.615646437227,
    3.637252331689,
    3.657860673072,
    3.677556175853,
    3.696413349185,
    3.714498061375,
    3.731868816887,
    3.748577806831,
    3.764671779385,
    3.780192765841,
    3.795178690014,
    3.809663882747,
    3.823679518639,
    3.837253988676,
    3.850413219673,
    3.863180949380,
    3.875578964405,
    3.887627306809,
    3.899344454180,
    3.910747477169,
    3.921852177757,
    3.932673211031,
    3.943224192754,
    3.953517794659,
    3.963565829129,
    3.973379324619,
    3.982968593028,
    3.992343290009,
    4.001512469103,
    4.010484630418,
    4.019267764515,
    4.027869392046,
    4.036296599626,
    4.044556072366,
    4.052654123421,
    4.060596720889,
    4.068389512322,
    4.076037847122,
    4.083546797007,
    4.090921174773,
    4.098165551497,
    4.105284272340,
    4.112281471089,
    4.119161083553,
    4.125926859914,
    4.132582376138,
    4.139131044527,
    4.145576123483,
    4.151920726562,
    4.158167830873,
    4.164320284881,
    4.170380815658,
    4.176352035642,
    4.182236448921,
    4.188036457107,
    4.193754364808,
    4.199392384750,
    4.204952642564,
    4.210437181260,
    4.215847965430,
    4.221186885180,
    4.226455759824,
    4.231656341347,
    4.236790317671,
    4.241859315710,
    4.246864904262,
    4.251808596723,
    4.256691853645,
    4.261516085155,
    4.266282653231,
    4.270992873854,
    4.275648019044,
    4.280249318781,
    4.284797962827,
    4.289295102445,
    4.293741852035,
    4.298139290676,
    4.302488463597,
)
_NEMENYI_Q_ALPHA_LOOKUP_END_K = _NEMENYI_Q_ALPHA_LOOKUP_START_K + len(_NEMENYI_Q_ALPHA_LOOKUP) - 1


def _nemenyi_q_alpha(k: int, alpha: float) -> float:
    if (
        alpha == _NEMENYI_ALPHA
        and _NEMENYI_Q_ALPHA_LOOKUP_START_K <= k <= _NEMENYI_Q_ALPHA_LOOKUP_END_K
    ):
        return _NEMENYI_Q_ALPHA_LOOKUP[k - _NEMENYI_Q_ALPHA_LOOKUP_START_K]

    from scipy.stats import studentized_range

    return studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0)


def _rank_to_x(rank: float, k: int, start_x: float, end_x: float) -> float:
    if k <= 1:
        return (start_x + end_x) / 2.0
    return end_x - (rank - 1.0) * (end_x - start_x) / (k - 1.0)


def _svg_line(parent: ET.Element, x1: float, y1: float, x2: float, y2: float, *, color: str = "black", width: float = STROKE_WIDTH) -> None:
    ET.SubElement(
        parent,
        "line",
        {
            "x1": f"{x1:.3f}",
            "y1": f"{y1:.3f}",
            "x2": f"{x2:.3f}",
            "y2": f"{y2:.3f}",
            "stroke": color,
            "stroke-width": f"{width:.3f}",
            "fill": "none",
        },
    )


def _svg_polyline(
    parent: ET.Element,
    points: Sequence[tuple[float, float]],
    *,
    color: str = "black",
    width: float = STROKE_WIDTH,
    linejoin: str = "miter",
) -> None:
    points_str = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
    ET.SubElement(
        parent,
        "polyline",
        {
            "points": points_str,
            "stroke": color,
            "stroke-width": f"{width:.3f}",
            "fill": "none",
            "stroke-linejoin": linejoin,
        },
    )


def _svg_rect(parent: ET.Element, x: float, y: float, width: float, height: float, *, fill: str = "red") -> None:
    ET.SubElement(
        parent,
        "rect",
        {
            "x": f"{x:.3f}",
            "y": f"{y:.3f}",
            "width": f"{width:.3f}",
            "height": f"{height:.3f}",
            "fill": fill,
        },
    )


def _svg_text(
    parent: ET.Element,
    text: str,
    x: float,
    y: float,
    *,
    anchor: str = "middle",
    color: str = "black",
    dominant_baseline: str | None = None,
) -> None:
    attrib = {
        "x": f"{x:.3f}",
        "y": f"{y:.3f}",
        "font-size": str(FONT_SIZE),
        "font-family": FONT_FAMILY,
        "text-anchor": anchor,
        "fill": color,
    }
    if dominant_baseline is not None:
        attrib["dominant-baseline"] = dominant_baseline

    node = ET.SubElement(parent, "text", attrib)
    node.text = text


def _draw_ruler(
    parent: ET.Element,
    k: int,
    axis_y: float,
    start_x: float,
    end_x: float,
) -> None:
    tick_len = 8.0
    _svg_line(parent, start_x, axis_y, end_x, axis_y)
    for tick in range(1, k + 1):
        x = _rank_to_x(float(tick), k, start_x, end_x)
        _svg_line(parent, x, axis_y - tick_len / 2.0, x, axis_y + tick_len / 2.0, width=STROKE_WIDTH / 2.0)
        _svg_text(parent, str(tick), x, axis_y - tick_len / 2.0 - FONT_SIZE)


def _draw_models(
    parent: ET.Element,
    labels: list[str],
    avg_ranks: list[float],
    k: int,
    axis_y: float,
    start_x: float,
    end_x: float,
    labels_base_y: float,
) -> None:
    n = len(labels)
    if n == 0:
        return
    half = (n + 1) // 2
    row_spacing = FONT_SIZE + 6
    h_gap = 4.0
    text_pad = 2.0
    marker_w = STROKE_WIDTH
    marker_h = 2.0 * STROKE_WIDTH  # taller than the axis stroke so the marker is visible

    for i, (label, rank) in enumerate(zip(labels, avg_ranks)):
        x_marker = _rank_to_x(rank, k, start_x, end_x)
        _svg_rect(
            parent,
            x_marker - marker_w / 2.0,
            axis_y - marker_h / 2.0,
            marker_w,
            marker_h,
            fill="black",
        )

        if i < half:
            # Right side: best ranks, topmost row closest to ruler is best.
            row_index = i
            label_x_edge = end_x + h_gap
            label_x_text = label_x_edge + text_pad
            anchor = "start"
        else:
            # Left side: worst rank at the topmost left row.
            j = i - half
            row_index = (n - half - 1) - j
            label_x_edge = start_x - h_gap
            label_x_text = label_x_edge - text_pad
            anchor = "end"

        row_y = labels_base_y + row_index * row_spacing

        # Single polyline per classifier so the L-corner has a proper
        # stroke-linejoin (avoids the notch/gap produced by two separate lines).
        _svg_polyline(
            parent,
            [
                (x_marker, axis_y + marker_h / 2.0),
                (x_marker, row_y),
                (label_x_edge, row_y),
            ],
            width=STROKE_WIDTH / 2.0,
            linejoin="miter",
        )
        _svg_text(
            parent,
            label,
            label_x_text,
            row_y,
            anchor=anchor,
            dominant_baseline="middle",
        )


def _compute_nonsignificant_groups(sorted_avg_ranks: list[float], cd: float) -> list[tuple[float, float]]:
    raw: list[tuple[int, int]] = []
    for i in range(len(sorted_avg_ranks)):
        j = i + 1
        while j < len(sorted_avg_ranks) and sorted_avg_ranks[j] - sorted_avg_ranks[i] <= cd:
            j += 1
        if j - i > 1:
            raw.append((i, j - 1))

    # Keep only maximal groups: drop any whose right endpoint is not strictly
    # greater than the running max right endpoint seen in left-to-right order.
    # A subset clique (same rightmost member, different leftmost) would otherwise
    # render as a near-duplicate bar on top of its maximal parent.
    maximal: list[tuple[float, float]] = []
    max_right = -1
    for (i, r) in raw:
        if r > max_right:
            maximal.append((sorted_avg_ranks[i], sorted_avg_ranks[r]))
            max_right = r
    return maximal


def _draw_cd_bar(parent: ET.Element, cd: float, k: int, start_x: float, end_x: float, cd_y: float) -> None:
    cd_start = 1.0
    cd_end = 1.0 + cd
    x1 = _rank_to_x(cd_start, k, start_x, end_x)
    x2 = _rank_to_x(cd_end, k, start_x, end_x)
    tick_len = 8.0  # same as ruler ticks
    _svg_line(parent, x1, cd_y, x2, cd_y, color="red", width=STROKE_WIDTH / 2.0)
    _svg_line(parent, x1, cd_y - tick_len / 2.0, x1, cd_y + tick_len / 2.0, color="red", width=STROKE_WIDTH / 2.0)
    _svg_line(parent, x2, cd_y - tick_len / 2.0, x2, cd_y + tick_len / 2.0, color="red", width=STROKE_WIDTH / 2.0)
    _svg_text(parent, f"CD={cd:.2f}", (x1 + x2) / 2.0, cd_y - FONT_SIZE - 3)


def _assign_group_rows(
    groups: list[tuple[float, float]], k: int, start_x: float, end_x: float
) -> tuple[list[int], int]:
    """Pack groups into the minimum number of rows (greedy interval scheduling)."""
    if not groups:
        return [], 0
    rows: list[list[tuple[float, float]]] = []
    row_indices: list[int] = []
    for rank_start, rank_end in groups:
        x1 = _rank_to_x(rank_start, k, start_x, end_x)
        x2 = _rank_to_x(rank_end, k, start_x, end_x)
        x_left, x_right = min(x1, x2), max(x1, x2)
        placed = False
        for r, row_intervals in enumerate(rows):
            if not any(
                not (x_right + 2.0 < ix_left or x_left - 2.0 > ix_right)
                for ix_left, ix_right in row_intervals
            ):
                row_intervals.append((x_left, x_right))
                row_indices.append(r)
                placed = True
                break
        if not placed:
            rows.append([(x_left, x_right)])
            row_indices.append(len(rows) - 1)
    return row_indices, len(rows)


def _render_groups(
    parent: ET.Element,
    groups: list[tuple[float, float]],
    group_rows: list[int],
    k: int,
    start_x: float,
    end_x: float,
    axis_y: float,
    compact_spacing: float,
) -> None:
    gap = 8.0
    cap_half = compact_spacing * 0.4
    for (rank_start, rank_end), row in zip(groups, group_rows):
        y = axis_y + gap + row * compact_spacing
        x1 = _rank_to_x(rank_start, k, start_x, end_x)
        x2 = _rank_to_x(rank_end, k, start_x, end_x)
        _svg_line(parent, x1, y, x2, y, color="red", width=STROKE_WIDTH / 2.0)
        _svg_line(parent, x1, y - cap_half, x1, y + cap_half, color="red", width=STROKE_WIDTH / 2.0)
        _svg_line(parent, x2, y - cap_half, x2, y + cap_half, color="red", width=STROKE_WIDTH / 2.0)


def _render_cd_diagram(
    cd: float,
    avg_ranks: list[float],
    labels: list[str],
    title: str | None = None,
    fig_size: tuple[int, int] | None = None,
) -> ET.Element:
    rank_label_pairs = sorted(zip(avg_ranks, labels), key=lambda item: item[0])
    sorted_avg_ranks = [item[0] for item in rank_label_pairs]
    sorted_labels = [item[1] for item in rank_label_pairs]
    groups = _compute_nonsignificant_groups(sorted_avg_ranks, cd)

    k = len(sorted_avg_ranks)

    # Label margins
    max_label_len = max((len(l) for l in sorted_labels), default=1)
    max_label_px = max(24.0, 0.62 * FONT_SIZE * max_label_len) + 8.0
    page_margin = 12.0
    min_ruler_span = 220.0

    if fig_size is None:
        width = max(512, int(2 * (max_label_px + page_margin) + min_ruler_span))
    else:
        width = max(fig_size[0], int(2 * (max_label_px + page_margin) + min_ruler_span))

    start_x = max_label_px + page_margin
    end_x = width - max_label_px - page_margin

    # Pack non-significant groups into minimum rows
    group_row_indices, n_group_rows = _assign_group_rows(groups, k, start_x, end_x)
    compact_group_spacing = 8.0
    groups_below_h = n_group_rows * compact_group_spacing + 8.0  # height used below axis

    # Fixed vertical layout (top → bottom):
    #   title | CD bar | axis ticks | ruler | groups | labels
    _TITLE_H = 30.0
    _CD_BAR_H = 50.0    # space for bar + "CD=X.XX" text above it
    _TICKS_H = 22.0     # space above axis for tick number labels
    axis_y = _TITLE_H + _CD_BAR_H + _TICKS_H
    cd_bar_y = _TITLE_H + _CD_BAR_H * 0.5

    label_rows_needed = (k + 1) // 2
    row_spacing_lbl = FONT_SIZE + 6
    labels_base_y = axis_y + groups_below_h + 10.0
    labels_h = label_rows_needed * row_spacing_lbl

    if fig_size is None:
        height = max(256, int(labels_base_y + labels_h + 20))
    else:
        height = fig_size[1]

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(width),
            "height": str(height),
            "style": "background-color:white",
            "font-family": FONT_FAMILY,
        },
    )

    if title:
        _svg_text(svg, title, width / 2.0, _TITLE_H / 2.0, color="black")
    _draw_ruler(svg, k, axis_y, start_x, end_x)
    _draw_cd_bar(svg, cd, k, start_x, end_x, cd_bar_y)
    _draw_models(
        svg,
        sorted_labels,
        sorted_avg_ranks,
        k,
        axis_y,
        start_x,
        end_x,
        labels_base_y,
    )
    # Groups painted last so clique bars render on top of everything else.
    _render_groups(svg, groups, group_row_indices, k, start_x, end_x, axis_y, compact_group_spacing)

    return svg


def _to_numpy_2d(samples: object) -> np.ndarray:
    if isinstance(samples, np.ndarray):
        arr = samples.astype(float, copy=False)
    elif hasattr(samples, "to_numpy"):
        # Supports DataFrame-like objects without importing pandas.
        arr = np.asarray(samples.to_numpy(), dtype=float)
    elif hasattr(samples, "values"):
        arr = np.asarray(samples.values, dtype=float)
    else:
        arr = np.asarray(samples, dtype=float)

    if arr.ndim != 2:
        raise ValueError("samples must be a 2D array-like object")
    return arr


def draw_cd_diagram(
    samples: object,
    labels: Sequence[str],
    title: str | None = None,
    out_file: str | None = None,
    fig_size: tuple[int, int] | None = None,
) -> ET.Element | None:
    alpha = 0.05

    samples_ = _to_numpy_2d(samples)
    labels_ = list(labels)

    from scipy.stats import friedmanchisquare, rankdata

    _, pvalue = friedmanchisquare(*samples_.T)
    if pvalue >= alpha:
        warnings.warn(
            "The null hypothesis of the Friedman test cannot be rejected.",
            stacklevel=2,
        )
        return None

    N, k = samples_.shape
    if len(labels_) != k:
        raise ValueError("labels length must match number of model columns")

    q_alpha = _nemenyi_q_alpha(k, alpha)
    cd = q_alpha * np.sqrt((k * (k + 1)) / (6 * N))

    avg_ranks = rankdata(-samples_, axis=1, method="average").mean(axis=0)
    sorted_indices = np.argsort(avg_ranks)

    svg = _render_cd_diagram(
        cd,
        avg_ranks[sorted_indices].tolist(),
        [labels_[i] for i in sorted_indices],
        title,
        fig_size,
    )

    if out_file is not None:
        tree = ET.ElementTree(svg)
        tree.write(Path(out_file), encoding="utf-8", xml_declaration=True)

    return svg
