from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree as ET

import numpy as np


STROKE_WIDTH = 3.0
FONT_SIZE = 10
START_Y_PERC = 0.4

__all__ = ["draw_cd_diagram"]


def _rank_to_x(rank: float, k: int, start_x: float, end_x: float) -> float:
    if k <= 1:
        return (start_x + end_x) / 2.0
    return start_x + (rank - 1.0) * (end_x - start_x) / (k - 1.0)


def _estimate_label_rows(labels: list[str], k: int, start_x: float, end_x: float) -> int:
    if k <= 1 or not labels:
        return 1
    avg_rank_gap_px = (end_x - start_x) / (k - 1)
    max_label_len = max(len(label) for label in labels)
    est_label_width_px = max(24.0, 0.58 * FONT_SIZE * max_label_len)
    rows = int(np.ceil(est_label_width_px / max(1.0, avg_rank_gap_px)))
    return int(np.clip(rows, 1, 6))


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
        "text-anchor": anchor,
        "fill": color,
        "stroke": color,
        "stroke-width": "1",
    }
    if dominant_baseline is not None:
        attrib["dominant-baseline"] = dominant_baseline

    node = ET.SubElement(parent, "text", attrib)
    node.text = text


def _draw_ruler(parent: ET.Element, k: int, width: int, height: int, axis_y: float) -> tuple[float, float]:
    start_x = 0.2 * width
    end_x = 0.8 * width

    _svg_line(parent, start_x, axis_y, end_x, axis_y)
    tick_len = 0.035 * height
    for tick in range(1, k + 1):
        x = _rank_to_x(float(tick), k, start_x, end_x)
        _svg_line(parent, x, axis_y - tick_len / 2.0, x, axis_y + tick_len / 2.0, width=STROKE_WIDTH / 2.0)
        _svg_text(parent, str(tick), x, axis_y - tick_len / 2.0 - FONT_SIZE)

    return start_x, end_x


def _draw_models(
    parent: ET.Element,
    labels: list[str],
    avg_ranks: list[float],
    k: int,
    width: int,
    height: int,
    axis_y: float,
    start_x: float,
    end_x: float,
) -> None:
    n_rows = _estimate_label_rows(labels, k, start_x, end_x)
    row_spacing = FONT_SIZE + 8
    base_offset = 0.075 * height

    for i, (label, rank) in enumerate(zip(labels, avg_ranks)):
        x = _rank_to_x(rank, k, start_x, end_x)
        marker_size = STROKE_WIDTH
        _svg_rect(parent, x - marker_size / 2.0, axis_y - marker_size / 2.0, marker_size, marker_size, fill="black")

        label_row = i % n_rows
        label_y = axis_y + base_offset + label_row * row_spacing
        _svg_line(parent, x, axis_y + marker_size / 2.0, x, label_y - FONT_SIZE / 2.0, width=STROKE_WIDTH / 2.0)
        _svg_text(parent, label, x, label_y, dominant_baseline="hanging")


def _compute_nonsignificant_groups(sorted_avg_ranks: list[float], cd: float) -> list[tuple[float, float]]:
    groups: list[tuple[float, float]] = []
    for i in range(len(sorted_avg_ranks)):
        j = i + 1
        while j < len(sorted_avg_ranks) and sorted_avg_ranks[j] - sorted_avg_ranks[i] <= cd:
            j += 1
        if j - i > 1:
            groups.append((sorted_avg_ranks[i], sorted_avg_ranks[j - 1]))
    return groups


def _draw_cd_bar(parent: ET.Element, cd: float, k: int, start_x: float, end_x: float, cd_y: float) -> None:
    cd_start = 1.0
    cd_end = 1.0 + cd
    x1 = _rank_to_x(cd_start, k, start_x, end_x)
    x2 = _rank_to_x(cd_end, k, start_x, end_x)
    _svg_line(parent, x1, cd_y, x2, cd_y, color="red", width=STROKE_WIDTH / 2.0)
    tick_len = 0.02 * (end_x - start_x)
    _svg_line(parent, x1, cd_y - tick_len / 2.0, x1, cd_y + tick_len / 2.0, color="red", width=STROKE_WIDTH / 2.0)
    _svg_line(parent, x2, cd_y - tick_len / 2.0, x2, cd_y + tick_len / 2.0, color="red", width=STROKE_WIDTH / 2.0)
    _svg_text(parent, f"CD={cd:.2f}", (x1 + x2) / 2.0, cd_y - FONT_SIZE - 3)


def _render_groups(
    parent: ET.Element,
    groups: list[tuple[float, float]],
    k: int,
    start_x: float,
    end_x: float,
    groups_start_y: float,
    group_spacing: float,
) -> None:
    for i, (rank_start, rank_end) in enumerate(groups):
        y = groups_start_y + i * group_spacing
        x1 = _rank_to_x(rank_start, k, start_x, end_x)
        x2 = _rank_to_x(rank_end, k, start_x, end_x)
        _svg_line(parent, x1, y, x2, y, color="red", width=STROKE_WIDTH / 2.0)
        _svg_line(parent, x1, y - STROKE_WIDTH / 2.0, x1, y + STROKE_WIDTH / 2.0, color="red", width=STROKE_WIDTH / 2.0)
        _svg_line(parent, x2, y - STROKE_WIDTH / 2.0, x2, y + STROKE_WIDTH / 2.0, color="red", width=STROKE_WIDTH / 2.0)


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

    delta = 18
    offset_height = 170 + (len(groups) * 10)
    if fig_size is None:
        width, height = 512, max(256, len(labels) * delta + offset_height)
    else:
        width, height = fig_size

    ruler_step = 6
    number = len(str(len(labels))) * ruler_step
    min_ruler_width = number * len(labels)
    width = max(width, int(min_ruler_width / 0.6))

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(width),
            "height": str(height),
            "style": "background-color:white",
        },
    )

    _svg_text(svg, title or "", width / 2.0, 0.1 * height, color="black")
    group_spacing = max(8.0, 0.03 * height)
    groups_start_y = 0.26 * height
    axis_y = groups_start_y + (max(0, len(groups) - 1) * group_spacing) + 0.08 * height
    start_x, end_x = _draw_ruler(svg, len(sorted_avg_ranks), width, height, axis_y)
    _draw_cd_bar(svg, cd, len(sorted_avg_ranks), start_x, end_x, 0.18 * height)
    _render_groups(svg, groups, len(sorted_avg_ranks), start_x, end_x, groups_start_y, group_spacing)

    _draw_models(
        svg,
        sorted_labels,
        sorted_avg_ranks,
        len(sorted_avg_ranks),
        width,
        height,
        axis_y,
        start_x,
        end_x,
    )

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

    from scipy.stats import friedmanchisquare, rankdata, studentized_range

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

    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
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
