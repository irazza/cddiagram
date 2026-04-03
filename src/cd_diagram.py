from __future__ import annotations

from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree as ET

import numpy as np
from scipy.stats import friedmanchisquare, rankdata


STROKE_WIDTH = 3.0
FONT_SIZE = 10
START_Y_PERC = 0.4

__all__ = ["draw_cd_diagram"]


def _get_relative_x(value: float, n_items: int, interval_len: float) -> float:
    return ((n_items - value + 1.0) / n_items) * interval_len


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


def _draw_ruler(parent: ET.Element, n_items: int, width: int, height: int) -> None:
    start_y = START_Y_PERC * height
    start_x = 0.2 * width
    end_x = 0.8 * width

    _svg_line(parent, start_x, start_y, end_x, start_y)

    n_lines = n_items * 2
    step = (end_x - start_x) / n_lines
    for i in range(n_lines + 1):
        x = start_x + i * step
        if i % 2 == 0:
            bar_len = 0.05 * height
            _svg_line(parent, x, start_y + STROKE_WIDTH / 2.0, x, start_y - bar_len)
            number = n_items - (i // 2) + 1
            _svg_text(parent, str(number), x, start_y - bar_len - FONT_SIZE)
        else:
            bar_len = 0.025 * height
            _svg_line(parent, x, start_y + STROKE_WIDTH / 2.0, x, start_y - bar_len)


def _draw_models(parent: ET.Element, labels: list[str], avg_ranks: list[float], lowest_clique: float, width: int, height: int) -> None:
    start_y = START_Y_PERC * height
    start_x = 0.2 * width
    end_x = 0.8 * width

    half_count = len(labels) // 2
    for i, (label, value) in enumerate(zip(labels, avg_ranks)):
        x = start_x + _get_relative_x(value, len(labels), end_x - start_x)
        color = "gray" if i % 2 == 0 else "black"

        if i < half_count:
            end_y = (
                lowest_clique
                + (i * (height - lowest_clique)) / (half_count + 1)
                + FONT_SIZE / 2.0
                + STROKE_WIDTH
            )
            _svg_line(parent, x, start_y, x, end_y, color=color, width=STROKE_WIDTH / 2.0)
            _svg_line(parent, x, end_y, start_x - 0.01 * width, end_y, color=color, width=STROKE_WIDTH / 2.0)
            _svg_text(
                parent,
                label,
                start_x - 0.015 * width,
                end_y,
                anchor="end",
                color=color,
                dominant_baseline="middle",
            )
        else:
            end_y = (
                lowest_clique
                + ((len(labels) - i - 1) * (height - lowest_clique)) / (half_count + 1)
                + FONT_SIZE / 2.0
                + STROKE_WIDTH
            )
            _svg_line(parent, x, start_y, x, end_y, color=color, width=STROKE_WIDTH / 2.0)
            _svg_line(parent, x, end_y, end_x + 0.01 * width, end_y, color=color, width=STROKE_WIDTH / 2.0)
            _svg_text(
                parent,
                label,
                end_x + 0.015 * width,
                end_y,
                anchor="start",
                color=color,
                dominant_baseline="middle",
            )


def _draw_clique(parent: ET.Element, start_x: float, start_y: float, clique_len: float, dry_run: bool) -> float:
    if dry_run:
        return start_y

    _svg_rect(parent, start_x - STROKE_WIDTH, start_y - STROKE_WIDTH / 2.0, STROKE_WIDTH, STROKE_WIDTH)
    _svg_line(parent, start_x, start_y, start_x + clique_len, start_y, color="red", width=STROKE_WIDTH / 2.0)
    _svg_rect(parent, start_x + clique_len, start_y - STROKE_WIDTH / 2.0, STROKE_WIDTH, STROKE_WIDTH)
    return start_y


def _draw_cliques(parent: ET.Element, cd: float, avg_ranks: list[float], width: int, height: int, *, dry_run: bool) -> float:
    start_y = (START_Y_PERC - 0.15) * height
    start_x = 0.2 * width
    end_x = 0.8 * width
    cd_len = (end_x - start_x) * cd / len(avg_ranks)

    lowest_clique = _draw_clique(parent, start_x - cd_len / 2.0, start_y + (0.01 * height), cd_len, dry_run)

    _svg_text(parent, f"CD={cd:.2f}", start_x, start_y)

    height_stride_perc = 1.0 / (len(avg_ranks) * 3)
    cliques_start_y = (START_Y_PERC + 0.02) * height
    h = 0
    last_x2 = None
    for i in range(len(avg_ranks) - 1, -1, -1):
        count = 0
        for j in range(i - 1, -1, -1):
            if abs(avg_ranks[i] - avg_ranks[j]) < cd:
                count += 1
            else:
                break
        if count > 0:
            x1 = start_x + _get_relative_x(avg_ranks[i], len(avg_ranks), end_x - start_x)
            x2 = start_x + _get_relative_x(avg_ranks[i - count], len(avg_ranks), end_x - start_x)
            if last_x2 is None or abs(last_x2 - x2) > 1e-9:
                last_x2 = x2
                y = cliques_start_y + height_stride_perc * (h * height)
                lowest_clique = max(lowest_clique, _draw_clique(parent, x2, y, abs(x1 - x2), dry_run))
                h += 1

    return lowest_clique


def _render_cd_diagram(
    cd: float,
    avg_ranks: list[float],
    labels: list[str],
    title: str | None = None,
    out_file: str | None = None,
    fig_size: tuple[int, int] | None = None,
) -> None:
    delta = 8
    offset_height = 32
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
    _draw_ruler(svg, len(avg_ranks) - 1, width, height)
    lowest_clique = _draw_cliques(svg, cd, avg_ranks, width, height, dry_run=True)
    _draw_models(svg, labels, avg_ranks, lowest_clique, width, height)
    _draw_cliques(svg, cd, avg_ranks, width, height, dry_run=False)

    output = Path(out_file or "image.svg")
    tree = ET.ElementTree(svg)
    tree.write(output, encoding="utf-8", xml_declaration=True)


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
) -> None:
    alpha = 0.05

    samples_ = _to_numpy_2d(samples)
    labels_ = list(labels)

    _, pvalue = friedmanchisquare(*samples_.T)
    if pvalue < alpha:
        N, k = samples_.shape
        if len(labels_) != k:
            raise ValueError("labels length must match number of model columns")
        if k >= len(qstu_0_05) or np.isnan(qstu_0_05[k]):
            raise ValueError(f"unsupported number of models for lookup table: {k}")

        q_alpha = qstu_0_05[k]
        cd = q_alpha * np.sqrt((k * (k + 1)) / (6 * N))

        avg_ranks = rankdata(-samples_, axis=1, method="average").mean(axis=0)
        sorted_indices = np.argsort(-avg_ranks)

        _render_cd_diagram(
            cd,
            avg_ranks[sorted_indices].tolist(),
            [labels_[i] for i in sorted_indices],
            title,
            out_file,
            fig_size,
        )
    else:
        print("The null hypothesis of Friedman Test cannot be rejected")


qstu_0_05 = [np.nan, np.nan, 1.959964233, 2.343700476, 2.569032073, 2.727774717, 2.849705382, 2.948319908, 3.030878867, 3.10173026, 3.16368342, 3.218653901, 3.268003591, 3.312738701, 3.353617959, 3.391230382, 3.426041249, 3.458424619, 3.488684546, 3.517072762, 3.543799277, 3.569040161, 3.592946027, 3.615646276, 3.637252631, 3.657860551, 3.677556303, 3.696413427, 3.71449839, 3.731869175, 3.748578108, 3.764671858, 3.780192852, 3.795178566, 3.809663649, 3.823679212, 3.837254248, 3.850413505, 3.863181025, 3.875578729, 3.887627121, 3.899344587, 3.910747391, 3.921852503, 3.932673359, 3.943224099, 3.953518159, 3.963566147, 3.973379375, 3.98296845, 3.992343271, 4.001512325, 4.010484803, 4.019267776, 4.02786973, 4.036297029, 4.044556036, 4.05265453, 4.060596753, 4.068389777, 4.076037844, 4.083547318, 4.090921028, 4.098166044, 4.105284488, 4.112282016, 4.119161458, 4.125927056, 4.132582345, 4.139131568, 4.145576139, 4.151921008, 4.158168297, 4.164320833, 4.170380738, 4.176352255, 4.182236797, 4.188036487, 4.19375486, 4.199392622, 4.204952603, 4.21043763, 4.215848411, 4.221187067, 4.22645572, 4.23165649, 4.236790793, 4.241859334, 4.246864943, 4.251809034, 4.256692313, 4.261516196, 4.266282802, 4.270992841, 4.275648432, 4.280249575, 4.284798393, 4.289294885, 4.29374188, 4.298139377, 4.302488791]
