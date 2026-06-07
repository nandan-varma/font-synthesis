"""Stroke processing and SVG rendering."""

from typing import List, Optional
import numpy as np
import svgwrite
from scipy.signal import savgol_filter


def align(coords: np.ndarray) -> np.ndarray:
    coords = np.copy(coords)
    x = np.concatenate([np.ones([coords.shape[0], 1]), coords[:, :1]], axis=1)
    coef = np.linalg.lstsq(x, coords[:, 1:2], rcond=None)[0]
    offset, slope = coef.squeeze()
    theta = float(np.arctan(slope))
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    coords[:, :2] = coords[:, :2] @ R - offset
    return coords


def denoise(coords: np.ndarray) -> np.ndarray:
    strokes = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0)
    out = []
    for s in strokes:
        if len(s) == 0:
            continue
        if len(s) >= 7:
            s = s.copy()
            s[:, 0] = savgol_filter(s[:, 0], 7, 3, mode='nearest')
            s[:, 1] = savgol_filter(s[:, 1], 7, 3, mode='nearest')
        out.append(s)
    return np.vstack(out) if out else coords


def normalize(offsets: np.ndarray) -> np.ndarray:
    offsets = np.copy(offsets)
    norms = np.linalg.norm(offsets[:, :2], axis=1)
    median = float(np.median(norms[norms > 0])) or 1.0
    offsets[:, :2] /= median
    return offsets


def offsets_to_coords(offsets: np.ndarray) -> np.ndarray:
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)


def coords_to_offsets(coords: np.ndarray) -> np.ndarray:
    offsets = np.concatenate([coords[1:, :2] - coords[:-1, :2], coords[1:, 2:3]], axis=1)
    return np.concatenate([np.array([[0, 0, 1]]), offsets], axis=0)


def draw_svg(
    strokes: List[np.ndarray],
    lines: List[str],
    filename: str,
    stroke_colors: Optional[List[str]] = None,
    stroke_widths: Optional[List[float]] = None,
) -> None:
    stroke_colors = stroke_colors or ['black'] * len(lines)
    stroke_widths = stroke_widths or [2] * len(lines)

    line_height = 60
    view_width = 1000
    view_height = line_height * (len(strokes) + 1)

    dwg = svgwrite.Drawing(filename=filename)
    dwg.viewbox(width=view_width, height=view_height)
    dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))

    initial_coord = np.array([0.0, -(3 * line_height / 4)])
    for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):
        if not line:
            initial_coord[1] -= line_height
            continue

        offsets = offsets.copy()
        offsets[:, :2] *= 1.5
        coords = offsets_to_coords(offsets)
        coords = denoise(coords)
        coords[:, :2] = align(coords[:, :2])

        coords[:, 1] *= -1
        coords[:, :2] -= coords[:, :2].min(axis=0) + initial_coord
        coords[:, 0] += (view_width - coords[:, 0].max()) / 2

        prev_eos = 1.0
        p = "M0,0 "
        for x, y, eos in zip(*coords.T):
            p += f"{'M' if prev_eos == 1.0 else 'L'}{x:.2f},{y:.2f} "
            prev_eos = eos

        path = svgwrite.path.Path(p)
        path = path.stroke(color=color, width=width, linecap='round').fill("none")
        dwg.add(path)
        initial_coord[1] -= line_height

    dwg.save()
