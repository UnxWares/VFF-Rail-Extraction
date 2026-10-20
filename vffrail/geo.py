"""Web Mercator geometry (EPSG:3857, IGN "PM" tile grid) and rasterisation.

IGN "PM" tiles follow the standard XYZ grid exactly (256 px, origin at the
top left). All "pixel" coordinates are global pixels at level z:
    px = (X + R) / res(z),  py = (R - Y) / res(z),  res(z) = 2R / (256 · 2^z)
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

R = 20037508.342789244  # Web Mercator half-extent (m)
TILE = 256


def res(z: int) -> float:
    """Size of a Web Mercator pixel at level z (Mercator m)."""
    return 2 * R / (TILE * 2**z)


def ground_res(z: int, lat: float) -> float:
    """Real ground size of a pixel (m) at the given latitude."""
    return res(z) * math.cos(math.radians(lat))


def merc_to_px(x, y, z: int):
    r = res(z)
    return (np.asarray(x) + R) / r, (R - np.asarray(y)) / r


def px_to_merc(px, py, z: int):
    r = res(z)
    return np.asarray(px) * r - R, R - np.asarray(py) * r


def lonlat_to_merc(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = lon * R / 180.0
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R / math.pi
    return x, y


def merc_to_lonlat(x, y):
    lon = np.asarray(x) / R * 180.0
    lat = np.degrees(2 * np.arctan(np.exp(np.asarray(y) / R * math.pi)) - np.pi / 2)
    return lon, lat


def merc_scale(lat: float) -> float:
    """Mercator factor: 1 ground metre = 1/cos(lat) Mercator metres."""
    return 1.0 / math.cos(math.radians(lat))


def draw_lines(size: tuple[int, int], lines_px, width: float, value: int = 1, canvas=None):
    """Rasterises polylines (lists of (px, py) relative to the canvas corner)."""
    img = canvas if canvas is not None else Image.new('L', size, 0)
    d = ImageDraw.Draw(img)
    w = max(1, int(round(width)))
    for pts in lines_px:
        if len(pts) < 2:
            continue
        flat = [(float(a), float(b)) for a, b in pts]
        d.line(flat, fill=value, width=w, joint='curve')
        if w >= 3:
            r = w / 2
            for a, b in (flat[0], flat[-1]):
                d.ellipse([a - r, b - r, a + r, b + r], fill=value)
    return img
