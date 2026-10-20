"""Registration field for the georeferencing of an old basemap.

Observation: at a point of a precisely known track (ground truth), the model sees the
track drawing shifted by `d` metres along the normal `n`. The local shift (dx, dy) is
estimated by robust weighted least squares on the neighbouring observations (Gaussian
kernel), which solves the aperture problem (a straight track only constrains the
normal component) as soon as tracks of varied orientations are nearby.

Convention: drawing position = true position + (dx, dy) (ground metres, x towards
the east, y towards the north). To correct an extracted trace: subtract (dx, dy).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def normal_profiles(prob, lines_px, search_px, step_px=6.0, half_tan=8.0, min_peak=0.5):
    """Probability peak along the normal to the true axes.

    prob: (H, W) float; lines_px: list of (n, 2) arrays (column, row).
    Returns a (k, 6) array: x, y, nx, ny, d (px, + = side of the normal), peak.
    """
    from scipy.ndimage import map_coordinates

    H, W = prob.shape
    offs = np.arange(-search_px, search_px + 0.01, 0.5)
    out = []
    for pts in lines_px:
        pts = np.asarray(pts, float)
        if len(pts) < 2:
            continue
        seg = np.diff(pts, axis=0)
        L = np.hypot(seg[:, 0], seg[:, 1])
        cum = np.concatenate([[0], np.cumsum(L)])
        if cum[-1] < 2 * half_tan:
            continue
        s = np.arange(half_tan, cum[-1] - half_tan, step_px)
        if not len(s):
            continue
        px = np.interp(s, cum, pts[:, 0])
        py = np.interp(s, cum, pts[:, 1])
        ax = np.interp(s - half_tan, cum, pts[:, 0])
        ay = np.interp(s - half_tan, cum, pts[:, 1])
        bx = np.interp(s + half_tan, cum, pts[:, 0])
        by = np.interp(s + half_tan, cum, pts[:, 1])
        tx, ty = bx - ax, by - ay
        tn = np.hypot(tx, ty)
        ok = tn > 0.5 * half_tan  # no point in a cusp
        tx, ty = tx / np.maximum(tn, 1e-6), ty / np.maximum(tn, 1e-6)
        nx, ny = -ty, tx
        inside = ((px - search_px - 2 > 0) & (px + search_px + 2 < W) & (py - search_px - 2 > 0) & (py + search_px + 2 < H))
        sel = ok & inside
        if not sel.any():
            continue
        px, py, nx, ny = px[sel], py[sel], nx[sel], ny[sel]
        cx = px[:, None] + nx[:, None] * offs[None, :]
        cy = py[:, None] + ny[:, None] * offs[None, :]
        prof = map_coordinates(prob, [cy.ravel(), cx.ravel()], order=1, mode='nearest').reshape(cx.shape)
        peak = prof.max(axis=1)
        # Centroid of the mass above 0.3 (robust to plateaus), with a single mode:
        # the mass must be concentrated (standard deviation <= search / 4) and not touch the edges.
        w = np.clip(prof - 0.3, 0, None) ** 2
        sw = w.sum(axis=1)
        good = (peak >= min_peak) & (sw > 0)
        for i in np.nonzero(good)[0]:
            mu = float((w[i] * offs).sum() / sw[i])
            sd = float(np.sqrt((w[i] * (offs - mu) ** 2).sum() / sw[i]))
            edge = w[i][:4].sum() + w[i][-4:].sum()
            if sd > search_px / 4 or edge > 0.05 * sw[i]:
                continue
            out.append((px[i], py[i], nx[i], ny[i], mu, peak[i]))
    return np.asarray(out, float).reshape(-1, 6)


class OffsetField:
    """(dx, dy) field in ground metres, from observations in Mercator coordinates."""

    def __init__(self, x, y, nx, ny, d_m, w=None, group=None, sigma_m=3000.0, radius_m=9000.0):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.nx, self.ny = np.asarray(nx, float), np.asarray(ny, float)
        self.d = np.asarray(d_m, float)
        self.w = np.ones_like(self.d) if w is None else np.asarray(w, float)
        self.group = None if group is None else np.asarray(group)
        self.sigma, self.radius = sigma_m, radius_m
        self.tree = cKDTree(np.stack([self.x, self.y], 1)) if len(self.x) else None

    def __len__(self):
        return len(self.x)

    def at(self, x, y, merc_scale=1.0, exclude_group=None, min_obs=30, huber_m=6.0):
        """Returns (dx, dy, n_obs, sd_m) or None if poorly constrained. Mercator -> ground distances: / merc_scale."""
        if self.tree is None:
            return None
        idx = self.tree.query_ball_point([x, y], self.radius * merc_scale)
        if not idx:
            return None
        idx = np.asarray(idx)
        if exclude_group is not None and self.group is not None:
            idx = idx[self.group[idx] != exclude_group]
        if len(idx) < min_obs:
            return None
        dist = np.hypot(self.x[idx] - x, self.y[idx] - y) / merc_scale
        w = self.w[idx] * np.exp(-0.5 * (dist / self.sigma) ** 2)
        A = np.stack([self.nx[idx], self.ny[idx]], 1)
        b = self.d[idx]
        sol = np.zeros(2)
        for _ in range(5):
            r = A @ sol - b
            rw = w / np.maximum(1.0, np.abs(r) / huber_m)
            M = (A * rw[:, None]).T @ A
            if np.linalg.cond(M) > 25:  # orientations too uniform: tangential component unknown
                return None
            sol = np.linalg.solve(M, (A * rw[:, None]).T @ b)
        r = A @ sol - b
        sd = float(np.sqrt(np.average(np.minimum(r ** 2, (3 * huber_m) ** 2), weights=w)))
        return float(sol[0]), float(sol[1]), int(len(idx)), sd
