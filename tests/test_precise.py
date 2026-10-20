"""Tests without GPU or database of vffrail.precise / precise_write (synthetic probabilities).

    python -I -m unittest discover -s tests -p 'test_precise.py' -v
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import shapely  # noqa: E402
from shapely import LineString  # noqa: E402

from vffrail import precise as P  # noqa: E402
from vffrail import precise_write as W  # noqa: E402
from vffrail.geo import lonlat_to_merc, merc_scale, res  # noqa: E402

LAT = 48.0
S = merc_scale(LAT)
R = 20037508.342789244
X0, Y0 = (float(v) / S for v in lonlat_to_merc(-1.0, LAT))  # local ground frame


def synth_mosaic(tracks, half_w=7.0, size=900, noise=0.0, seed=0):
    """Synthetic z16 mosaic: probability 1 within half_w m of a track, decreasing over 3 m."""
    r = res(16)
    ox = int((X0 * S + R) / r) - size // 2
    oy = int((R - Y0 * S) / r) - size // 2
    cols, rows = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    gx = ((cols + ox) * r - R) / S
    gy = (R - (rows + oy) * r) / S
    pts = shapely.points(np.stack([gx.ravel(), gy.ravel()], 1))
    d = np.full(pts.shape, np.inf)
    for t in tracks:
        d = np.minimum(d, shapely.distance(pts, LineString(t)))
    p = np.clip(1.0 - (d - half_w) / 3.0, 0.0, 1.0).reshape(size, size)
    if noise:
        p = np.clip(p + np.random.default_rng(seed).normal(0, noise, p.shape), 0, 1)
    return P.Mosaic(None, None, LAT, arr=p.astype(np.float32), origin_px=(ox, oy))


def arc_branch(s_switch, radius, length, side=1.0):
    """Real branch: leaves tangent to the axis y = Y0 at x = X0 + s_switch, arc of radius `radius` then straight."""
    out = []
    th_max = 0.12  # ~ tan 1/8 at the arc exit
    for s in np.arange(0, length, 1.0):
        th = min(s / radius, th_max)
        if s / radius <= th_max:
            x = radius * math.sin(th)
            y = radius * (1 - math.cos(th))
        else:
            s_arc = th_max * radius
            x = radius * math.sin(th_max) + (s - s_arc) * math.cos(th_max)
            y = radius * (1 - math.cos(th_max)) + (s - s_arc) * math.sin(th_max)
        out.append((X0 + s_switch + x, Y0 + side * y))
    return np.asarray(out)


class PlateauTest(unittest.TestCase):
    def test_subpixel_centre(self):
        u = np.arange(-30, 30.01, 0.4)
        for c in (-2.3, 0.0, 1.3, 4.9):
            prof = np.clip(1.0 - (np.abs(u - c) - 7.0) / 3.0, 0, 1)[None]
            out = P.plateau_centres(prof, u, 8.0)
            self.assertAlmostEqual(out[0, 0], c, delta=0.25)
            self.assertAlmostEqual(out[0, 2], 2 * 8.5, delta=1.0)  # width at half height

    def test_open_plateau_rejected(self):
        u = np.arange(-10, 10.01, 0.4)
        prof = np.ones((1, len(u)))
        self.assertTrue(np.isnan(P.plateau_centres(prof, u, 8.0)[0, 0]))

    def test_low_probability_rejected(self):
        u = np.arange(-30, 30.01, 0.4)
        prof = 0.2 * np.clip(1.0 - (np.abs(u) - 7.0) / 3.0, 0, 1)[None]
        self.assertTrue(np.isnan(P.plateau_centres(prof, u, 8.0)[0, 0]))

    def test_other_plateau_detected(self):
        u = np.arange(-30, 30.01, 0.4)
        prof = np.maximum(np.clip(1.0 - (np.abs(u) - 4.0) / 2.0, 0, 1), np.clip(1.0 - (np.abs(u - 18) - 4.0) / 2.0, 0, 1))[None]
        c = P.plateau_centres(prof, u, 8.0)[:, 0]
        oth = P.other_plateaus(prof, u, c)
        self.assertEqual(len(oth[0]), 1)
        self.assertAlmostEqual(oth[0][0], 18.0, delta=0.5)


class FitTest(unittest.TestCase):
    def test_fit_recovers_offset_and_smooths(self):
        rng = np.random.default_rng(1)
        ref = np.stack([np.arange(0, 600, 2.0), np.zeros(300)], 1)
        obs = 2.0 + rng.normal(0, 0.8, 300)
        fit, off, _ = P.fit_rail(ref, obs, np.ones(300))
        self.assertLess(np.abs(off[20:-20] - 2.0).max(), 0.6)
        self.assertLess(P.max_turn_rate(fit), 1.0 / 500)  # 0.8 m noise: apparent radius > 500 m

    def test_fit_robust_to_outliers(self):
        ref = np.stack([np.arange(0, 400, 2.0), np.zeros(200)], 1)
        obs = np.full(200, 1.0)
        obs[90:95] = 15.0  # another track picked up 10 m away
        _, off, _ = P.fit_rail(ref, obs, np.ones(200))
        self.assertLess(abs(off[92] - 1.0), 2.5)

    def test_turn_rate_detects_kink(self):
        a = np.stack([np.arange(0, 200, 2.0), np.zeros(100)], 1)
        b = a[-1] + np.stack([np.arange(2, 200, 2.0), np.arange(2, 200, 2.0)], 1) / math.sqrt(2)
        self.assertGreater(P.max_turn_rate(np.vstack([a, b])), 1.0 / 60)
        self.assertLess(P.max_turn_rate(a), 1e-6)

    def test_min_radius_matches_engine(self):
        self.assertEqual(P.min_radius('principal', 1435), 190.0)
        self.assertEqual(P.min_radius('secondaire', None), 60.0)
        self.assertEqual(P.min_radius('principal', 1000), 60.0)
        self.assertEqual(P.min_radius('militaire', None), 25.0)
        self.assertEqual(P.min_radius(None, 600), 25.0)
        self.assertEqual(P.min_radius('secondaire', 1435), 150.0)


class SwitchTest(unittest.TestCase):
    def setUp(self):
        self.G = np.stack([X0 - 600 + np.arange(0, 1400, 5.0), np.full(280, Y0)], 1)
        self.B = arc_branch(0.0, 500.0, 500.0)
        self.mos = synth_mosaic([self.G, self.B])

    def test_leg_tangent_exact_on_partner(self):
        Bl = LineString(self.B)
        k = int(np.argmax(shapely.distance(shapely.points(self.B), LineString(self.G)) >= P.SEP_M + 2))
        Q = self.B[k]
        uQ = self.B[k - 1] - self.B[k]
        uQ /= np.hypot(*uQ)
        leg = P.switch_leg(LineString(self.G), Q, uQ, self.mos, np.zeros(2), 190.0)
        self.assertIsNotNone(leg)
        c = leg['curve']
        # absolute continuity: the first vertex IS a point of the partner track
        self.assertLess(LineString(self.G).distance(shapely.Point(c[0])), 1e-6)
        # tangent: starting heading parallel to the partner track (less than 0.5°)
        d0 = c[min(3, len(c) - 1)] - c[0]
        self.assertLess(abs(math.degrees(math.atan2(d0[1], d0[0]))), 0.5)
        # arrival at Q, tangent to the branch
        self.assertLess(np.hypot(*(c[-1] - Q)), 1e-6)
        # shape: within 1.5 m of the true branch (switch searched via the edge of the common plateau)
        self.assertLess(float(shapely.distance(shapely.points(c), shapely.union_all([Bl, LineString(self.G)])).max()), 1.0)
        self.assertLess(leg['e_model'], 2.0)
        self.assertLess(abs(leg['s_on_G'] - 600.0), 25.0)
        self.assertLessEqual(P.max_turn_rate(c, 20.0), 1.0 / 190.0 + 1e-9)

    def test_edge_obs_follow_branch(self):
        G = LineString(self.G)
        sig, w = P.edge_obs(self.mos, G, self.B[-1], np.zeros(2), 450, 900)
        true = shapely.distance(shapely.points([G.interpolate(x).coords[0] for x in sig]), LineString(self.B))
        ok = np.isfinite(w) & (true < 12) & (sig > 600)
        self.assertGreater(ok.sum(), 10)
        self.assertLess(np.median(np.abs(w[ok] - true[ok])), 1.0)
        before = np.isfinite(w) & (sig < 580)
        self.assertTrue((w[before] < 0.8).all())


class ProcessTest(unittest.TestCase):
    def test_branch_ends_on_partner(self):
        G = np.stack([X0 - 600 + np.arange(0, 1400, 5.0), np.full(280, Y0)], 1)
        B = arc_branch(0.0, 500.0, 900.0)
        mos = synth_mosaic([G, B], noise=0.05)
        feat = B + np.array([0.0, 1.2])  # extracted path offset by 1.2 m, stuck to the track near the switch
        coarse = B + np.array([0.0, 2.0])
        brs, _ = P.process_target({}, [feat], mos, [{'g': G, 'line_id': 2, 'part_id': 20, 'src': 'bdtopo'}], [], 190.0,
                                  truth_g=[B], other_g=[G], coarse_g=coarse)
        self.assertEqual(len(brs), 1)
        br = brs[0]
        self.assertTrue(any(e['leg'] is not None for e in br['ends']))
        geom = P.assemble(br)
        self.assertLess(LineString(G).distance(shapely.Point(geom[0])), 1e-6)  # shared vertex
        self.assertLess(np.nanpercentile(br['err'], 90), 1.0)  # sub-pixel axis: the 1.2 m offset is corrected
        # no bend over the whole assembled geometry
        self.assertLessEqual(P.max_turn_rate(geom, 20.0), 1.0 / 190.0 * 1.2)


class WriteTest(unittest.TestCase):
    def test_runs_share_vertices(self):
        c = np.stack([np.arange(10.0), np.zeros(10)], 1)
        lab = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1], bool)
        runs = W.runs_of(c, lab)
        self.assertEqual([r[0] for r in runs], [True, False, True])
        for (_, a), (_, b) in zip(runs, runs[1:]):
            self.assertTrue(np.array_equal(a[-1], b[0]))  # continuity: shared vertex
        self.assertTrue(np.array_equal(runs[0][1][0], c[0]) and np.array_equal(runs[-1][1][-1], c[-1]))

    def test_axis_filter(self):
        rows = np.zeros((4, 17))
        rows[:, 8] = [1, 5, 1, 1]       # dco
        rows[:, 5] = [1, 1, 0.5, 1]     # coverage
        rows[:, 13] = [0.9, 0.9, 0.9, 0.9]
        rows[:, 9] = [15, 15, 15, 30]   # width
        self.assertEqual(W.axis_precise(rows).tolist(), [True, False, False, False])

    def test_write_refused(self):
        with self.assertRaises(SystemExit):
            W.main(['--stations', 'x', '--branches', 'y', '--out', str(Path(tempfile.gettempdir()) / 'vffrail-none'), '--write'])


if __name__ == '__main__':
    unittest.main()
