"""Tests without GPU or database: corridors, 15 m cut, reconciliation, extraction on synthetic probabilities.

    ./run.sh python -I /app/tests/test_junctions.py -v    (VFF_RAIL_NO_GPU=1)
"""
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import shapely  # noqa: E402
from PIL import Image  # noqa: E402
from shapely import LineString  # noqa: E402

from vffrail import snap  # noqa: E402
from vffrail.geo import TILE, lonlat_to_merc, merc_scale, merc_to_lonlat, merc_to_px, px_to_merc  # noqa: E402
from vffrail.junctions import _merge  # noqa: E402

LAT = 48.0
S = merc_scale(LAT)          # Mercator m per ground m
X0, Y0 = (float(v) for v in lonlat_to_merc(-1.0, LAT))


def M(pts_m):
    """Points in ground metres around (X0, Y0) -> Mercator."""
    a = np.asarray(pts_m, float)
    return np.stack([X0 + a[:, 0] * S, Y0 + a[:, 1] * S], 1)


def dist_m(p, line_pts_m):
    return LineString(line_pts_m).distance(shapely.Point(p))


class MergeIntervals(unittest.TestCase):
    def test_overlap_and_gap(self):
        out = _merge([(0, 1500, [('a', 0)]), (1000, 2500, [('b', 1700)]), (2600, 3000, [('c', 2800)]), (4000, 4500, [('d', 4200)])])
        self.assertEqual([(a, b) for a, b, _ in out], [(0, 2500), (2600, 3000), (4000, 4500)])
        self.assertEqual(sorted(j for j, _ in out[0][2]), ['a', 'b'])
        out = _merge([(0, 1500, [('a', 0)]), (1540, 2000, [('b', 1700)])])   # gap <= 50 m: merged
        self.assertEqual([(a, b) for a, b, _ in out], [(0, 2000)])


class Heading(unittest.TestCase):
    def test_cardinal(self):
        self.assertAlmostEqual(snap.heading((0, 0), (0, 1)), 0.0)
        self.assertAlmostEqual(snap.heading((0, 0), (1, 0)), 90.0)
        self.assertAlmostEqual(snap.heading((0, 0), (0, -1)), 180.0)
        self.assertAlmostEqual(snap.heading((0, 0), (-1, 0)), 270.0)


class CutAtPartners(unittest.TestCase):
    partner_m = [(-2000, 0), (2000, 0)]          # precise track E-W

    def partners(self, line_id=9):
        return [{'line_id': line_id, 'src': 'bdtopo', 'part_id': 1, 'coords': M(self.partner_m).tolist()}]

    def test_branch_cut_at_15m(self):
        # branch descending at an angle (~10°) towards the track and joining it at x = 0
        br = [(-1000 * math.cos(math.radians(10)), 1000 * math.sin(math.radians(10))), (0, 0)]
        out = snap.cut_at_partners(M(br), S, self.partners(), [M([(0, 0)])[0]])
        self.assertEqual(len(out), 1)
        c, rac = out[0]
        tip = (np.asarray(c[-1]) - (X0, Y0)) / S
        self.assertLessEqual(dist_m(tip, self.partner_m), 15.0)               # stops at the first approach within <= 15 m
        self.assertGreater(dist_m(tip, self.partner_m), 14.0)
        self.assertEqual(len(rac), 1)
        r = rac[0]
        self.assertEqual(r['bout'], 'fin')
        self.assertLessEqual(r['raccord_ecart_m'], 15.0)
        self.assertAlmostEqual(r['raccord_ecart_m'], dist_m(tip, self.partner_m), delta=0.1)
        self.assertAlmostEqual(r['cap_branche_deg'], 100.0, delta=1.5)        # towards east-south-east
        self.assertAlmostEqual(r['cap_voie_deg'], 90.0, delta=0.5)           # track oriented like the branch
        self.assertEqual(r['partner_line_id'], 9)
        rac_m = (np.asarray(r['raccord_merc']) - (X0, Y0)) / S
        self.assertAlmostEqual(rac_m[1], 0.0, delta=0.2)                      # connection ON the track
        self.assertLess(r['jonction_dist_m'], 100)

    def test_parallel_within_tol_dropped(self):
        out = snap.cut_at_partners(M([(-800, 10), (800, 10)]), S, self.partners(), [])
        self.assertEqual(out, [])

    def test_far_branch_kept_without_raccord(self):
        out = snap.cut_at_partners(M([(-800, 60), (800, 60)]), S, self.partners(), [])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1], [])

    def test_short_free_piece_without_raccord_dropped(self):
        self.assertEqual(snap.cut_at_partners(M([(0, 60), (150, 60)]), S, self.partners(), []), [])

    def test_short_raccord_piece_min_50m(self):
        # 40 m free before the track: refused; 80 m: kept
        self.assertEqual(snap.cut_at_partners(M([(0, 55), (0, 0)]), S, self.partners(), []), [])
        out = snap.cut_at_partners(M([(0, 95), (0, 0)]), S, self.partners(), [])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1][0]['bout'], 'fin')

    def test_crossing_gives_two_pieces_with_raccords(self):
        out = snap.cut_at_partners(M([(0, 500), (0, -500)]), S, self.partners(), [])
        self.assertEqual(len(out), 2)
        self.assertEqual({r['bout'] for _, rac in out for r in rac}, {'debut', 'fin'})

    def test_no_partner_keeps_piece(self):
        out = snap.cut_at_partners(M([(0, 0), (500, 0)]), S, [], [])
        self.assertEqual(len(out), 1)


class PartnersFor(unittest.TestCase):
    def test_selection(self):
        prec = [{'line_id': 1, 'src': 'osm', 'coords': [[0, 0], [1, 1]]}, {'line_id': 2, 'src': 'osm', 'coords': [[0, 0], [1, 1]]},
                {'line_id': 3, 'src': 'osm', 'coords': [[0, 0], [1, 1]]}]
        t = {'line_id': 1, 'precise': prec, 'junctions': [{'kind': 'extremite', 'relation': 'autre-ligne', 'partners': [2]}]}
        self.assertEqual([p['line_id'] for p in snap.partners_for(t)], [2])
        t['junctions'].append({'kind': 'extremite', 'relation': 'meme-ligne', 'partners': []})
        self.assertEqual([p['line_id'] for p in snap.partners_for(t)], [1, 2])
        t['junctions'] = [{'kind': 'detecteur', 'relation': 'V1', 'partners': []}]
        self.assertEqual(len(snap.partners_for(t)), 3)


class MergeWithExisting(unittest.TestCase):
    def test_replace_when_covered(self):
        drop, rest = snap.merge_with_existing([M([(0, 0), (1000, 0)])], [M([(100, 2), (900, 2)])], S)
        self.assertEqual(drop, [0])
        self.assertEqual(len(rest[0]), 1)

    def test_trim_when_partial(self):
        drop, rest = snap.merge_with_existing([M([(0, 0), (1000, 0)])], [M([(-3000, 1), (300, 1)])], S)
        self.assertEqual(drop, [])
        self.assertEqual(len(rest[0]), 1)
        L = LineString(rest[0][0]).length / S
        self.assertAlmostEqual(L, 1000 - 308, delta=3)

    def test_no_existing(self):
        drop, rest = snap.merge_with_existing([M([(0, 0), (1000, 0)])], [], S)
        self.assertEqual((drop, len(rest[0])), ([], 1))


class ProcessTargetSynthetic(unittest.TestCase):
    """End-to-end on CPU: synthetic probabilities, junction corridor, precise track of another line blocked
    except at the junction point, then cut at 15 m."""

    def test_extract_and_cut(self):
        from vffrail import extract

        z = 16
        true_m = np.array([(-1400, 300), (-700, 150), (0, 0)], float)        # real branch -> junction at (0, 0)
        trunk_m = np.array([(-2500, 0), (1500, 0)], float)                   # precise track of line 9
        coarse_m = np.array([(-1400, 325), (0, 22)], float)                  # coarse trace: chord offset by ~20 m
        tmp = Path(tempfile.mkdtemp())
        allp = np.concatenate([M(true_m), M(trunk_m)])
        px, py = merc_to_px(allp[:, 0], allp[:, 1], z)
        tx0, tx1 = int(px.min() // TILE) - 2, int(px.max() // TILE) + 2
        ty0, ty1 = int(py.min() // TILE) - 2, int(py.max() // TILE) + 2
        from PIL import ImageDraw
        W, H = (tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE
        im = Image.new('L', (W, H), 5)
        dr = ImageDraw.Draw(im)
        for line in (true_m, trunk_m):
            a, b = merc_to_px(M(line)[:, 0], M(line)[:, 1], z)
            dr.line([(x - tx0 * TILE, y - ty0 * TILE) for x, y in zip(a, b)], fill=240, width=3)
        arr = np.asarray(im)
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                p = extract.prob_path(tmp, z, tx, ty)
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE, (tx - tx0) * TILE:(tx - tx0 + 1) * TILE]).save(p)
        blockers = [(9, M(trunk_m))]
        extract._init_worker(tmp, z, blockers, np.zeros((0, 9)), False, False)
        junction = M([(0, 0)])[0].tolist()
        t = {'tid': 'j1-1-0', 'part_id': 1, 'line_id': 5, 'source': next(iter(extract.BASE_W)), 'coarse': M(coarse_m),
             'release': [M([(0, 22)])[0].tolist()], 'truth': None,
             'junctions': [{'jid': 'c1', 'kind': 'extremite', 'relation': 'autre-ligne', 'partners': [9], 'pt': junction}],
             'precise': [{'line_id': 9, 'src': 'bdtopo', 'part_id': 2, 'coords': M(trunk_m).tolist()}]}
        st, feats = extract._process_target(t)
        self.assertTrue(feats, st)
        truth = LineString(true_m)
        for f in feats:
            ll = np.asarray(f['coords'])
            mx, my = lonlat_to_merc(ll[:, 0], ll[:, 1])
            pts = np.stack([(mx - X0) / S, (my - Y0) / S], 1)
            d = shapely.distance(shapely.points(pts), truth)
            self.assertLess(float(np.median(d)), 3.0)                       # follows the real track, not the chord
            # cut at 15 m from the precise track
            out = snap.cut_at_partners(np.stack([mx, my], 1), S, snap.partners_for(t), [junction])
            self.assertTrue(out)
            c, rac = max(out, key=lambda o: len(o[0]))
            self.assertTrue(rac)
            tip = (np.asarray(c[-1 if rac[0]['bout'] == 'fin' else 0]) - (X0, Y0)) / S
            self.assertLessEqual(abs(tip[1]), 15.0 + 0.5)
            self.assertGreater(abs(tip[1]), 12.0)
            self.assertLess(abs(tip[0]), 200)                               # near the real junction


if __name__ == '__main__':
    unittest.main()
