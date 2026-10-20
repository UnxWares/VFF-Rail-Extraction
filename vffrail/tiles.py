"""Disk cache and POLITE download of IGN WMTS tiles (Geoplateforme, no key needed).

- User-Agent "VFF-Bot/1.0 (+https://vff-project.org)";
- only a few simultaneous connections (4 by default) and a capped rate;
- growing global pause on 429 / 5xx;
- disk cache: a tile is never downloaded twice (including tiles outside
  coverage, remembered by a ".none" file).

Licence of the basemaps: Licence Ouverte 2.0 (Etalab) - credit "IGN - <layer>".
"""

from __future__ import annotations

import io
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

UA = 'VFF-Bot/1.0 (+https://vff-project.org)'
WMTS = 'https://data.geopf.fr/wmts'


@dataclass(frozen=True)
class Layer:
    key: str          # short name (folders, source_traces.layer)
    layer: str        # WMTS identifier
    style: str
    fmt: str          # image/jpeg | image/png
    zoom: int         # working level
    years: str        # period represented
    label: str        # provenance credit
    recalage: bool = False  # georeferencing registration field (maps); useless and harmful on orthophotos

    @property
    def ext(self) -> str:
        return 'jpg' if self.fmt == 'image/jpeg' else 'png'


LAYERS = {
    'scan50-1950': Layer('scan50-1950', 'GEOGRAPHICALGRIDSYSTEMS.MAPS.SCAN50.1950', 'SCAN50_1950', 'image/jpeg', 15,
                         '1920-1960', 'IGN - SCAN 50 historique (1950)', True),
    'ortho-1950': Layer('ortho-1950', 'ORTHOIMAGERY.ORTHOPHOTOS.1950-1965', 'BDORTHOHISTORIQUE', 'image/png', 16,
                        '1950-1965', 'IGN - BD ORTHO historique 1950-1965'),
    'ortho-1965': Layer('ortho-1965', 'ORTHOIMAGERY.ORTHOPHOTOS.1965-1980', 'BDORTHOHISTORIQUE', 'image/png', 16,
                        '1965-1980', 'IGN - BD ORTHO historique 1965-1980'),
    'ortho-1980': Layer('ortho-1980', 'ORTHOIMAGERY.ORTHOPHOTOS.1980-1995', 'BDORTHOHISTORIQUE', 'image/png', 16,
                        '1980-1995', 'IGN - BD ORTHO historique 1980-1995'),
    'etatmajor40': Layer('etatmajor40', 'GEOGRAPHICALGRIDSYSTEMS.ETATMAJOR40', 'normal', 'image/jpeg', 15,
                         '1820-1866', 'IGN - Carte de l\'état-major (1820-1866)', True),
}


class TileCache:
    def __init__(self, root: str | Path, layer: Layer):
        self.root = Path(root) / layer.key
        self.layer = layer

    def path(self, z: int, x: int, y: int) -> Path:
        return self.root / str(z) / str(x) / f'{y}.{self.layer.ext}'

    def none_path(self, z: int, x: int, y: int) -> Path:
        return self.root / str(z) / str(x) / f'{y}.none'

    def has(self, z, x, y) -> bool:
        return self.path(z, x, y).exists() or self.none_path(z, x, y).exists()

    def read(self, z: int, x: int, y: int) -> np.ndarray | None:
        p = self.path(z, x, y)
        if not p.exists():
            return None
        try:
            im = Image.open(p)
            im = im.convert('RGB')
            return np.asarray(im)
        except Exception:
            return None

    def mosaic(self, z: int, x0: int, y0: int, nx: int, ny: int):
        """Assembles nx × ny tiles; returns (uint8 HxWx3 image, HxW validity mask)."""
        img = np.zeros((ny * 256, nx * 256, 3), np.uint8)
        valid = np.zeros((ny * 256, nx * 256), bool)
        for j in range(ny):
            for i in range(nx):
                t = self.read(z, x0 + i, y0 + j)
                if t is not None and t.shape[:2] == (256, 256):
                    img[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = t
                    valid[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = True
        return img, valid


class PoliteFetcher:
    """Rate-limited parallel download, with a global pause on 429 / 5xx."""

    def __init__(self, cache: TileCache, workers: int = 4, max_rps: float = 8.0, log=print):
        import requests  # imported here: only the download needs it

        self.requests = requests
        self.cache = cache
        self.workers = workers
        self.min_interval = 1.0 / max_rps
        self.lock = threading.Lock()
        self.next_slot = 0.0
        self.pause_until = 0.0
        self.penalty = 0
        self.local = threading.local()
        self.log = log
        self.stats = {'ok': 0, 'none': 0, 'err': 0, 'bytes': 0, 'retries': 0}

    def session(self):
        s = getattr(self.local, 's', None)
        if s is None:
            s = self.requests.Session()
            s.headers['User-Agent'] = UA
            self.local.s = s
        return s

    def _wait_slot(self):
        while True:
            with self.lock:
                now = time.monotonic()
                start = max(now, self.next_slot, self.pause_until)
                self.next_slot = start + self.min_interval
            delay = start - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            if time.monotonic() >= self.pause_until:
                return

    def _backoff(self):
        with self.lock:
            self.penalty = min(self.penalty + 1, 6)
            wait = min(5 * 2 ** (self.penalty - 1), 120) + random.random()
            self.pause_until = max(self.pause_until, time.monotonic() + wait)
            self.stats['retries'] += 1
        self.log(f'  pause {wait:.0f} s (server error, penalty {self.penalty})')

    def fetch_one(self, z: int, x: int, y: int) -> str:
        if self.cache.has(z, x, y):
            return 'cached'
        L = self.cache.layer
        params = {'SERVICE': 'WMTS', 'REQUEST': 'GetTile', 'VERSION': '1.0.0', 'LAYER': L.layer, 'STYLE': L.style,
                  'TILEMATRIXSET': 'PM', 'TILEMATRIX': str(z), 'TILEROW': str(y), 'TILECOL': str(x), 'FORMAT': L.fmt}
        for attempt in range(8):
            self._wait_slot()
            try:
                r = self.session().get(WMTS, params=params, timeout=60)
            except Exception:
                self._backoff()
                continue
            if r.status_code == 200 and r.headers.get('content-type', '').startswith('image/'):
                p = self.cache.path(z, x, y)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix('.part')
                tmp.write_bytes(r.content)
                os.replace(tmp, p)
                with self.lock:
                    self.stats['ok'] += 1
                    self.stats['bytes'] += len(r.content)
                    self.penalty = max(0, self.penalty - 1) if self.stats['ok'] % 50 == 0 else self.penalty
                return 'ok'
            if r.status_code in (400, 404) or (r.status_code == 200):
                # Outside coverage (or XML exception response): remembered so as not to ask again.
                p = self.cache.none_path(z, x, y)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(str(r.status_code))
                with self.lock:
                    self.stats['none'] += 1
                return 'none'
            if r.status_code == 429 or r.status_code >= 500:
                self._backoff()
                continue
            break
        with self.lock:
            self.stats['err'] += 1
        return 'err'

    def fetch_all(self, tiles, z: int, progress_every: int = 500):
        todo = [(x, y) for x, y in tiles if not self.cache.has(z, x, y)]
        self.log(f'  {len(tiles)} tiles requested, {len(tiles) - len(todo)} already cached, {len(todo)} to download')
        t0 = time.monotonic()
        done = 0
        with ThreadPoolExecutor(self.workers) as ex:
            futs = [ex.submit(self.fetch_one, z, x, y) for x, y in todo]
            for f in as_completed(futs):
                f.result()
                done += 1
                if done % progress_every == 0:
                    el = time.monotonic() - t0
                    self.log(f'  {done}/{len(todo)} ({done / el:.1f} t/s, {self.stats["bytes"] / 1e6:.0f} MB, '
                             f'{self.stats["none"]} outside coverage, {self.stats["err"]} errors)')
        return self.stats


def blank_ratio(img: np.ndarray) -> float:
    """Share of pure white/black pixels (coverage edge)."""
    g = img.mean(axis=2)
    return float(((g > 250) | (g < 3)).mean())


def to_png_bytes(arr: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format='PNG')
    return b.getvalue()
