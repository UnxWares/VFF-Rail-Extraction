"""Local configuration: names of the trace sources in the database and working folder.

The "coarse" (imprecise) traces to correct carry, in the line_reference_parts table, the name of
their source. These names depend on the database: they are not hard-coded.

Order of precedence:
1. environment variables
   VFF_RAIL_COARSE_SOURCES="name:half-width_m,name:half-width_m,..."  (e.g. "imprecise:120,ohm:120,sncf:200")
   VFF_RAIL_WORK=/path/to/working/folder
2. JSON file (VFF_RAIL_CONFIG, otherwise rail.local.json at the repository root, unpublished):
   {"coarse_sources": {"imprecise": 120, "ohm": 120, "sncf": 200}, "work_dir": "/work"}
3. neutral defaults below.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

DEFAULT_COARSE = {'imprecise': 120.0, 'ohm': 120.0, 'sncf': 200.0}
DEFAULT_WORK = '/work'
LOCAL_FILE = Path(__file__).resolve().parents[1] / 'rail.local.json'
_NAME = re.compile(r'^[A-Za-z0-9_.-]+$')


def _file() -> dict:
    p = Path(os.environ.get('VFF_RAIL_CONFIG') or LOCAL_FILE)
    if p.is_file():
        return json.loads(p.read_text())
    return {}


def parse_sources(spec: str) -> dict[str, float]:
    """'a:120,b:200' -> {'a': 120.0, 'b': 200.0} (default half-width 150 m)."""
    out = {}
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        name, _, w = item.partition(':')
        out[name.strip()] = float(w) if w.strip() else 150.0
    return out


def coarse_sources() -> dict[str, float]:
    """Coarse trace sources -> base half-width of the corridor (m)."""
    env = os.environ.get('VFF_RAIL_COARSE_SOURCES')
    if env:
        src = parse_sources(env)
    else:
        src = {k: float(v) for k, v in (_file().get('coarse_sources') or DEFAULT_COARSE).items()}
    bad = [k for k in src if not _NAME.match(k)]
    if bad or not src:
        raise SystemExit(f'invalid source names: {bad or "none"}')
    return src


def coarse_sql_list() -> str:
    """Literal SQL list (validated names): ('a','b')."""
    return '(' + ','.join(f"'{k}'" for k in coarse_sources()) + ')'


def work_dir() -> Path:
    return Path(os.environ.get('VFF_RAIL_WORK') or _file().get('work_dir') or DEFAULT_WORK)
