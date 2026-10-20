"""PostgreSQL / PostGIS access (host network: 127.0.0.1:5432, credentials from the dev .env)."""

from __future__ import annotations

import os

import numpy as np
import shapely
from shapely import wkb


def connect():
    import psycopg

    return psycopg.connect(
        host=os.environ.get('DB_HOST', '127.0.0.1'), port=int(os.environ.get('DB_PORT', '5432')),
        dbname=os.environ.get('DB_DATABASE', 'vff'), user=os.environ.get('DB_USERNAME', 'vff'),
        password=os.environ.get('DB_PASSWORD', ''), autocommit=True,
    )


def env_sql(bbox) -> str:
    lon0, lat0, lon1, lat1 = bbox
    return f'ST_MakeEnvelope({lon0:.6f}, {lat0:.6f}, {lon1:.6f}, {lat1:.6f}, 4326)'


def geoms(rows, col: int):
    """Decodes a WKB column (bytes) into shapely geometries."""
    return [wkb.loads(bytes(r[col])) if r[col] is not None else None for r in rows]


def lines_of(g) -> list[np.ndarray]:
    """List of the LineStrings of any geometry, as (n, 2) arrays."""
    if g is None or g.is_empty:
        return []
    out = []
    for part in shapely.get_parts(g):
        if part.geom_type == 'LineString':
            out.append(np.asarray(part.coords)[:, :2])
        elif part.geom_type in ('MultiLineString', 'GeometryCollection'):
            out.extend(lines_of(part))
    return out
