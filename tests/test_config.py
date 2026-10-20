"""Configuration of the source names (vffrail.config)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vffrail import config  # noqa: E402


class ConfigTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(config.parse_sources('a:120, b:200,c'), {'a': 120.0, 'b': 200.0, 'c': 150.0})

    def test_env_wins(self):
        with mock.patch.dict(os.environ, {'VFF_RAIL_COARSE_SOURCES': 'x:100,y:250'}):
            self.assertEqual(config.coarse_sources(), {'x': 100.0, 'y': 250.0})
            self.assertEqual(config.coarse_sql_list(), "('x','y')")

    def test_file_and_default(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'c.json'
            f.write_text(json.dumps({'coarse_sources': {'z': 90}, 'work_dir': d}))
            env = {'VFF_RAIL_CONFIG': str(f)}
            with mock.patch.dict(os.environ, env):
                os.environ.pop('VFF_RAIL_COARSE_SOURCES', None)
                os.environ.pop('VFF_RAIL_WORK', None)
                self.assertEqual(config.coarse_sources(), {'z': 90.0})
                self.assertEqual(config.work_dir(), Path(d))
            with mock.patch.dict(os.environ, {'VFF_RAIL_CONFIG': str(Path(d) / 'absent.json')}):
                os.environ.pop('VFF_RAIL_COARSE_SOURCES', None)
                os.environ.pop('VFF_RAIL_WORK', None)
                self.assertEqual(config.coarse_sources(), config.DEFAULT_COARSE)
                self.assertEqual(config.work_dir(), Path('/work'))

    def test_injection_refused(self):
        with mock.patch.dict(os.environ, {'VFF_RAIL_COARSE_SOURCES': "a');DROP TABLE x;--"}):
            with self.assertRaises(SystemExit):
                config.coarse_sources()


if __name__ == '__main__':
    unittest.main()
