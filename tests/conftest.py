"""Fixtures compartidas para la suite de pruebas."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def temp_json(tmp_path):
    import json

    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    return write
