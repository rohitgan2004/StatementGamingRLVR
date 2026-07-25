"""Small IO helpers: JSONL append/read and directory helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)
