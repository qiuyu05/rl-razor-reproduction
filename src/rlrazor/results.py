"""Results store: one JSON file per (run_id, step), safe for parallel writers."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _row_path(results_dir, run_id: str, step) -> Path:
    if "/" in str(run_id):
        raise ValueError(f"run_id must not contain '/': {run_id!r}")
    return Path(results_dir) / f"{run_id}__step{step}.json"


def has_row(results_dir, run_id: str, step) -> bool:
    return _row_path(results_dir, run_id, step).exists()


def append_row(results_dir, row: dict) -> bool:
    path = _row_path(results_dir, row["run_id"], row["step"])
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(row, sort_keys=True))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return True


def load_rows(results_dir) -> list[dict]:
    d = Path(results_dir)
    if not d.is_dir():
        return []
    rows = [json.loads(p.read_text()) for p in d.glob("*.json")]
    return sorted(rows, key=lambda r: (r["run_id"], r["step"]))


def export_jsonl(results_dir, out_path) -> int:
    rows = load_rows(results_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return len(rows)


def get_row(results_dir, run_id: str, step) -> dict | None:
    path = _row_path(results_dir, run_id, step)
    return json.loads(path.read_text()) if path.exists() else None
