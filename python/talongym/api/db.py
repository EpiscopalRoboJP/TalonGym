from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from talongym import paths
from talongym.presets.loader import (
    LoadedPresets,
    PresetError,
    list_presets,
    load_json,
    load_preset,
    preset_index,
    validate_document,
)

_LOCK = threading.Lock()
_CONN: Any = None
_BACKEND = "sqlite"


class _PgShim:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, sql: str, params: tuple = ()):
        q = sql.replace("?", "%s")
        return self._conn.execute(q, params)

    def executescript(self, script: str) -> None:
        self._conn.execute(script.replace("?", "%s"))

    def commit(self) -> None:
        self._conn.commit()


def db_path() -> Path:
    paths.VAR_DIR.mkdir(parents=True, exist_ok=True)
    return paths.VAR_DIR / "talongym.db"


def disconnect() -> None:
    """Close the process-global connection so tests can retarget VAR_DIR."""
    global _CONN, _BACKEND
    with _LOCK:
        conn = _CONN
        _CONN = None
        _BACKEND = "sqlite"
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        pass


def backend_name() -> str:
    return _BACKEND


def _ensure() -> Any:
    global _CONN, _BACKEND
    if _CONN is None:
        url = os.environ.get("TALONGYM_DATABASE_URL") or ""
        if url.startswith("postgres"):
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:
                raise RuntimeError("Postgres URL set; pip install -e '.[postgres]'") from exc
            raw = psycopg.connect(url, row_factory=dict_row)
            _CONN = _PgShim(raw)
            _BACKEND = "postgres"
        else:
            _CONN = sqlite3.connect(db_path(), check_same_thread=False)
            _CONN.row_factory = sqlite3.Row
            _CONN.execute("PRAGMA journal_mode=WAL")
            _BACKEND = "sqlite"
        _init(_CONN)
        seed_disk_presets(_CONN)
    return _CONN


def connect() -> Any:
    with _LOCK:
        return _ensure()


@contextmanager
def _session() -> Iterator[Any]:
    """Hold the lock for the whole statement: the connection is shared across request threads."""
    with _LOCK:
        yield _ensure()


def _init(conn: Any) -> None:
    script = """
        CREATE TABLE IF NOT EXISTS presets (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            document TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            config TEXT NOT NULL,
            state TEXT NOT NULL,
            metrics TEXT,
            log TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            report TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS replays (
            id TEXT PRIMARY KEY,
            meta TEXT NOT NULL,
            frames TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            state TEXT NOT NULL,
            payload TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS robot_drafts (
            id TEXT PRIMARY KEY,
            document TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    if _BACKEND == "postgres":
        for stmt in script.split(";"):
            if stmt.strip():
                conn.execute(stmt)
    else:
        conn.executescript(script)
    conn.commit()


def seed_disk_presets(conn: sqlite3.Connection) -> None:
    idx = preset_index(refresh=True)
    for kind, mapping in idx.items():
        for pid, path in mapping.items():
            doc = json.dumps(load_json(path))
            conn.execute(
                "INSERT INTO presets(id, kind, document) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET document=excluded.document, kind=excluded.kind",
                (pid, kind, doc),
            )
    conn.commit()


def get_preset(preset_id: str) -> dict[str, Any] | None:
    with _session() as conn:
        row = conn.execute("SELECT kind, document FROM presets WHERE id=?", (preset_id,)).fetchone()
    if not row:
        return None
    doc = json.loads(row["document"])
    doc["_kind"] = row["kind"]
    return doc


def load_runtime_document(kind: str, preset_id: str) -> dict[str, Any]:
    """Load a Lab-saved SQLite preset, falling back to a shipped disk file."""
    doc = get_preset(preset_id)
    if doc and doc.get("_kind") == kind:
        data = {key: value for key, value in doc.items() if key != "_kind"}
        errs = validate_document(kind, data)
        if errs:
            raise PresetError(f"{kind} preset {preset_id} invalid: " + "; ".join(errs[:8]))
        return data
    return load_preset(kind, preset_id)


def load_runtime_bundle(
    field_id: str | None = None,
    robot_id: str | None = None,
    scoring_id: str | None = None,
    training_id: str | None = None,
) -> LoadedPresets:
    """Resolve the field/robot/scoring/training bundle used by Lab jobs.

    Catalog robots saved from the builder live in SQLite, not on disk. Disk
    `load_bundle` cannot see them, so training would keep the shipped bot.
    """
    from talongym.presets.defaults import get_defaults

    defaults = get_defaults()
    field_id = field_id or defaults["fieldId"]
    robot_id = robot_id or defaults["robotId"]
    scoring_id = scoring_id or defaults["scoringId"]
    training_id = training_id or defaults.get("trainingId")
    field = load_runtime_document("field", field_id)
    robot = load_runtime_document("robot", robot_id)
    scoring = load_runtime_document("scoring", scoring_id)
    training = load_runtime_document("training", training_id) if training_id else None
    idx = preset_index()
    return LoadedPresets(
        field=field,
        robot=robot,
        scoring=scoring,
        training=training,
        field_path=idx["field"].get(field_id),
        robot_path=idx["robot"].get(robot_id),
        scoring_path=idx["scoring"].get(scoring_id),
    )


def upsert_preset(kind: str, document: dict[str, Any]) -> str:
    pid = document["id"]
    with _session() as conn:
        conn.execute(
            "INSERT INTO presets(id, kind, document) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET document=excluded.document, kind=excluded.kind, updated_at=CURRENT_TIMESTAMP",
            (pid, kind, json.dumps(document)),
        )
        conn.commit()
    return pid


def get_robot_draft(preset_id: str) -> dict[str, Any] | None:
    with _session() as conn:
        row = conn.execute("SELECT document FROM robot_drafts WHERE id=?", (preset_id,)).fetchone()
    if not row:
        return None
    return json.loads(row["document"])


def upsert_robot_draft(preset_id: str, document: dict[str, Any]) -> str:
    payload = dict(document)
    payload["id"] = preset_id
    with _session() as conn:
        conn.execute(
            "INSERT INTO robot_drafts(id, document) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET document=excluded.document, updated_at=CURRENT_TIMESTAMP",
            (preset_id, json.dumps(payload)),
        )
        conn.commit()
    return preset_id


def delete_robot_draft(preset_id: str) -> bool:
    with _session() as conn:
        cur = conn.execute("DELETE FROM robot_drafts WHERE id=?", (preset_id,))
        conn.commit()
        return cur.rowcount > 0


def delete_preset(preset_id: str) -> bool:
    with _session() as conn:
        used = conn.execute("SELECT id FROM runs WHERE config LIKE ?", (f"%{preset_id}%",)).fetchone()
        if used:
            return False
        cur = conn.execute("DELETE FROM presets WHERE id=?", (preset_id,))
        conn.commit()
        return cur.rowcount > 0


def list_kind(kind: str) -> list[dict[str, Any]]:
    with _session() as conn:
        rows = conn.execute("SELECT id, document FROM presets WHERE kind=?", (kind,)).fetchall()
    from talongym.presets.loader import LATEST_KNOWN_MANUAL

    disk_ids = set(preset_index().get(kind, {}))
    out = []
    for row in rows:
        doc = json.loads(row["document"])
        season = (doc.get("season") or {}).get("slug")
        revision = (doc.get("provenance") or {}).get("manualRevision")
        stale = bool(
            season and revision and LATEST_KNOWN_MANUAL.get(season) and revision != LATEST_KNOWN_MANUAL[season]
        )
        out.append(
            {
                "id": row["id"],
                "displayName": doc.get("displayName"),
                "season": season,
                "manualRevision": revision,
                "verifyAgainstManual": (doc.get("provenance") or {}).get("verifyAgainstManual", False),
                "stale": stale,
                "computeProfile": doc.get("computeProfile"),
                "shipped": row["id"] in disk_ids,
            }
        )
    if not out:
        return list_presets(kind)
    return out


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def save_replay(frames: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> str:
    rid = new_id()
    with _session() as conn:
        conn.execute(
            "INSERT INTO replays(id, meta, frames) VALUES(?,?,?)",
            (rid, json.dumps(meta or {}), json.dumps(frames)),
        )
        conn.commit()
    return rid


def get_replay(replay_id: str) -> dict[str, Any] | None:
    with _session() as conn:
        row = conn.execute("SELECT meta, frames FROM replays WHERE id=?", (replay_id,)).fetchone()
    if not row:
        return None
    return {"id": replay_id, "meta": json.loads(row["meta"]), "frames": json.loads(row["frames"])}


def list_replays() -> list[dict[str, Any]]:
    with _session() as conn:
        rows = conn.execute("SELECT id, meta FROM replays ORDER BY created_at DESC").fetchall()
    return [{"id": r["id"], **json.loads(r["meta"])} for r in rows]


RUN_NAME_MAX = 80


def clean_run_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())[:RUN_NAME_MAX]
    return text or None


def set_run_name(run_id: str, name: str | None) -> dict[str, Any] | None:
    row = get_run(run_id)
    if row is None:
        return None
    config = dict(row.get("config") or {})
    cleaned = clean_run_name(name)
    if cleaned:
        config["name"] = cleaned
    else:
        config.pop("name", None)
    with _session() as conn:
        conn.execute("UPDATE runs SET config=? WHERE id=?", (json.dumps(config), run_id))
        conn.commit()
    row["config"] = config
    return row


def save_run(run_id: str, config: dict, state: str, metrics: dict | None = None, log: str | None = None) -> None:
    with _session() as conn:
        if state == "running":
            row = conn.execute("SELECT state FROM runs WHERE id=?", (run_id,)).fetchone()
            if row and row["state"] == "cancelling":
                state = "cancelling"
        conn.execute(
            "INSERT INTO runs(id, config, state, metrics, log) VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state, metrics=excluded.metrics, log=excluded.log",
            (run_id, json.dumps(config), state, json.dumps(metrics or {}), log),
        )
        conn.commit()


def get_run(run_id: str) -> dict[str, Any] | None:
    with _session() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "config": json.loads(row["config"]),
        "state": row["state"],
        "metrics": json.loads(row["metrics"] or "{}"),
        "log": row["log"],
    }


def list_runs() -> list[dict[str, Any]]:
    with _session() as conn:
        rows = conn.execute("SELECT id, state, config, metrics FROM runs ORDER BY created_at DESC").fetchall()
    return [
        {
            "id": r["id"],
            "state": r["state"],
            "config": json.loads(r["config"]),
            "metrics": json.loads(r["metrics"] or "{}"),
        }
        for r in rows
    ]


def fail_orphan_runs() -> int:
    """In-process workers die with the Lab process; leftover rows must not block Start."""
    with _session() as conn:
        cur = conn.execute("UPDATE runs SET state='failed' WHERE state IN ('queued','running','cancelling')")
        conn.commit()
        return int(cur.rowcount or 0)


def save_evaluation(report: dict[str, Any], run_id: str | None = None) -> str:
    eid = new_id()
    slim = {k: v for k, v in report.items() if k != "bestFrames"}
    with _session() as conn:
        conn.execute(
            "INSERT INTO evaluations(id, run_id, report) VALUES(?,?,?)",
            (eid, run_id, json.dumps(slim)),
        )
        conn.commit()
    return eid


def get_evaluation(eid: str) -> dict[str, Any] | None:
    with _session() as conn:
        row = conn.execute("SELECT * FROM evaluations WHERE id=?", (eid,)).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "runId": row["run_id"],
        "createdAt": row["created_at"],
        "report": json.loads(row["report"]),
    }


def list_evaluations() -> list[dict[str, Any]]:
    with _session() as conn:
        rows = conn.execute(
            "SELECT id, run_id, report, created_at FROM evaluations ORDER BY created_at DESC"
        ).fetchall()
    return [
        {"id": r["id"], "runId": r["run_id"], "createdAt": r["created_at"], "report": json.loads(r["report"])}
        for r in rows
    ]


def save_artifact(run_id: str, kind: str, payload: str) -> str:
    aid = new_id()
    with _session() as conn:
        conn.execute(
            "INSERT INTO artifacts(id, run_id, kind, payload) VALUES(?,?,?,?)",
            (aid, run_id, kind, payload),
        )
        conn.commit()
    return aid


def list_artifacts(run_id: str) -> list[dict[str, Any]]:
    with _session() as conn:
        rows = conn.execute(
            "SELECT id, run_id, kind, payload FROM artifacts WHERE run_id=? ORDER BY id",
            (run_id,),
        ).fetchall()
    return [{"id": r["id"], "runId": r["run_id"], "kind": r["kind"], "payload": r["payload"]} for r in rows]


def enqueue_job(job_id: str, run_id: str, payload: dict[str, Any], state: str = "queued") -> None:
    """Mirror run state into the jobs table. Not a queue: no consumer reads these rows."""
    with _session() as conn:
        conn.execute(
            "INSERT INTO jobs(id, run_id, state, payload) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state, payload=excluded.payload",
            (job_id, run_id, state, json.dumps(payload)),
        )
        conn.commit()
