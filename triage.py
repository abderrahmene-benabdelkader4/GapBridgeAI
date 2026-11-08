"""Researcher decisions on gaps (keep / dismiss / later + notes), stored per project in SQLite."""
import sqlite3, time
import pipeline
def _db(project):
    con = sqlite3.connect(pipeline.pdir(project) / "triage.sqlite"); con.execute("CREATE TABLE IF NOT EXISTS decisions (key TEXT PRIMARY KEY, status TEXT, note TEXT, updated TEXT)"); return con
def get_all(project):
    con = _db(project)
    try: return {k: dict(status=s, note=n or "") for k, s, n in con.execute("SELECT key, status, note FROM decisions")}
    finally: con.close()
def set_(project, key, status=None, note=None):
    con = _db(project)
    try:
        cur = con.execute("SELECT status, note FROM decisions WHERE key=?", (key,)).fetchone() or ("new", "")
        con.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?)", (key, status if status is not None else cur[0], note if note is not None else cur[1], time.strftime("%Y-%m-%d %H:%M:%S"))); con.commit()
    finally: con.close()
