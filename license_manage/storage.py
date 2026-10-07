import sqlite3
import time
import os
import sys
from pathlib import Path

def _db_path():
    base = Path(__file__).resolve().parent
    if getattr(sys, "frozen", False):
        appdata = Path(os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(Path.home()))
        target = appdata / "RSNA" / "LicenseManager"
        target.mkdir(parents=True, exist_ok=True)
        return target / "licenses.db"
    return base / "licenses.db"

DB_PATH = _db_path()

def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS licenses (id INTEGER PRIMARY KEY AUTOINCREMENT, cpu_id TEXT, expires_ts INTEGER, created_ts INTEGER, license TEXT)")
    conn.commit()
    conn.close()

def add_record(cpu_id, expires_ts, license_text):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("INSERT INTO licenses (cpu_id, expires_ts, created_ts, license) VALUES (?, ?, ?, ?)", (cpu_id, int(expires_ts), int(time.time()), license_text))
    conn.commit()
    conn.close()

def list_records(limit=100):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT id, cpu_id, expires_ts, created_ts, license FROM licenses ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows
