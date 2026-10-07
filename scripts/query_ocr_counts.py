import os
import sqlite3
import sys
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "medical_imaging.db")
DB_PATH = os.path.abspath(DB_PATH)

def resolve_project_id(cur, arg):
    if not arg:
        return None
    # If looks like id, return directly
    if arg.startswith("project_"):
        return arg
    # Else try by name
    cur.execute("SELECT project_id FROM projects WHERE name = ? LIMIT 1", (arg,))
    row = cur.fetchone()
    return row[0] if row else None

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    pid = resolve_project_id(cur, arg)
    if not pid:
        # fallback to latest project
        cur.execute("SELECT project_id, name FROM projects ORDER BY created_at DESC LIMIT 1")
        r = cur.fetchone()
        if r:
            pid = r["project_id"]
            pname = r["name"]
        else:
            print(json.dumps({"error": "no projects found"}, ensure_ascii=False))
            return
    else:
        cur.execute("SELECT name FROM projects WHERE project_id = ?", (pid,))
        r = cur.fetchone()
        pname = r["name"] if r else pid
    # counts
    cur.execute("SELECT COUNT(*) FROM ocr_sessions WHERE project_id = ?", (pid,))
    ocr_sessions = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(*) 
        FROM ocr_results r
        JOIN ocr_sessions s ON r.session_id = s.session_id
        WHERE s.project_id = ?
    """, (pid,))
    ocr_results_rows = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(DISTINCT r.session_id) 
        FROM ocr_results r
        JOIN ocr_sessions s ON r.session_id = s.session_id
        WHERE s.project_id = ?
    """, (pid,))
    sessions_with_results = cur.fetchone()[0]
    print(json.dumps({
        "project_id": pid,
        "project_name": pname,
        "ocr_sessions": ocr_sessions,
        "ocr_results_rows": ocr_results_rows,
        "sessions_with_results": sessions_with_results
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
