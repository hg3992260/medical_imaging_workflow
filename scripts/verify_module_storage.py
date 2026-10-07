import os
import sqlite3
import sys
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "medical_imaging.db")
DB_PATH = os.path.abspath(DB_PATH)

def resolve_project(cur, arg):
    if not arg:
        cur.execute("SELECT project_id, name FROM projects ORDER BY created_at DESC LIMIT 1")
        r = cur.fetchone()
        return (r[0], r[1]) if r else (None, None)
    if arg.startswith("project_"):
        cur.execute("SELECT name FROM projects WHERE project_id = ? LIMIT 1", (arg,))
        r = cur.fetchone()
        return (arg, r[0] if r else arg)
    cur.execute("SELECT project_id, name FROM projects WHERE name = ? LIMIT 1", (arg,))
    r = cur.fetchone()
    if r:
        return (r[0], r[1])
    return (None, None)

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    pid, pname = resolve_project(cur, arg)
    if not pid:
        print(json.dumps({"error": "project not found"}, ensure_ascii=False))
        return
    # DICOM ROI识别
    cur.execute("SELECT COUNT(*) FROM dicom_sessions WHERE project_id=?", (pid,))
    dicom_sessions = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(*) 
        FROM roi_data rd JOIN dicom_sessions ds ON rd.session_id = ds.session_id
        WHERE ds.project_id = ?
    """, (pid,))
    roi_rows = cur.fetchone()[0]
    # Magic Seg
    cur.execute("""
        SELECT COUNT(*) 
        FROM roi_data rd JOIN dicom_sessions ds ON rd.session_id = ds.session_id
        WHERE ds.project_id = ? AND (rd.source = 'magic_seg_sam' OR rd.source = 'magic_seg_interactive')
    """, (pid,))
    magic_seg_rows = cur.fetchone()[0]
    # OCR识别
    cur.execute("SELECT COUNT(*) FROM ocr_sessions WHERE project_id=?", (pid,))
    ocr_sessions = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(*) 
        FROM ocr_results r JOIN ocr_sessions s ON r.session_id = s.session_id
        WHERE s.project_id = ?
    """, (pid,))
    ocr_results = cur.fetchone()[0]
    # 文本处理
    cur.execute("SELECT COUNT(*) FROM text_sessions WHERE project_id=?", (pid,))
    text_sessions = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(*) 
        FROM documents d JOIN text_sessions ts ON d.session_id = ts.session_id
        WHERE ts.project_id = ?
    """, (pid,))
    documents = cur.fetchone()[0]
    print(json.dumps({
        "project_id": pid,
        "project_name": pname,
        "dicom_sessions": dicom_sessions,
        "roi_rows": roi_rows,
        "magic_seg_rows": magic_seg_rows,
        "ocr_sessions": ocr_sessions,
        "ocr_results": ocr_results,
        "text_sessions": text_sessions,
        "documents": documents
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
