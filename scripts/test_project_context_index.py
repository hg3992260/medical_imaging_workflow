import json
import os
import sqlite3
import uuid
from pathlib import Path

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.knowledge_base import KnowledgeBase
from src.ai.project_context_indexer import ProjectContextIndexer
from src.core.app_config import AppConfig
from src.services.database_service import DatabaseService


def main():
    test_db = os.path.abspath("database/_tmp_rag_index.db")
    Path(os.path.dirname(test_db)).mkdir(parents=True, exist_ok=True)
    if os.path.exists(test_db):
        os.remove(test_db)

    cfg = AppConfig()
    cfg.set("database.path", test_db)
    db = DatabaseService(cfg)
    db.initialize_database()

    conn = sqlite3.connect(test_db)
    cur = conn.cursor()

    cur.execute("INSERT OR IGNORE INTO projects (project_id, name) VALUES (?,?)", ("p1", "P1"))
    cur.execute(
        "INSERT OR IGNORE INTO user_ids (user_id, display_name, project_id, is_active) VALUES (?,?,?,1)",
        ("u1", "U1", "p1"),
    )
    cur.execute(
        "INSERT INTO text_sessions (session_id, user_id, project_id, operation_type, file_path, char_count, word_count, line_count) VALUES (?,?,?,?,?,?,?,?)",
        ("ts1", "u1", "p1", "text_processing", "/f", 0, 0, 0),
    )
    cur.execute(
        "INSERT INTO documents (doc_id, session_id, file_path, file_name, format, content, analysis_result, keywords, analysis_file_path) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            uuid.uuid4().hex,
            "ts1",
            "/f",
            "a.txt",
            ".txt",
            "patient has lung nodule. follow-up recommended.",
            json.dumps({"k": "v"}, ensure_ascii=False),
            json.dumps(["lung", "nodule"], ensure_ascii=False),
            "/ana",
        ),
    )
    cur.execute(
        "INSERT INTO ocr_sessions (session_id, user_id, project_id, image_path) VALUES (?,?,?,?)",
        ("os1", "u1", "p1", "/img"),
    )
    cur.execute(
        "INSERT INTO ocr_results (result_id, session_id, recognized_text, confidence, bounding_boxes, processing_time) VALUES (?,?,?,?,?,?)",
        (
            uuid.uuid4().hex,
            "os1",
            "CT shows consolidation in RLL.",
            0.9,
            "[]",
            0.1,
        ),
    )
    conn.commit()
    conn.close()

    kb = KnowledgeBase()
    idx = ProjectContextIndexer(db, kb)
    res = idx.sync_project("p1", force=True)
    print("indexed", res)

    ctx = kb.query_project("lung nodule consolidation", project_id="p1", limit=5)
    print("query_result_len", len(ctx))
    print(ctx[:600])

    os.remove(test_db)


if __name__ == "__main__":
    main()
