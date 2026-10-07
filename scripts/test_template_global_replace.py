import json
import os
import sqlite3
import sys
import tempfile

import lancedb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.knowledge_base import KnowledgeBase
from ui.widgets.ai_summary_widget import AISummaryWidget


class _DBService:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_connection(self):
        return sqlite3.connect(self.db_path)


class _Label:
    def __init__(self):
        self.value = ""

    def setText(self, text: str):
        self.value = text


def _build_projects_db() -> str:
    fd, db_path = tempfile.mkstemp(prefix="template_global_", suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE projects (project_id TEXT PRIMARY KEY, metadata TEXT, updated_at TEXT)")
    cur.executemany(
        "INSERT INTO projects(project_id, metadata, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
        [
            ("p1", json.dumps({"draft_template": {"template_name": "OldA", "constraints": {"a": 1}}}, ensure_ascii=False),),
            ("p2", json.dumps({"draft_template": {"template_name": "OldB", "constraints": {"b": 1}}}, ensure_ascii=False),),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


def test_save_updates_all_projects():
    db_path = _build_projects_db()
    holder = type("Holder", (), {})()
    holder.db_service = _DBService(db_path)
    holder.current_project_id = "p1"
    holder.selected_template_name = "GlobalTemplate"
    holder.selected_template_constraints = {"source": "pdf_template", "version": 2}
    holder.log = lambda message: None

    AISummaryWidget._save_template_to_project_metadata(holder)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT project_id, metadata FROM projects ORDER BY project_id")
    rows = cur.fetchall()
    conn.close()

    assert len(rows) == 2
    for _, metadata in rows:
        meta = json.loads(metadata)
        assert meta["draft_template"]["template_name"] == "GlobalTemplate"
        assert meta["draft_template"]["constraints"]["version"] == 2


def test_load_prefers_global_cache():
    db_path = _build_projects_db()
    holder = type("Holder", (), {})()
    holder.db_service = _DBService(db_path)
    holder.current_project_id = "p2"
    holder.selected_template_name = ""
    holder.selected_template_constraints = {}
    holder.selected_template_constraints_json = ""
    holder.template_label = _Label()
    holder._load_global_template_cache = lambda: {
        "template_name": "NewestGlobal",
        "constraints": {"source": "pdf_template", "global": True},
    }

    AISummaryWidget._load_template_from_project_metadata(holder)

    assert holder.selected_template_name == "NewestGlobal"
    assert holder.selected_template_constraints["global"] is True
    assert "(全局)" in holder.template_label.value


def test_clear_all_template_pdf_vectors():
    tmp_dir = tempfile.mkdtemp(prefix="template_lancedb_")
    db = lancedb.connect(tmp_dir)
    records = [
        {
            "id": "a",
            "project_id": "p1",
            "user_id": "u",
            "source_type": "template_pdf",
            "source_id": "t1",
            "chunk_index": 0,
            "content_hash": "h1",
            "updated_at": "now",
            "text": "template-a",
            "vector": [0.1, 0.2],
        },
        {
            "id": "b",
            "project_id": "p2",
            "user_id": "u",
            "source_type": "template_pdf",
            "source_id": "t2",
            "chunk_index": 0,
            "content_hash": "h2",
            "updated_at": "now",
            "text": "template-b",
            "vector": [0.3, 0.4],
        },
        {
            "id": "c",
            "project_id": "p2",
            "user_id": "u",
            "source_type": "project_doc",
            "source_id": "doc",
            "chunk_index": 0,
            "content_hash": "h3",
            "updated_at": "now",
            "text": "project-doc",
            "vector": [0.5, 0.6],
        },
    ]
    db.create_table("project_context", data=records, mode="overwrite")

    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.db_uri = tmp_dir

    result = KnowledgeBase.clear_all_template_pdf_vectors(kb)
    assert result["success"] is True

    rows = db.open_table("project_context").search().limit(10).to_list()
    assert len(rows) == 1
    assert rows[0]["source_type"] == "project_doc"


def main():
    test_save_updates_all_projects()
    test_load_prefers_global_cache()
    test_clear_all_template_pdf_vectors()


if __name__ == "__main__":
    main()
