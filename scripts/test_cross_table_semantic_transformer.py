import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.app_config import AppConfig
from src.services.database_service import DatabaseService
from src.services.project_storage_service import ProjectStorageService
from src.services.cross_table_semantic_transformer import CrossTableSemanticTransformer


def main():
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "t.db")
        cfg_path = os.path.join(d, "app_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"database": {"type": "sqlite", "path": db_path}}, f, ensure_ascii=False)

        db = DatabaseService(AppConfig(config_file=cfg_path))
        db.initialize_database()

        project_id = "p1"
        db.execute_update("INSERT INTO projects (project_id, name, description) VALUES (?, ?, ?)", (project_id, "StudyX", ""))

        storage = ProjectStorageService(db)
        storage.create_project_storage(project_id)

        db.execute_update(
            "INSERT INTO dicom_sessions (session_id, user_id, project_id, file_path, created_at, dicom_info) VALUES (?, ?, ?, ?, ?, ?)",
            ("ds1", "S1", project_id, os.path.join(d, "dcm1"), "now", json.dumps({"SliceThickness": 1.0, "KVP": 120})),
        )
        db.execute_update(
            "INSERT INTO dicom_sessions (session_id, user_id, project_id, file_path, created_at, dicom_info) VALUES (?, ?, ?, ?, ?, ?)",
            ("ds2", "S2", project_id, os.path.join(d, "dcm2"), "now", json.dumps({"SliceThickness": 2.0, "KVP": 80})),
        )

        db.execute_update(
            "INSERT INTO roi_data (roi_id, session_id, roi_type, roi_name, coordinates, area, perimeter, properties) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("r1", "ds1", "detected", "x", "[]", 10.0, 5.0, json.dumps({"Mean_Int": 1.0})),
        )
        db.execute_update(
            "INSERT INTO roi_data (roi_id, session_id, roi_type, roi_name, coordinates, area, perimeter, properties) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("r2", "ds2", "detected", "x", "[]", 20.0, 8.0, json.dumps({"Mean_Int": 2.0})),
        )

        leaf = {
            "metadata": {"user_id": "S1", "branch_path": "StudyX/S1/file/Sheet1", "domain_tag": "Tabular"},
            "sheet": {"sheet_name": "Sheet1", "headers": ["featA"], "rows": [["featA"], ["1.0"], ["2.0"]], "kv": {}},
        }
        fp = os.path.join(d, "leaf.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(leaf, f, ensure_ascii=False)
        file_id = storage._generate_file_id(fp)
        storage._record_file_storage(project_id, file_id, fp, "tabular", 10, {"branch_path": "StudyX/S1/file/Sheet1"})

        cts = CrossTableSemanticTransformer(db)
        txt, structured = cts.summarize_project(project_id, schema_cache={}, max_chars=6000)
        assert txt
        assert isinstance(structured, dict)


if __name__ == "__main__":
    main()

