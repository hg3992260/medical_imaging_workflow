import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.app_config import AppConfig
from src.services.database_service import DatabaseService
from src.services.project_storage_service import ProjectStorageService
from src.services.schema_cache_service import SchemaCacheService
from src.services.tabular_branch_service import TabularBranchService
from src.ai.project_context_indexer import ProjectContextIndexer


def main():
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "t.db")
        cfg_path = os.path.join(d, "app_config.json")
        os.makedirs(os.path.join(d, "config"), exist_ok=True)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"database": {"type": "sqlite", "path": db_path}}, f, ensure_ascii=False)

        db = DatabaseService(AppConfig(config_file=cfg_path))
        db.initialize_database()

        project_id = "p1"
        db.execute_update("INSERT INTO projects (project_id, name, description) VALUES (?, ?, ?)", (project_id, "StudyX", ""))

        storage = ProjectStorageService(db)
        assert storage.create_project_storage(project_id)

        tab_fp = os.path.join(d, "leaf.json")
        with open(tab_fp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {"user_id": "S1", "branch_path": "StudyX/S1/file/Sheet1"},
                    "sheet": {"sheet_name": "Sheet1", "headers": ["A", "B"], "rows": [["A", "B"], ["1", "2"]], "kv": {}},
                },
                f,
                ensure_ascii=False,
            )

        file_id = storage._generate_file_id(f"{project_id}::tabular_test")
        storage._record_file_storage(project_id, file_id, tab_fp, "tabular", 10, {"branch_path": "StudyX/S1/file/Sheet1"})

        idx = ProjectContextIndexer(db, kb=None)
        list(idx._load_tabular_artifacts(project_id))
        list(idx._load_schema_cache(project_id))

        scs = SchemaCacheService(db, storage_service=storage)
        scs.ensure_schema_cache(project_id, llm_call=lambda p: "{}", max_new_branches=8)

        tbs = TabularBranchService(db)
        tbs.build_tree(project_id, schema_cache={})


if __name__ == "__main__":
    main()

