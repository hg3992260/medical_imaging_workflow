import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.schema_cache_service import SchemaCacheService


class StubDB:
    def __init__(self, project_id, project_name, tabular_rows):
        self._project_id = project_id
        self._project_name = project_name
        self._tabular_rows = tabular_rows
        self._files = []

    def get_connection(self):
        raise RuntimeError("not needed for this test")

    def execute_query(self, q, params=None):
        if "FROM projects" in q:
            return [{"name": self._project_name}]
        if "FROM storage_files" in q:
            return self._tabular_rows
        return []

    def execute_update(self, q, params=None):
        return 1


class FakeStorage:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._cache = {}

    def load_schema_cache(self, project_id: str):
        return self._cache.get(project_id, {})

    def store_schema_cache(self, project_id: str, schema_cache: dict):
        self._cache[project_id] = schema_cache
        return os.path.join(self.root_dir, f"{project_id}_schema_cache.json")


def main():
    with tempfile.TemporaryDirectory() as d:
        project_id = "p1"
        tab_fp = os.path.join(d, "leaf.json")
        leaf = {
            "metadata": {"user_id": "S1", "branch_path": "P/S1/t.csv/CSV", "domain_tag": "Tabular", "priority_weight": 2.0},
            "sheet": {"sheet_name": "CSV", "headers": ["Mean_Int", "Volume(mm3)"], "rows": [["Mean_Int", "Volume(mm3)"], ["1.0", "10"], ["2.0", "20"]], "kv": {}},
        }
        with open(tab_fp, "w", encoding="utf-8") as f:
            json.dump(leaf, f, ensure_ascii=False)

        tab_rows = [{"file_id": "t1", "file_path": tab_fp, "metadata": "{}", "created_at": "now"}]
        db = StubDB(project_id, "LungCTStudy", tab_rows)
        svc = SchemaCacheService(db, storage_service=FakeStorage(d))

        def llm_stub(prompt: str) -> str:
            assert "Schema_Cache" in prompt or "canonical_mappings" in prompt
            return json.dumps(
                {
                    "canonical_mappings": [
                        {
                            "canonical_name": "mean_intensity",
                            "aliases": [{"path": "P/S1/t.csv/CSV", "raw_name": "Mean_Int"}],
                            "unit_logic": "NA",
                            "scientific_role": "特征",
                        }
                    ]
                },
                ensure_ascii=False,
            )

        cache1, preview1 = svc.ensure_schema_cache(project_id, llm_call=llm_stub, max_new_branches=8)
        assert "mean_intensity" in preview1

        cache2, preview2 = svc.ensure_schema_cache(project_id, llm_call=lambda p: "{}", max_new_branches=8)
        assert preview2.count("mean_intensity") >= 1


if __name__ == "__main__":
    main()
