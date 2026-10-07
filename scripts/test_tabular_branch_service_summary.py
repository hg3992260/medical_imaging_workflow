import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.tabular_branch_service import TabularBranchService


class StubDB:
    def __init__(self, file_rows, project_name="MyProject"):
        self._rows = file_rows
        self._project_name = project_name

    def execute_query(self, q, params=None):
        if "FROM projects" in q:
            return [{"name": self._project_name}]
        return self._rows

    def fetch_one(self, q, params):
        return {"name": self._project_name}

    def fetch_all(self, q, params):
        return self._rows


def main():
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "s1.json")
        obj = {
            "metadata": {"user_id": "S1", "branch_path": "P/S1/table/Sheet1", "domain_tag": "Tabular", "priority_weight": 2.0, "n_rows": 3, "n_cols": 2},
            "sheet": {"sheet_name": "Sheet1", "headers": ["x", "y"], "records": [{"x": "1", "y": "2"}, {"x": "2", "y": "4"}, {"x": "3", "y": "6"}], "kv": {}},
        }
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

        db = StubDB([{"file_id": "f1", "file_path": fp, "metadata": "{}", "created_at": "now"}], project_name="LungCTStudy")
        svc = TabularBranchService(db)
        s = svc.summarize_project("p1", max_chars=4000)
        assert "[TABULAR_TREE]" in s
        assert "[DATA_DENSITY]" in s


if __name__ == "__main__":
    main()
