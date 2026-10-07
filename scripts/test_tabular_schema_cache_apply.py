import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.tabular_branch_service import TabularBranchService


class StubDB:
    def __init__(self, project_name, rows):
        self._project_name = project_name
        self._rows = rows

    def execute_query(self, q, params=None):
        if "FROM projects" in q:
            return [{"name": self._project_name}]
        return self._rows


def main():
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "leaf.json")
        obj = {
            "metadata": {"user_id": "S1", "branch_path": "P/S1/file/Sheet1", "domain_tag": "Tabular", "priority_weight": 2.0},
            "sheet": {"sheet_name": "Sheet1", "headers": ["Mean_Int", "Other"], "records": [{"Mean_Int": "1.0", "Other": "x"}, {"Mean_Int": "2.0", "Other": "y"}], "kv": {}},
        }
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

        db = StubDB("StudyX", [{"file_id": "t1", "file_path": fp, "metadata": "{}", "created_at": "now"}])
        schema_cache = {
            "canonical_mappings": [
                {
                    "canonical_name": "mean_intensity",
                    "aliases": [{"path": "P/S1/file/Sheet1", "raw_name": "Mean_Int"}],
                    "unit_logic": "NA",
                    "scientific_role": "特征",
                }
            ]
        }
        svc = TabularBranchService(db)
        tree = svc.build_tree("p1", schema_cache=schema_cache)
        branches = tree.get("samples")[0]["branches"]
        df = branches[0]["df"]
        assert "mean_intensity" in list(df.columns)


if __name__ == "__main__":
    main()

