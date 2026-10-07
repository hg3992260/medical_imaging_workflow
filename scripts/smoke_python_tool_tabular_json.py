import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.tools.python_code_executor import PythonCodeExecutorToolGroup


def main():
    ws = str((ROOT / "project_storage" / "projects" / "tool_smoke").resolve())
    tabular_root = Path(ws) / "tabular"
    tabular_root.mkdir(parents=True, exist_ok=True)
    sample = tabular_root / "sample.json"
    if not sample.exists():
        sample.write_text('{"metadata":{"branch_path":"SampleSheet","n_rows":2,"n_cols":2},"sheet":{"sheet_name":"Sample","headers":["a","b"],"rows":[["a","b"],["1","2"]],"records":[{"a":"1","b":"2"}],"n_rows":1,"n_cols":2}}', encoding="utf-8")

    tool = PythonCodeExecutorToolGroup(ws, timeout_s=20.0, artifact_dir="exports")
    code = r"""
paths = list_tabular_json_paths()
print("n_json", len(paths))
print("first5", paths[:5])
if paths:
    obj = load_tabular_json(paths[0])
    print("top_keys", list(obj.keys())[:10])
    meta = obj.get("metadata") or {}
    print("meta_branch_path", meta.get("branch_path"))
"""
    res = tool.python(code)
    print("success", res.get("success"))
    print("stdout", (res.get("stdout") or "")[:1200])
    print("stderr", (res.get("stderr") or "")[:600])


if __name__ == "__main__":
    main()
