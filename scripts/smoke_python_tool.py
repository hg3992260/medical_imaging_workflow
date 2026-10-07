import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.database_service import DatabaseService
from src.services.project_storage_service import ProjectStorageService
from src.ai.tools.python_code_executor import PythonCodeExecutorToolGroup


def main():
    project_id = "tool_smoke"
    db = DatabaseService()
    ps = ProjectStorageService(db)
    ps.create_project_storage(project_id)
    ws = ps.get_project_storage_path(project_id)
    tabular_dir = ws / "tabular"
    tabular_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = tabular_dir / "patients.xlsx"
    csv_path = tabular_dir / "patients.csv"

    try:
        from openpyxl import Workbook

        wb = Workbook()
        sheet = wb.active
        sheet.title = "Sheet1"
        sheet.append(["patient_id", "dose_mgy", "noise_sd", "cnr"])
        for i in range(1, 21):
            sheet.append([f"P{i:03d}", 2.0 + 0.05 * i, 18.0 - 0.2 * i, 1.2 + 0.03 * i])
        wb.save(xlsx_path)
        print("xlsx_written", str(xlsx_path))
    except Exception as e:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("patient_id,dose_mgy,noise_sd,cnr\n")
            for i in range(1, 21):
                f.write(f"P{i:03d},{2.0 + 0.05 * i},{18.0 - 0.2 * i},{1.2 + 0.03 * i}\n")
        print("xlsx_unavailable_fallback_csv", type(e).__name__, str(e))
        print("csv_written", str(csv_path))

    tool = PythonCodeExecutorToolGroup(str(ws), timeout_s=25.0, artifact_dir="exports")
    code = r"""
import os
print("WORKSPACE", WORKSPACE)
print("ARTIFACT_DIR", ARTIFACT_DIR)

xlsx_path = os.path.join("tabular", "patients.xlsx")
csv_path = os.path.join("tabular", "patients.csv")

try:
    import pandas as pd
    if os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
    else:
        df = pd.read_csv(csv_path)
    print("rows", len(df))
    print("dose_mgy_mean", float(df["dose_mgy"].mean()))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        ax.plot(df["dose_mgy"], df["noise_sd"])
        ax.set_xlabel("dose_mgy")
        ax.set_ylabel("noise_sd")
        out = os.path.join("exports", "dose_vs_noise.png")
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        ARTIFACTS.append(out)
        print("saved", out)
    except Exception as e:
        print("matplotlib_failed", type(e).__name__, str(e))
except Exception as e:
    print("pandas_failed", type(e).__name__, str(e))
"""

    res = tool.python(code)
    print("exec_success", res.get("success"))
    print("artifacts", res.get("artifacts"))
    print("stdout", (res.get("stdout") or "")[:900])
    print("stderr", (res.get("stderr") or "")[:600])


if __name__ == "__main__":
    main()
