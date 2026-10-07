import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.services.llm_guided_sklearn_pipeline import run_llm_guided_sklearn_pipeline


class DummyLLM:
    def generate(self, model, prompt, timeout=600):
        if "FIGURE_CONTEXT_JSON" in prompt:
            return {
                "success": True,
                "response": json.dumps(
                    {
                        "captions": {"exports/agent_sklearn_pca_scree.png": "PCA解释方差"},
                        "notes": "- test note for figures",
                    },
                    ensure_ascii=False,
                ),
            }
        return {"success": False, "error": "dummy"}


def main():
    project = os.path.join(ROOT, "project_storage", "projects", "batch_20260422041802_009_f6bba6")
    out_dir = os.path.join(project, "exports", "_ml_test_pca_explain2")
    res = run_llm_guided_sklearn_pipeline(
        llm_service=DummyLLM(),
        model="dummy",
        materials_text="test",
        project_storage_path=project,
        workspace=project,
        out_dir=out_dir,
    )
    print("success", res.get("success"))
    print("figure_notes", bool(res.get("figure_notes")))
    rp = os.path.join(out_dir, "agent_sklearn_report.json")
    mp = os.path.join(out_dir, "agent_ml_artifacts.json")
    obj = json.load(open(rp, "r", encoding="utf-8"))
    man = json.load(open(mp, "r", encoding="utf-8"))
    print("report_has_figure_explanations", "figure_explanations" in obj)
    first_art = (man.get("artifacts") or [{}])[0]
    print("manifest_first_artifact_keys", sorted(list(first_art.keys())))


if __name__ == "__main__":
    main()
