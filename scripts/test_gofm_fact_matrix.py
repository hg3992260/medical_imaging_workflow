import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.agent_core_loop import _fit_fact_matrix, _missing_fact_keys


def main():
    gofm = {
        "gofm_metadata": {"source_scope": "all_retrieved_chunks", "density_threshold": "high"},
        "objective_data": {
            "technical_entities": [{"name": "Scanner", "spec": "3T", "role": "device", "source_id": "r1"}],
            "parameters_and_metrics": [
                {"term": "SliceThickness", "value": "0.6", "unit": "mm", "condition": "T1", "source_id": "r1"},
                {"term": "slice thickness", "value": "0.8", "unit": "mm", "condition": "T1", "source_id": "r9"},
                {"term": "TR", "value": "2000", "unit": "ms", "condition": "T1", "source_id": "r2"},
                {"term": "TR", "value": "2000 ", "unit": " ms", "condition": "T1", "source_id": "r2b"},
                {"term": "AUC", "value": "0.91", "unit": "", "condition": "", "source_id": "r3"},
            ],
            "methodological_logic": [{"step": "1", "action": "Normalize", "input": "DICOM", "output": "Tensor", "source_id": "r4"}],
            "evidence_and_claims": [{"claim": "Performance improved", "confidence": "p<0.05", "source_id": "r3"}],
            "logical_conflicts": [{"point": "TR differs", "variants": [{"val": "1500ms", "src": "r5"}, {"val": "2000ms", "src": "r2"}]}],
        },
    }
    raw = json.dumps(gofm, ensure_ascii=False)
    fitted = _fit_fact_matrix(raw, 220, ["parameters_and_metrics", "evidence_and_claims"])
    obj = json.loads(fitted)
    assert "gofm_metadata" in obj
    assert "objective_data" in obj
    assert set(obj["objective_data"].keys()) == {"parameters_and_metrics", "evidence_and_claims"}
    assert len(fitted) <= 220

    missing = _missing_fact_keys(raw, ["SliceThickness", "PixelSpacing", "AUC", "PValue"])
    assert "PixelSpacing" in missing
    assert "SliceThickness" not in missing
    assert "AUC" not in missing

    fitted2 = _fit_fact_matrix(raw, 2000, ["parameters_and_metrics", "logical_conflicts"])
    obj2 = json.loads(fitted2)
    pms = obj2["objective_data"]["parameters_and_metrics"]
    assert sum(1 for x in pms if x.get("term") == "TR" and x.get("value") == "2000") == 1
    tr = next(x for x in pms if x.get("term") == "TR" and x.get("value") == "2000")
    assert "r2" in (tr.get("source_id") or "")
    assert "r2b" in (tr.get("source_id") or "")
    lcs = obj2["objective_data"]["logical_conflicts"]
    assert any(("SliceThickness" in (c.get("point") or "")) and len((c.get("variants") or [])) >= 2 for c in lcs)


if __name__ == "__main__":
    main()
