import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.agent_core_loop import _build_term_to_ids, _validate_citations_deterministic


def main():
    gofm = {
        "gofm_metadata": {"source_scope": "all_retrieved_chunks", "density_threshold": "high"},
        "objective_data": {
            "parameters_and_metrics": [
                {"term": "TR", "value": "2000", "unit": "ms", "condition": "T1", "source_id": "r2"},
                {"term": "TE", "value": "30", "unit": "ms", "condition": "T1", "source_id": "r3"},
            ],
            "technical_entities": [],
            "methodological_logic": [],
            "evidence_and_claims": [],
            "logical_conflicts": [],
        },
    }
    fm = json.dumps(gofm, ensure_ascii=False)
    term_to_ids = _build_term_to_ids(fm, "")
    allowed_ids = ["r2", "r3"]

    text_ok = "TR was 2000 ms [Source: r2]\nTE was 30 ms [Source: r3]"
    assert _validate_citations_deterministic(text_ok, allowed_ids, term_to_ids) == []

    text_bad_id = "TR was 2000 ms [Source: r999]"
    issues = _validate_citations_deterministic(text_bad_id, allowed_ids, term_to_ids)
    assert any(x.startswith("invalid_ids:") for x in issues)

    text_mismatch = "TR was 2000 ms [Source: r3]"
    issues2 = _validate_citations_deterministic(text_mismatch, allowed_ids, term_to_ids)
    assert "term_mismatch:TR" in issues2

    text_multi = "TR was 2000 ms [Source: r2]. TE was 30 ms [Source: r2]."
    issues3 = _validate_citations_deterministic(text_multi, allowed_ids, term_to_ids)
    assert "term_mismatch:TE" in issues3

    text_comma = "TR was 2000 ms [Source: r2], TE was 30 ms [Source: r3]."
    assert _validate_citations_deterministic(text_comma, allowed_ids, term_to_ids) == []

    text_and = "TR was 2000 ms [Source: r2] and TE was 30 ms [Source: r3]."
    assert _validate_citations_deterministic(text_and, allowed_ids, term_to_ids) == []


if __name__ == "__main__":
    main()
