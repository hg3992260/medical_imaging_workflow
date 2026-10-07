import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.agent_core_loop import _pack_weighted_chunks_for_extraction, _fit_fact_matrix


def main():
    records = [
        {"id": "r0", "source_type": "stats", "source_id": "project", "chunk_index": 0, "text": "[PROJECT_STATS]\nCounts:\n- a: 1\n"},
        {"id": "r1", "source_type": "dicom", "source_id": "s1", "chunk_index": 1, "text": "[DICOM_SESSION]\nkey_dicom_fields:\n- SliceThickness: 0.6 mm\n\nPage 1 of 2\n"},
    ]
    weighted = _pack_weighted_chunks_for_extraction(records, 500)
    assert "[ID: r0]" in weighted
    assert "[ID: r1]" in weighted
    assert "Page 1 of 2" not in weighted.lower()

    fm = "A\n" * 500
    fitted = _fit_fact_matrix(fm, 200)
    assert len(fitted) <= 200


if __name__ == "__main__":
    main()

