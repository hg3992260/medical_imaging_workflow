import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.agent_core_loop import _pack_weighted_chunks_for_extraction_scoped


def main():
    recs = [
        {"id": "s", "source_type": "stats", "source_id": "project", "chunk_index": 0, "text": "[PROJECT_STATS]\nCounts:\n- a: 1\n"},
        {"id": "m1", "source_type": "dicom", "source_id": "x", "chunk_index": 1, "text": "[DICOM_SESSION]\nscope: method\nkey_dicom_fields:\n- TR: 3.2\n"},
        {"id": "r1", "source_type": "roi", "source_id": "y", "chunk_index": 2, "text": "[ROI]\nscope: results\nproperties_key_fields:\n- diam_mm: 10\n"},
        {"id": "o1", "source_type": "document", "source_id": "d", "chunk_index": 0, "text": "[DOCUMENT]\nscope: background\nfoo\n"},
    ]
    out = _pack_weighted_chunks_for_extraction_scoped(recs, 1000)
    assert "[SCOPE: STATS]" in out
    assert "[SCOPE: METHOD]" in out
    assert "[SCOPE: RESULTS]" in out
    assert "[ID: s]" in out
    assert "[ID: m1]" in out
    assert "[ID: r1]" in out


if __name__ == "__main__":
    main()

