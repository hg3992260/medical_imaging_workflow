import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.agent_core_loop import _pack_project_context_records


def main():
    recs = [
        {
            "id": "stats",
            "source_type": "stats",
            "source_id": "project",
            "chunk_index": 0,
            "text": "[PROJECT_STATS]\nCounts:\n- rois: 10\n",
        },
        {
            "id": "a0",
            "source_type": "dicom",
            "source_id": "s1",
            "chunk_index": 0,
            "text": "[DICOM_SESSION]\nscope: method\nkey_dicom_fields:\n- SliceThickness: 0.6 mm\n\nStart.\n",
        },
        {
            "id": "a1",
            "source_type": "dicom",
            "source_id": "s1",
            "chunk_index": 1,
            "text": "[DICOM_SESSION]\nscope: method\nkey_dicom_fields:\n- SliceThickness: 0.6 mm\n\nMiddle.\n",
        },
        {
            "id": "a2",
            "source_type": "dicom",
            "source_id": "s1",
            "chunk_index": 2,
            "text": "[DICOM_SESSION]\nscope: method\nkey_dicom_fields:\n- SliceThickness: 0.6 mm\n\nEnd.\n",
        },
    ]

    packed = _pack_project_context_records(recs, 500)
    assert "[PROJECT_STATS]" in packed
    assert "Start." in packed or "Middle." in packed or "End." in packed
    assert ("Start." in packed and "Middle." in packed) or ("Middle." in packed and "End." in packed)


if __name__ == "__main__":
    main()

