import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.agent_core_loop import _pack_project_context


def main():
    ctx = "\n\n".join(
        [
            "[DOCUMENT]\ntext: something long " + ("x" * 1200),
            "[PROJECT_STATS]\nCounts:\n- documents: 10\n- ocr: 20\n",
            "[DICOM_SESSION]\nscope: method\nkey_dicom_fields:\n- SliceThickness: 0.6 mm\n" + ("y" * 1200),
            "[OCR_RESULT]\nscope: results\nconfidence: 0.9\nFindings ...\n",
        ]
    )
    packed = _pack_project_context(ctx, 800)
    assert "[PROJECT_STATS]" in packed
    assert "key_dicom_fields" in packed
    assert len(packed) <= 800


if __name__ == "__main__":
    main()

