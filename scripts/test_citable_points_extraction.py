import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.agent_core_loop import _extract_citable_points_from_abstract


def main():
    abs1 = (
        "BACKGROUND: MRI acquisition can be slow. "
        "METHODS: We retrospectively studied 120 patients. "
        "RESULTS: AUC improved from 0.82 to 0.91 (p<0.001). "
        "CONCLUSION: The proposed method significantly improved performance."
    )
    pts = _extract_citable_points_from_abstract(abs1, 2, 240)
    assert len(pts) == 2
    assert "AUC" in pts[0] or "AUC" in pts[1]
    assert "CONCLUSION" not in pts[0].upper()

    abs2 = "Objective: Evaluate TR and TE. Results: TR was 2000 ms and TE was 30 ms."
    pts2 = _extract_citable_points_from_abstract(abs2, 2, 240)
    assert len(pts2) >= 1
    assert any("2000" in p or "30" in p for p in pts2)

    assert _extract_citable_points_from_abstract("", 2, 240) == []


if __name__ == "__main__":
    main()

