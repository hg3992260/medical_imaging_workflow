import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.agent_core_loop import _detect_expected_fact_keys, _missing_fact_keys, _section_anchor, _has_source_citations


def main():
    src = "[ID: x]\nkey_dicom_fields:\n- SliceThickness: 0.6 mm\n- PixelSpacing: 0.4\\0.4\nTR=3.2 TE=1.2 kVp=120 n=30 25%\n"
    expected = _detect_expected_fact_keys(src)
    assert "SliceThickness" in expected
    assert "PixelSpacing" in expected
    assert "TR" in expected
    assert "TE" in expected
    assert "kVp" in expected
    assert "SampleSize" in expected
    assert "Percent" in expected

    fm_ok = "SliceThickness: 0.6 mm [ID: x]\nPixelSpacing 0.4 [ID: x]\nTR 3.2 [ID: x]\nTE 1.2 [ID: x]\nkVp 120 [ID: x]\nn=30 [ID: x]\n25% [ID: x]\n"
    assert _missing_fact_keys(fm_ok, expected) == []

    fm_bad = "TR 3.2 [ID: x]\n"
    missing = _missing_fact_keys(fm_bad, expected)
    assert "SliceThickness" in missing
    assert "PixelSpacing" in missing

    assert _section_anchor("Methods") == "## 2. Method"
    assert _has_source_citations("abc [Source: x] def") is True
    assert _has_source_citations("no cite") is False


if __name__ == "__main__":
    main()

