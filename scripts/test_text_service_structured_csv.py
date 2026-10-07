import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.text_service import TextService


def main():
    svc = TextService()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write("k,v\nalpha,1\nbeta,2\n")
        r = svc.read_file_structured(p)
        assert r.get("format") == ".csv"
        assert r.get("tabular", {}).get("type") == "csv"
        sheet = (r.get("tabular", {}).get("sheets") or [])[0]
        assert sheet.get("headers") == ["k", "v"]
        assert isinstance(sheet.get("kv"), dict) and sheet.get("kv", {}).get("alpha") == "1"
        assert "=== CSV ===" in (r.get("text") or "")


if __name__ == "__main__":
    main()

