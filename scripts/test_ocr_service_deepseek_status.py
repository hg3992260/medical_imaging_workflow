import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.ocr_service import get_ocr_service


def main():
    s = get_ocr_service()
    t0 = time.time()
    while getattr(s, "deepseek_adapter", None) is not None and getattr(s.deepseek_adapter, "loading", False) and time.time() - t0 < 30:
        time.sleep(0.5)
    print("deepseek_adapter:", bool(getattr(s, "deepseek_adapter", None)))
    print("deepseek_loading:", bool(getattr(getattr(s, "deepseek_adapter", None), "loading", False)))
    print("deepseek_available:", bool(getattr(s, "deepseek_available", False)))
    print("deepseek_error:", getattr(getattr(s, "deepseek_adapter", None), "load_error", None))


if __name__ == "__main__":
    main()
