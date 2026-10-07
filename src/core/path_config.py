import os
import sys
from pathlib import Path

def get_base_dir() -> Path:
    """
    Get the base directory of the application.
    - If frozen (PyInstaller), returns the directory containing the executable.
    - If running from source, returns the project root.
    """
    if getattr(sys, 'frozen', False):
        return Path(os.path.dirname(sys.executable))
    else:
        # Assuming this file is in src/core/path_config.py
        # Project root is ../../
        return Path(__file__).resolve().parent.parent.parent

def get_coco_data_dir() -> Path:
    """
    Get the directory for CocoIndex data (sources, lancedb, etc.)
    Defaults to 'coco_data' inside the base directory.
    Can be overridden by COCO_DATA_DIR environment variable.
    """
    env_path = os.environ.get("COCO_DATA_DIR")
    if env_path:
        return Path(env_path)

    base = get_base_dir()

    if getattr(sys, "frozen", False):
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(base)
        target = Path(root) / "MedicalImagingWorkflow" / "coco_data"
        try:
            (target / "sources" / "references").mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        bundled = base / "coco_data"
        try:
            src_ref = bundled / "sources" / "references"
            dst_ref = target / "sources" / "references"
            if src_ref.exists() and src_ref.is_dir() and dst_ref.exists():
                for p in src_ref.glob("*"):
                    if not p.is_file():
                        continue
                    out = dst_ref / p.name
                    if not out.exists():
                        try:
                            import shutil

                            shutil.copy2(str(p), str(out))
                        except Exception:
                            pass
        except Exception:
            pass

        return target

    return base / "coco_data"

def get_models_dir() -> Path:
    """
    Get the directory for models.
    """
    return get_base_dir() / "models"
