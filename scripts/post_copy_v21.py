import os
import shutil
from pathlib import Path

ROOT = Path(r"F:\RSNA\medical_imaging_workflow")
DIST_DIR = ROOT / "dist_v21" / "MedicalImagingWorkflow"
ASSETS_SRC = ROOT / "assets"
ASSETS_DST = DIST_DIR / "assets"
MODELS_DST = DIST_DIR / "models"

def main():
    print("Copying assets...")
    ASSETS_DST.mkdir(parents=True, exist_ok=True)
    for item in ASSETS_SRC.iterdir():
        dst = ASSETS_DST / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)
    print(f"Assets copied to {ASSETS_DST}")
    print("Ensuring models folders...")
    (MODELS_DST / "deepseek_ocr").mkdir(parents=True, exist_ok=True)
    (MODELS_DST / "all-MiniLM-L6-v2").mkdir(parents=True, exist_ok=True)
    print("Done.")
    bat = DIST_DIR / "run_offline.bat"
    with open(bat, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write("set HF_HUB_OFFLINE=1\n")
        f.write("set TRANSFORMERS_OFFLINE=1\n")
        f.write("set OLLAMA_NUM_GPU=0\n")
        f.write("set OLLAMA_NUM_CTX=2048\n")
        f.write("MedicalImagingWorkflow.exe\n")
        f.write("pause\n")

if __name__ == "__main__":
    main()
