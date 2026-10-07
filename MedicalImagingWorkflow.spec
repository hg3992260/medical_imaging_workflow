# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path

PROJECT_ROOT = Path(r"F:\RSNA\medical_imaging_workflow")

# ── Helper: collect a directory tree including all files ──
def tree_datas(src_rel: str) -> list:
    full = PROJECT_ROOT / src_rel
    if not full.exists():
        return []
    out = []
    for root, dirs, files in os.walk(str(full)):
        rel = os.path.relpath(root, str(PROJECT_ROOT))
        dest = str(Path(rel).parent) if os.path.isfile(str(full)) else rel
        for f in files:
            if f.startswith(".") or f.startswith("~"):
                continue
            out.append((os.path.join(root, f), rel))
    return out


# ── Collect third-party hidden imports ──
hidden = [
    "_cffi_backend",
    "safetensors", "safetensors.torch",
    "pydicom", "pydicom.dataset",
    "easydict",
    "mpl_toolkits.mplot3d",
]

# ── Exclude unused heavy packages ──
excludes = [
    "tkinter", "PyQt6", "PySide2", "PySide6",
    "notebook", "jupyter", "jupyter_client", "ipython",
    "tensorflow", "tensorboard",
    "bokeh", "plotly", "dash",
    "sphinx", "docutils",
    "devtools", "black", "isort", "pylint",
    "pytest", "unittest",
    "Cython",
    "mypy",
    "pythoncom", "pywintypes", "win32com",
]

block_cipher = None

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=sum([
        tree_datas("config"),
        tree_datas("assets"),
        tree_datas("assets/ollama"),
        tree_datas("assets/models/deepseek_ocr"),
        tree_datas("assets/icon.ico"),
        tree_datas("assets/medlogo.png"),
        tree_datas("models/embedding"),
        tree_datas("models/sam"),
        tree_datas("poppler"),
        tree_datas("cocoindex"),
        tree_datas("segment-anything-main"),
        tree_datas("markitdown-main"),
        tree_datas("resources"),
    ], []),
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=True,
    optimize=0,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="MedicalImagingWorkflow",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "assets/icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MedicalImagingWorkflow",
)
