# -*- mode: python ; coding: utf-8 -*-
"""
Medical Imaging Workflow — Lite build spec (Release V1.0)

由首次运行初始化面板负责下载模型权重，因此打包时**不包含**：
  assets/models*、assets/models_home、assets/ollama、CUDA 独立库、第三方源码副本。
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(r"F:\RSNA\medical_imaging_workflow_release")


def tree_datas(src_rel: str) -> list:
    full = PROJECT_ROOT / src_rel
    if not full.exists():
        return []
    out = []
    excluded_dirs = {".cache", "build", "dist", "CMakeFiles", ".ipynb_checkpoints",
                     "__pycache__", ".deepseek_ocr_backup", ".deepseek_ocr_download"}
    for root, dirs, files in os.walk(str(full)):
        parts = root.replace("\\", "/").split("/")
        if any(d in parts for d in excluded_dirs):
            continue
        rel = os.path.relpath(root, str(PROJECT_ROOT))
        for f in files:
            if f.startswith(".") or f.startswith("~") or f.endswith(".incomplete"):
                continue
            if f.endswith((".pyc", ".pyo")) or f in ("licenses.db",):
                continue
            out.append((os.path.join(root, f), rel))
    return out


hidden = [
    "cffi", "_cffi_backend", "cryptography", "cryptography.hazmat",
    "cryptography.hazmat.backends", "cryptography.hazmat.backends.openssl",
    "cryptography.hazmat.bindings", "cryptography.hazmat.bindings._rust",
    "pyexpat", "xml.parsers.expat", "lxml", "lxml.etree", "lxml.isoschematron",
    "pypdfium2", "pypdfium2_raw",
    "scipy._lib", "scipy.spatial.transform", "scipy.linalg",
    "scipy.sparse.linalg", "scipy.sparse.csgraph",
    "sklearn.utils._weight_vector",
    "safetensors", "safetensors.torch", "simplejson", "simplejson.errors", "fsspec",
    "PIL", "PIL.Image", "PIL._imaging",
    "pyarrow", "pyarrow.lib",
    "sentence_transformers", "sentence_transformers.models",
    "tokenizers", "tokenizers.decoders",
    "addict", "easydict", "pydicom", "pydicom.dataset", "mpl_toolkits.mplot3d",
    "packaging", "packaging.version", "packaging.specifiers", "yaml", "docx",
    "aiohttp", "multipart", "multipart.multipart",
]

# numpy 2.x + PyInstaller: numpy.f2py 需显式收集，否则运行期报 No module named 'numpy.f2py'
try:
    from PyInstaller.utils.hooks import collect_submodules
    hidden += collect_submodules("numpy.f2py")
    hidden += collect_submodules("numpy._core")
except Exception:
    hidden += ["numpy.f2py", "numpy.f2py._backends", "numpy._core"]

# 保守排重：仅排除明确未使用的 GUI/文档/构建工具链
excludes = [
    "tkinter", "PyQt6", "PySide2", "PySide6",
    "notebook", "jupyter", "jupyter_client", "ipython", "IPython",
    "tensorboard", "sphinx", "docutils",
    "black", "isort", "pylint", "mypy",
    # 本项目未导入、且会拖入体积/损坏 hook 的重型可选依赖
    "sentry_sdk", "wandb", "gradio", "streamlit", "plotly", "dash", "bokeh",
    "panel", "pyvista", "vtk", "open3d", "monai", "nnunetv2",
    "paddle", "paddleocr", "paddlex", "totalsegmentator", "TotalSegmentator",
    "simpleitk", "SimpleITK", "triton", "cupy", "numba", "llvmlite",
    "onnxruntime", "onnx", "tensorflow", "keras", "jax", "jaxlib",
    "torchaudio", "torchmetrics", "pytorch_lightning", "lightning",
    "datasets", "visdom", "sqlalchemy", "alembic",
    "nvidia", "cupy", "cupy_backends", "cuda",
]

datas = sum([
    tree_datas("config"),
    tree_datas("assets"),
    tree_datas("license_manage"),
    tree_datas("ui/resources"),
    tree_datas("src/api/static"),
    tree_datas("src/api/templates"),
    tree_datas("coco_data/sources"),
], [])

block_cipher = None

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(PROJECT_ROOT / "build_scripts" / "runtime_hook.py")],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# ---------------------------------------------------------------------------
# 本机 torch 为 CPU 版 (2.9.1+cpu)：移除被误收集的 CUDA 运行时 DLL / nvidia 包，
# 否则会凭空多出 ~1.4GB 无用体积。
# ---------------------------------------------------------------------------
_CUDA_KEYS = (
    "cublas", "cusparse", "cusolver", "cudnn", "cufft", "curand", "nccl",
    "cudart", "nvrtc", "nvjitlink", "nvtoolext", "nvjpeg", "cupti", "npp",
    "thrust", "nvfatbin", "cufile",
)


def _norm(s):
    return str(s or "").lower().replace("\\", "/")


def _drop_cuda(entries):
    return [e for e in entries
            if not any(k in _norm(e[0]) for k in _CUDA_KEYS)
            and not _norm(e[0]).startswith("nvidia/")]


a.binaries = _drop_cuda(a.binaries)
a.datas = [d for d in a.datas
           if "/nvidia/" not in _norm(d[0]) and not _norm(d[0]).startswith("nvidia/")
           and "/nvidia/" not in _norm(d[1]) and not _norm(d[1]).startswith("nvidia/")]

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
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
    icon=str(PROJECT_ROOT / "assets" / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MedicalImagingWorkflow_v1.0",
)
