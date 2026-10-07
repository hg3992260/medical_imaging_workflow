import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--runtime-kind", choices=["source", "dist"], required=True)
    parser.add_argument("--check", choices=["ocr", "magic_seg"], required=True)
    parser.add_argument("--python-exe", default="")
    parser.add_argument("--expected", default="Hello RSNA")
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _prepare_runtime(runtime_root: Path, runtime_kind: str, python_exe: str):
    runtime_root = runtime_root.resolve()
    restore = {}
    extra_sys_paths = []
    if runtime_kind == "dist":
        internal_dir = runtime_root / "_internal"
        sys.path.insert(0, str(internal_dir))
        extra_sys_paths.append(str(internal_dir))
        restore["frozen"] = getattr(sys, "frozen", None)
        restore["executable"] = sys.executable
        restore["meipass"] = getattr(sys, "_MEIPASS", None)
        sys.frozen = True
        sys.executable = str(runtime_root / "MedicalImagingWorkflow_v41.exe")
        sys._MEIPASS = str(internal_dir)
    else:
        sys.path.insert(0, str(runtime_root))
        extra_sys_paths.append(str(runtime_root))
    for candidate in [
        runtime_root / "models" / "segment-anything",
        Path(r"F:\RSNA\segment-anything-main"),
    ]:
        try:
            if candidate.exists():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                    extra_sys_paths.append(candidate_str)
        except Exception:
            pass
    restore["extra_sys_paths"] = extra_sys_paths
    if python_exe:
        os.environ["DEEPANALYZE_EXEC_PYTHON"] = python_exe
        os.environ["DEEPANALYZE_PYTHON"] = python_exe
        os.environ["RSNA_OCR_PYTHON"] = python_exe
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return restore


def _restore_runtime(runtime_kind: str, restore: dict):
    for candidate in restore.get("extra_sys_paths") or []:
        try:
            while candidate in sys.path:
                sys.path.remove(candidate)
        except Exception:
            pass
    if runtime_kind == "dist":
        if restore.get("frozen") is None and hasattr(sys, "frozen"):
            delattr(sys, "frozen")
        elif restore.get("frozen") is not None:
            sys.frozen = restore["frozen"]
        sys.executable = restore["executable"]
        if restore.get("meipass") is None and hasattr(sys, "_MEIPASS"):
            delattr(sys, "_MEIPASS")
        elif restore.get("meipass") is not None:
            sys._MEIPASS = restore["meipass"]


def _create_ocr_test_image(expected: str) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="step1_ocr_selfcheck_"))
    image_path = tmp_dir / "step1_ocr_test.png"
    image = Image.new("RGB", (560, 220), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((32, 84), expected, fill="black")
    image.save(image_path)
    return image_path


def _build_magic_seg_test_image() -> np.ndarray:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[48:208, 48:208] = 40
    image[88:176, 88:176] = 255
    image[100:164, 100:164] = 180
    return image


def run_step1_ocr(expected: str, base_size: int, image_size: int) -> dict:
    image_path = _create_ocr_test_image(expected)
    cfg = {
        "engine": "deepseek_ocr",
        "preprocessing": False,
        "deepseek_mode": "free",
        "base_size": base_size,
        "image_size": image_size,
        "crop_mode": True,
        "test_compress": True,
    }
    if getattr(sys, "frozen", False):
        from src.services.ocr_service import OCRService

        service = OCRService()
        result = service.extract_text_from_image(str(image_path), cfg)
        adapter = getattr(service, "deepseek_adapter", None)
        return {
            "status": str(result.get("status") or ""),
            "text": str(result.get("text") or result.get("extracted_text") or ""),
            "engine": str(result.get("engine") or ""),
            "ocr_engine": str(result.get("ocr_engine") or ""),
            "error": str(result.get("error") or ""),
            "adapter_class": adapter.__class__.__name__ if adapter is not None else "",
            "worker_path": str(getattr(adapter, "worker_path", "") or ""),
            "model_path": str(getattr(adapter, "model_path", "") or ""),
            "available": bool(getattr(adapter, "available", False)) if adapter is not None else False,
        }

    python_exe = os.environ.get("DEEPANALYZE_EXEC_PYTHON") or os.environ.get("DEEPANALYZE_PYTHON") or ""
    if not python_exe:
        return {"status": "skip", "error": "未提供 python_exe，无法执行 external worker 自检"}
    from src.adapters.deepseek_ocr_external_adapter import DeepSeekOCRExternalAdapter

    runtime_root = Path(__file__).resolve().parent.parent
    worker_path = runtime_root / "src" / "adapters" / "deepseek_ocr_external_worker.py"
    model_path = runtime_root / "assets" / "models" / "deepseek_ocr"
    adapter = DeepSeekOCRExternalAdapter(
        python_exe=python_exe,
        model_kind="ocr",
        model_path=str(model_path),
        worker_path=str(worker_path),
    )
    result = adapter.extract_text_from_image(str(image_path), cfg)
    return {
        "status": str(result.get("status") or ""),
        "text": str(result.get("text") or result.get("extracted_text") or ""),
        "engine": str(result.get("engine") or ""),
        "ocr_engine": str(result.get("ocr_engine") or ""),
        "error": str(result.get("error") or ""),
        "adapter_class": adapter.__class__.__name__,
        "worker_path": str(getattr(adapter, "worker_path", "") or ""),
        "model_path": str(getattr(adapter, "model_path", "") or ""),
        "available": bool(getattr(adapter, "available", False)),
    }


def run_magic_seg() -> dict:
    try:
        import torch
        from segment_anything import sam_model_registry, SamPredictor
    except Exception as exc:
        return {"status": "skip", "error": f"segment_anything/torch 不可用: {exc}"}

    runtime_root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent

    override = (os.environ.get("RSNA_SAM_MODEL") or os.environ.get("RSNA_SAM_CHECKPOINT") or "").strip()
    candidates = []
    if override:
        candidates.append(Path(override))
    candidates.extend(
        [
            runtime_root / "_internal" / "assets" / "models" / "sam" / "sam_vit_b_01ec64.pth",
            runtime_root / "_internal" / "models" / "sam" / "sam_vit_b_01ec64.pth",
            runtime_root / "assets" / "models" / "sam" / "sam_vit_b_01ec64.pth",
            runtime_root / "models" / "sam" / "sam_vit_b_01ec64.pth",
            Path(r"F:\RSNA\segment-anything-main\sam_vit_b_01ec64.pth"),
            runtime_root / "_internal" / "assets" / "models" / "sam" / "sam_vit_h_4b8939.pth",
            runtime_root / "_internal" / "models" / "sam" / "sam_vit_h_4b8939.pth",
            runtime_root / "assets" / "models" / "sam" / "sam_vit_h_4b8939.pth",
            runtime_root / "models" / "sam" / "sam_vit_h_4b8939.pth",
            Path(r"F:\RSNA\segment-anything-main\sam_vit_h_4b8939.pth"),
        ]
    )

    checkpoint = None
    for p in candidates:
        try:
            if p and p.exists() and p.is_file():
                checkpoint = p
                break
        except Exception:
            continue

    if checkpoint is None:
        return {
            "status": "skip",
            "error": "SAM checkpoint 未找到",
            "searched": [str(p) for p in candidates[:8]],
        }

    model_type = "vit_h" if checkpoint.name.lower().endswith("sam_vit_h_4b8939.pth") else "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image = _build_magic_seg_test_image()
    predictor.set_image(image)
    points = np.array([[128, 128], [24, 24]], dtype=np.float32)
    labels = np.array([1, 0], dtype=np.int32)
    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )
    if masks is None or scores is None or len(scores) == 0:
        return {
            "status": "error",
            "error": "No masks returned",
            "checkpoint_path": str(checkpoint),
            "device": device,
            "model_type": model_type,
        }
    best_index = int(np.argmax(scores))
    best_mask = masks[best_index]
    return {
        "status": "ok",
        "available": True,
        "device": device,
        "model_type": model_type,
        "checkpoint_path": str(checkpoint),
        "mask_count": int(len(scores)),
        "best_score": float(scores[best_index]),
        "best_mask_sum": int(np.asarray(best_mask).astype(np.uint8).sum()),
    }


def main():
    args = parse_args()
    runtime_root = Path(args.runtime_root)
    restore = _prepare_runtime(runtime_root, args.runtime_kind, args.python_exe)
    try:
        if args.check == "ocr":
            payload = run_step1_ocr(args.expected, args.base_size, args.image_size)
        else:
            payload = run_magic_seg()
    except Exception as exc:
        payload = {
            "status": "error",
            "error": str(exc),
        }
    finally:
        _restore_runtime(args.runtime_kind, restore)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
