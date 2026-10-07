import os
import shutil
from pathlib import Path


def _copytree_files(src: Path, dst: Path) -> None:
    for root, dirs, files in os.walk(src):
        root_p = Path(root)
        rel = root_p.relative_to(src)
        if rel.parts and rel.parts[0] in {".ipynb_checkpoints", ".git"}:
            dirs[:] = []
            continue
        dst_root = dst / rel
        dst_root.mkdir(parents=True, exist_ok=True)
        for fn in files:
            if fn.endswith(".incomplete") or fn.lower().endswith(".tmp"):
                continue
            src_f = root_p / fn
            dst_f = dst_root / fn
            if dst_f.exists():
                continue
            shutil.copy2(src_f, dst_f)


def _try_hardlink(src: Path, dst: Path) -> bool:
    try:
        if dst.exists():
            return True
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        return True
    except Exception:
        return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "models" / "DeepSeek-OCR"
    dst_dir = repo_root / "models" / "deepseek_ocr"

    if not src_dir.is_dir():
        raise FileNotFoundError(f"source model dir not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    _copytree_files(src_dir, dst_dir)

    weight_name = "model-00001-of-000001.safetensors"
    weight_dst = dst_dir / weight_name
    if not weight_dst.exists():
        candidates = [
            repo_root / "dist_v41_onefile_assets_fix7" / "MedicalImagingWorkflow_v41_Final" / "_internal" / "assets" / "models" / "deepseek_ocr" / weight_name,
            repo_root / "dist" / "MedicalImagingWorkflow_v41_Final" / "_internal" / "assets" / "models" / "deepseek_ocr" / weight_name,
        ]
        weight_src = next((p for p in candidates if p.is_file()), None)
        if weight_src is None:
            raise FileNotFoundError(f"weight shard not found in candidates: {candidates}")
        if not _try_hardlink(weight_src, weight_dst):
            shutil.copy2(weight_src, weight_dst)

    print(f"OK: {dst_dir}")
    print(f"- config: {(dst_dir / 'config.json').is_file()}")
    print(f"- tokenizer: {(dst_dir / 'tokenizer.json').is_file()}")
    print(f"- weights: {weight_dst.is_file()} size={weight_dst.stat().st_size if weight_dst.exists() else 0}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

