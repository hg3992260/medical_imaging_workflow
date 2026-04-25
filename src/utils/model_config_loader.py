import json
import os
import sys
import logging

logger = logging.getLogger(__name__)

def _looks_like_model_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    has_config = os.path.isfile(os.path.join(path, "config.json"))
    has_tokenizer = os.path.isfile(os.path.join(path, "tokenizer.json")) or os.path.isfile(os.path.join(path, "tokenizer_config.json"))
    has_weights = any(
        os.path.isfile(os.path.join(path, name))
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "model-00001-of-000001.safetensors",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )
    return bool(has_config and (has_tokenizer or has_weights))

def _find_model_dir(path: str, max_depth: int = 4) -> str:
    if _looks_like_model_dir(path):
        return os.path.normpath(path)
    if not path or not os.path.isdir(path):
        return ""
    base_depth = os.path.normpath(path).count(os.sep)
    for root, dirs, _ in os.walk(path):
        depth = os.path.normpath(root).count(os.sep) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        if _looks_like_model_dir(root):
            return os.path.normpath(root)
    return ""

def get_model_paths():
    """
    Get the absolute paths for external models from configuration.
    Defaults to 'models/' directory in the application root.
    """
    # Default relative paths
    paths = {
        "deepseek_ocr": "assets/models/deepseek_ocr",
        "embedding_model": "models/embedding/all-MiniLM-L6-v2",
        "ollama_models": "assets/models_home",
        "ollama_exe": "assets/ollama/ollama.exe",
        "sam_model": "assets/models/sam_vit_b_01ec64.pth"
    }
    
    # Determine base directory
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
        internal_dir = getattr(sys, "_MEIPASS", os.path.join(app_dir, "_internal"))
    else:
        # Running from source (src/utils/model_config_loader.py -> ... -> root)
        # f:\RSNA\medical_imaging_workflow\src\utils\model_config_loader.py
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        internal_dir = app_dir
        
    config_path = os.path.join(app_dir, 'config', 'model_paths.json')
    
    # Try to load from config file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Update only if key exists in defaults
                for k, v in config.items():
                    if k in paths:
                        paths[k] = v
        except Exception as e:
            logger.error(f"Error loading model config from {config_path}: {e}")
            
    # Resolve to absolute paths
    resolved_paths = {}
    for key, path in paths.items():
        if key == "deepseek_ocr":
            override_env = "RSNA_DEEPSEEK_OCR_DIR"
            override_dir = os.environ.get(override_env)
            if override_dir:
                override_dir = os.path.normpath(str(override_dir))
                discovered = _find_model_dir(override_dir)
                if discovered:
                    resolved_paths[key] = discovered
                    continue
            candidates = []
            if os.path.isabs(path):
                candidates.append(os.path.normpath(path))
            else:
                candidates.append(os.path.normpath(os.path.join(app_dir, path)))
                candidates.append(os.path.normpath(os.path.join(internal_dir, path)))
            candidates.extend(
                [
                    os.path.normpath(os.path.join(app_dir, "assets", "models", "deepseek_ocr")),
                    os.path.normpath(os.path.join(internal_dir, "assets", "models", "deepseek_ocr")),
                    os.path.normpath(os.path.join(app_dir, "_internal", "assets", "models", "deepseek_ocr")),
                    os.path.normpath(os.path.join(app_dir, "models", "deepseek_ocr1")),
                    os.path.normpath(os.path.join(internal_dir, "models", "deepseek_ocr1")),
                ]
            )
            chosen = ""
            for cand in candidates:
                discovered = _find_model_dir(cand)
                if discovered:
                    chosen = discovered
                    break
            resolved_paths[key] = chosen if chosen else os.path.normpath(path)
            continue
        if not os.path.isabs(path):
            app_candidate = os.path.normpath(os.path.join(app_dir, path))
            internal_candidate = os.path.normpath(os.path.join(internal_dir, path))
            candidates = [app_candidate, internal_candidate]

            path_norm = str(path).replace("\\", "/")
            if getattr(sys, "frozen", False) and path_norm.startswith("models/"):
                alt_rel = "assets/models/" + path_norm[len("models/"):]
                candidates.append(os.path.normpath(os.path.join(app_dir, alt_rel)))
                candidates.append(os.path.normpath(os.path.join(internal_dir, alt_rel)))
                candidates.append(os.path.normpath(os.path.join(app_dir, "_internal", alt_rel)))
                candidates.append(os.path.normpath(os.path.join(app_dir, "_internal", "assets", "models", path_norm[len("models/"):])))

            chosen = None
            for cand in candidates:
                if os.path.exists(cand):
                    chosen = cand
                    break
            resolved_paths[key] = chosen if chosen else app_candidate
        else:
            resolved_paths[key] = os.path.normpath(path)
            
    return resolved_paths
