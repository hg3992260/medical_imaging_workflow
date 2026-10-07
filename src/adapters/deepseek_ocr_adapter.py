import os
import sys
import time
from typing import Dict, Any
import importlib.metadata
import gc
import glob
import contextlib
import tempfile
import logging

try:
    hf_home_default = os.path.join(tempfile.gettempdir(), "medical_imaging_workflow_hf_home")
    os.environ.setdefault("HF_HOME", hf_home_default)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home_default, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home_default, "transformers"))
    os.environ.setdefault("HF_MODULES_CACHE", os.path.join(hf_home_default, "modules"))
except Exception:
    pass

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    import torch
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
except Exception as e:
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(e)
    print(f"Warning: PyTorch or Transformers not found. DeepSeekOCRAdapter will be unavailable. Error: {e}")

import threading
from src.utils.model_config_loader import get_model_paths
import re

logger = logging.getLogger(__name__)

class DeepSeekOCRAdapter:
    def __init__(self, model_path: str = None, model_kind: str = "ocr", hub_id: str = None):
        # 强制使用 OCR1，忽略任何 ocr2 参数
        self.model_kind = "ocr"
        self.strict_ocr1 = os.environ.get("DEEPSEEK_OCR_STRICT_OCR1", "1").strip().lower() in ("1", "true", "yes")
        self._explicit_model_path = model_path is not None
        if model_path is None:
            paths = get_model_paths()
            self.model_path = paths.get("deepseek_ocr") or ""
        else:
            self.model_path = model_path
            
        if hub_id is not None:
            # 强制指向 OCR1 的 HF 仓库
            self.hub_id = "deepseek-ai/DeepSeek-OCR"
        else:
            self.hub_id = "deepseek-ai/DeepSeek-OCR"

        self.model = None
        self.tokenizer = None
        self.loading = False
        self.load_error = None
        self.loaded_model_path = None
        
        # Set device early for debugging and consistency
        device_pref = os.environ.get("DEEPSEEK_OCR_DEVICE", "").strip().lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if device_pref == "cpu":
            self.device = "cpu"
        
        if not TORCH_AVAILABLE:
            self.load_error = f"Missing dependencies: {TORCH_IMPORT_ERROR}"
            return

        # 启动异步加载线程
        self.loading_thread = threading.Thread(target=self._load_model_thread, daemon=True)
        self.loading_thread.start()

    def _format_ocr_text(self, text: str, mode: str) -> str:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        if mode in ("raw", "none", ""):
            return t.strip()

        t = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", t, flags=re.DOTALL)
        t = re.sub(r"<\|det\|>.*?<\|/det\|>", "", t, flags=re.DOTALL)
        t = re.sub(r"\n{3,}", "\n\n", t).strip()

        if mode in ("markdown_clean", "md_clean"):
            t = re.sub(r"[ \t]+\n", "\n", t)
            t = re.sub(r"\n[ \t]+", "\n", t)
            return t.strip()

        if mode in ("plain_clean", "text_clean"):
            lines = [ln.strip() for ln in t.split("\n")]
            lines = [ln for ln in lines if ln]
            return "\n".join(lines).strip()

        return t.strip()

    def _load_model_thread(self):
        """
        在后台线程中加载模型
        """
        try:
            self.loading = True
            logger.info("DeepSeek-OCR: background load started (device=%s)", getattr(self, "device", ""))
            print("正在后台加载DeepSeek-OCR模型...")
            self._load_model()
        except Exception as e:
            self.load_error = str(e)
            logger.exception("DeepSeek-OCR: background load failed: %s", e)
            print(f"DeepSeek-OCR模型加载失败: {e}")
        finally:
            self.loading = False

    def _load_model(self):
        """
        加载DeepSeek-OCR模型 (严格对齐 DeepSeek-OCR-main 的 HF 推理)
        """
        # [Fix] Set offline mode immediately to prevent any connection attempts
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        try:
            try:
                import torchvision.transforms  # noqa: F401
            except Exception:
                import types
                import enum
                tv = sys.modules.get("torchvision") or types.ModuleType("torchvision")
                transforms = types.ModuleType("torchvision.transforms")
                functional = types.ModuleType("torchvision.transforms.functional")
                presets = types.ModuleType("torchvision.transforms._presets")

                class InterpolationMode(enum.Enum):
                    NEAREST = 0
                    BILINEAR = 2
                    BICUBIC = 3
                    LANCZOS = 1
                    BOX = 4
                    HAMMING = 5

                transforms.InterpolationMode = InterpolationMode
                tv.transforms = transforms
                sys.modules["torchvision"] = tv
                sys.modules["torchvision.transforms"] = transforms
                sys.modules["torchvision.transforms.functional"] = functional
                sys.modules["torchvision.transforms._presets"] = presets

            device_pref = os.environ.get("DEEPSEEK_OCR_DEVICE", "").strip().lower()
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if device_pref == "cpu":
                self.device = "cpu"
            elif device_pref in ("cuda", "cuda:0", "gpu"):
                self.device = "cuda:0"
                
            # Configure device_map and max_memory based on device availability
            # ... (保持原有的加载逻辑不变)
            # FORCE device_map to None to avoid accelerate splitting model across devices or using meta tensors
            device_map = None 
            
            # Check for accelerate
            try:
                import accelerate
            except ImportError:
                print("Warning: 'accelerate' package not found.")

            max_memory = {"cpu": "30GiB"}
            if str(self.device).startswith("cuda"):
                try:
                    total_vram = torch.cuda.get_device_properties(0).total_memory
                    total_vram_gb = total_vram / (1024**3)
                    print(f"GPU VRAM: {total_vram_gb:.1f}GB")
                    if not device_pref and total_vram_gb < 8.0:
                        print(
                            f"GPU VRAM {total_vram_gb:.1f}GB may be insufficient for DeepSeek-OCR; defaulting to CPU. "
                            f"Set DEEPSEEK_OCR_DEVICE=cuda to force GPU."
                        )
                        self.device = "cpu"
                except Exception as e:
                    print(f"Error checking VRAM: {e}")
            
            force_ocr_fp32 = os.environ.get("DEEPSEEK_OCR_FORCE_FP32", "").strip() in ("1", "true", "True")
            if str(self.device).startswith("cuda"):
                if force_ocr_fp32:
                    dtype_default = torch.float32
                else:
                    try:
                        dtype_default = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    except Exception:
                        dtype_default = torch.float16
            else:
                dtype_default = torch.float32

            # 优先尝试本地路径（从之前的修复中得知我们有 bundled model）
            import sys
            
            # 定义候选路径列表
            candidate_paths = []
            strict_local = os.environ.get("DEEPSEEK_OCR_STRICT_LOCAL", "").strip().lower() in ("1", "true", "yes")
            if getattr(sys, "frozen", False) and os.environ.get("DEEPSEEK_OCR_STRICT_LOCAL") is None:
                strict_local = True

            def _has_any_weights(model_dir: str) -> bool:
                try:
                    if not model_dir or not os.path.isdir(model_dir):
                        return False
                    direct = [
                        "model.safetensors",
                        "model.safetensors.index.json",
                        "pytorch_model.bin",
                        "pytorch_model.bin.index.json",
                    ]
                    if any(os.path.exists(os.path.join(model_dir, f)) for f in direct):
                        return True
                    for root, _, files in os.walk(model_dir):
                        for f in files:
                            lf = f.lower()
                            if lf.endswith(".safetensors") or lf.endswith(".bin"):
                                return True
                        break
                    return False
                except Exception:
                    return False

            def _add_variants(p: str):
                if not p:
                    return
                cand = p
                if (
                    cand not in candidate_paths
                    and os.path.exists(cand)
                    and _has_any_weights(cand)
                ):
                    candidate_paths.append(cand)

            # 0. 显式指定路径优先（仅当调用方明确传入 model_path 时）
            if self._explicit_model_path and self.model_path:
                _add_variants(self.model_path)
                if os.path.exists(self.model_path):
                    print(f"Adding explicit path: {self.model_path}")

            # 0. 优先尝试配置文件中的路径
            try:
                paths = get_model_paths()
                config_path = paths.get("deepseek_ocr")
                if config_path:
                    _add_variants(config_path)
                    if os.path.exists(config_path):
                        print(f"Adding configured path: {config_path}")
            except Exception as e:
                print(f"Failed to get configured path: {e}")
            
            # 1. 优先尝试 PyInstaller 打包路径
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
                
                # Try to find _internal directory
                internal_dir = os.path.join(base_dir, '_internal')
                if not os.path.exists(internal_dir):
                    # Maybe it's not named _internal?
                    possible_internals = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and 'internal' in d]
                    if possible_internals:
                        internal_dir = os.path.join(base_dir, possible_internals[0])

                meipass_dir = getattr(sys, "_MEIPASS", None)
                if meipass_dir:
                    meipass_assets = os.path.join(
                        meipass_dir, 'assets', 'models', 'deepseek_ocr'
                    )
                    _add_variants(meipass_assets)
                    meipass_path = os.path.join(
                        meipass_dir, 'models', 'deepseek_ocr'
                    )
                    _add_variants(meipass_path)
                
                internal_assets = os.path.join(
                    internal_dir, 'assets', 'models', 'deepseek_ocr'
                )
                _add_variants(internal_assets)
                
                internal_path = os.path.join(
                    internal_dir, 'models', 'deepseek_ocr'
                )
                _add_variants(internal_path)
                
                base_assets = os.path.join(
                    base_dir, 'assets', 'models', 'deepseek_ocr'
                )
                _add_variants(base_assets)
                
                # Legacy paths
                legacy_path = os.path.join(base_dir, 'models', 'deepseek_ocr')
                _add_variants(legacy_path)
            else:
                try:
                    app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    _add_variants(os.path.join(app_root, 'assets', 'models', 'deepseek_ocr'))
                    _add_variants(os.path.join(app_root, 'models', 'deepseek_ocr'))
                except Exception:
                    pass

            # 2. 尝试本地 HuggingFace 缓存路径
            try:
                allow_hf_cache_env = os.environ.get("DEEPSEEK_OCR_ALLOW_HF_CACHE")
                if allow_hf_cache_env is None:
                    allow_hf_cache = not getattr(sys, "frozen", False)
                else:
                    allow_hf_cache = allow_hf_cache_env.strip().lower() in ("1", "true", "yes")

                if allow_hf_cache:
                    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME") or "", "hub")
                    if not hub:
                        hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
                    hub_dir = "models--" + str(self.hub_id).replace("/", "--")
                    snaps = os.path.join(hub, hub_dir, "snapshots", "*")
                    cand = sorted([p for p in glob.glob(snaps) if os.path.isdir(p)])
                    if cand:
                        snap = cand[-1]
                        if snap not in candidate_paths:
                            candidate_paths.append(snap)
                else:
                    if getattr(sys, "frozen", False):
                        logger.info("DeepSeek-OCR: skipping HF cache snapshots in frozen build (DEEPSEEK_OCR_ALLOW_HF_CACHE not enabled).")
            except Exception:
                pass
            
            # 3. 尝试源码路径 (已移除错误路径，改为直接依赖 HF 缓存)
            # source_path = r"F:\RSNA\DeepSeek-OCR-main\DeepSeek-OCR-main" 
            # This path is source code only, not model weights.
            # We skip it to avoid "Unrecognized model" error.

            if self.model_path and self.model_path not in candidate_paths:
                candidate_paths.append(self.model_path)
            if self.hub_id and self.hub_id not in candidate_paths and (not strict_local):
                candidate_paths.append(self.hub_id)

            if self.strict_ocr1:
                hub_norm = str(self.hub_id or "").replace("\\", "/").lower()
                keep = []
                for p in candidate_paths:
                    if not p:
                        continue
                    if p == self.hub_id:
                        keep.append(p)
                        continue
                    pn = str(p).replace("\\", "/").lower()
                    if "deepseek_ocr" in pn or "assets/models/deepseek_ocr" in pn:
                        keep.append(p)
                        continue
                    if hub_norm and "models--" in pn and "snapshots" in pn and hub_norm.replace("/", "--") in pn:
                        keep.append(p)
                        continue
                candidate_paths = keep

            # 遍历尝试加载
            model_loaded = False
            last_error = None
            loaded_path = None

            def _looks_like_model_dir(p: str) -> bool:
                try:
                    if not p or not os.path.isdir(p):
                        return False
                    return (
                        os.path.exists(os.path.join(p, "config.json"))
                        or os.path.exists(os.path.join(p, "tokenizer_config.json"))
                        or os.path.exists(os.path.join(p, "tokenizer.json"))
                        or os.path.exists(os.path.join(p, "model.safetensors"))
                        or os.path.exists(os.path.join(p, "pytorch_model.bin"))
                    )
                except Exception:
                    return False
            
            for path in candidate_paths:
                try:
                    print(f"尝试从路径加载模型: {path}")
                    if os.path.isabs(path) and not os.path.exists(path):
                        raise FileNotFoundError(f"模型路径不存在: {path}")
                    if os.path.isabs(path) and os.path.exists(path) and not _looks_like_model_dir(path):
                        raise RuntimeError(f"模型目录缺少必要文件: {path}")
                    
                    # 本地路径无需注入自定义模块；依赖 transformers 的 trust_remote_code 加载
                    if os.path.exists(path) and os.path.isdir(path):
                        # Mock easydict
                        if 'easydict' not in sys.modules:
                            try:
                                import easydict
                            except ImportError:
                                print("Warning: easydict not found, installing fallback...")
                                class EasyDict(dict):
                                    def __getattr__(self, name):
                                        if name in self: return self[name]
                                        raise AttributeError(name)
                                    def __setattr__(self, name, value):
                                        self[name] = value
                                sys.modules['easydict'] = types.ModuleType('easydict')
                                sys.modules['easydict'].EasyDict = EasyDict

                    # 加载 Tokenizer
                    # [Fix] Force local_files_only=True to prevent online check
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        path, 
                        trust_remote_code=True, 
                        use_fast=True,
                        local_files_only=True
                    )
                    try:
                        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                    except Exception:
                        pass
                    
                    # 加载 Model（严格对齐 DeepSeek-OCR-main 的 HF 推理方式）
                    use_cuda = str(self.device).startswith("cuda")
                    attn_impls = ["sdpa", "eager"] if use_cuda else ["eager"]
                    if use_cuda:
                        try:
                            import importlib.util as _importlib_util
                            if _importlib_util.find_spec("flash_attn") is not None:
                                attn_impls = ["flash_attention_2"] + attn_impls
                        except Exception:
                            pass

                    kwargs = dict(
                        trust_remote_code=True,
                        use_safetensors=True,
                        local_files_only=True,
                    )
                    last_attn_error = None
                    config = None

                    def _try_patch_config_from_weights(model_dir: str):
                        nonlocal config
                        if not model_dir or not os.path.isdir(model_dir):
                            return
                        if config is None:
                            try:
                                config = AutoConfig.from_pretrained(
                                    model_dir,
                                    trust_remote_code=True,
                                    local_files_only=True,
                                )
                            except Exception:
                                config = None
                                return
                        try:
                            import json as _json
                            from safetensors.torch import safe_open as _safe_open

                            index_path = os.path.join(model_dir, "model.safetensors.index.json")
                            weight_map = {}
                            if os.path.isfile(index_path):
                                with open(index_path, "r", encoding="utf-8") as f:
                                    idx = _json.load(f) or {}
                                weight_map = idx.get("weight_map") or {}

                            def _get_shape(key: str):
                                fname = weight_map.get(key)
                                if not fname:
                                    for cand in ("model.safetensors", "model-00001-of-000001.safetensors"):
                                        p = os.path.join(model_dir, cand)
                                        if os.path.isfile(p):
                                            fname = cand
                                            break
                                if not fname:
                                    return None
                                p = os.path.join(model_dir, fname)
                                if not os.path.isfile(p):
                                    return None
                                with _safe_open(p, framework="pt", device="cpu") as sf:
                                    if key not in sf.keys():
                                        return None
                                    t = sf.get_tensor(key)
                                    return tuple(t.shape)

                            proj_shape = _get_shape("model.projector.layers.weight")
                            sam_shape = _get_shape("model.sam_model.net_3.weight")
                            if not proj_shape and not sam_shape:
                                return

                            patched = False

                            if proj_shape and len(proj_shape) == 2:
                                input_dim = int(proj_shape[1])
                                pc = getattr(config, "projector_config", None)
                                if isinstance(pc, dict) and int(pc.get("input_dim") or 0) != input_dim:
                                    pc["input_dim"] = input_dim
                                    logger.warning(f"DeepSeek-OCR: patched projector_config.input_dim from {pc.get('input_dim')} to {input_dim}")
                                    patched = True

                            if sam_shape and len(sam_shape) >= 1:
                                last_ch = int(sam_shape[0])
                                vc = getattr(config, "vision_config", None)
                                if isinstance(vc, dict):
                                    width = vc.get("width") or {}
                                    sam_cfg = width.get("sam_vit_b") or {}
                                    ch = sam_cfg.get("downsample_channels")
                                    if isinstance(ch, list) and ch:
                                        if int(ch[-1]) != last_ch:
                                            ch[-1] = last_ch
                                            sam_cfg["downsample_channels"] = ch
                                            width["sam_vit_b"] = sam_cfg
                                            vc["width"] = width
                                            logger.warning(f"DeepSeek-OCR: patched downsample_channels to {ch}")
                                            patched = True

                            vision_key_sample = "model.vision_model.embeddings.class_embedding"
                            has_vision_weights = _get_shape(vision_key_sample) is not None
                            current_use_vision = getattr(config, "use_vision_model", True)
                            if not has_vision_weights and current_use_vision is not False:
                                setattr(config, "use_vision_model", False)
                                logger.warning("DeepSeek-OCR: vision_model weights missing, setting use_vision_model=false")
                                patched = True
                            elif has_vision_weights and current_use_vision is not True:
                                setattr(config, "use_vision_model", True)
                                logger.warning("DeepSeek-OCR: vision_model weights found, setting use_vision_model=true")
                                patched = True

                            if patched:
                                logger.warning("Detected DeepSeek-OCR config/weights mismatch. Patching config to match checkpoint shapes.")
                                print("Detected DeepSeek-OCR config/weights mismatch. Patching config to match checkpoint shapes.")
                        except Exception:
                            return

                    def _clear_model():
                        try:
                            self.model = None
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                    
                    def _set_attn_impl(cfg, impl: str) -> bool:
                        if cfg is None:
                            return False
                        for attr in ("attn_implementation", "_attn_implementation"):
                            try:
                                if hasattr(cfg, attr):
                                    setattr(cfg, attr, impl)
                                    return True
                            except Exception:
                                pass
                        try:
                            setattr(cfg, "_attn_implementation", impl)
                            return True
                        except Exception:
                            return False

                    try:
                        if os.path.isdir(path):
                            _try_patch_config_from_weights(path)
                        for attn_impl in attn_impls:
                            try:
                                kwargs_try = dict(kwargs)
                                if config is not None:
                                    _set_attn_impl(config, attn_impl)
                                else:
                                    kwargs_try["_attn_implementation"] = attn_impl
                                if config is not None:
                                    self.model = AutoModel.from_pretrained(path, config=config, **kwargs_try)
                                else:
                                    self.model = AutoModel.from_pretrained(path, **kwargs_try)
                                last_attn_error = None
                                break
                            except TypeError as e:
                                msg = str(e)
                                if "_attn_implementation" in msg and "unexpected keyword argument" in msg:
                                    try:
                                        kwargs_try.pop("_attn_implementation", None)
                                    except Exception:
                                        pass
                                    try:
                                        if config is not None:
                                            if hasattr(config, "attn_implementation"):
                                                setattr(config, "attn_implementation", attn_impl)
                                            elif hasattr(config, "_attn_implementation"):
                                                setattr(config, "_attn_implementation", attn_impl)
                                    except Exception:
                                        pass
                                    try:
                                        if config is not None:
                                            self.model = AutoModel.from_pretrained(path, config=config, **kwargs_try)
                                        else:
                                            self.model = AutoModel.from_pretrained(path, **kwargs_try)
                                        logger.warning("DeepSeek-OCR: model does not accept _attn_implementation; loaded without it (attn_impl=%s).", attn_impl)
                                        last_attn_error = None
                                        break
                                    except Exception as e2:
                                        last_attn_error = e2
                                        self.model = None
                                        continue
                                last_attn_error = e
                                self.model = None
                                continue
                            except Exception as e:
                                last_attn_error = e
                                self.model = None
                                continue
                        if self.model is None and last_attn_error is not None:
                            raise last_attn_error

                        self.model.eval()
                        if str(self.device).startswith("cuda"):
                            self.model = self.model.cuda()
                            try:
                                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                            except Exception:
                                dtype = torch.float16
                            self.model = self.model.to(dtype)
                        else:
                            self.model = self.model.cpu().to(torch.float32)
                    except Exception as e:
                        _clear_model()
                        raise e
                    
                    pretty_name = "DeepSeek-OCR"
                    print(f"成功从 {path} 加载 {pretty_name} 模型")
                    logger.info("DeepSeek-OCR: loaded model from %s", path)
                    model_loaded = True
                    loaded_path = path
                    break
                    
                except Exception as e:
                    print(f"从 {path} 加载失败: {e}")
                    last_error = e
                    continue
            
            if not model_loaded:
                raise last_error or Exception("所有路径均加载失败")

            if self.strict_ocr1:
                cfg_name = None
                try:
                    cfg_name = getattr(getattr(self.model, "config", None), "_name_or_path", None)
                except Exception:
                    cfg_name = None
                cfg_name_norm = str(cfg_name or "").replace("\\", "/").lower()
                hub_norm = str(self.hub_id or "").replace("\\", "/").lower()
                ok = False
                if cfg_name == self.hub_id:
                    ok = True
                elif hub_norm and hub_norm in cfg_name_norm:
                    ok = True
                elif "models--deepseek-ai--deepseek-ocr" in cfg_name_norm:
                    ok = True
                elif loaded_path and os.path.isdir(loaded_path):
                    try:
                        import json as _json
                        cfg_file = os.path.join(loaded_path, "config.json")
                        if os.path.isfile(cfg_file):
                            with open(cfg_file, "r", encoding="utf-8") as f:
                                raw_cfg = _json.load(f) or {}
                            name_or_path = str(raw_cfg.get("_name_or_path") or "").replace("\\", "/")
                            if name_or_path == self.hub_id:
                                ok = True
                            elif hub_norm and hub_norm in name_or_path.lower():
                                ok = True
                    except Exception:
                        pass
                if not ok:
                    raise RuntimeError(f"严格 OCR1 模式下模型不匹配: _name_or_path={cfg_name}")
            
            # Require tokenizer to be present; try loading from the successful model path if missing
            if not self.tokenizer and loaded_path:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        loaded_path, trust_remote_code=True, use_fast=True, local_files_only=True
                    )
                    if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                except Exception as e:
                    logger.warning("DeepSeek-OCR: tokenizer reload failed from %s: %s", loaded_path, e)

            if not self.tokenizer:
                raise Exception("Tokenizer not available in local model path")
            pad_id = getattr(self.tokenizer, 'pad_token_id', None)
            if pad_id is None:
                eos_id = getattr(self.tokenizer, 'eos_token_id', None)
                if eos_id is not None:
                    try:
                        self.tokenizer.pad_token_id = eos_id
                        pad_id = eos_id
                    except Exception:
                        pad_id = eos_id
                if pad_id is None and hasattr(self.model, 'config'):
                    pad_id = getattr(self.model.config, 'pad_token_id', None) or getattr(self.model.config, 'eos_token_id', None) or 0
            if hasattr(self.model, 'generation_config'):
                try:
                    self.model.generation_config.pad_token_id = pad_id
                    self.model.generation_config.do_sample = False
                except Exception:
                    pass
            
            self.model.eval()
            self.available = True
            self.loaded_model_path = loaded_path
            print(f"DeepSeek-OCR model loaded successfully (dtype={getattr(dtype_default, '__name__', str(dtype_default))})")
            
        except Exception as e:
            self.load_error = str(e)
            logger.exception("DeepSeek-OCR init failed: %s", e)
            print(f"DeepSeek-OCR init failed: {e}")
            import traceback
            traceback.print_exc()
            self.available = False

    def extract_text_from_image(self, image_path: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        start = time.time()
        config = config or {}
        engine_name = "deepseek_ocr"
        pretty_name = "DeepSeek-OCR"
        
        # 检查加载状态
        if self.loading:
            return {
                'text': '',
                'text_boxes': [],
                'engine': engine_name,
                'confidence': 0.0,
                'processing_time': 0.0,
                'error': f'{pretty_name} 模型正在加载中',
                'status': 'loading',
            }
            
        if not self.available:
            error_msg = f"{pretty_name} 不可用: {self.load_error}" if self.load_error else f"{pretty_name} 不可用"
            return {
                'text': '',
                'text_boxes': [],
                'engine': engine_name,
                'confidence': 0.0,
                'processing_time': time.time() - start,
                'error': error_msg,
                'status': 'unavailable',
            }
            
        # [Fix] 显式禁用网络连接，防止联网检查导致的超时
        # 我们已经确认模型在本地，强制离线模式
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        mode = (config.get("deepseek_mode") or config.get("mode") or "free").strip().lower()
        prompt = (config.get("deepseek_prompt") or "").strip()
        if not prompt:
            if getattr(self, "strict_ocr1", False):
                prompt = "<image>\n<|grounding|>Convert the document to markdown."
            else:
                prompt = "<image>\nFree OCR."
        try:
            print(f"DeepSeek OCR Inferencing on: {image_path}")
            
            # === 关键修复：强制单线程推理 ===
            # 防止 MKL 多线程冲突导致的 RangeChecks 异常
            # 并在 CPU 上进行额外的保护
            with torch.no_grad():
                # 再次确保模型处于 eval 模式
                self.model.eval()
                
                # 调用模型推理
                output_dir = os.path.dirname(image_path)
                if not output_dir: # Handle relative path like "test.png" -> ""
                    output_dir = "."

                # We are forcing GPU now, so we don't need to mask CPU availability
                # force_cpu = False
                # try:
                #    p = next(self.model.parameters())
                #    force_cpu = getattr(p, "device", None) is None or p.device.type == "cpu"
                # except Exception:
                #    force_cpu = self.device == "cpu"
                
                try:
                    use_autocast = False
                    if str(self.device).startswith("cuda"):
                        try:
                            use_autocast = next(self.model.parameters()).dtype == torch.float16
                        except Exception:
                            use_autocast = False
                    autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if use_autocast else contextlib.nullcontext()
                    orig_masked_scatter = None
                    if str(self.device).startswith("cuda") and self.model_kind != "ocr2":
                        try:
                            orig_masked_scatter = torch.Tensor.masked_scatter_
                            def _masked_scatter_cast(self_tensor, mask, source):
                                if getattr(self_tensor, "dtype", None) is not None and getattr(source, "dtype", None) is not None:
                                    if self_tensor.dtype != source.dtype:
                                        source = source.to(self_tensor.dtype)
                                return orig_masked_scatter(self_tensor, mask, source)
                            torch.Tensor.masked_scatter_ = _masked_scatter_cast
                        except Exception:
                            orig_masked_scatter = None
                    # Ensure model is on the correct device and sync all inputs
                    model_device = next(self.model.parameters()).device
                    print(f"Model device: {model_device}, self.device: {self.device}")
                    
                    # Force model to self.device if there's a mismatch
                    if str(model_device) != self.device:
                        print(f"Moving model from {model_device} to {self.device}")
                        if str(self.device).startswith("cuda"):
                            self.model = self.model.cuda()
                        else:
                            self.model = self.model.cpu()
                        model_device = next(self.model.parameters()).device
                    
                    # Handle CUDA availability based on actual system capability
                    original_cuda_available = None
                    if self.device == "cpu" or not torch.cuda.is_available():
                        original_cuda_available = torch.cuda.is_available
                        torch.cuda.is_available = lambda: False
                    
                    try:
                        with autocast_ctx:
                            base_size = config.get("base_size")
                            image_size = config.get("image_size")
                            crop_mode = config.get("crop_mode")
                            test_compress = config.get("test_compress")

                            base_size = 1024 if base_size is None else base_size
                            image_size = 640 if image_size is None else image_size
                            crop_mode = True if crop_mode is None else crop_mode

                            if test_compress is None:
                                test_compress = False

                            output_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")
                            # 强制 save_results=True 以确保如果 infer 返回空，我们能从文件恢复
                            res = self.model.infer(
                                self.tokenizer,
                                prompt=prompt,
                                image_file=image_path,
                                output_path=output_dir,
                                base_size=int(base_size),
                                image_size=int(image_size),
                                crop_mode=bool(crop_mode),
                                test_compress=bool(test_compress),
                                save_results=True, 
                                eval_mode=True,
                            )
                    finally:
                        if orig_masked_scatter is not None:
                            try:
                                torch.Tensor.masked_scatter_ = orig_masked_scatter
                            except Exception:
                                pass
                        # Restore original CUDA availability function
                        if original_cuda_available is not None:
                            try:
                                torch.cuda.is_available = original_cuda_available
                            except Exception:
                                pass
                except RuntimeError as e:
                    msg = str(e)
                    if "device meta" in msg or "expected device cuda" in msg:
                        self.load_error = f"推理失败（检测到 meta 张量/设备不一致），建议关闭 device_map 自动分配或改为整模上 GPU: {e}"
                    if "at least two devices" in msg and "cpu and cuda" in msg:
                        self.load_error = f"推理失败（CPU/GPU 混用导致设备不一致），建议确认模型与输入在同一设备: {e}"
                    print(f"DeepSeek OCR inference failed: {e}")
                    print(f"Model device: {next(self.model.parameters()).device if hasattr(self.model, 'parameters') else 'unknown'}")
                    print(f"Self device: {self.device}")
                    raise
            
            text = ""
            if res is None:
                text = ""
            elif isinstance(res, str):
                text = res
            elif isinstance(res, dict):
                # 增加对更多可能键名的支持
                text = (res.get("text") or res.get("extracted_text") or res.get("result") or 
                        res.get("output") or res.get("response") or "").strip()
            else:
                text = str(res)

            raw_text = text
            if not text.strip() or text == "None":
                # 优化文件搜索逻辑，递归搜索 output_dir
                print(f"DeepSeek OCR infer returned empty, searching output_dir: {output_dir}")
                found_files = []
                for root, dirs, files in os.walk(output_dir):
                    for fname in files:
                        if fname.lower() in ("result.md", "result.mmd", "result.txt"):
                            found_files.append(os.path.join(root, fname))
                
                # 按照优先级排序
                priority = {"result.md": 0, "result.mmd": 1, "result.txt": 2}
                found_files.sort(key=lambda x: priority.get(os.path.basename(x).lower(), 99))
                
                for p in found_files:
                    try:
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                        if text.strip():
                            print(f"Successfully recovered OCR text from file: {p}")
                            break
                    except Exception as e:
                        print(f"Failed to read OCR result file {p}: {e}")
                        continue

            try:
                text = (
                    text.replace("<｜begin▁of▁sentence｜>", "")
                    .replace("<｜end▁of▁sentence｜>", "")
                    .strip()
                )
            except Exception:
                pass

            fmt = (config.get("deepseek_output_format") or "").strip().lower()
            if not fmt:
                fmt = (os.environ.get("DEEPSEEK_OCR_OUTPUT_FORMAT") or "").strip().lower()
            if not fmt:
                fmt = "markdown_clean" if getattr(self, "strict_ocr1", False) else "plain_clean"
            text = self._format_ocr_text(text, fmt)
            
            # Clean up memory after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return {
                'text': text.strip(),
                'text_raw': (raw_text or "").strip(),
                'extracted_text': text.strip(),
                'text_boxes': [],
                'engine': engine_name,
                'ocr_engine': engine_name,
                'confidence': 0.0,
                'confidence_score': 0.0,
                'processing_time': time.time() - start
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"DeepSeek OCR inference failed: {e}")
            
            # 尝试返回友好的错误信息而不是崩溃
            return {
                'text': '', 
                'extracted_text': '',
                'text_boxes': [], 
                'engine': engine_name, 
                'ocr_engine': engine_name,
                'confidence': 0.0, 
                'confidence_score': 0.0,
                'processing_time': time.time() - start,
                'error': f"{pretty_name} 推理失败: {str(e)}",
                'status': 'error',
            }
        finally:
            # Clean up memory even on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
