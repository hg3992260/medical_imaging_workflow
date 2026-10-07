import subprocess
import re
import time
import logging
import threading
import queue
import requests
import json
import os
import sys
from typing import List, Optional
from src.utils.model_config_loader import get_model_paths
from src.utils.gpu_manager import gpu_manager

logger = logging.getLogger(__name__)

def _project_root_dir() -> str:
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        internal_dir = getattr(sys, "_MEIPASS", os.path.join(base_dir, "_internal"))
        internal_root = os.path.join(internal_dir, "assets", "models_home")
        if os.path.isdir(os.path.join(internal_root, "blobs")):
            return internal_dir
        root = os.path.join(base_dir, "assets", "models_home")
        if os.path.isdir(os.path.join(root, "blobs")):
            return base_dir
        cwd_root = os.path.join(os.getcwd(), "assets", "models_home")
        if os.path.isdir(os.path.join(cwd_root, "blobs")):
            return os.getcwd()
        return internal_dir
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _get_frozen_internal_dir() -> str:
    base_dir = os.path.dirname(sys.executable)
    return getattr(sys, "_MEIPASS", os.path.join(base_dir, "_internal"))


def _iter_models_home_candidates() -> list:
    return [os.path.join(_project_root_dir(), "assets", "models_home")]


def _iter_ollama_binary_candidates() -> list:
    base_dir = os.path.dirname(sys.executable)
    internal_dir = _get_frozen_internal_dir()
    candidates = [
        os.path.join(_project_root_dir(), "assets", "ollama", "ollama.exe"),
        os.path.join(base_dir, "_internal", "assets", "ollama", "ollama.exe"),
        os.path.join(internal_dir, "assets", "ollama", "ollama.exe"),
        os.path.join(base_dir, "assets", "ollama", "ollama.exe"),
    ]
    return list(dict.fromkeys(candidates))


def _bool_env(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() not in ["0", "false", "no", "off", ""]


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _gpu_layers_ladder(max_layers: int) -> List[int]:
    if max_layers == -1:
        return [-1]
    ladder = [999, 80, 72, 64, 56, 48, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0]
    out = []
    for x in ladder:
        if x <= max_layers and x not in out:
            out.append(x)
    if max_layers not in out:
        out.insert(0, max_layers)
    return out


def _is_oom_error(msg: str) -> bool:
    s = (msg or "").lower()
    patterns = [
        "out of memory",
        "cuda out of memory",
        "cublas",
        "cuda error",
        "failed to allocate",
        "failed to alloc",
        "ggml_cuda",
        "cuda malloc",
        "allocation failed",
        "not enough memory",
    ]
    return any(p in s for p in patterns)

def _is_empty_generation_error(msg: str) -> bool:
    s = (msg or "").strip().lower()
    return s in {"empty response", "empty cli response"}


def _query_nvidia_smi_memory(main_gpu: int = 0) -> tuple:
    try:
        creation_flags = 0x08000000 if sys.platform == 'win32' else 0
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=3,
            creationflags=creation_flags,
        )
        if p.returncode != 0:
            return (0, 0)
        rows = [x.strip() for x in (p.stdout or "").splitlines() if x.strip()]
        if not rows:
            return (0, 0)
        idx = max(0, min(int(main_gpu), len(rows) - 1))
        parts = [s.strip() for s in rows[idx].split(",")]
        if len(parts) < 2:
            return (0, 0)
        total = int(float(parts[0]))
        used = int(float(parts[1]))
        return (max(total, 0), max(used, 0))
    except Exception:
        return (0, 0)


class OllamaLocalService:
    def __init__(self, binary: str = "", base_url: str = None):
        self.logger = logger # Add logger ref for instance methods
        
        # === ISOLATION FIX: REMOVED to avoid conflict with system Ollama ===
        # We prefer to use the system instance if available (port 11434).
        # Only if not available, we might start our own.
        # if getattr(sys, 'frozen', False):
        #    base_url = "http://localhost:11435"
        #    os.environ['OLLAMA_HOST'] = "127.0.0.1:11435"
        #    logger.info(f"Frozen environment detected. Forcing OLLAMA_HOST to {os.environ['OLLAMA_HOST']}")
        
        # 优先使用环境变量中的 OLLAMA_HOST (仅在非打包模式，或 base_url 仍为空时)
        if not base_url:
            env_host = os.environ.get('OLLAMA_HOST')
            if env_host:
                # 确保有 http 前缀
                if not env_host.startswith('http'):
                    base_url = f"http://{env_host}"
                else:
                    base_url = env_host
            else:
                base_url = "http://localhost:11434"
        
        logger.info(f"OllamaService initializing with base_url: {base_url}")
        
        # Register with GPU Manager
        gpu_manager.register("ollama", self.unload_model)

        # Default: only "force bundled" when a bundled binary is actually present.
        # In the open-source build there is no bundled Ollama, so we fall back to a
        # user-installed Ollama (or an external OpenAI-compatible API).
        self._force_bundled = _bool_env("RSNA_OLLAMA_FORCE_BUNDLED", False)
        for cand in _iter_ollama_binary_candidates():
            if os.path.exists(cand):
                binary = cand
                logger.info(f"Using bundled Ollama binary: {binary}")
                break
        if self._force_bundled and not binary:
            logger.error("Bundled Ollama binary not found. Expected assets/ollama/ollama.exe")
            binary = ""
        
        self.binary = binary
        self.base_url = base_url
        
        # Cache for list_models to reduce overhead
        self._models_cache = []
        self._last_cache_time = 0
        self._cache_ttl = 60 # seconds
        self._gpu_layers_cache = {}
        try:
            self._gpu_budget_ratio = float(os.environ.get("OLLAMA_GPU_BUDGET_RATIO", "0.97"))
        except Exception:
            self._gpu_budget_ratio = 0.97
        self._gpu_budget_ratio = max(0.7, min(0.99, self._gpu_budget_ratio))
        try:
            total_mib, _ = _query_nvidia_smi_memory(0)
            auto_cap_gb = max(9.0, (total_mib / 1024) * 0.99)
        except Exception:
            auto_cap_gb = 9.0
        env_cap = os.environ.get("OLLAMA_GPU_VRAM_CAP_GB", "")
        cap_gb = float(env_cap) if env_cap else auto_cap_gb
        self._gpu_vram_cap_mib = int(max(2048, cap_gb * 1024))

    def _emit_status(self, status_callback, message: str):
        if not callable(status_callback):
            return
        try:
            status_callback(str(message or ""))
        except Exception:
            pass

    def _extract_model_b(self, model: str) -> float:
        s = str(model or "").lower()
        m = re.search(r'(\d+(?:\.\d+)?)\s*b\b', s)
        if not m:
            m = re.search(r'[:\-](\d+(?:\.\d+)?)b\b', s)
        if not m:
            m = re.search(r'(\d+(?:\.\d+)?)b\b', s)
        if not m:
            return 0.0
        try:
            return float(m.group(1))
        except Exception:
            return 0.0

    def _should_force_hybrid(self, model: str, effective_total_mib: int) -> bool:
        model_b = self._extract_model_b(model)
        if model_b <= 8.0:
            return False
        return effective_total_mib <= 10240

    def _apply_gpu_budget(self, opts: dict, model: str = "") -> dict:
        options = dict(opts or {})
        main_gpu = 0
        try:
            main_gpu = int(options.get("main_gpu", os.environ.get("OLLAMA_MAIN_GPU", "0")))
        except Exception:
            main_gpu = 0
        total_mib, used_mib = _query_nvidia_smi_memory(main_gpu)
        if total_mib <= 0:
            return options

        # Enforce hard cap so Ollama (weights + KV cache) stays under 9.5G budget envelope
        effective_total_mib = min(total_mib, self._gpu_vram_cap_mib)
        budget_mib = int(effective_total_mib * self._gpu_budget_ratio)
        # Fixed budget: use guaranteed VRAM allocation instead of subtracting current usage.
        # PyTorch's cached memory is invisible to nvidia-smi's "used" counter,
        # so usable_mib must be based on absolute budget, not budget - used.
        usable_mib = budget_mib

        try:
            original_layers = int(options.get("num_gpu", options.get("gpu_layers", -1)))
        except Exception:
            original_layers = -1
            
        capped_layers = original_layers
        if self._should_force_hybrid(model, effective_total_mib) and original_layers != 0:
            capped_layers = -1
        # Let Ollama natively decide the GPU/CPU split based on OLLAMA_MAX_VRAM and OLLAMA_GPU_OVERHEAD.
        # We pass the original requested layers (-1 or 999) and let Ollama handle hybrid mode natively.
        
        try:
            original_ctx = int(options.get("num_ctx", 8192))
        except Exception:
            original_ctx = 8192
        # Global ceiling for KV cache growth under 9.5G VRAM cap (with q8_0 quantization)
        capped_ctx = min(original_ctx, 8192) # Increased ceiling slightly as q8_0 allows more context
        if usable_mib < 256:
            capped_ctx = min(capped_ctx, 1024)
        elif usable_mib < 512:
            capped_ctx = min(capped_ctx, 2048)
        elif usable_mib < 2048:
            capped_ctx = min(capped_ctx, 4096)
        elif usable_mib < 4096:
            capped_ctx = min(capped_ctx, 6144)

        options["num_gpu"] = int(capped_layers)
        options["num_ctx"] = int(capped_ctx)
        
        # Remove old key if present
        if "gpu_layers" in options:
            del options["gpu_layers"]
            
        logger.info(
            f"Ollama GPU budget applied: model={model} gpu={main_gpu} total={total_mib}MiB used={used_mib}MiB "
            f"effective_total={effective_total_mib}MiB cap={self._gpu_vram_cap_mib}MiB "
            f"budget={budget_mib}MiB usable={usable_mib}MiB num_gpu={options['num_gpu']} num_ctx={options['num_ctx']}"
        )
        return options

    def unload_model(self):
        """Force unload model from GPU to free resources"""
        try:
            logger.info("Ollama: Unloading model to free GPU...")
            # Unload all currently loaded models via the ps + generate route
            try:
                import urllib.request, json as _json
                ps_req = urllib.request.Request(f"{self.base_url}/api/ps", headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(ps_req, timeout=3) as r:
                    models_info = _json.loads(r.read().decode())
                if isinstance(models_info, dict) and models_info.get("models"):
                    for mi in models_info["models"]:
                        mname = mi.get("name", "")
                        if mname:
                            requests.post(f"{self.base_url}/api/generate", json={"model": mname, "keep_alive": 0}, timeout=5)
                            logger.info(f"Ollama: unloaded model={mname}")
            except Exception:
                pass
            requests.post(f"{self.base_url}/api/generate", json={"keep_alive": 0}, timeout=3)
            import time
            for _ in range(3):
                time.sleep(1.0)
                _, used = _query_nvidia_smi_memory(0)
                if used < 7000:
                    break
            logger.info("Ollama: GPU release sequence completed")
        except Exception as e:
            logger.debug(f"Ollama unload request failed (expected if not running): {e}")

    def ensure_model_ready(self, model: str, status_callback=None, timeout: int = 180):
        self._emit_status(status_callback, f"Ollama 本地模型 `{model}` 正在加载到显存，请稍候…")
        gpu_manager.request_gpu("ollama")
        import time
        vram_freed = False
        for i in range(15):
            time.sleep(1.0)
            total, used = _query_nvidia_smi_memory(0)
            free_pct = 100 * (total - used) / max(total, 1)
            logger.info(f"VRAM wait: {i+1}/15s used={used}MiB/{total}MiB free={free_pct:.1f}%")
            if used < 7000:
                vram_freed = True
                break
        if not vram_freed:
            logger.warning("VRAM not freed within 15s, Ollama will load in CPU mode (num_gpu=0)")
            self._emit_status(status_callback, "等待显存释放超时，将以 CPU 为主加载 Ollama 模型，生成速度会较慢")
        available_models = self.list_models()
        resolved_model = model
        if available_models and resolved_model not in available_models:
            best_match = None
            for m in available_models:
                if resolved_model in m or m in resolved_model:
                    best_match = m
                    break
            if not best_match and available_models:
                best_match = available_models[0]
            if best_match:
                resolved_model = best_match
                self._emit_status(status_callback, f"Ollama 已切换到可用模型 `{resolved_model}` 并开始预热…")

        warm_options = self._apply_gpu_budget({
            "num_predict": 8,
            "num_ctx": min(_int_env("OLLAMA_NUM_CTX", 8192), 1024),
            "num_gpu": int(os.environ.get("OLLAMA_GPU_LAYERS", "-1")),
            "main_gpu": int(os.environ.get("OLLAMA_MAIN_GPU", "0")),
        }, model=resolved_model)
        payload = {
            "model": resolved_model,
            "prompt": "Reply with OK only.",
            "stream": False,
            "options": warm_options,
            "keep_alive": "30m",
        }
        session = requests.Session()
        session.trust_env = False
        try:
            resp = session.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout)
            if resp.status_code != 200:
                err = f"API Error {resp.status_code}: {resp.text}"
                self._emit_status(status_callback, f"Ollama 模型加载失败：{err}；建议释放另一个模型后重试。")
                return {"success": False, "model": resolved_model, "error": err}
            self._emit_status(status_callback, f"Ollama 本地模型 `{resolved_model}` 已加载完成并可用。")
            return {"success": True, "model": resolved_model, "error": ""}
        except Exception as e:
            err = str(e)
            self._emit_status(status_callback, f"Ollama 模型加载失败：{err}；建议释放另一个模型后重试。")
            return {"success": False, "model": resolved_model, "error": err}

    def _clean_ansi(self, text: str) -> str:
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def clear_models_cache(self):
        self._models_cache = []
        self._last_cache_time = 0

    def list_models(self):
        # Check cache first
        if self._models_cache and (time.time() - self._last_cache_time < self._cache_ttl):
            return self._models_cache

        # 优先使用编译版内置路径，其次配置路径
        # 增加对当前工作目录下 assets/models_home 的检测（兼容用户手动复制模型的情况）
        if getattr(sys, 'frozen', False) or self._force_bundled:
            model_path = _iter_models_home_candidates()[0]
            if os.path.exists(model_path):
                os.environ["OLLAMA_MODELS"] = model_path
                logger.info(f"Using OLLAMA_MODELS: {model_path}")
            else:
                try:
                    paths = get_model_paths()
                    models_path = paths.get("ollama_models")
                    if models_path and os.path.exists(models_path):
                        os.environ["OLLAMA_MODELS"] = models_path
                        logger.info(f"Using configured OLLAMA_MODELS: {models_path}")
                except Exception as e:
                    logger.warning(f"Failed to set OLLAMA_MODELS from config: {e}")
        else:
            try:
                paths = get_model_paths()
                models_path = paths.get("ollama_models")
                if models_path and os.path.exists(models_path):
                    os.environ["OLLAMA_MODELS"] = models_path
                    logger.info(f"Using configured OLLAMA_MODELS: {models_path}")
            except Exception as e:
                logger.warning(f"Failed to set OLLAMA_MODELS from config: {e}")

        # === CRITICAL: Ensure Ollama service is running BEFORE listing models ===
        # If we are in a frozen app, we might need to kickstart it if it died or wasn't started
        if getattr(sys, 'frozen', False):
            # Check if API is responsive first
            try:
                 session = requests.Session()
                 session.trust_env = False
                 session.get(f"{self.base_url}/api/tags", timeout=0.2)
            except Exception:
                 # If not responsive, try to start it
                 logger.warning("Ollama API unresponsive, attempting proactive startup...")
                 self._start_bundled_ollama()

        # Prefer API, fallback to CLI if API fails
        try:
            # Use very short timeout for localhost check (500ms)
            session = requests.Session()
            session.trust_env = False
            resp = session.get(f"{self.base_url}/api/tags", timeout=0.5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m['name'] for m in data.get('models', [])]
                
                # === AGGRESSIVE FIX: If API works but returns NO models, it might be the wrong Ollama instance ===
                if not models and (getattr(sys, 'frozen', False) or self._force_bundled):
                    logger.warning("Ollama API returned 0 models! This might be a system instance conflict.")
                    logger.warning("Attempting to KILL existing Ollama and restart with correct env...")
                    
                    # 1. Kill existing
                    try:
                        subprocess.run("taskkill /F /IM ollama.exe", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        time.sleep(1.0) # Wait for death
                    except Exception as e:
                        logger.error(f"Failed to kill ollama: {e}")

                    # 2. Restart bundled
                    self._start_bundled_ollama()
                    
                    # 3. Retry listing (give it a moment to boot)
                    time.sleep(2.0)
                    try:
                        resp2 = requests.get(f"{self.base_url}/api/tags", timeout=1.0)
                        if resp2.status_code == 200:
                            models = [m['name'] for m in resp2.json().get('models', [])]
                    except: pass

                # Update cache
                self._models_cache = models
                self._last_cache_time = time.time()
                return models
        except Exception as e:
            # Log as debug to avoid noise, since we have fallback
            logger.debug(f"Ollama API list_models quick check failed: {e}")
            
            # === CRITICAL FIX: If API fails, it might mean Ollama died or hasn't started yet ===
            # We should try to restart it if we are in a bundled environment
            if getattr(sys, 'frozen', False) or self._force_bundled:
                 logger.warning("Ollama API failed, attempting to ensure service is running...")
                 self._start_bundled_ollama()
                 # Retry listing?
                 time.sleep(1.0)
                 try:
                    # One last try via API
                    resp = requests.get(f"{self.base_url}/api/tags", timeout=0.5)
                    if resp.status_code == 200:
                         models = [m['name'] for m in resp.json().get('models', [])]
                         return models
                 except: pass

        # Fallback to CLI
        if not self.binary:
            logger.error("Ollama bundled binary unavailable; CLI list fallback disabled.")
            return []
        try:
            # IMPORTANT: Pass environment to CLI subprocess too!
            env = os.environ.copy()
            if getattr(sys, 'frozen', False) or self._force_bundled:
                models_path = ""
                models_path = _iter_models_home_candidates()[0]
                if os.path.exists(models_path):
                    env["OLLAMA_MODELS"] = models_path
                    logger.info(f"Ollama CLI fallback: OLLAMA_MODELS={models_path}")

            res = subprocess.run(
                [self.binary, "list"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW,
                env=env  # Add environment here
            )
            if res.returncode != 0:
                logger.error(f"Ollama CLI list failed with return code {res.returncode}: {res.stderr}")
                return []
            
            lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
            models = []
            for line in lines:
                if line.lower().startswith("name "): continue
                parts = line.split()
                if parts: models.append(parts[0])
            
            # Update cache if successful
            if models:
                self._models_cache = models
                self._last_cache_time = time.time()
                return models
            # Offline parse manifests
            offline_models = []
            try:
                mp = env.get("OLLAMA_MODELS", "")
                manifests_dir = os.path.join(mp, "manifests")
                if os.path.isdir(manifests_dir):
                    for root, _, files in os.walk(manifests_dir):
                        for f in files:
                            if f.endswith(".json"):
                                p = os.path.join(root, f)
                                try:
                                    with open(p, "r", encoding="utf-8") as fh:
                                        data = json.load(fh)
                                        name = data.get("name")
                                        if name and name not in offline_models:
                                            offline_models.append(name)
                                except Exception:
                                    pass
            except Exception:
                pass
            if offline_models:
                self._models_cache = offline_models
                self._last_cache_time = time.time()
                return offline_models
            return []
        except Exception as e:
            logger.error(f"Ollama CLI list exception: {e}")
            return []

    def _start_bundled_ollama(self):
        """Helper to start the bundled Ollama instance with correct environment"""
        try:
            base_dir = os.path.dirname(sys.executable)
            bundled_ollama = ""
            for cand in _iter_ollama_binary_candidates():
                if os.path.exists(cand):
                    bundled_ollama = cand
                    break
            if bundled_ollama:
                creation_flags = 0x08000000 if sys.platform == 'win32' else 0
                
                env = os.environ.copy()
                if "OLLAMA_MODELS" not in env:
                    env["OLLAMA_MODELS"] = _iter_models_home_candidates()[0]
                
                # === MEMORY OPTIMIZATION ===
                # Optimize for 8GB-9.5GB VRAM scenarios to prevent fallback to CPU
                env["OLLAMA_NO_GPU"] = env.get("OLLAMA_NO_GPU", "0")
                # Remove OLLAMA_LLM_LIBRARY and CUDA_VISIBLE_DEVICES overrides to allow native auto-detection
                if "OLLAMA_LLM_LIBRARY" in env:
                    del env["OLLAMA_LLM_LIBRARY"]
                if "CUDA_VISIBLE_DEVICES" in env:
                    del env["CUDA_VISIBLE_DEVICES"]
                    
                env["OLLAMA_GPU_OVERHEAD"] = "0"
                env["OLLAMA_MAX_LOADED_MODELS"] = os.environ.get("OLLAMA_MAX_LOADED_MODELS", "1")
                env["OLLAMA_NUM_PARALLEL"] = os.environ.get("OLLAMA_NUM_PARALLEL", "1")
                env["OLLAMA_FLASH_ATTENTION"] = os.environ.get("OLLAMA_FLASH_ATTENTION", "1")
                env["OLLAMA_GPU_BUDGET_RATIO"] = os.environ.get("OLLAMA_GPU_BUDGET_RATIO", "0.97")
                try:
                    main_gpu = int(os.environ.get("OLLAMA_MAIN_GPU", "0"))
                    total_mib, _ = _query_nvidia_smi_memory(main_gpu)
                    if total_mib > 0:
                        effective_total_mib = min(total_mib, self._gpu_vram_cap_mib)
                        reserve_mib = int(effective_total_mib * (1.0 - self._gpu_budget_ratio))
                        # OLLAMA_GPU_OVERHEAD expects bytes! Convert MiB to bytes
                        reserve_bytes = reserve_mib * 1024 * 1024
                        max_vram_bytes = int(effective_total_mib * self._gpu_budget_ratio) * 1024 * 1024
                        env["OLLAMA_KV_CACHE_TYPE"] = os.environ.get("OLLAMA_KV_CACHE_TYPE", "q8_0")
                        # 强制覆写为全局安全的 9.5G 上限，防止动态测算越界
                        env["OLLAMA_MAX_VRAM"] = str(self._gpu_vram_cap_mib * 1024 * 1024)
                        env["OLLAMA_GPU_OVERHEAD"] = str(max(0, reserve_bytes))
                except Exception:
                    pass
                # env["CUDA_VISIBLE_DEVICES"] = "0"       # Optional: Force GPU 0 if multiple GPUs present

                logger.info(f"Starting bundled Ollama: {bundled_ollama} | OLLAMA_MODELS={env.get('OLLAMA_MODELS')}")
                
                # Set CWD to ollama directory so it can find its libs
                ollama_cwd = os.path.dirname(bundled_ollama)
                
                # Redirect output to a log file for debugging GPU issues
                log_root = _project_root_dir() if not getattr(sys, 'frozen', False) else os.path.dirname(base_dir)
                log_file = os.path.join(log_root, 'logs', 'ollama_server.log')
                try:
                    os.makedirs(os.path.dirname(log_file), exist_ok=True)
                    out_handle = open(log_file, 'a')
                    err_handle = subprocess.STDOUT # Merge stderr to stdout
                except:
                    out_handle = subprocess.DEVNULL
                    err_handle = subprocess.DEVNULL

                subprocess.Popen(
                        [bundled_ollama, 'serve'], 
                        stdout=out_handle, 
                        stderr=err_handle,
                        creationflags=creation_flags,
                        env=env,
                        cwd=ollama_cwd # CRITICAL: Run from ollama dir to find libs
                    )
                time.sleep(2.0) # Wait for startup
        except Exception as e:
            logger.error(f"Failed to start bundled Ollama: {e}")

    def _rank(self, name: str) -> int:
        m = re.search(r'(\d+)\s*b', name.lower())
        if m:
            try:
                return int(m.group(1))
            except:
                return 9999
        return 9999

    def pick_fallback(self, failed_model: str, tried: Optional[List[str]] = None):
        models = self.list_models()
        tried_set = {str(x).strip() for x in (tried or []) if str(x).strip()}
        candidates = [m for m in models if m != failed_model and m not in tried_set]
        if not candidates:
            return ""
        preferred_raw = os.environ.get("OLLAMA_EMPTY_RESPONSE_FALLBACK_MODELS", "")
        preferred = [x.strip() for x in preferred_raw.split(",") if x.strip()]
        if not preferred:
            preferred = ["llama3", "qwen3.5", "qwen3", "mistral", "qwen", "gpt-oss"]

        def _pref_score(name: str):
            nl = str(name or "").lower()
            for idx, key in enumerate(preferred):
                if key.lower() in nl:
                    return idx
            return len(preferred) + 1

        ranked = sorted(candidates, key=lambda n: (_pref_score(n), self._rank(n), str(n).lower()))
        return ranked[0]

    def _resolve_num_predict(self, num_predict: int, caller_options: dict = None) -> int:
        caller_options = caller_options or {}
        for key in ("num_predict", "max_tokens", "max_new_tokens"):
            try:
                value = caller_options.get(key)
                if value is not None and int(value) >= 0:
                    return int(value)
            except Exception:
                pass
        try:
            if num_predict is not None and int(num_predict) >= 0:
                return int(num_predict)
        except Exception:
            pass
        return max(256, _int_env("OLLAMA_NUM_PREDICT_DEFAULT", 4096))

    def generate(self, model: str, prompt: str, timeout: int = None, num_predict: int = -1, options: dict = None, stream_callback=None, response_format: dict = None, _fallback_tried: Optional[List[str]] = None, _fallback_depth: int = 0):
        """
        Generate using Ollama API /api/generate with keep_alive to avoid reloading models.
        Timeout may be None, but generation length still has a finite token cap.
        """
        # Request GPU access for Ollama (this will unload MagicSeg if needed)
        gpu_manager.request_gpu("ollama")

        # 1. Validate/Switch Model if needed
        available_models = self.list_models()
        if available_models and model not in available_models:
            # Try to find a close match (e.g., 'llama3' -> 'llama3:8b')
            best_match = None
            for m in available_models:
                if model in m or m in model:
                    best_match = m
                    break
            
            if not best_match:
                # Fallback to first available model if specific one not found
                # Prefer models with 'llama', 'qwen', 'mistral' in name
                for preferred in ['llama', 'qwen', 'mistral']:
                    for m in available_models:
                        if preferred in m.lower():
                            best_match = m
                            break
                    if best_match: break
                
                if not best_match:
                    best_match = available_models[0]
            
            logger.warning(f"Model '{model}' not found. Switching to '{best_match}'.")
            if stream_callback:
                try:
                    stream_callback(f"> ⚠️ Model '{model}' not found. Auto-switching to '{best_match}'...\n")
                except: pass
            model = best_match

        # === SPECIFIC FIX for Vision Models ===
        # If the model name implies it is a Vision-Language Model (like 'vl'), it might not support the standard /generate endpoint
        # or might behave differently. However, standard Ollama VL models usually support /generate.
        # But if the user tries to use a pure embedding model or a non-chat model, we should warn.
        
        logger.info(f"Ollama generating with model: {model} (Timeout: {timeout if timeout else 'Infinite'}s)")
        if stream_callback:
            try:
                stream_callback(f"\n> [Debug] Requesting Ollama API (Model: {model}) at {self.base_url}...\n")
            except: pass
            
        logs = []
        caller_options = dict(options or {})
        effective_num_predict = self._resolve_num_predict(num_predict, caller_options)

        num_ctx = 8192
        try:
            num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
        except Exception:
            num_ctx = 8192

        max_gpu_layers = -1
        try:
            max_gpu_layers = int(os.environ.get("OLLAMA_GPU_LAYERS", "-1"))
        except Exception:
            max_gpu_layers = -1

        dynamic_gpu_layers = _bool_env("OLLAMA_GPU_LAYERS_DYNAMIC", True) and ("num_gpu" not in caller_options)
        max_tries = 5
        try:
            max_tries = int(os.environ.get("OLLAMA_GPU_LAYERS_MAX_TRIES", "5"))
        except Exception:
            max_tries = 5

        start_gpu_layers = max_gpu_layers
        if dynamic_gpu_layers:
            try:
                cached = int(self._gpu_layers_cache.get(model, max_gpu_layers))
                start_gpu_layers = min(max_gpu_layers, max(-1, cached))
            except Exception:
                start_gpu_layers = max_gpu_layers
        elif "num_gpu" in caller_options:
            try:
                start_gpu_layers = int(caller_options.get("num_gpu"))
            except Exception:
                start_gpu_layers = max_gpu_layers
        elif "gpu_layers" in caller_options:
            try:
                start_gpu_layers = int(caller_options.get("gpu_layers"))
            except Exception:
                start_gpu_layers = max_gpu_layers

        base_options = {
            "num_predict": effective_num_predict,
            "num_ctx": num_ctx,
            "num_gpu": start_gpu_layers,
            "main_gpu": int(os.environ.get("OLLAMA_MAIN_GPU", "0")),
        }
        base_options.update(caller_options)
        base_options = self._apply_gpu_budget(base_options, model=model)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": base_options,
            "keep_alive": "30m",
        }
        if response_format and isinstance(response_format, dict):
            fmt_type = response_format.get("type", "")
            if fmt_type == "json_object":
                payload["format"] = "json"
            elif fmt_type in ("json_schema",):
                payload["format"] = response_format
            else:
                payload["format"] = response_format
        
        # 代理免疫辅助方法：使用系统底层的 urllib 请求，而不是 requests，这样能彻底无视一切上层代理环境变量
        import urllib.request
        import json
        import time

        def _safe_post(url, payload_dict, timeout_sec):
            req = urllib.request.Request(url, method="POST")
            req.add_header('Content-Type', 'application/json')
            data = json.dumps(payload_dict).encode('utf-8')
            # 建立空的 ProxyHandler 来彻底无视所有系统代理
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            return opener.open(req, data=data, timeout=timeout_sec)
        
        def _safe_get(url, timeout_sec):
            req = urllib.request.Request(url, method="GET")
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            return opener.open(req, timeout=timeout_sec)
        if "OLLAMA_NUM_GPU" in os.environ:
            try:
                if not self._should_force_hybrid(model, self._gpu_vram_cap_mib):
                    payload["options"]["num_gpu"] = int(os.environ["OLLAMA_NUM_GPU"])
            except Exception:
                pass
        
        def _api_generate(_payload):
            full_response = []
            start_time = time.time()
            try:
                logger.debug(f"Initiating safe urllib.post to {self.base_url}/api/generate")
                try:
                    resp = _safe_post(f"{self.base_url}/api/generate", _payload, timeout)
                    status_code = resp.getcode()
                except urllib.error.HTTPError as e:
                    resp = e
                    status_code = e.code

                logger.debug(f"safe urllib.post returned, status: {status_code}")
                if stream_callback:
                    try:
                        stream_callback(f"> [Debug] API Status Code: {status_code}\n")
                    except:
                        pass

                if status_code != 200:
                    try:
                        err_body = resp.read().decode('utf-8', errors='ignore')
                    except Exception:
                        err_body = str(resp)
                    error_msg = f"API Error {status_code}: {err_body}"
                    logger.warning(f"Ollama generation failed with {model}. Error: {error_msg}")
                    logs.append(f"api_error={status_code}")
                    if stream_callback:
                        try:
                            stream_callback(f"> [Debug] API Failed: {error_msg}\n")
                        except:
                            pass
                    fatal = bool(status_code == 400 and "does not support generate" in err_body)
                    return {"success": False, "response": "", "error": error_msg, "logs": logs, "fatal": fatal}

                chunk_count = 0
                for line in resp:
                    if timeout and (time.time() - start_time > timeout):
                        logger.error("Ollama generation timed out during stream")
                        return {"success": False, "response": "".join(full_response), "error": "timeout", "logs": logs, "fatal": False}

                    line = line.strip()
                    if line:
                        chunk_count += 1
                        try:
                            chunk = json.loads(line)
                            content = ""
                            if "response" in chunk:
                                content = chunk.get("response") or ""
                            elif isinstance(chunk.get("message"), dict):
                                # Compatibility path for chat-style chunk payloads.
                                content = chunk.get("message", {}).get("content") or ""
                            if content:
                                full_response.append(content)
                                if len(content.strip()) > 0:
                                    clean_content = content.replace('\n', '\\n')
                                    logger.debug(f"[Ollama Stream]: {clean_content}")
                                if stream_callback:
                                    try:
                                        stream_callback(content)
                                    except Exception:
                                        pass

                            if chunk.get("done", False):
                                end_time = time.time()
                                elapsed = end_time - start_time
                                elapsed_msg = f"\n[Generation completed in {elapsed:.2f}s, Chunks: {chunk_count}]\n"
                                logger.info(elapsed_msg.strip())
                                if stream_callback:
                                    try:
                                        stream_callback(elapsed_msg)
                                    except Exception:
                                        pass
                                break
                        except Exception as e:
                            logger.warning(f"Failed to parse chunk: {e} | Line: {line}")

                    if chunk_count == 0:
                        if stream_callback:
                            try:
                                stream_callback(f"> [Debug] Stream ended with 0 chunks received! (Possible empty response)\n")
                            except:
                                pass
                        return {"success": False, "response": "", "error": "Empty Stream", "logs": logs, "fatal": False}

                joined = "".join(full_response)
                if not joined.strip():
                    logger.warning(
                        "Ollama API returned a completed stream with empty content. model=%s chunk_count=%s",
                        model,
                        chunk_count,
                    )
                    return {"success": False, "response": "", "error": "Empty Response", "logs": logs, "fatal": False}
                return {"success": True, "response": joined, "error": "", "logs": logs, "fatal": False}

            except Exception as e:
                logger.warning(f"Ollama API connection failed: {e}")
                logs.append(f"connection_error={str(e)}")
                if stream_callback:
                    try:
                        stream_callback(f"> [Debug] Connection Error: {e}\n")
                    except:
                        pass
                return {"success": False, "response": "", "error": str(e), "logs": logs, "fatal": False}

        try:
            effective_start_gpu_layers = int(payload.get("options", {}).get("num_gpu", start_gpu_layers))
        except Exception:
            effective_start_gpu_layers = start_gpu_layers

        candidates = [effective_start_gpu_layers]
        if dynamic_gpu_layers and int(effective_start_gpu_layers) >= 0:
            ladder = _gpu_layers_ladder(max_gpu_layers)
            try:
                if effective_start_gpu_layers in ladder:
                    candidates = ladder[ladder.index(effective_start_gpu_layers):]
                else:
                    candidates = [effective_start_gpu_layers] + [x for x in ladder if x < effective_start_gpu_layers]
            except Exception:
                candidates = [effective_start_gpu_layers]

        last_error = ""
        for gi, gl in enumerate(candidates[: max(1, max_tries)]):
            # Ollama new API expects num_gpu; gpu_layers is ignored and can force CPU fallback.
            payload["options"]["num_gpu"] = int(gl)
            if stream_callback:
                try:
                    stream_callback(f"\n> [Debug] num_gpu={gl}\n")
                except:
                    pass
            res = _api_generate(payload)
            if res.get("success"):
                if dynamic_gpu_layers:
                    self._gpu_layers_cache[model] = int(gl)
                return res
            if res.get("fatal"):
                return res
            last_error = res.get("error", "")
            if dynamic_gpu_layers and int(gl) > 0 and _is_oom_error(last_error):
                continue
            break

        return self._handle_fallback(
            model,
            prompt,
            timeout,
            logs,
            last_error or "generation_failed",
            stream_callback,
            response_format=response_format,
            _fallback_tried=_fallback_tried,
            _fallback_depth=_fallback_depth,
        )

    def _handle_fallback(self, model, prompt, timeout, logs, error_msg, stream_callback, response_format: dict = None, _fallback_tried: Optional[List[str]] = None, _fallback_depth: int = 0):
        tried = list(_fallback_tried or [])
        if model and model not in tried:
            tried.append(model)
        max_depth = 2
        try:
            max_depth = int(os.environ.get("OLLAMA_FALLBACK_MAX_DEPTH", "2"))
        except Exception:
            max_depth = 2
        if _fallback_depth >= max_depth:
            return {"success": False, "response": "", "error": error_msg, "logs": logs}
        if _is_empty_generation_error(error_msg):
            fb = self.pick_fallback(model, tried=tried)
            if fb and fb not in tried:
                logger.warning(f"Ollama model '{model}' returned empty content. Auto-switching to fallback model '{fb}'.")
                logs.append(f"empty_fallback_model={fb}")
                if stream_callback:
                    try:
                        stream_callback(f"\n> [Fallback] Model '{model}' returned empty content. Switching to '{fb}'...\n")
                    except Exception:
                        pass
                return self.generate(
                    fb,
                    prompt,
                    timeout=timeout,
                    stream_callback=stream_callback,
                    response_format=response_format,
                    _fallback_tried=tried,
                    _fallback_depth=_fallback_depth + 1,
                )
        # Try CLI generation with the original model first
        try:
            cli_res = self._generate_via_cli(model, prompt, timeout, stream_callback=stream_callback)
            if cli_res.get("success"):
                return cli_res
            else:
                logs.append(f"cli_error={cli_res.get('error')}")
        except Exception as e:
            logs.append(f"cli_exception={str(e)}")
        
        # Then try a smaller fallback model via API again
        fb = self.pick_fallback(model, tried=tried)
        if fb and fb not in tried:
            logger.info(f"Attempting fallback model: {fb}")
            logs.append(f"fallback_model={fb}")
            return self.generate(
                fb,
                prompt,
                timeout=timeout,
                stream_callback=stream_callback,
                response_format=response_format,
                _fallback_tried=tried,
                _fallback_depth=_fallback_depth + 1,
            )
        
        return {"success": False, "response": "", "error": error_msg, "logs": logs}

    def _generate_via_cli(self, model: str, prompt: str, timeout: int, stream_callback=None):
        """
        Fallback: use ollama CLI to run generation when API is unavailable.
        """
        if not self.binary:
            return {"success": False, "response": "", "error": "Bundled Ollama binary missing", "logs": []}
        try:
            # Prefer '-p' to pass prompt safely; Windows cmd quoting can be tricky, rely on subprocess arg list
            args = [self.binary, "run", model, "-p", prompt]
            start_time = time.time()
             
            # Add environment for CLI generation
            env = os.environ.copy()
            if getattr(sys, 'frozen', False) or self._force_bundled:
                # Ensure OLLAMA_MODELS is set if not present
                if "OLLAMA_MODELS" not in env:
                    env["OLLAMA_MODELS"] = _iter_models_home_candidates()[0]
                
                logger.info(f"Ollama CLI fallback: OLLAMA_MODELS={env.get('OLLAMA_MODELS')}")
 
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore",
                creationflags=subprocess.CREATE_NO_WINDOW,
                env=env # Pass env
            )
            full = []
            # Stream stdout
            while True:
                if timeout and (time.time() - start_time) > timeout:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return {"success": False, "response": "".join(full), "error": "timeout", "logs": []}
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    # Avoid busy loop
                    time.sleep(0.01)
                    continue
                clean = self._clean_ansi(line)
                full.append(clean)
                if stream_callback:
                    try:
                        stream_callback(clean)
                    except Exception:
                        pass
            # Read any remaining stderr for diagnostics
            try:
                err_tail = proc.stderr.read() or ""
                if err_tail:
                    logger.debug(f"Ollama CLI stderr: {err_tail.strip()}")
            except Exception:
                pass
            joined = "".join(full)
            if not joined.strip():
                logger.warning(f"Ollama CLI fallback returned empty content for model={model}")
                return {"success": False, "response": "", "error": "Empty CLI Response", "logs": []}
            return {"success": True, "response": joined, "error": "", "logs": []}
        except Exception as e:
            return {"success": False, "response": "", "error": str(e), "logs": []}

    def generate_with_images(self, model: str, prompt: str, images: List[str], timeout: int = None, num_predict: int = -1, options: dict = None):
        gpu_manager.request_gpu("ollama")

        available_models = self.list_models()
        if available_models and model not in available_models:
            best_match = None
            for m in available_models:
                if model in m or m in model:
                    best_match = m
                    break
            if not best_match:
                for m in available_models:
                    ml = m.lower()
                    if "deepseek" in ml and "ocr" in ml:
                        best_match = m
                        break
            model = best_match or available_models[0]

        num_ctx = 8192
        try:
            num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
        except Exception:
            num_ctx = 8192

        gpu_layers = -1
        try:
            gpu_layers = int(os.environ.get("OLLAMA_GPU_LAYERS", "-1"))
        except Exception:
            pass

        base_options = {
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "num_gpu": gpu_layers,
            "main_gpu": int(os.environ.get("OLLAMA_MAIN_GPU", "0")),
        }
        base_options.update(dict(options or {}))
        base_options = self._apply_gpu_budget(base_options, model=model)
        if "OLLAMA_NUM_GPU" in os.environ:
            try:
                if not self._should_force_hybrid(model, self._gpu_vram_cap_mib):
                    base_options["num_gpu"] = int(os.environ["OLLAMA_NUM_GPU"])
            except Exception:
                pass

        payload = {
            "model": model,
            "prompt": prompt,
            "images": list(images or []),
            "stream": False,
            "options": base_options,
            "keep_alive": "30m",
        }
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout)
            if resp.status_code != 200:
                return {"success": False, "response": "", "error": f"API Error {resp.status_code}: {resp.text}", "logs": []}
            obj = resp.json() if resp.text else {}
            text = ""
            if isinstance(obj, dict):
                text = obj.get("response") or ""
            return {"success": True, "response": text, "error": "", "logs": []}
        except Exception as e:
            return {"success": False, "response": "", "error": str(e), "logs": []}


class CustomLLMService:
    """
    OpenAI-API-compatible LLM service wrapper.
    Supports any provider that implements the /v1/chat/completions endpoint
    (DeepSeek, OpenAI, Groq, SiliconFlow, etc.).

    DeepSeek-specific capabilities:
      - Thinking Mode (reasoning_content separation)
      - Multi-turn Conversation (stateful message history)
      - Chat Prefix Completion / FIM Completion (Beta)
      - JSON Output (response_format = json_object)
      - Tool Calls
      - Context Caching (automatic, server-side)
    """

    # ------------------------------------------------------------------ #
    # Profile schema (stored in custom_api_profiles.json)
    # ------------------------------------------------------------------ #
    PROFILE_FIELDS = ("name", "base_url", "api_key", "model",
                       "mode",         # chat | thinking | completion | fim | json
                       "temperature", "max_tokens", "top_p",
                       "api_type")     # openai | deepseek (affects reasoning_content handling)

    # ------------------------------------------------------------------ #
    # Instance init — profile fields become persistent instance attributes
    # ------------------------------------------------------------------ #
    def __init__(self, base_url: str = "", api_key: str = "", model: str = "",
                 mode: str = "chat", temperature: float = 0.7,
                 max_tokens: int = 4096, top_p: float = 1.0,
                 api_type: str = "deepseek", **kwargs):
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "")
        self.model = str(model or "")
        self._mode = mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.api_type = api_type
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------ #
    # Public generate() — mirrors OllamaLocalService.generate() signature
    # ------------------------------------------------------------------ #
    def generate(
        self,
        model: str,
        prompt: str,
        timeout: int = None,
        num_predict: int = -1,
        options: dict = None,
        stream_callback=None,
        response_format: dict = None,
        _fallback_tried: list = None,
        _fallback_depth: int = 0,
    ) -> dict:
        options = dict(options or {})
        target_model = str(model or self.model or "").strip()
        if not target_model:
            return {"success": False, "response": "", "error": "No model specified", "logs": []}

        # Resolve mode from profile or options
        mode = options.pop("_ds_mode", None) or getattr(self, "_mode", "chat") or "chat"

        # Build messages
        if mode == "thinking":
            messages = [{"role": "user", "content": prompt}]
        elif mode == "fim":
            # FIM is handled by a separate endpoint; fall back to chat for safety
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

        # Apply DeepSeek-specific thinking mode
        extra_body = {}
        if mode == "thinking":
            extra_body["thinking"] = {"type": "enabled"}

        # JSON output mode
        if mode == "json" and response_format is None:
            response_format = {"type": "json_object"}

        result = self._chat_complete(
            model=target_model,
            messages=messages,
            stream_callback=stream_callback,
            response_format=response_format,
            extra_body=extra_body,
            options=options,
            num_predict=num_predict,
            timeout=timeout,
        )

        # Extract reasoning_content for thinking mode
        if mode == "thinking" and result.get("success"):
            rc = result.pop("reasoning_content", "")
            result["response"] = rc + ("\n\n" + result["response"] if result.get("response") else rc)

        return result

    # ------------------------------------------------------------------ #
    # chat/completions (OpenAI-compatible)
    # ------------------------------------------------------------------ #
    def chat_complete(
        self,
        model: str,
        messages: List[dict],
        stream_callback=None,
        response_format: dict = None,
        extra_body: dict = None,
        options: dict = None,
        num_predict: int = -1,
        timeout: int = None,
    ) -> dict:
        return self._chat_complete(
            model=model,
            messages=messages,
            stream_callback=stream_callback,
            response_format=response_format,
            extra_body=extra_body,
            options=options,
            num_predict=num_predict,
            timeout=timeout,
        )

    def _chat_complete(
        self,
        model: str,
        messages: List[dict],
        stream_callback,
        response_format,
        extra_body,
        options,
        num_predict,
        timeout,
    ) -> dict:
        import urllib.request
        import urllib.error
        import json
        import time

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if num_predict and num_predict > 0:
            payload["max_tokens"] = num_predict
        elif options.get("num_predict", -1) > 0:
            payload["max_tokens"] = options["num_predict"]
        if options.get("temperature"):
            payload["temperature"] = float(options["temperature"])
        if options.get("top_p"):
            payload["top_p"] = float(options["top_p"])
        if response_format and isinstance(response_format, dict):
            payload["response_format"] = response_format
        if extra_body:
            payload["extra_body"] = extra_body

        # Determine endpoint
        endpoint = f"{self.base_url}/v1/chat/completions"
        if not endpoint.endswith("/v1/chat/completions"):
            endpoint = f"{self.base_url}/chat/completions"

        full_response: List[str] = []
        reasoning_content_buf: List[str] = []
        start_time = time.time()

        def _safe_post(url, payload_dict, timeout_sec):
            req = urllib.request.Request(url, method="POST")
            for k, v in headers.items():
                req.add_header(k, v)
            data = json.dumps(payload_dict).encode("utf-8")
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            return opener.open(req, data=data, timeout=timeout_sec)

        try:
            resp = _safe_post(endpoint, payload, timeout or 120)
            status_code = resp.getcode()
        except urllib.error.HTTPError as e:
            resp = e
            status_code = e.code
        except Exception as e:
            self.logger.error(f"CustomLLM request failed: {e}")
            return {"success": False, "response": "", "error": str(e), "logs": []}

        if status_code != 200:
            try:
                err_body = resp.read().decode("utf-8")
            except Exception:
                err_body = ""
            self.logger.error(f"CustomLLM API error {status_code}: {err_body}")
            return {"success": False, "response": "", "error": f"HTTP {status_code}: {err_body[:200]}", "logs": []}

        try:
            for line in resp:
                line_text = line.decode("utf-8", errors="replace").strip()
                if not line_text or line_text == "data: [DONE]":
                    continue
                if line_text.startswith("data: "):
                    line_text = line_text[6:].strip()
                if not line_text:
                    continue
                try:
                    chunk = json.loads(line_text)
                except json.JSONDecodeError:
                    continue

                delta = ""
                reasoning_delta = ""
                if isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = (choices[0].get("delta", {}) or {}).get("content", "")
                        if delta is None:
                            delta = ""
                        reasoning_delta = (choices[0].get("delta", {}) or {}).get("reasoning_content", "")
                        if reasoning_delta is None:
                            reasoning_delta = ""

                if delta:
                    full_response.append(delta)
                    if stream_callback:
                        try:
                            stream_callback(delta)
                        except Exception:
                            pass
                if reasoning_delta:
                    reasoning_content_buf.append(reasoning_delta)
        except Exception as e:
            self.logger.warning(f"Stream parse error: {e}")

        elapsed = time.time() - start_time
        result_text = "".join(full_response)
        reasoning_text = "".join(reasoning_content_buf)
        if not str(result_text or "").strip():
            try:
                payload2 = dict(payload)
                payload2["stream"] = False
                resp2 = _safe_post(endpoint, payload2, timeout or 120)
                body = resp2.read().decode("utf-8", errors="replace")
                obj = json.loads(body) if body else {}
                content2 = ""
                reasoning2 = ""
                if isinstance(obj, dict):
                    choices = obj.get("choices") or []
                    if isinstance(choices, list) and choices:
                        c0 = choices[0] if isinstance(choices[0], dict) else {}
                        msg = c0.get("message") or {}
                        if isinstance(msg, dict):
                            content2 = msg.get("content") or ""
                            reasoning2 = msg.get("reasoning_content") or ""
                        if not content2:
                            content2 = c0.get("text") or ""
                if content2:
                    result_text = str(content2)
                    reasoning_text = str(reasoning2 or "")
            except Exception:
                pass
        self.logger.info(f"CustomLLM.generate done ({elapsed:.1f}s, {len(result_text)} chars)")
        return {
            "success": True,
            "response": result_text,
            "reasoning_content": reasoning_text,
            "error": "",
            "model": model,
            "elapsed": elapsed,
            "logs": [],
        }

    # ------------------------------------------------------------------ #
    # FIM Completion (Beta) — /v1/completions with prompt + suffix
    # ------------------------------------------------------------------ #
    def fim_complete(
        self,
        model: str,
        prompt: str,
        suffix: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream_callback=None,
        timeout: int = None,
    ) -> dict:
        import urllib.request
        import urllib.error
        import json
        import time

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "prompt": prompt,
            "suffix": suffix,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        endpoint = f"{self.base_url}/v1/completions"
        if "beta" not in self.base_url:
            endpoint = self.base_url.rstrip("/") + "/beta/v1/completions"

        full_response: List[str] = []
        start_time = time.time()

        def _safe_post(url, payload_dict, timeout_sec):
            req = urllib.request.Request(url, method="POST")
            for k, v in headers.items():
                req.add_header(k, v)
            data = json.dumps(payload_dict).encode("utf-8")
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            return opener.open(req, data=data, timeout=timeout_sec)

        try:
            resp = _safe_post(endpoint, payload, timeout or 60)
            status_code = resp.getcode()
        except urllib.error.HTTPError as e:
            resp = e
            status_code = e.code
        except Exception as e:
            self.logger.error(f"FIM request failed: {e}")
            return {"success": False, "response": "", "error": str(e), "logs": []}

        if status_code != 200:
            try:
                err_body = resp.read().decode("utf-8")
            except Exception:
                err_body = ""
            return {"success": False, "response": "", "error": f"HTTP {status_code}: {err_body[:200]}", "logs": []}

        try:
            for line in resp:
                line_text = line.decode("utf-8", errors="replace").strip()
                if not line_text or line_text == "data: [DONE]":
                    continue
                if line_text.startswith("data: "):
                    line_text = line_text[6:].strip()
                if not line_text:
                    continue
                try:
                    chunk = json.loads(line_text)
                except json.JSONDecodeError:
                    continue
                delta = ""
                if isinstance(chunk, dict):
                    delta = (chunk.get("choices", [{}])[0].get("delta", {}) or {}).get("text", "")
                    if delta is None:
                        delta = ""
                if delta:
                    full_response.append(delta)
                    if stream_callback:
                        try:
                            stream_callback(delta)
                        except Exception:
                            pass
        except Exception as e:
            self.logger.warning(f"FIM stream parse error: {e}")

        elapsed = time.time() - start_time
        result_text = "".join(full_response)
        self.logger.info(f"CustomLLM.fim_complete done ({elapsed:.1f}s, {len(result_text)} chars)")
        return {"success": True, "response": result_text, "error": "", "model": model, "elapsed": elapsed, "logs": []}

    # ------------------------------------------------------------------ #
    # Chat Prefix Completion (Beta) — prefix role=assistant in messages
    # ------------------------------------------------------------------ #
    def chat_prefix_complete(
        self,
        model: str,
        messages: List[dict],
        stop: list = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream_callback=None,
        timeout: int = None,
    ) -> dict:
        """
        messages[-1] must have role=assistant with prefix=True to signal
        the model to continue from there.
        Beta base_url (with /beta) is auto-selected.
        """
        import urllib.request
        import urllib.error
        import json
        import time

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Inject prefix flag into last message
        prefixed_messages = [dict(m) for m in messages]
        if prefixed_messages:
            prefixed_messages[-1]["prefix"] = True

        payload = {
            "model": model,
            "messages": prefixed_messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        # Use Beta endpoint
        base = self.base_url.rstrip("/")
        if "/beta" not in base:
            endpoint = base + "/beta/v1/chat/completions"
        else:
            endpoint = base + "/v1/chat/completions"

        full_response: List[str] = []
        start_time = time.time()

        def _safe_post(url, payload_dict, timeout_sec):
            req = urllib.request.Request(url, method="POST")
            for k, v in headers.items():
                req.add_header(k, v)
            data = json.dumps(payload_dict).encode("utf-8")
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            return opener.open(req, data=data, timeout=timeout_sec)

        try:
            resp = _safe_post(endpoint, payload, timeout or 120)
            status_code = resp.getcode()
        except urllib.error.HTTPError as e:
            resp = e
            status_code = e.code
        except Exception as e:
            self.logger.error(f"Chat prefix complete failed: {e}")
            return {"success": False, "response": "", "error": str(e), "logs": []}

        if status_code != 200:
            try:
                err_body = resp.read().decode("utf-8")
            except Exception:
                err_body = ""
            return {"success": False, "response": "", "error": f"HTTP {status_code}: {err_body[:200]}", "logs": []}

        try:
            for line in resp:
                line_text = line.decode("utf-8", errors="replace").strip()
                if not line_text or line_text == "data: [DONE]":
                    continue
                if line_text.startswith("data: "):
                    line_text = line_text[6:].strip()
                if not line_text:
                    continue
                try:
                    chunk = json.loads(line_text)
                except json.JSONDecodeError:
                    continue
                delta = ""
                if isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = (choices[0].get("delta", {}) or {}).get("content", "")
                        if delta is None:
                            delta = ""
                if delta:
                    full_response.append(delta)
                    if stream_callback:
                        try:
                            stream_callback(delta)
                        except Exception:
                            pass
        except Exception as e:
            self.logger.warning(f"Chat prefix stream parse error: {e}")

        elapsed = time.time() - start_time
        result_text = "".join(full_response)
        self.logger.info(f"CustomLLM.chat_prefix_complete done ({elapsed:.1f}s)")
        return {"success": True, "response": result_text, "error": "", "model": model, "elapsed": elapsed, "logs": []}

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def list_models(self) -> List[str]:
        if not self.base_url or not self.model:
            return []
        return [self.model]

    @classmethod
    def profile_defaults(cls) -> dict:
        return {
            "name": "",
            "base_url": "https://api.deepseek.com",
            "api_key": "",
            "model": "deepseek-v4-flash",
            "mode": "chat",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1.0,
            "api_type": "deepseek",
        }
