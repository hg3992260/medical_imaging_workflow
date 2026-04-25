import os
import sys

# Ensure project root is in sys.path BEFORE any other imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def _boottrace(message: str):
    try:
        base_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
        with open(os.path.join(base_dir, "boot_trace.log"), "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")
    except Exception:
        pass

_boottrace("boot:start")

def _disable_fastapi_runtime():
    """
    硬禁用 FastAPI 通道：
    1) 设置显式环境标记，供下游代码判断；
    2) 清理常见 FastAPI/uvicorn 监听端口，避免外部服务“生效”。
    """
    try:
        os.environ["RSNA_DISABLE_FASTAPI"] = "1"
        os.environ["FASTAPI_DISABLE"] = "1"
        os.environ["UVICORN_DISABLE"] = "1"
    except Exception:
        pass

    # Windows 下尝试清理常见 FastAPI 端口监听进程
    if sys.platform.startswith("win"):
        try:
            import subprocess
            check_ports = (8000, 8001, 8080)
            for p in check_ports:
                try:
                    out = subprocess.check_output(
                        f'netstat -ano | findstr ":{p} "',
                        shell=True,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        encoding="utf-8",
                        errors="ignore",
                    )
                except Exception:
                    continue
                pids = set()
                for ln in (out or "").splitlines():
                    parts = ln.split()
                    if len(parts) >= 5 and parts[-1].isdigit():
                        pids.add(parts[-1])
                for pid in pids:
                    if pid == str(os.getpid()):
                        continue
                    try:
                        proc_info = subprocess.check_output(
                            f'tasklist /FI "PID eq {pid}"',
                            shell=True,
                            stderr=subprocess.DEVNULL,
                            text=True,
                            encoding="utf-8",
                            errors="ignore",
                        ).lower()
                    except Exception:
                        proc_info = ""
                    # 仅清理典型 FastAPI 载体进程，避免误伤其它服务
                    if ("python" in proc_info) or ("uvicorn" in proc_info):
                        try:
                            subprocess.run(
                                f"taskkill /F /PID {pid}",
                                shell=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        except Exception:
                            pass
        except Exception:
            pass

_disable_fastapi_runtime()

# === 关键修复：彻底禁用 faulthandler (必须在任何导入之前) ===
# 在某些 Windows 环境（特别是打包后），Python 的 faulthandler 会与 ntdll.dll 冲突
# 导致 "RangeChecks 检测代码检测到超出范围的数组访问" 异常
# 必须显式禁用，防止 PyInstaller bootloader 自动启用
try:
    import faulthandler
    faulthandler.disable()
except ImportError:
    pass
os.environ['PYTHONFAULTHANDLER'] = '0'
# 全局禁用 Hugging Face 在线访问
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_INTERNET"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# ========================================================

# 设置环境变量以规避常见的 DLL 冲突和访问违规
# 必须在导入任何第三方库（尤其是 PyQt5, cv2, torch）之前设置

# 1. 禁用 Intel OpenMP (KMP) 重复初始化错误和 Fork 时的初始化冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_INIT_AT_FORK"] = "FALSE" # 修复 ntdll.dll 数组越界

# 1.5 解决 PyTorch 独占显存导致 Ollama 判定可用为 0B 的问题
# 通过 expandable_segments 和 fraction 限制，强迫 PyTorch 将未使用的显存交还给驱动
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

# 2. 禁用 PyTorch 的 mkldnn 加速（常见于 PyInstaller 打包后的 AVX 指令集冲突）
os.environ["LRU_CACHE_CAPACITY"] = "1"
os.environ["TORCH_MKLDNN_MATMUL_OP_MATH_MODE"] = "FP32"
os.environ["OMP_NUM_THREADS"] = "1"  # Force OpenMP to single thread for stability
# Force MKL to load the correct DLL if possible, though PyTorch usually handles this.
# os.environ["MKL_THREADING_LAYER"] = "GNU" 

# 3. 强制使用软件渲染，避免 OpenGL 驱动冲突 (0xC0000005 的常见原因之一)
# 很多显卡驱动与 Qt 的 OpenGL 引擎在打包后不兼容
# 但是，如果您希望启用 Windows 上的 GPU 硬件加速界面（通常在现代环境没问题），
# 可以注释掉下面这行，或者设置为 desktop / dynamic
# os.environ["QT_OPENGL"] = "software"
os.environ["QT_ANGLE_PLATFORM"] = "warp" # 使用 WARP 软件光栅化，也可以根据需要调整
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1" # 防止高 DPI 缩放导致的崩溃

# 解决 c10.dll 加载失败的关键修复
# 必须手动添加 torch/lib 到 DLL 搜索路径
if getattr(sys, 'frozen', False):
    # 确定应用程序根目录和内部目录
    app_dir = os.path.dirname(sys.executable)
    internal_dir = os.path.join(app_dir, '_internal')
    if not os.path.isdir(internal_dir):
        # 如果 _internal 不存在，可能是在单文件模式或开发模式，或者直接在根目录
        internal_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else app_dir

    def _add_dll_dir(p: str) -> bool:
        if not p or not os.path.isdir(p):
            return False
        try:
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(p)
            return True
        except Exception as e:
            print(f"Failed to add DLL directory {p}: {e}")
            return False

    system32_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32")
    _add_dll_dir(system32_dir)

    # 构造可能的 torch lib 路径
    # 优先使用打包到 assets 下的 Torch DLL 目录（避免与其它库冲突）
    torch_lib_paths = [
        os.path.join(internal_dir, 'torch', 'lib'),
        os.path.join(internal_dir, 'lib'),
        os.path.join(app_dir, 'torch', 'lib'),
    ]

    def _existing_dirs(paths):
        return [p for p in paths if p and os.path.isdir(p)]

    def _preload_first_found(dll_name, search_dirs):
        for d in _existing_dirs(search_dirs):
            dll_path = os.path.join(d, dll_name)
            if os.path.exists(dll_path):
                try:
                    import ctypes
                    ctypes.WinDLL(dll_path)
                    print(f"Successfully preloaded {dll_name} from {d}")
                    return True
                except Exception as e:
                    print(f"Failed to preload {dll_name} from {d}: {e}")
                    return False
        return False
    
    # 必须添加 internal_dir 到 DLL 搜索路径，因为 zlibwapi.dll, VCOMP140.dll 等可能在这里
    if _add_dll_dir(internal_dir):
        print(f"Added base DLL directory: {internal_dir}")

    torch_lib_existing = _existing_dirs(torch_lib_paths)
    cuda_bundled_dirs = [
        os.path.join(internal_dir, "assets", "cuda_libs"),
        os.path.join(app_dir, "assets", "cuda_libs"),
        os.path.join(app_dir, "_internal", "assets", "cuda_libs"),
        os.path.join(internal_dir, "assets", "ollama", "lib", "ollama", "cuda_v12"),
        os.path.join(internal_dir, "assets", "ollama", "lib", "ollama", "cuda_v13"),
    ]
    for cp in cuda_bundled_dirs:
        if _add_dll_dir(cp):
            print(f"Added CUDA bundled DLL directory: {cp}")

    preload_search_dirs = torch_lib_existing + [internal_dir] + [p for p in cuda_bundled_dirs if os.path.isdir(p)]
    for dll_name in ['zlibwapi.dll', 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'dbghelp.dll']:
        _preload_first_found(dll_name, preload_search_dirs)

    # 添加 torch/lib 到 PATH 和 DLL 搜索目录
    for p in torch_lib_paths:
        if _add_dll_dir(p):
            print(f"Added DLL directory: {p}")
    
    need_cuda_fallback = True
    for d in [internal_dir] + [p for p in cuda_bundled_dirs if os.path.isdir(p)]:
        for dll_name in ("c10_cuda.dll", "cudart64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"):
            if os.path.exists(os.path.join(d, dll_name)):
                need_cuda_fallback = False
                break
        if not need_cuda_fallback:
            break

    if need_cuda_fallback:
        cuda_paths = []
        for env_name in ("CUDA_PATH", "CUDA_PATH_V12_8"):
            val = os.environ.get(env_name)
            if val:
                cuda_paths.append(os.path.join(val, "bin"))
                cuda_paths.append(os.path.join(val, "lib", "x64"))
        default_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
        default_cuda_lib = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64"
        for cp in (default_cuda_bin, default_cuda_lib):
            if os.path.isdir(cp):
                cuda_paths.append(cp)
        seen = set()
        for cp in cuda_paths:
            if not cp or cp in seen:
                continue
            seen.add(cp)
            try:
                os.environ["PATH"] = cp + os.pathsep + os.environ["PATH"]
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(cp)
                print(f"Added CUDA DLL directory: {cp}")
            except Exception as e:
                print(f"Failed to add CUDA DLL directory {cp}: {e}")

# 4. 解决 OpenCV 和 Qt 的插件冲突
if getattr(sys, 'frozen', False):
    pass

import asyncio
import logging
_boottrace("boot:imported_asyncio_logging")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('startup.log', mode='w', encoding='utf-8')
    ]
)
_boottrace("boot:logging_configured")
logger = logging.getLogger(__name__)
logger.info("Environment variables set. Starting imports...")
_boottrace("boot:logger_ready")

try:
    import atexit, subprocess
    import socket
    
    # 始终尝试启动并配置内置 Ollama，即使是在源码运行环境下
    try:
        from src.core.path_config import get_base_dir, get_assets_dir
        base_dir = str(get_base_dir())
        assets_dir = str(get_assets_dir())
    except Exception:
        base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
        assets_dir = os.path.join(base_dir, "assets")
    
    def _first_free_port(start, end):
        for p in range(start, end + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.2)
                    if s.connect_ex(("127.0.0.1", p)) != 0:
                        return p
            except Exception:
                pass
        return start

    # 1. 优先使用编译版内置模型路径，其次使用配置路径
    potential_model_paths = [
        os.path.join(assets_dir, 'models_home'),
        os.path.join(base_dir, 'models', 'ollama_home'),
        os.path.join(os.getcwd(), 'models', 'ollama_home'),
        os.path.join(base_dir, 'assets', 'models_home'),
        os.path.join(os.getcwd(), 'assets', 'models_home'),
    ]
    
    model_path_set = False
    for p in potential_model_paths:
        if os.path.isdir(p):
            os.environ['OLLAMA_MODELS'] = p
            logger.info(f"Set OLLAMA_MODELS to auto-detected path: {p}")
            model_path_set = True
            break
    
    if not model_path_set:
        try:
            from src.utils.model_config_loader import get_model_paths
            paths = get_model_paths()
            models_home = paths.get("ollama_models")
            if models_home and os.path.isdir(models_home):
                os.environ['OLLAMA_MODELS'] = models_home
                logger.info(f"Set OLLAMA_MODELS to configured path: {models_home}")
            else:
                logger.warning(f"Configured OLLAMA_MODELS path not found: {models_home}")
        except Exception as e:
            logger.error(f"Failed to set OLLAMA_MODELS from config: {e}")

    # 1.5 强制启动内置 GPU 版 Ollama，不使用系统实例
    default_port = str(_first_free_port(11434, 11444))
    default_host = f"http://localhost:{default_port}"
    os.environ['OLLAMA_HOST'] = default_host
    os.environ['RSNA_OLLAMA_FORCE_BUNDLED'] = '1'
    logger.info(f"Set OLLAMA_HOST to bundled instance: {default_host}")

    # 2. 启动 Ollama 服务
    ollama_exe_paths = [
        os.path.join(assets_dir, 'ollama', 'ollama.exe'),
        os.path.join(base_dir, '_internal', 'assets', 'ollama', 'ollama.exe'),
    ]
    ollama_exe = None
    for p in ollama_exe_paths:
        if os.path.isfile(p):
            ollama_exe = p
            break
    
    _ollama_proc = None
    if ollama_exe:
        try:
            # 使用 creationflags=subprocess.CREATE_NO_WINDOW (0x08000000) 隐藏窗口
            creation_flags = 0x08000000 if sys.platform == 'win32' else 0
            
            logger.info(f"Starting bundled Ollama service on {default_host}: {ollama_exe}")
            
            log_file = open('ollama_startup.log', 'w', encoding='utf-8')
            os.environ['OLLAMA_NO_GPU'] = '0'
            os.environ['OLLAMA_GPU_BUDGET_RATIO'] = os.environ.get('OLLAMA_GPU_BUDGET_RATIO', '0.90')
            os.environ['OLLAMA_MAX_LOADED_MODELS'] = os.environ.get('OLLAMA_MAX_LOADED_MODELS', '1')
            os.environ['OLLAMA_NUM_PARALLEL'] = os.environ.get('OLLAMA_NUM_PARALLEL', '1')
            os.environ['OLLAMA_FLASH_ATTENTION'] = os.environ.get('OLLAMA_FLASH_ATTENTION', '1')
            
            # 强制限制 VRAM 开销和 KV Cache 到安全的 9.5GB 左右 (预留0.5G给系统显示等)
            os.environ['OLLAMA_MAX_VRAM'] = '10200547328'  # 9.5GB in bytes
            os.environ['OLLAMA_GPU_VRAM_CAP_GB'] = '9.5'   # 向下层同步显存上限
            os.environ['OLLAMA_KV_CACHE_TYPE'] = 'q8_0'    # 量化 KV cache 以节省显存
            
            if 'OLLAMA_GPU_LAYERS' not in os.environ:
                os.environ['OLLAMA_GPU_LAYERS'] = '999'
            
            ollama_env = os.environ.copy()
            ollama_env.pop('CUDA_VISIBLE_DEVICES', None)
            ollama_env.pop('OLLAMA_LLM_LIBRARY', None)

            ollama_dir = os.path.dirname(ollama_exe)
            ollama_lib = os.path.join(ollama_dir, 'lib', 'ollama')
            ollama_cuda_v13 = os.path.join(ollama_lib, 'cuda_v13')
            ollama_cuda_v12 = os.path.join(ollama_lib, 'cuda_v12')
            path_prefix = [ollama_dir, ollama_lib, ollama_cuda_v13, ollama_cuda_v12]
            path_prefix = [p for p in path_prefix if os.path.isdir(p)]
            existing_path = ollama_env.get('PATH', '')
            ollama_env['PATH'] = ';'.join(path_prefix + ([existing_path] if existing_path else []))
            
            _ollama_proc = subprocess.Popen(
                [ollama_exe, 'serve'],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
                env=ollama_env,
                cwd=ollama_dir,
            )
        except Exception as e:
            logger.error(f"Failed to start bundled Ollama: {e}")
            _ollama_proc = None
    else:
        logger.warning("Bundled Ollama executable not found. Please ensure assets/ollama/ollama.exe exists.")
        
    def _stop_ollama():
        try:
            if _ollama_proc and _ollama_proc.poll() is None:
                logger.info("Stopping bundled Ollama service...")
                _ollama_proc.terminate()
        except Exception:
            pass
    atexit.register(_stop_ollama)
except Exception as e:
    logger.error(f"Error in Ollama setup: {e}")

# === 关键修复：强制单线程模式 ===
# MKL/OpenMP 在 PyInstaller 打包环境下极易出现多线程竞争导致的 RangeChecks 异常
# 使用 threadpoolctl 强制限制为单线程可以规避绝大多数此类问题
# try:
#     import threadpoolctl
#     threadpoolctl.threadpool_limits(limits=1)
#     logger.info("Threadpool limits set to 1 (single-threaded mode) to prevent MKL crashes.")
# except ImportError:
#     logger.warning("threadpoolctl not found, skipping thread limit setting.")
# ==============================

# === 关键修复：逐步导入并记录日志 ===
def safe_import(module_name):
    try:
        logger.info(f"Attempting to import {module_name}...")
        __import__(module_name)
        logger.info(f"Successfully imported {module_name}.")
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {e}")
    except Exception as e:
        logger.critical(f"CRITICAL ERROR importing {module_name}: {e}")
        # 这里不退出，尝试继续，以便记录更多信息

safe_import("torch")
safe_import("numpy")
safe_import("cv2")
try:
    import torch  # noqa: F401
    try:
        cuda_ok = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_ok else 0
        names = []
        if cuda_ok:
            for i in range(device_count):
                try:
                    # torch.cuda.set_per_process_memory_fraction(0.4, i)
                    names.append(torch.cuda.get_device_name(i))
                except Exception:
                    names.append("unknown")
        logger.info(
            "Torch CUDA check: torch=%s torch_cuda=%s cuda_available=%s device_count=%s devices=%s",
            str(getattr(torch, "__version__", "")),
            str(getattr(getattr(torch, "version", None), "cuda", None)),
            cuda_ok,
            device_count,
            "; ".join(names),
        )
    except Exception as e:
        logger.warning(f"Torch CUDA check failed: {e}")
except Exception as e:
    logger.warning(f"Torch import failed after safe_import: {e}")
# ==============================

def _start_deepseek_preload_thread(db_service=None):
    try:
        import threading
        from src.services.ocr_service import get_ocr_service

        def _preload_deepseek_ocr():
            try:
                svc = get_ocr_service(db_service)
                svc.ensure_deepseek_ocr_adapter()
                try:
                    import torch
                    logger.info(
                        "DeepSeek-OCR preload torch check: cuda_available=%s torch_cuda=%s device_count=%s",
                        bool(torch.cuda.is_available()),
                        str(getattr(torch.version, "cuda", None)),
                        int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                    )
                except Exception as e:
                    logger.warning(f"DeepSeek-OCR preload torch check failed: {e}")
                logger.info("Triggered DeepSeek-OCR preload (background loading thread started).")
            except Exception as e:
                logger.warning(f"DeepSeek-OCR preload failed: {e}")

        threading.Thread(target=_preload_deepseek_ocr, daemon=True).start()
        logger.info("DeepSeek-OCR preload thread scheduled immediately after torch/bootstrap initialization.")
    except Exception as e:
        logger.warning(f"Failed to start DeepSeek-OCR preload thread: {e}")

try:
    import traceback
    logger.info("Importing PyQt5...")
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from PyQt5.QtCore import Qt, QTimer
    import qdarkstyle
    logger.info("PyQt5 imported.")
except Exception as e:
    logger.critical(f"Failed to import PyQt5: {e}")
    sys.exit(1)

# --- 全局异常捕获 ---
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """
    捕获未处理的异常，防止程序闪退，并记录日志
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logging.critical("Uncaught exception:\n%s", error_msg)
    
    # 尝试弹窗显示错误（如果 QApplication 已初始化）
    app = QApplication.instance()
    if app:
        try:
            QMessageBox.critical(None, "程序错误", f"发生未处理的异常，程序可能需要重启。\n\n错误信息:\n{exc_value}")
        except:
            pass

sys.excepthook = global_exception_handler
# ---------------------

# --- Compatibility Layer Injection ---
# Since cocoindex requires Python 3.11+ and we are on an older version,
# we inject a mock compatibility layer if the real module is missing.
try:
    import cocoindex
except (ImportError, ModuleNotFoundError):
    loaded = False
    try:
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
            for p in [os.path.join(base_dir, "cocoindex"), os.path.join(base_dir, "_internal", "cocoindex")]:
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)
            import importlib
            try:
                _ci_engine = importlib.import_module("cocoindex._engine")
                cocoindex = _ci_engine
                sys.modules["cocoindex"] = _ci_engine
                loaded = True
            except Exception:
                loaded = False
    except Exception:
        loaded = False
    if not loaded:
        local_ci_root = r"F:\RSNA\cocoindex-main"
        # Try importing from python subdir if it exists (source layout)
        local_ci_python = os.path.join(local_ci_root, "python")
        
        paths_to_try = []
        if os.path.isdir(local_ci_python):
            paths_to_try.append(local_ci_python)
        if os.path.isdir(local_ci_root):
            paths_to_try.append(local_ci_root)
            
        for p in paths_to_try:
            try:
                if p not in sys.path:
                    sys.path.insert(0, p)
                import cocoindex as _ci_local
                cocoindex = _ci_local
                loaded = True
                print(f"✅ Loaded local CocoIndex from: {p}")
                break
            except Exception as e:
                # print(f"Failed to load from {p}: {e}")
                pass
                
    if not loaded:
        # Compatibility for PyInstaller
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
        else:
            base_dir = os.path.dirname(__file__)
            
        src_path = os.path.join(base_dir, "src")
        if src_path not in sys.path:
            sys.path.append(src_path)
            
        try:
            import src.cocoindex_compat as cocoindex
            sys.modules["cocoindex"] = cocoindex
            sys.modules["cocoindex.sources"] = cocoindex.sources
            sys.modules["cocoindex.functions"] = cocoindex.functions
            sys.modules["cocoindex.targets"] = cocoindex.targets
            sys.modules["cocoindex.targets.lancedb"] = cocoindex.targets.lancedb
            logger.info("CocoIndex not found or incompatible. Using compatibility layer.")
        except ImportError as e:
            logger.warning(f"Failed to load compatibility layer: {e}")
# -------------------------------------

_boottrace("boot:before_top_level_app_imports")
try:
    from ui.widgets.main_window import MainWindow
    from src.core.app_config import AppConfig
    _boottrace("boot:top_level_app_imports_ok")
except Exception as e:
    _boottrace(f"boot:top_level_app_imports_failed:{type(e).__name__}:{e}")
    raise

def main():
    _boottrace("boot:main_enter")
    # Windows platform specific setup
    if sys.platform == 'win32':
        # Set AppUserModelID for taskbar icon
        import ctypes
        myappid = 'rsna.medical.imaging.workflow.v41.medlogo'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        
        # Enable high DPI scaling
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Handle bundled dependencies
    if getattr(sys, 'frozen', False):
        # Add bundled poppler to PATH
        base_dir = os.path.dirname(sys.executable)
        base_path = sys._MEIPASS
        poppler_candidates = [
            os.path.join(base_dir, "_internal", "assets", "poppler_bin"),
            os.path.join(base_path, 'assets', 'poppler_bin'),
            os.path.join(base_path, 'assets', 'poppler', 'bin'),
            os.path.join(base_path, 'poppler_bin'),
            os.path.join(base_dir, 'assets', 'poppler_bin'),
            os.path.join(base_dir, 'poppler_bin'),
        ]
        for poppler_path in poppler_candidates:
            if os.path.isdir(poppler_path):
                os.environ["PATH"] = poppler_path + os.pathsep + os.environ.get("PATH", "")
                try:
                    if hasattr(os, "add_dll_directory"):
                        os.add_dll_directory(poppler_path)
                except Exception:
                    pass
                print(f"Added poppler to PATH: {poppler_path}")
                break

    # 初始化数据库
    try:
        from src.services.database_service import DatabaseService
        _boottrace("boot:database_import_ok")
        db_service = DatabaseService()
        
        # 强制执行完整的初始化流程
        logger = logging.getLogger(__name__)
        logger.info("正在强制初始化数据库...")
        db_service.initialize_database()
        
        # 验证表是否存在
        with db_service.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='projects'")
            if not cursor.fetchone():
                logger.error("数据库初始化后 'projects' 表仍不存在！尝试再次初始化...")
                db_service.initialize_database()
            else:
                logger.info("数据库 'projects' 表验证通过")
                
    except Exception as e:
        logging.getLogger(__name__).error(f"数据库初始化失败: {e}")

    _start_deepseek_preload_thread(db_service if 'db_service' in locals() else None)

    # 启动应用
    app = QApplication(sys.argv)
    _boottrace("boot:qapplication_created")
    app.setStyle('Fusion')
    try:
        from PyQt5.QtGui import QIcon

        icon_candidates = []
        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(sys.executable)
            internal_dir = getattr(sys, "_MEIPASS", os.path.join(base_dir, "_internal"))
            icon_candidates.extend(
                [
                    os.path.join(base_dir, "_internal", "assets", "icon.ico"),
                    os.path.join(base_dir, "_internal", "assets", "medlogo.png"),
                    os.path.join(internal_dir, "assets", "icon.ico"),
                    os.path.join(internal_dir, "assets", "medlogo.png"),
                    os.path.join(base_dir, "assets", "icon.ico"),
                    os.path.join(base_dir, "assets", "medlogo.png"),
                ]
            )
        icon_candidates.extend(
            [
                os.path.join(os.getcwd(), "assets", "icon.ico"),
                os.path.join(os.getcwd(), "assets", "medlogo.png"),
                os.path.join(os.path.dirname(__file__), "assets", "icon.ico"),
                os.path.join(os.path.dirname(__file__), "assets", "medlogo.png"),
            ]
        )
        for p in icon_candidates:
            if p and os.path.exists(p):
                app.setWindowIcon(QIcon(p))
                break
    except Exception:
        pass

    # Initialize theme
    from src.utils.theme_manager import ThemeManager
    
    # Initialize config
    config = AppConfig()
    _boottrace("boot:appconfig_created")
    
    # Retrieve saved theme (default to light)
    saved_theme = config.get("theme", "light")
    ThemeManager.apply_theme(app, saved_theme)
    
    # === License Verification ===
    # Force verification on startup. If invalid or machine fingerprint changed,
    # show activation dialog.
    try:
        from license_manage.integration import validate_license_strict
        from license_manage.activation_ui import LicenseActivationDialog
        _boottrace("boot:license_imports_ok")
        
        is_valid, msg, remaining, exp_ts = validate_license_strict()
        
        if not is_valid:
            logger.warning(f"License invalid: {msg}. Showing activation dialog.")
            
            # Show activation dialog
            dlg = LicenseActivationDialog()
            if dlg.exec_() == LicenseActivationDialog.Accepted:
                # Re-validate after activation
                is_valid, msg, remaining, exp_ts = validate_license_strict()
                if not is_valid:
                    logger.error(f"License still invalid after activation: {msg}")
                    sys.exit(1)
            else:
                logger.info("Activation cancelled by user.")
                sys.exit(0)
            
    except Exception as e:
        logger.error(f"License check failed: {e}")
        # If critical error in license system, fail safe (don't run)
        QMessageBox.critical(None, "License Error", f"License system error: {e}")
        sys.exit(1)

    window = MainWindow(remaining_seconds=remaining, license_exp_ts=exp_ts)
    _boottrace("boot:mainwindow_created")
    window.show()
    _boottrace("boot:mainwindow_shown")

    # Create asyncio loop for async tasks integration if needed
    loop = asyncio.get_event_loop()
    
    # 运行应用
    exit_code = app.exec_()
    
    logging.getLogger(__name__).info("Application event loop finished. Exiting...")
    
    # === 关键修复：使用 os._exit 避免退出崩溃 ===
    # PyTorch/Qt/OpenCV 在 Python 解释器关闭时的析构顺序经常冲突
    # 导致 ntdll.dll 0xC0000005 或 RangeChecks 错误
    # os._exit(0) 会立即终止进程，跳过清理步骤，规避此类崩溃
    # 这在打包的 exe 中是标准做法
    logging.shutdown()
    os._exit(exit_code)
    # ===========================================

if __name__ == '__main__':
    # Add multiprocessing freeze_support for Windows PyInstaller packaging
    import multiprocessing
    multiprocessing.freeze_support()
    
    from PyQt5.QtCore import Qt 
    main()
