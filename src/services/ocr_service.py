#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR识别服务模块

负责图像文字识别、结果处理和存储等功能。
"""

import os
import sys
import json
import base64
import cv2
import numpy as np
import gc
import threading
import time
import subprocess
from datetime import datetime
from src.core.event_bus import get_event_bus, EventNames
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageEnhance, ImageFilter

from src.services.database_service import DatabaseService
from src.services.project_storage_service import ProjectStorageService
from src.utils.logger import get_logger
from src.utils.gpu_manager import gpu_manager
try:
    from src.adapters.deepseek_ocr_external_adapter import DeepSeekOCRExternalAdapter
except Exception:
    DeepSeekOCRExternalAdapter = None

class OCRService:
    """OCR识别服务类"""
    
    @property
    def deepseek_available(self):
        """动态检查DeepSeek OCR是否可用"""
        if self.deepseek_adapter:
            return self.deepseek_adapter.available
        return False

    def __init__(self, db_service: DatabaseService = None):
        """初始化OCR服务"""
        self.db_service = db_service or DatabaseService()
        self.storage_service = ProjectStorageService(self.db_service)
        self.logger = get_logger(__name__)
        self._image_cache = {}
        self.ollama_service = None
        self._ollama_ocr_failures = 0
        self._ollama_ocr_dead_until = 0.0
        self._ollama_ocr_last_warn_at = 0.0
        self._ollama_ocr_ever_succeeded = False
        
        self.deepseek_adapter = None
        self.deepseek_init_error = None
        self.deepseek_ocr_init_error = None
        try:
            gpu_manager.register("deepseek_ocr", self.release_gpu_resources)
        except Exception:
            pass


        # DeepSeek-OCR 是唯一支持的引擎

    def ensure_deepseek_adapter(self) -> bool:
        return self.ensure_deepseek_ocr_adapter()

    def _get_external_python(self) -> Optional[str]:
        for k in ("DEEPANALYZE_EXEC_PYTHON", "DEEPANALYZE_PYTHON", "RSNA_OCR_PYTHON"):
            v = os.environ.get(k)
            if v:
                return v
        return None

    def _get_deepseek_model_path_for_external(self, model_kind: str) -> str:
        kind = (model_kind or "ocr").strip().lower()
        if getattr(sys, "frozen", False):
            override_dir = os.environ.get("RSNA_DEEPSEEK_OCR_DIR")
            if override_dir and os.path.isdir(override_dir):
                return override_dir
            base_dir = os.path.dirname(sys.executable)
            internal_dir = os.path.join(base_dir, "_internal")
            return os.path.join(internal_dir, "assets", "models", "deepseek_ocr")
        try:
            from src.utils.model_config_loader import get_model_paths

            paths = get_model_paths()
            return paths.get("deepseek_ocr") or ""
        except Exception:
            return ""

    def ensure_deepseek_ocr_adapter(self) -> bool:
        if self.deepseek_adapter is not None:
            return True
        external_py = self._get_external_python()
        if getattr(sys, "frozen", False) and external_py and DeepSeekOCRExternalAdapter is not None:
            self.deepseek_adapter = DeepSeekOCRExternalAdapter(
                python_exe=external_py,
                model_kind="ocr",
                model_path=self._get_deepseek_model_path_for_external("ocr"),
            )
            self.deepseek_init_error = None
            self.deepseek_ocr_init_error = None
            self.logger.info("DeepSeek-OCR external adapter initialized")
            return True
        try:
            from src.adapters.deepseek_ocr_adapter import DeepSeekOCRAdapter
            from src.utils.model_config_loader import get_model_paths

            model_paths = get_model_paths()
            model_path = model_paths.get("deepseek_ocr")
            self.deepseek_adapter = DeepSeekOCRAdapter(model_kind="ocr", model_path=model_path)
            self.deepseek_init_error = None
            self.deepseek_ocr_init_error = None
            self.logger.info("DeepSeek-OCR adapter initialized (loading in background)")
            return True
        except Exception as e:
            if external_py and DeepSeekOCRExternalAdapter is not None:
                self.deepseek_adapter = DeepSeekOCRExternalAdapter(
                    python_exe=external_py,
                    model_kind="ocr",
                    model_path=self._get_deepseek_model_path_for_external("ocr"),
                )
                self.deepseek_init_error = None
                self.deepseek_ocr_init_error = None
                self.logger.info("DeepSeek-OCR external adapter initialized after in-process failure")
                return True
            self.deepseek_init_error = str(e)
            self.deepseek_ocr_init_error = str(e)
            self.logger.warning(f"DeepSeek-OCR 初始化失败: {e}")
            self.deepseek_adapter = None
            return False

    def _get_deepseek_wait_seconds(self, config: Dict[str, Any] = None) -> float:
        cfg = dict(config or {})
        try:
            device = str(getattr(getattr(self, "deepseek_adapter", None), "device", "") or "").lower()
            default_wait = 90.0 if device.startswith("cpu") else 45.0
            return float(cfg.get("deepseek_wait_seconds", default_wait))
        except Exception:
            return 90.0

    def _emit_status(self, status_callback, message: str) -> None:
        if not callable(status_callback):
            return
        try:
            status_callback(str(message or ""))
        except Exception:
            pass

    def _wait_for_deepseek_ready(self, wait_s: float, status_callback=None) -> None:
        adapter = getattr(self, "deepseek_adapter", None)
        if adapter is None or wait_s <= 0:
            return
        if not bool(getattr(adapter, "loading", False)):
            if bool(getattr(adapter, "available", False)):
                self._emit_status(status_callback, "DeepSeek-OCR 模型已可用，开始执行 OCR。")
            return

        start = time.time()
        logged = False
        while (time.time() - start) < float(wait_s):
            if bool(getattr(adapter, "available", False)):
                if logged:
                    self.logger.info("DeepSeek-OCR 已完成后台加载，用时 %.1fs", time.time() - start)
                self._emit_status(status_callback, "DeepSeek-OCR 模型已加载到显存并可用。")
                return

            loading = bool(getattr(adapter, "loading", False))
            load_thread = getattr(adapter, "loading_thread", None)
            if load_thread is not None and not load_thread.is_alive() and not bool(getattr(adapter, "available", False)):
                if not getattr(adapter, "load_error", None):
                    adapter.load_error = "background loading thread finished without ready state"
                try:
                    adapter.loading = False
                except Exception:
                    pass
                break

            if not loading:
                break

            if not logged:
                self.logger.info("DeepSeek-OCR 正在后台加载，等待最多 %.1f 秒后再执行 OCR", float(wait_s))
                self._emit_status(status_callback, "DeepSeek-OCR 模型正在加载到显存，请稍候…")
                logged = True
            time.sleep(0.25)
        if getattr(adapter, "load_error", None):
            self._emit_status(status_callback, f"DeepSeek-OCR 加载失败：{adapter.load_error}")
        else:
            self._emit_status(
                status_callback,
                "DeepSeek-OCR 等待超时，暂未就绪。建议释放另一个模型后重试。"
            )

    # DeepSeek-OCR 是唯一支持的引擎，无需检查其他引擎

    def extract_text_from_image(self, image_path: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """从图像提取文本"""
        config = config or {}
        status_callback = config.get("status_callback")
        engine = (config.get('engine', 'deepseek_ocr') or 'deepseek_ocr').strip().lower()
        backend = (config.get("deepseek_backend") or os.environ.get("RSNA_DEEPSEEK_OCR_BACKEND") or "").strip().lower()
        if engine in ("auto", "best", "default"):
            engine = "deepseek_ocr"
        if engine in ("ollama", "ollama_ocr", "ollama-deepseek-ocr", "ollama_deepseek_ocr", "deepseek_ocr_ollama"):
            engine = "ollama_deepseek_ocr"
        if engine in ("deepseek", "deepseek-ocr", "deepseek_ocr"):
            engine = "deepseek_ocr"
        if engine in ("deepseek2", "deepseek-2"):
            engine = "deepseek_ocr"

        if backend in ("ollama", "ollama_deepseek_ocr", "ollama-ocr"):
            if engine == "deepseek_ocr":
                engine = "ollama_deepseek_ocr"
        
        # 只有在需要对图像进行预处理（例如传统OCR）时才读取为 NumPy 数组
        # 对于 DeepSeek-OCR，直接使用原始路径以避免转码损失和额外的 I/O
        if isinstance(image_path, str) and engine not in ("deepseek_ocr", "ollama_deepseek_ocr"):
            try:
                # 优先尝试使用 numpy 读取以支持中文路径
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                image_array = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is None:
                    # 如果 numpy 读取失败，尝试原生读取
                    image = cv2.imread(image_path)
            except Exception as e:
                self.logger.warning(f"使用imdecode读取图像失败，尝试cv2.imread: {e}")
                image = cv2.imread(image_path)
        else:
            image = image_path
            
        if image is None:
            return {
                'text': '',
                'extracted_text': '',
                'text_boxes': [],
                'engine': engine,
                'ocr_engine': engine,
                'confidence': 0.0,
                'confidence_score': 0.0,
                'processing_time': 0.0,
                'error': '无法读取图像，请检查文件路径或格式',
                'status': 'error',
            }

        # 预处理 (DeepSeek-OCR 是多模态大模型，不应进行灰度化等传统 OCR 预处理，否则会丢失 RGB 特征导致性能退化)
        if config.get('preprocessing', True) and engine not in ("deepseek_ocr", "ollama_deepseek_ocr"):
            image = self._preprocess_image(image, config)

        if engine == "ollama_deepseek_ocr":
            self._emit_status(status_callback, "Ollama OCR 模型正在加载到显存，请稍候…")
            res = self.extract_text_ollama_deepseek_ocr(image, config)
            if isinstance(res, dict) and (res.get("error") or not (res.get("text") or "").strip()):
                err = str(res.get("error") or "")
                now = time.time()
                if now - float(getattr(self, "_ollama_ocr_last_warn_at", 0.0) or 0.0) > 15:
                    self._ollama_ocr_last_warn_at = now
                    self.logger.warning(f"Ollama DeepSeek OCR 识别失败，尝试回退: {err}")
            else:
                self._emit_status(status_callback, "Ollama OCR 模型已可用，开始识别。")
                return res

            if self.deepseek_available:
                return self.extract_text_deepseek_ocr(image, config)
            return res

        if engine in ("deepseek", "deepseek_ocr"):
            if backend in ("vllm", "vllm_image", "vllm-image"):
                self._emit_status(status_callback, "正在启动 vLLM 推理子进程（首次运行可能较慢）…")
                return self.extract_text_deepseek_ocr_vllm(image, config)
            self._emit_status(status_callback, "正在为 DeepSeek-OCR 协调显存资源…")
            try:
                gpu_manager.request_gpu("deepseek_ocr")
            except Exception:
                pass
            self.ensure_deepseek_adapter()
            wait_s = self._get_deepseek_wait_seconds(config)
            self._wait_for_deepseek_ready(wait_s, status_callback=status_callback)

            if self.deepseek_available:
                self._emit_status(status_callback, "DeepSeek-OCR 模型已可用，开始 OCR 识别。")
                t0 = time.time()
                while True:
                    res = self.extract_text_deepseek_ocr(image, config)
                    if isinstance(res, dict):
                        status = (res.get("status") or "").strip().lower()
                        text_val = (res.get("extracted_text") or res.get("text") or "").strip()
                        if text_val:
                            return res
                        if status in ("empty", "ok") and not (res.get("error") or ""):
                            return res
                        if status in ("loading",) and wait_s > 0 and (time.time() - t0) < wait_s:
                            time.sleep(0.25)
                            continue
                        reason = res.get("error") or status or "empty"
                        self.logger.warning(f"DeepSeek-OCR 识别失败: {reason}")
                        # 严格使用 deepseek_ocr_1 时，不回退
                        if getattr(self.deepseek_adapter, "strict_ocr1", False):
                            self._emit_status(status_callback, f"DeepSeek-OCR 识别失败：{reason}")
                            return res
                        self.logger.warning(f"DeepSeek-OCR 识别失败，尝试回退: {reason}")
                        self._emit_status(status_callback, f"DeepSeek-OCR 识别失败，准备回退其他 OCR：{reason}")
                        break
                    if str(res).strip():
                        return res
                    
                    if getattr(self.deepseek_adapter, "strict_ocr1", False):
                        return {'text': '', 'error': 'empty', 'status': 'error', 'engine': 'deepseek_ocr'}
                    self.logger.warning("DeepSeek-OCR 识别失败，尝试回退: empty")
                    break
            else:
                reason = "Unknown error"
                details = ""
                if self.deepseek_adapter:
                    if getattr(self.deepseek_adapter, "loading", False):
                        reason = "Loading..."
                    else:
                        reason = self.deepseek_adapter.load_error or "Unavailable"
                    try:
                        lp = getattr(self.deepseek_adapter, "loaded_model_path", None)
                        if lp:
                            details += f", path={lp}"
                    except Exception:
                        pass
                    try:
                        import transformers, torch
                        details += f", transformers={getattr(transformers, '__version__', '?')}, torch={getattr(torch, '__version__', '?')}"
                    except Exception:
                        pass
                
                # 严格模式下，即使服务加载中或者不可用，也不回退
                if getattr(self.deepseek_adapter, "strict_ocr1", False):
                    self.logger.warning(f"DeepSeek-OCR 严格模式：服务不可用 ({reason}{details})，拒绝回退")
                    self._emit_status(status_callback, f"DeepSeek-OCR 当前不可用：{reason}")
                    return {
                        'text': '',
                        'extracted_text': '',
                        'text_boxes': [],
                        'engine': 'deepseek_ocr',
                        'ocr_engine': 'deepseek_ocr',
                        'confidence': 0.0,
                        'confidence_score': 0.0,
                        'processing_time': 0.0,
                        'error': f"严格模式要求DeepSeek-OCR，但当前不可用: {reason}",
                        'status': 'unavailable',
                    }
                    
                self.logger.warning(f"DeepSeek-OCR 请求被调用但服务不可用 ({reason}{details})")
                self._emit_status(status_callback, f"DeepSeek-OCR 当前不可用：{reason}")

            return {
                'text': '',
                'extracted_text': '',
                'text_boxes': [],
                'engine': 'deepseek_ocr',
                'ocr_engine': 'deepseek_ocr',
                'confidence': 0.0,
                'confidence_score': 0.0,
                'processing_time': 0.0,
                'error': 'DeepSeek不可用',
                'status': 'unavailable',
            }

        # 仅支持 DeepSeek-OCR
        if self.deepseek_available:
            return self.extract_text_deepseek(image, config)
        else:
            return {
                'text': '',
                'extracted_text': '',
                'text_boxes': [],
                'engine': engine,
                    'ocr_engine': engine,
                    'confidence': 0.0,
                    'confidence_score': 0.0,
                    'processing_time': 0.0,
                    'error': '没有可用的OCR引擎',
                    'status': 'unavailable',
                }

    def extract_text_deepseek(self, image, config=None):
        return self.extract_text_deepseek_ocr(image, config)

    def _ensure_ollama_service(self) -> bool:
        if self.ollama_service is not None:
            return True
        try:
            from src.services.ollama_local_service import OllamaLocalService

            self.ollama_service = OllamaLocalService()
            return True
        except Exception as e:
            self.ollama_service = None
            self.logger.warning(f"Ollama OCR 初始化失败: {e}")
            return False

    def _image_to_png_base64(self, image) -> str:
        if isinstance(image, str):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        fmt = (os.environ.get("OLLAMA_OCR_IMAGE_FORMAT", "jpg") or "jpg").strip().lower()
        if fmt in ("png",):
            ok, buf = cv2.imencode(".png", image)
        else:
            q = 80
            try:
                q = int(os.environ.get("OLLAMA_OCR_JPEG_QUALITY", "80"))
            except Exception:
                q = 80
            q = max(30, min(95, q))
            ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("failed to encode image to png")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def extract_text_ollama_deepseek_ocr(self, image, config=None):
        cfg = dict(config or {})
        if not self._ensure_ollama_service():
            return {"text": "", "error": "Ollama service unavailable", "ocr_engine": "ollama_deepseek_ocr"}

        now = time.time()
        try:
            dead_until = float(getattr(self, "_ollama_ocr_dead_until", 0.0) or 0.0)
        except Exception:
            dead_until = 0.0
        if dead_until and now < dead_until:
            return {"text": "", "error": "Ollama OCR temporarily disabled after recent timeouts", "ocr_engine": "ollama_deepseek_ocr"}

        model = (cfg.get("ollama_ocr_model") or os.environ.get("OLLAMA_OCR_MODEL") or "").strip()
        if not model:
            model = "deepseek-ocr"

        prompt = (cfg.get("ollama_prompt") or cfg.get("deepseek_prompt") or os.environ.get("OLLAMA_OCR_PROMPT") or "").strip()
        if not prompt:
            deepseek_mode = str(cfg.get("deepseek_mode") or "").strip().lower()
            if deepseek_mode in ("markdown", "md", "document", "doc"):
                prompt = "<image>\n<|grounding|>Convert the document to markdown."
            else:
                prompt = "<image>\nFree OCR."
        timeout = None
        try:
            if cfg.get("ollama_timeout_seconds") is not None:
                timeout = float(cfg.get("ollama_timeout_seconds"))
        except Exception:
            timeout = None
        if timeout is None:
            ever_ok = bool(getattr(self, "_ollama_ocr_ever_succeeded", False))
            if not ever_ok:
                try:
                    timeout = float(os.environ.get("OLLAMA_OCR_FIRST_TIMEOUT_SECONDS", "600"))
                except Exception:
                    timeout = 600.0
            else:
                try:
                    timeout = float(os.environ.get("OLLAMA_OCR_TIMEOUT_SECONDS", "90"))
                except Exception:
                    timeout = 90.0

        try:
            img_b64 = self._image_to_png_base64(image)
        except Exception as e:
            return {"text": "", "error": f"image encode failed: {e}", "ocr_engine": "ollama_deepseek_ocr"}

        try:
            opt = cfg.get("ollama_options")
            opt = opt if isinstance(opt, dict) else {}
            num_predict = int(cfg.get("ollama_num_predict", -1))
        except Exception:
            opt = {}
            num_predict = -1

        try:
            opt.setdefault("temperature", 0.0)
        except Exception:
            pass

        res = self.ollama_service.generate_with_images(model=model, prompt=prompt, images=[img_b64], timeout=timeout, num_predict=num_predict, options=opt)
        if not res or not res.get("success"):
            err = str(res.get("error") if isinstance(res, dict) else "ollama failed")
            if "Read timed out" in err or "read timeout" in err or "timeout" == err:
                try:
                    self._ollama_ocr_failures = int(getattr(self, "_ollama_ocr_failures", 0) or 0) + 1
                except Exception:
                    self._ollama_ocr_failures = 1
                try:
                    threshold = int(os.environ.get("OLLAMA_OCR_CIRCUIT_BREAKER_FAILS", "1"))
                except Exception:
                    threshold = 1
                try:
                    cool_s = float(os.environ.get("OLLAMA_OCR_CIRCUIT_BREAKER_SECONDS", "300"))
                except Exception:
                    cool_s = 300.0
                if self._ollama_ocr_failures >= max(1, threshold):
                    self._ollama_ocr_dead_until = time.time() + max(10.0, cool_s)
            return {"text": "", "error": res.get("error") if isinstance(res, dict) else "ollama failed", "ocr_engine": "ollama_deepseek_ocr"}

        text = (res.get("response") or "").strip()
        self._ollama_ocr_failures = 0
        self._ollama_ocr_dead_until = 0.0
        self._ollama_ocr_ever_succeeded = True
        return {"text": text, "confidence_score": 0.0, "confidence": 0.0, "engine": "ollama_deepseek_ocr", "ocr_engine": "ollama_deepseek_ocr"}

    def extract_text_deepseek_ocr(self, image, config=None):
        try:
            import tempfile

            if self.deepseek_adapter is None or not self.deepseek_available:
                raise Exception("DeepSeek-OCR不可用")
            
            cfg = dict(config or {})
            cfg.setdefault("deepseek_mode", "markdown")
            if not (cfg.get("deepseek_prompt") or "").strip():
                # 严格对齐官方：默认使用 "<image>\n<|grounding|>Convert the document to markdown. "
                cfg["deepseek_prompt"] = "<image>\n<|grounding|>Convert the document to markdown. "
            cfg.setdefault("base_size", 1024)
            cfg.setdefault("image_size", 640)
            cfg.setdefault("crop_mode", True)
            cfg.setdefault("test_compress", True)

            if isinstance(image, str):
                image_path = image
                is_temp = False
            else:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    image_path = temp_file.name
                cv2.imwrite(image_path, image)
                is_temp = True

            try:
                print(f"[OCRService] Requesting DeepSeek OCR for image: {image_path}")
                res = self.deepseek_adapter.extract_text_from_image(image_path, cfg)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    self.logger.warning("DeepSeek OCR CUDA内存不足，尝试清理内存后重试")
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    gc.collect()
                    time.sleep(1)
                    res = self.deepseek_adapter.extract_text_from_image(image_path, cfg)
                else:
                    raise

            try:
                if isinstance(res, dict):
                    if "extracted_text" not in res and (res.get("text") or "").strip():
                        res["extracted_text"] = res.get("text") or ""
                    if "text" not in res and (res.get("extracted_text") or "").strip():
                        res["text"] = res.get("extracted_text") or ""
                    res.setdefault("ocr_engine", "deepseek_ocr")
                    res.setdefault("engine", "deepseek_ocr")
                    res.setdefault("confidence_score", res.get("confidence", 0.0) or 0.0)
                    if "status" not in res:
                        t = (res.get("extracted_text") or res.get("text") or "").strip()
                        res["status"] = "ok" if t else "empty"
            except Exception:
                pass

            if is_temp:
                try:
                    if os.path.exists(image_path):
                        os.unlink(image_path)
                except Exception:
                    pass
            return res
        except Exception as e:
            self.logger.error(f"DeepSeek OCR提取失败: {e}")
            return {
                'text': '',
                'extracted_text': '',
                'text_boxes': [],
                'engine': 'deepseek_ocr',
                'ocr_engine': 'deepseek_ocr',
                'confidence': 0.0,
                'confidence_score': 0.0,
                'processing_time': 0.0
                ,
                'error': str(e),
                'status': 'error',
            }

    def _get_deepseek_ocr_vllm_dir(self) -> str:
        v = (os.environ.get("RSNA_DEEPSEEK_OCR_VLLM_DIR") or "").strip()
        if v and os.path.isdir(v):
            return v
        try:
            app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            repo_root = os.path.abspath(os.path.join(app_root, ".."))
            cand = os.path.join(
                repo_root,
                "DeepSeek-OCR-main",
                "DeepSeek-OCR-main",
                "DeepSeek-OCR-master",
                "DeepSeek-OCR-vllm",
            )
            if os.path.isdir(cand):
                return cand
        except Exception:
            pass
        return ""

    def extract_text_deepseek_ocr_vllm(self, image, config=None):
        cfg = dict(config or {})
        status_callback = cfg.get("status_callback")
        start_time = time.time()

        if not isinstance(image, str) or not os.path.isfile(image):
            return {
                'text': '',
                'extracted_text': '',
                'text_boxes': [],
                'engine': 'deepseek_ocr_vllm',
                'ocr_engine': 'deepseek_ocr_vllm',
                'confidence': 0.0,
                'confidence_score': 0.0,
                'processing_time': 0.0,
                'error': 'vLLM 后端仅支持传入图像文件路径',
                'status': 'error',
            }

        vllm_dir = self._get_deepseek_ocr_vllm_dir()
        if not vllm_dir:
            return {
                'text': '',
                'extracted_text': '',
                'text_boxes': [],
                'engine': 'deepseek_ocr_vllm',
                'ocr_engine': 'deepseek_ocr_vllm',
                'confidence': 0.0,
                'confidence_score': 0.0,
                'processing_time': 0.0,
                'error': '未找到 DeepSeek-OCR-vllm 目录。可通过环境变量 RSNA_DEEPSEEK_OCR_VLLM_DIR 指定。',
                'status': 'error',
            }

        prompt = (cfg.get("deepseek_prompt") or "").strip()
        if not prompt:
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
        model_path = (cfg.get("deepseek_model_path") or cfg.get("model_path") or "deepseek-ai/DeepSeek-OCR").strip()
        crop_mode = cfg.get("crop_mode")
        if crop_mode is None:
            crop_mode = True
        gpu_util = cfg.get("vllm_gpu_memory_utilization")
        try:
            gpu_util = float(gpu_util) if gpu_util is not None else 0.75
        except Exception:
            gpu_util = 0.75

        timeout_s = cfg.get("deepseek_vllm_timeout_seconds")
        try:
            timeout_s = float(timeout_s) if timeout_s is not None else 900.0
        except Exception:
            timeout_s = 900.0

        self._emit_status(status_callback, "vLLM 推理进行中…")

        py_code = (
            "import os,sys,asyncio,time\n"
            "os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')\n"
            "os.environ.setdefault('OMP_NUM_THREADS','1')\n"
            "os.environ.setdefault('MKL_NUM_THREADS','1')\n"
            "os.environ.setdefault('VLLM_USE_V1','0')\n"
            f"os.chdir({json.dumps(vllm_dir)})\n"
            f"sys.path.insert(0,{json.dumps(vllm_dir)})\n"
            "from vllm import AsyncLLMEngine, SamplingParams\n"
            "from vllm.engine.arg_utils import AsyncEngineArgs\n"
            "from vllm.model_executor.models.registry import ModelRegistry\n"
            "from deepseek_ocr import DeepseekOCRForCausalLM\n"
            "from process.image_process import DeepseekOCRProcessor\n"
            "from process.ngram_norepeat import NoRepeatNGramLogitsProcessor\n"
            "from PIL import Image, ImageOps\n"
            "ModelRegistry.register_model('DeepseekOCRForCausalLM', DeepseekOCRForCausalLM)\n"
            f"image_path={json.dumps(os.path.abspath(image))}\n"
            f"prompt={json.dumps(prompt)}\n"
            f"model_path={json.dumps(model_path)}\n"
            f"crop_mode={('True' if bool(crop_mode) else 'False')}\n"
            f"gpu_mem_util={json.dumps(gpu_util)}\n"
            "img=ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')\n"
            "image_features='' \n"
            "if '<image>' in prompt:\n"
            "    image_features=DeepseekOCRProcessor().tokenize_with_images(images=[img], bos=True, eos=True, cropping=crop_mode)\n"
            "async def _run():\n"
            "    engine_args=AsyncEngineArgs(\n"
            "        model=model_path,\n"
            "        hf_overrides={'architectures':['DeepseekOCRForCausalLM']},\n"
            "        block_size=256,\n"
            "        max_model_len=8192,\n"
            "        enforce_eager=False,\n"
            "        trust_remote_code=True,\n"
            "        tensor_parallel_size=1,\n"
            "        gpu_memory_utilization=float(gpu_mem_util),\n"
            "    )\n"
            "    engine=AsyncLLMEngine.from_engine_args(engine_args)\n"
            "    logits_processors=[NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90, whitelist_token_ids={128821,128822})]\n"
            "    sampling_params=SamplingParams(temperature=0.0, max_tokens=8192, logits_processors=logits_processors, skip_special_tokens=False)\n"
            "    req={'prompt': prompt}\n"
            "    if image_features and '<image>' in prompt:\n"
            "        req={'prompt': prompt, 'multi_modal_data': {'image': image_features}}\n"
            "    final=''\n"
            "    async for out in engine.generate(req, sampling_params, request_id='req'):\n"
            "        if out.outputs:\n"
            "            final=out.outputs[0].text\n"
            "    return final\n"
            "text=asyncio.run(_run())\n"
            "if text is None:\n"
            "    text=''\n"
            "text=str(text)\n"
            "text=text.replace('<｜begin▁of▁sentence｜>','').replace('<｜end▁of▁sentence｜>','').strip()\n"
            "sys.stdout.write(text)\n"
        )

        env = dict(os.environ)
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("VLLM_USE_V1", "0")

        try:
            proc = subprocess.run(
                [sys.executable, "-u", "-c", py_code],
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_s,
                cwd=vllm_dir,
            )
        except subprocess.TimeoutExpired:
            return {
                'text': '',
                'extracted_text': '',
                'text_boxes': [],
                'engine': 'deepseek_ocr_vllm',
                'ocr_engine': 'deepseek_ocr_vllm',
                'confidence': 0.0,
                'confidence_score': 0.0,
                'processing_time': time.time() - start_time,
                'error': f'vLLM 推理超时（{timeout_s:.0f}s）',
                'status': 'error',
            }
        except Exception as e:
            return {
                'text': '',
                'extracted_text': '',
                'text_boxes': [],
                'engine': 'deepseek_ocr_vllm',
                'ocr_engine': 'deepseek_ocr_vllm',
                'confidence': 0.0,
                'confidence_score': 0.0,
                'processing_time': time.time() - start_time,
                'error': f'vLLM 子进程启动失败: {e}',
                'status': 'error',
            }

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            if not err:
                err = f"vLLM 子进程退出码 {proc.returncode}"
            return {
                'text': '',
                'extracted_text': '',
                'text_boxes': [],
                'engine': 'deepseek_ocr_vllm',
                'ocr_engine': 'deepseek_ocr_vllm',
                'confidence': 0.0,
                'confidence_score': 0.0,
                'processing_time': time.time() - start_time,
                'error': err,
                'status': 'error',
            }

        text_out = (proc.stdout or "").strip()
        return {
            'text': text_out,
            'extracted_text': text_out,
            'text_boxes': [],
            'engine': 'deepseek_ocr_vllm',
            'ocr_engine': 'deepseek_ocr_vllm',
            'confidence': 0.0,
            'confidence_score': 0.0,
            'processing_time': time.time() - start_time,
            'status': 'ok' if text_out else 'empty',
        }

    # DeepSeek-OCR 是唯一支持的引擎

    def _preprocess_image(self, image, config):
        """图像预处理"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            return gray
        except Exception:
            return image
    def release_gpu_resources(self):
        try:
            import torch
            if self.deepseek_adapter:
                try:
                    if hasattr(self.deepseek_adapter, 'model') and self.deepseek_adapter.model:
                        try:
                            self.deepseek_adapter.model.to('cpu')
                        except Exception:
                            pass
                        self.deepseek_adapter.model = None
                    if hasattr(self.deepseek_adapter, 'tokenizer') and self.deepseek_adapter.tokenizer:
                        self.deepseek_adapter.tokenizer = None
                except Exception:
                    pass
                self.deepseek_adapter.available = False
                self.deepseek_adapter.loading = False
                self.deepseek_adapter.load_error = None
                self.deepseek_adapter = None
            import gc
            import time
            if torch.cuda.is_available():
                for _ in range(3):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.ipc_collect()
                    time.sleep(0.5)
            gc.collect()
        except Exception:
            pass

    def create_ocr_session(self, user_id: str, file_path: str, ocr_result: Dict, project_id: str = None) -> str:
        """
        创建OCR会话
        
        Args:
            user_id: 用户ID
            file_path: 文件路径
            ocr_result: OCR识别结果
            project_id: 项目ID，如果为None则使用默认项目
            
        Returns:
            会话ID
        """
        try:
            # 获取项目ID
            if project_id is None:
                project_id = 'default'
            
            # 1. 存储图像文件到项目存储
            if os.path.exists(file_path):
                stored_image_path = self.storage_service.store_ocr_image(
                    project_id=project_id,
                    file_path=file_path
                )
            else:
                stored_image_path = file_path  # 如果文件不存在（可能是临时文件已被删除），保留原路径

            # 2. 生成会话ID
            import uuid
            session_id = f"ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            # 3. 插入会话记录
            # 注意：数据库schema中使用image_path，而不是file_path
            session_query = """
                INSERT INTO ocr_sessions (session_id, user_id, project_id, image_path, created_at)
                VALUES (?, ?, ?, ?, ?)
            """
            
            self.db_service.execute_update(session_query, (
                session_id,
                user_id,
                project_id,
                stored_image_path,
                datetime.now().isoformat()
            ))
            
            # 4. 处理OCR结果
            # 提取关键信息用于ocr_results表
            recognized_text = ocr_result.get('text', '') or ocr_result.get('extracted_text', '')
            confidence = ocr_result.get('confidence_score', 0.0) or ocr_result.get('confidence', 0.0)
            processing_time = ocr_result.get('processing_time', 0.0)
            
            # 序列化完整结果
            if isinstance(ocr_result, dict):
                result_json = json.dumps(ocr_result, ensure_ascii=False)
            else:
                result_json = str(ocr_result)
            
            # 5. 插入OCR结果记录
            result_id = str(uuid.uuid4())
            
            # 提取bounding_boxes
            bounding_boxes = None
            if 'text_regions' in ocr_result:
                bounding_boxes = json.dumps(ocr_result['text_regions'], ensure_ascii=False)
            elif 'text_boxes' in ocr_result:
                bounding_boxes = json.dumps(ocr_result['text_boxes'], ensure_ascii=False)

            result_query = """
                INSERT INTO ocr_results (result_id, session_id, recognized_text, confidence, processing_time, bounding_boxes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            self.db_service.execute_update(result_query, (
                result_id,
                session_id,
                recognized_text,
                confidence,
                processing_time,
                bounding_boxes,
                datetime.now().isoformat()
            ))
            
            # 6. 将完整结果存储到项目文件存储（作为备份和详细数据）
            self.storage_service.store_ocr_result(
                project_id=project_id,
                result_data=ocr_result,
                filename=f"{session_id}_result.json"
            )
            try:
                get_event_bus().publish(EventNames.OCR_SESSION_CREATED, {"project_id": project_id, "session_id": session_id})
            except Exception:
                pass
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"创建OCR会话失败: {e}")
            return None

    def get_ocr_sessions(self, user_id: str, project_id: str = None) -> List[Dict]:
        """
        获取用户的OCR会话列表
        
        Args:
            user_id: 用户ID
            project_id: 项目ID，如果为None则不限制
            
        Returns:
            会话列表
        """
        try:
            # 联结 storage_files 表以获取元数据（原始文件名）
            # 注意：数据库字段是image_path，这里别名为file_path以保持兼容性
            if project_id:
                query = """
                    SELECT s.session_id, s.user_id, s.image_path as file_path, s.created_at, s.project_id, sf.metadata
                    FROM ocr_sessions s
                    LEFT JOIN storage_files sf ON s.image_path = sf.file_path
                    WHERE s.user_id = ? AND s.project_id = ?
                    ORDER BY s.created_at DESC
                """
                params = (user_id, project_id)
            else:
                query = """
                    SELECT s.session_id, s.user_id, s.image_path as file_path, s.created_at, s.project_id, sf.metadata
                    FROM ocr_sessions s
                    LEFT JOIN storage_files sf ON s.image_path = sf.file_path
                    WHERE s.user_id = ?
                    ORDER BY s.created_at DESC
                """
                params = (user_id,)
                
            results = self.db_service.execute_query(query, params)
            
            # 处理结果，解析 metadata
            for row in results:
                try:
                    fp = row.get("file_path") or ""
                    if isinstance(fp, str) and fp and not os.path.exists(fp):
                        migrated = self._migrate_legacy_storage_path(fp)
                        if migrated and migrated != fp:
                            row["file_path"] = migrated
                            row["migrated_file_path"] = True
                            if os.path.exists(migrated):
                                try:
                                    self.db_service.execute_update(
                                        "UPDATE ocr_sessions SET image_path=? WHERE session_id=?",
                                        (migrated, row.get("session_id")),
                                    )
                                except Exception:
                                    pass
                except Exception:
                    pass
                if row.get('metadata'):
                    try:
                        if isinstance(row['metadata'], str):
                            meta = json.loads(row['metadata'])
                        else:
                            meta = row['metadata']
                        row['original_filename'] = meta.get('original_filename')
                    except:
                        pass
            
            return results
            
        except Exception as e:
            self.logger.error(f"获取OCR会话列表失败: {e}")
            return []

    def _migrate_legacy_storage_path(self, file_path: str) -> str:
        try:
            p = str(file_path or "")
            if not p:
                return ""
            lower = p.replace("/", "\\").lower()
            marker = "\\project_storage\\"
            idx = lower.find(marker)
            if idx < 0:
                return ""
            rel = p.replace("/", "\\")[idx + len(marker):].lstrip("\\")
            base = str(getattr(self.storage_service, "base_storage_path", "") or "")
            if not base:
                return ""
            return os.path.normpath(os.path.join(base, rel))
        except Exception:
            return ""

    def get_ocr_session_details(self, session_id: str) -> Optional[Dict]:
        """
        获取OCR会话详情
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话详情
        """
        try:
            # 1. 获取会话基本信息
            session_query = """
                SELECT session_id, user_id, image_path as file_path, created_at, project_id
                FROM ocr_sessions
                WHERE session_id = ?
            """
            
            session_results = self.db_service.execute_query(session_query, (session_id,))
            
            if not session_results:
                return None
                
            session = session_results[0]
            try:
                fp = session.get("file_path") or ""
                if isinstance(fp, str) and fp and not os.path.exists(fp):
                    migrated = self._migrate_legacy_storage_path(fp)
                    if migrated:
                        session["file_path"] = migrated
                        if os.path.exists(migrated):
                            try:
                                self.db_service.execute_update(
                                    "UPDATE ocr_sessions SET image_path=? WHERE session_id=?",
                                    (migrated, session_id),
                                )
                            except Exception:
                                pass
            except Exception:
                pass
            
            # 2. 获取OCR结果
            # 尝试从数据库ocr_results表获取
            result_query = """
                SELECT recognized_text, confidence, processing_time, bounding_boxes
                FROM ocr_results
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """
            
            result_rows = self.db_service.execute_query(result_query, (session_id,))
            
            ocr_result = {}
            if result_rows:
                row = result_rows[0]
                text_content = row['recognized_text'] or ''
                ocr_result = {
                    'text': text_content,
                    'extracted_text': text_content,  # 添加 extracted_text 键以兼容 UI
                    'confidence': row['confidence'] or 0.0,
                    'processing_time': row['processing_time'] or 0.0
                }
                if row['bounding_boxes']:
                    try:
                        ocr_result['text_regions'] = json.loads(row['bounding_boxes'])
                    except:
                        pass
            else:
                # 如果数据库中没有详细结果，尝试加载项目存储中的JSON文件
                try:
                    project_id = session.get('project_id', 'default')
                    # 尝试查找对应的结果文件
                    # 这里简化处理，实际上可能需要遍历或有明确的文件ID
                    pass
                except:
                    pass
            
            session['ocr_result'] = ocr_result
            # 为了方便UI直接访问文本，也将extracted_text提升到session层级
            session['text'] = ocr_result.get('text', '')
            session['extracted_text'] = ocr_result.get('extracted_text', '')
            session['confidence'] = ocr_result.get('confidence', 0.0)
            session['processing_time'] = ocr_result.get('processing_time', 0.0)
                    
            return session
            
        except Exception as e:
            self.logger.error(f"获取OCR会话详情失败: {e}")
            return None

    def clear_all_ocr_sessions(self, user_id: str = None, project_id: str = None) -> bool:
        """
        清除指定用户或项目的所有OCR会话
        
        Args:
            user_id: 用户ID（可选）
            project_id: 项目ID（可选）
            
        Returns:
            清除是否成功
        """
        try:
            # 如果没有提供project_id，获取当前项目ID
            if project_id is None:
                project_id = 'default'
                
            # 构建查询条件
            where_conditions = ["project_id = ?"]
            params = [project_id]
            
            if user_id:
                where_conditions.append("user_id = ?")
                params.append(user_id)
                
            # 获取要删除的会话ID列表
            query = f"""
                SELECT session_id FROM ocr_sessions
                WHERE {' AND '.join(where_conditions)}
            """
            
            results = self.db_service.execute_query(query, tuple(params))
            session_ids = [row['session_id'] for row in results]
            
            if not session_ids:
                return True
                
            # 删除相关的结果记录
            # SQLite不支持批量删除 IN 语法很好的参数化，所以循环删除或构造大语句
            # 这里简单起见，使用事务循环删除
            
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 删除 ocr_results
                for sid in session_ids:
                    cursor.execute("DELETE FROM ocr_results WHERE session_id = ?", (sid,))
                    
                # 删除 ocr_sessions
                for sid in session_ids:
                    cursor.execute("DELETE FROM ocr_sessions WHERE session_id = ?", (sid,))
                    
                conn.commit()
                
            self.logger.info(f"成功清除项目 {project_id} 的 {len(session_ids)} 个OCR会话")
            return True
            
        except Exception as e:
            self.logger.error(f"清除OCR会话失败: {e}")
            return False

# 全局OCR服务实例
_global_ocr_service = None

def get_ocr_service(db_service: DatabaseService = None) -> OCRService:
    """获取全局OCR服务实例（单例模式）"""
    global _global_ocr_service
    if _global_ocr_service is None:
        _global_ocr_service = OCRService(db_service)
    return _global_ocr_service
