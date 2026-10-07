#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
首次运行初始化服务 (First-run setup service)

职责：
  1. 检测模型权重是否就绪（DeepSeek-OCR / SAM / 文本嵌入）；
  2. 提供模型权重下载（HTTP 直链 + HuggingFace 仓库快照）；
  3. 检测本地 Ollama 运行时与模型；
  4. 检测 / 保存 / 测试外部 OpenAI 兼容 API 配置；
  5. 记录首次初始化完成标记。

设计原则：
  * 不依赖 huggingface_hub（主程序默认 HF_HUB_OFFLINE=1），统一走 requests 直连，
    因此下载在离线默认环境中也能工作。
  * 所有路径基于 get_base_dir()（打包后=exe 同级目录；源码=项目根），
    与 model_config_loader 保持一致。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from typing import Callable, Dict, List, Optional

import requests

from src.core.path_config import get_base_dir, get_coco_data_dir
from src.utils.model_config_loader import get_model_paths

ProgressCb = Callable[[str, int, int], None]   # (resource_id, downloaded, total)
LogCb = Callable[[str], None]

# ---------------------------------------------------------------------------
# 模型权重清单
# ---------------------------------------------------------------------------
MODEL_SPECS: List[Dict] = [
    {
        "id": "deepseek_ocr",
        "name": "DeepSeek-OCR 权重",
        "desc": "OCR 文本识别模型（第三步/OCR 识别使用）",
        "kind": "hf_snapshot",
        "repo_id": "deepseek-ai/DeepSeek-OCR",
        "target": "assets/models/deepseek_ocr",
        "size_label": "≈ 6.8 GB",
        "required": False,
    },
    {
        "id": "embedding",
        "name": "文本嵌入模型 all-MiniLM-L6-v2",
        "desc": "知识库检索 / 语义嵌入（CocoIndex）",
        "kind": "hf_snapshot",
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "target": "models/embedding/all-MiniLM-L6-v2",
        "size_label": "≈ 90 MB",
        "required": False,
    },
    {
        "id": "sam_vit_b",
        "name": "SAM ViT-B 权重",
        "desc": "交互式分割（Magic Seg / DICOM ROI）",
        "kind": "url",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "target": "models/sam/sam_vit_b_01ec64.pth",
        "size_label": "≈ 375 MB",
        "required": False,
    },
]

OLLAMA_BINARY_CANDIDATES = [
    os.path.join("assets", "ollama", "ollama.exe"),
    os.path.join("_internal", "assets", "ollama", "ollama.exe"),
]

HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}?download=true"
HF_API = "https://huggingface.co/api/models/{repo}"


def _abs_target(rel: str) -> str:
    p = rel.replace("/", os.sep)
    if os.path.isabs(p):
        return p
    return os.path.join(str(get_base_dir()), p)


def _has_any(target_dir: str, names) -> bool:
    if not os.path.isdir(target_dir):
        return False
    for n in names:
        if os.path.isfile(os.path.join(target_dir, n)):
            return True
    return False


def _dir_size(path: str) -> int:
    total = 0
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except Exception:
            return 0
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except Exception:
                    pass
    return total


class SetupService:
    """首次运行初始化：状态检测 + 资源下载 + API 配置。"""

    def __init__(self):
        self._logger = None
        try:
            from src.utils.logger import get_logger
            self._logger = get_logger(__name__)
        except Exception:
            pass

    # ------------------------------------------------------------------ 日志
    def _log(self, msg: str):
        if self._logger:
            try:
                self._logger.info(msg)
            except Exception:
                pass

    # -------------------------------------------------------- 配置：完成标记
    def _config(self):
        from src.core.app_config import AppConfig
        return AppConfig()

    def is_setup_completed(self) -> bool:
        try:
            return bool(self._config().get("setup.completed", False))
        except Exception:
            return False

    def mark_setup_completed(self):
        try:
            cfg = self._config()
            cfg.set("setup.completed", True)
            cfg.set("setup.completed_at", int(time.time()))
            cfg.save()
            self._log("Setup marked as completed.")
        except Exception as e:
            print(f"mark_setup_completed failed: {e}")

    def needs_setup(self) -> bool:
        """是否需要弹出初始化面板：未完成过初始化即需要。"""
        if not self.is_setup_completed():
            return True
        # 已完成，但若所有推理后端都不可用，也提示一次
        try:
            st = self.collect_status()
            if not st["api"]["configured"] and not st["ollama"]["available"]:
                return True
        except Exception:
            pass
        return False

    # -------------------------------------------------------- 状态：模型权重
    def model_status(self, spec: Dict) -> Dict:
        target = _abs_target(spec["target"])
        rid = spec["id"]
        present = False
        if rid == "deepseek_ocr":
            present = os.path.isfile(os.path.join(target, "config.json")) and _has_any(
                target,
                ["model.safetensors", "model-00001-of-000001.safetensors",
                 "pytorch_model.bin", "model.safetensors.index.json"],
            )
        elif rid == "embedding":
            present = _has_any(target, ["config.json"]) and _has_any(
                target, ["model.safetensors", "pytorch_model.bin"]
            )
        else:
            present = os.path.isfile(target) and os.path.getsize(target) > 1_000_000
        return {
            "id": rid,
            "name": spec["name"],
            "desc": spec.get("desc", ""),
            "target": target,
            "size_label": spec.get("size_label", ""),
            "present": bool(present),
            "actual_bytes": _dir_size(target),
            "required": bool(spec.get("required", False)),
        }

    def all_model_statuses(self) -> List[Dict]:
        return [self.model_status(s) for s in MODEL_SPECS]

    # -------------------------------------------------------- 状态：Ollama
    def _ollama_host(self) -> str:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not host.startswith("http"):
            host = "http://" + host
        return host.rstrip("/")

    def _ollama_binary(self) -> Optional[str]:
        which = shutil.which("ollama")
        if which:
            return which
        for rel in OLLAMA_BINARY_CANDIDATES:
            p = os.path.join(str(get_base_dir()), rel)
            if os.path.isfile(p):
                return p
        return None

    def ollama_status(self) -> Dict:
        binary = self._ollama_binary()
        running = False
        models: List[str] = []
        try:
            r = requests.get(f"{self._ollama_host()}/api/tags", timeout=1.5)
            if r.status_code == 200:
                running = True
                models = [m.get("name") for m in r.json().get("models", []) if m.get("name")]
        except Exception:
            pass
        return {
            "installed": bool(binary),
            "binary": binary,
            "host": self._ollama_host(),
            "running": running,
            "models": models,
            "available": bool(running and models),
        }

    # -------------------------------------------------------- 状态：API Key
    def _profile_path(self) -> str:
        return os.path.join(str(get_coco_data_dir()), "custom_api_profiles.json")

    def load_api_profiles(self) -> List[Dict]:
        try:
            p = self._profile_path()
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            self._log(f"load_api_profiles failed: {e}")
        return []

    def api_status(self) -> Dict:
        profiles = self.load_api_profiles()
        valid = [
            p for p in profiles
            if str(p.get("base_url") or "").strip() and str(p.get("api_key") or "").strip()
        ]
        env_key = os.environ.get("RSNA_JUDGE_API_KEY", "").strip()
        env_url = os.environ.get("RSNA_JUDGE_BASE_URL", "").strip()
        return {
            "path": self._profile_path(),
            "count": len(profiles),
            "valid_count": len(valid),
            "profiles": profiles,
            "names": [str(p.get("name") or "Custom") for p in profiles],
            "env_key": bool(env_key),
            "env_url": env_url,
            "configured": bool(valid) or bool(env_key),
        }

    def save_api_profile(self, profile: Dict) -> bool:
        try:
            p = self._profile_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            profiles = self.load_api_profiles()
            name = str(profile.get("name") or "Custom").strip()
            idx = next((i for i, x in enumerate(profiles) if x.get("name") == name), -1)
            if idx >= 0:
                profiles[idx] = profile
            else:
                profiles.append(profile)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(profiles, f, ensure_ascii=False, indent=2)
            self._log(f"Saved API profile: {name}")
            return True
        except Exception as e:
            self._log(f"save_api_profile failed: {e}")
            return False

    def test_api_profile(self, profile: Dict) -> tuple:
        """向 OpenAI 兼容端点发一个轻量请求验证连通性。"""
        base = str(profile.get("base_url") or "").strip().rstrip("/")
        key = str(profile.get("api_key") or "").strip()
        if not base:
            return False, "Base URL 为空"
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            r = requests.get(f"{base}/models", headers=headers, timeout=10)
            if r.status_code < 500:
                return True, f"HTTP {r.status_code}（端点可达）"
            return False, f"HTTP {r.status_code}"
        except Exception as e:
            return False, f"连接失败: {e}"

    # -------------------------------------------------------- 汇总状态
    def collect_status(self) -> Dict:
        models = self.all_model_statuses()
        return {
            "models": models,
            "missing": [m for m in models if not m["present"]],
            "ollama": self.ollama_status(),
            "api": self.api_status(),
        }

    # -------------------------------------------------------- 下载
    def _download_url(self, url: str, dest: str, rid: str,
                      progress_cb: Optional[ProgressCb], cancel=None) -> bool:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if cancel is not None and cancel():
                            f.close()
                            try:
                                os.remove(tmp)
                            except Exception:
                                pass
                            return False
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        if progress_cb:
                            progress_cb(rid, done, total)
            if os.path.exists(dest):
                os.remove(dest)
            os.replace(tmp, dest)
            return True
        except Exception as e:
            self._log(f"download failed {url}: {e}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return False

    def _hf_list_files(self, repo_id: str) -> List[str]:
        r = requests.get(HF_API.format(repo=repo_id), timeout=30)
        r.raise_for_status()
        data = r.json()
        files = []
        for s in data.get("siblings", []):
            name = s.get("rfilename") or ""
            if not name or name.endswith(".gitattributes"):
                continue
            files.append(name)
        return files

    def download_resource(self, spec: Dict, progress_cb: Optional[ProgressCb] = None,
                          log_cb: Optional[LogCb] = None, cancel=None) -> bool:
        rid = spec["id"]
        target = _abs_target(spec["target"])
        emit = log_cb or (lambda m: None)

        if spec["kind"] == "url":
            emit(f"开始下载 {spec['name']} …")
            ok = self._download_url(spec["url"], target, rid, progress_cb, cancel)
            emit(("完成 " if ok else "失败 ") + spec["name"])
            return ok

        if spec["kind"] == "hf_snapshot":
            repo = spec["repo_id"]
            try:
                emit(f"获取文件列表 {repo} …")
                files = self._hf_list_files(repo)
            except Exception as e:
                emit(f"获取文件列表失败: {e}")
                return False
            if not files:
                emit("仓库文件列表为空")
                return False
            os.makedirs(target, exist_ok=True)
            ok_all = True
            for i, rel in enumerate(files, 1):
                if cancel is not None and cancel():
                    emit("已取消")
                    return False
                dest = os.path.join(target, rel.replace("/", os.sep))
                if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                    emit(f"[{i}/{len(files)}] 已存在，跳过 {rel}")
                    continue
                url = HF_RESOLVE.format(repo=repo, path=rel)
                emit(f"[{i}/{len(files)}] 下载 {rel}")
                if not self._download_url(url, dest, rid, progress_cb, cancel):
                    ok_all = False
            emit(("完成 " if ok_all else "部分失败 ") + spec["name"])
            return ok_all

        emit(f"未知资源类型: {spec.get('kind')}")
        return False

    def spec_by_id(self, rid: str) -> Optional[Dict]:
        return next((s for s in MODEL_SPECS if s["id"] == rid), None)
