import json
import logging
import os
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

import requests


logger = logging.getLogger(__name__)


def _normalize_openai_base_url(base_url: str) -> str:
    url = str(base_url or "").strip()
    if not url:
        return ""
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return url
    if "/v1/" in url or url.endswith("/v1"):
        return url
    return url + "/v1"


def _should_bypass_env_proxy(base_url: str) -> bool:
    # Reviewer/Judge requests should not inherit unstable IDE/system proxy settings.
    # This applies to both localhost and remote OpenAI-compatible providers.
    try:
        host = (urlparse(str(base_url or "")).hostname or "").strip().lower()
    except Exception:
        host = ""
    if not host:
        return True
    return True


class LLMJudgementClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout_s: float = 120.0,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = _normalize_openai_base_url(base_url)
        self.api_key = str(api_key or "").strip()
        self.timeout_s = float(timeout_s)
        self.default_headers = dict(default_headers or {})

    @classmethod
    def from_env(cls):
        base_url = os.environ.get("RSNA_JUDGE_BASE_URL", "").strip()
        api_key = os.environ.get("RSNA_JUDGE_API_KEY", "").strip()
        timeout_s = os.environ.get("RSNA_JUDGE_TIMEOUT_S", "120").strip()
        try:
            timeout_s = float(timeout_s)
        except Exception:
            timeout_s = 120.0
        return cls(base_url=base_url, api_key=api_key, timeout_s=timeout_s)

    def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.base_url:
            return {"success": False, "error": "empty_base_url"}
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        headers.update(self.default_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": str(model or "").strip(),
            "messages": messages,
            "temperature": float(temperature),
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if response_format is not None:
            payload["response_format"] = response_format
        if extra:
            payload.update(dict(extra))

        try:
            if _should_bypass_env_proxy(self.base_url):
                session = requests.Session()
                session.trust_env = False
                resp = session.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            else:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        except Exception as e:
            logger.warning(f"LLMJudgementClient request failed: {type(e).__name__}: {e}")
            return {"success": False, "error": f"request_failed:{type(e).__name__}", "exception": str(e)}

        try:
            data = resp.json()
        except Exception:
            data = None

        if resp.status_code >= 400:
            err_msg = ""
            try:
                if isinstance(data, dict):
                    err_msg = json.dumps(data, ensure_ascii=False)[:800]
                else:
                    err_msg = (resp.text or "")[:800]
            except Exception:
                err_msg = ""
            return {
                "success": False,
                "error": f"http_{resp.status_code}",
                "message": err_msg,
            }

        content = ""
        try:
            choice0 = (data or {}).get("choices", [])[0]
            content = (choice0.get("message", {}) or {}).get("content", "") or ""
        except Exception:
            content = ""

        return {"success": True, "content": content, "raw": data}
