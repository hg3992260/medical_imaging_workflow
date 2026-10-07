import json
import urllib.request
import urllib.error
import re

class OllamaService:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
    def list_models(self):
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                return models
        except Exception:
            return []
    def ping(self):
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return True
        except Exception:
            return False
    def show_model(self, name: str):
        try:
            url = f"{self.base_url}/api/show"
            payload = {"name": name}
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except Exception:
            return None
    def _rank_model_name(self, name: str) -> int:
        m = re.search(r'(\d+)\s*b', name.lower())
        if m:
            try:
                return int(m.group(1))
            except:
                return 9999
        return 9999
    def pick_fallback_model(self, failed_model: str) -> str:
        try:
            models = self.list_models()
            candidates = [m for m in models if m != failed_model]
            if not candidates:
                return ""
            ranked = sorted(candidates, key=lambda n: self._rank_model_name(n))
            return ranked[0]
        except Exception:
            return ""
    def generate(self, model: str, prompt: str, options: dict = None, timeout_secs: int = 300, retries: int = 1):
        logs = []
        try:
            if not self.ping():
                logs.append("ping_failed")
                return {"success": False, "response": "", "error": "Server unreachable", "logs": logs}
            mdl = self.show_model(model)
            if mdl is None:
                logs.append("model_not_found")
            else:
                logs.append("model_found")
            url = f"{self.base_url}/api/generate"
            payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "5m"}
            if options:
                payload["options"] = options
            body = json.dumps(payload).encode("utf-8")
            logs.append(f"POST {url}")
            logs.append(f"model={model}")
            logs.append(f"payload_bytes={len(body)}")
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_secs) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                logs.append(f"status={status}")
                data = json.loads(resp.read().decode("utf-8"))
                response = data.get("response", "")
                return {"success": True, "response": response, "error": "", "logs": logs}
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = ""
            logs.append(f"http_error={e.code}")
            if err_body:
                logs.append(f"error_body={err_body[:500]}")
            return {"success": False, "response": "", "error": f"HTTP {e.code}", "logs": logs}
        except urllib.error.URLError as e:
            logs.append(f"url_error={str(e)}")
            if retries > 0:
                logs.append("retrying")
                try:
                    url = f"{self.base_url}/api/generate"
                    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "5m"}
                    tuned_options = dict(options or {})
                    tuned_options.setdefault("num_predict", 256)
                    payload["options"] = tuned_options
                    if options:
                        payload["options"].update(options)
                    body = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
                    with urllib.request.urlopen(req, timeout=timeout_secs) as resp2:
                        status2 = getattr(resp2, "status", None) or resp2.getcode()
                        logs.append(f"status={status2}")
                        data2 = json.loads(resp2.read().decode("utf-8"))
                        response2 = data2.get("response", "")
                        return {"success": True, "response": response2, "error": "", "logs": logs}
                except Exception as e2:
                    logs.append(f"retry_error={str(e2)}")
            fb = self.pick_fallback_model(model)
            if fb:
                logs.append(f"fallback_model={fb}")
                try:
                    url = f"{self.base_url}/api/generate"
                    payload = {"model": fb, "prompt": prompt, "stream": False, "keep_alive": "5m"}
                    tuned_options = dict(options or {})
                    tuned_options.setdefault("num_predict", 256)
                    payload["options"] = tuned_options
                    body = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
                    with urllib.request.urlopen(req, timeout=timeout_secs) as resp3:
                        status3 = getattr(resp3, "status", None) or resp3.getcode()
                        logs.append(f"status={status3}")
                        data3 = json.loads(resp3.read().decode("utf-8"))
                        response3 = data3.get("response", "")
                        return {"success": True, "response": response3, "error": "", "logs": logs}
                except Exception as e3:
                    logs.append(f"fallback_error={str(e3)}")
            return {"success": False, "response": "", "error": "Connection error", "logs": logs}
        except Exception as e:
            logs.append(f"error={str(e)}")
            return {"success": False, "response": "", "error": "Unknown error", "logs": logs}
