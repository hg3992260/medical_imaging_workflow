import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.ollama_local_service import OllamaLocalService


class FakeSvc(OllamaLocalService):
    def __init__(self):
        super().__init__(binary="ollama", base_url="http://127.0.0.1:11434")
        self._models_cache = ["a", "b"]
        self._last_cache_time = 10**9
        self._cache_ttl = 10**9

    def list_models(self):
        return ["a", "b"]

    def _generate_via_cli(self, model: str, prompt: str, timeout: int, stream_callback=None):
        return {"success": False, "response": "", "error": "fail", "logs": []}

    def generate(self, model: str, prompt: str, timeout: int = None, num_predict: int = -1, options: dict = None, stream_callback=None, _fallback_tried=None, _fallback_depth: int = 0):
        return self._handle_fallback(model, prompt, timeout, [], "fail", stream_callback, _fallback_tried=_fallback_tried, _fallback_depth=_fallback_depth)


def main():
    os.environ["OLLAMA_FALLBACK_MAX_DEPTH"] = "2"
    svc = FakeSvc()
    res = svc.generate("a", "x")
    assert res.get("success") is False
    assert res.get("error") in ["fail", "generation_failed"]


if __name__ == "__main__":
    main()

