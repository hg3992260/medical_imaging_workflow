import asyncio
import os
import sys
import tempfile
import types

api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DeepAnalyze-main", "API"))
sys.path.insert(0, api_dir)

fastapi_mod = types.ModuleType("fastapi")


class _HTTPException(Exception):
    pass


class _APIRouter:
    def __init__(self, *args, **kwargs):
        pass

    def post(self, *args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator


fastapi_mod.HTTPException = _HTTPException
fastapi_mod.APIRouter = _APIRouter
fastapi_mod.Body = lambda default=None, **kwargs: default
sys.modules["fastapi"] = fastapi_mod

responses_mod = types.ModuleType("fastapi.responses")
responses_mod.StreamingResponse = lambda content, media_type=None: {"content": content, "media_type": media_type}
sys.modules["fastapi.responses"] = responses_mod

httpx_mod = types.ModuleType("httpx")
httpx_mod.Client = lambda *args, **kwargs: object()
httpx_mod.AsyncClient = lambda *args, **kwargs: object()
sys.modules["httpx"] = httpx_mod

openai_mod = types.ModuleType("openai")
openai_mod.OpenAI = lambda *args, **kwargs: types.SimpleNamespace()
openai_mod.AsyncOpenAI = lambda *args, **kwargs: types.SimpleNamespace()
sys.modules["openai"] = openai_mod

api_config_mod = types.ModuleType("api_config")
api_config_mod.API_BASE = "http://127.0.0.1:11434/v1"
api_config_mod.DEFAULT_TEMPERATURE = 0.4
api_config_mod.STOP_TOKEN_IDS = []
api_config_mod.MAX_NEW_TOKENS = 2048
sys.modules["api_config"] = api_config_mod

models_mod = types.ModuleType("models")
models_mod.ChatCompletionRequest = object
models_mod.ChatCompletionResponse = object
models_mod.ChatCompletionChoice = object
sys.modules["models"] = models_mod

storage_mod = types.ModuleType("storage")
storage_mod.storage = types.SimpleNamespace(
    create_thread=lambda metadata=None: types.SimpleNamespace(id="thread-test"),
    get_thread=lambda thread_id: None,
    get_file=lambda fid: None,
    files={},
)
sys.modules["storage"] = storage_mod

utils_mod = types.ModuleType("utils")
utils_mod.get_thread_workspace = lambda thread_id: tempfile.mkdtemp(prefix="chat_api_guard_stub_")
utils_mod.prepare_vllm_messages = lambda messages, workspace_dir: [{"role": "user", "content": "run"}]
utils_mod.execute_code_safe = lambda *args, **kwargs: "ok"


async def _stub_execute_code_safe_async(code_str, workspace_dir):
    return "ok"


utils_mod.execute_code_safe_async = _stub_execute_code_safe_async
utils_mod.WorkspaceTracker = lambda workspace_dir, generated_dir: types.SimpleNamespace(diff_and_collect=lambda: [])
utils_mod.render_file_block = lambda *args, **kwargs: ""
utils_mod.generate_report_from_messages = lambda *args, **kwargs: ""
utils_mod.extract_code_from_segment = lambda content: "print(1)"
utils_mod.uniquify_path = lambda path: path
sys.modules["utils"] = utils_mod

import chat_api


class _Chunk:
    def __init__(self, content, finish_reason=None):
        delta = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(delta=delta, finish_reason=finish_reason)
        self.choices = [choice]


class _Response:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        value = self._chunks[self._index]
        self._index += 1
        return value


class _Completions:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return _Response([
            _Chunk("<Code>print(1)</Code>"),
            _Chunk(None, "stop"),
        ])


async def _fake_execute(code_str, workspace_dir):
    return "ok"


async def main():
    os.environ["DEEPANALYZE_MAX_EXEC_TURNS"] = "3"
    workspace_dir = tempfile.mkdtemp(prefix="chat_api_guard_")
    completions = _Completions()

    chat_api.storage.create_thread = lambda metadata=None: types.SimpleNamespace(id="thread-test")
    chat_api.get_thread_workspace = lambda thread_id: workspace_dir
    chat_api.prepare_vllm_messages = lambda messages, wd: [{"role": "user", "content": "run"}]
    chat_api.execute_code_safe_async = _fake_execute
    chat_api.render_file_block = lambda *args, **kwargs: ""
    chat_api.generate_report_from_messages = lambda *args, **kwargs: ""
    chat_api.vllm_client_async = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=completions)
    )

    result = await chat_api.chat_completions(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
    )

    content = result["choices"][0]["message"]["content"]
    assert "Execution halted: repeated code block detected." in content
    assert completions.calls == 2


if __name__ == "__main__":
    asyncio.run(main())
