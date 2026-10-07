import contextlib
import io
import os
import re
import signal
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional


def _extract_python_fence(code: str) -> str:
    s = str(code or "")
    m = re.search(r"```python\s*([\s\S]*?)```", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*([\s\S]*?)```", s)
    if m:
        return m.group(1).strip()
    return s.strip()


@dataclass
class PythonExecResult:
    success: bool
    stdout: str
    stderr: str
    artifacts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "artifacts": list(self.artifacts or []),
        }


def _target(result_queue: Queue, workspace: str, code: str, artifact_dir: str):
    workspace_path = Path(workspace).resolve()
    artifact_path = (workspace_path / artifact_dir).resolve()
    artifact_path.mkdir(parents=True, exist_ok=True)

    stdout_io = io.StringIO()
    stderr_io = io.StringIO()

    def _deny_import(name, *args, **kwargs):
        top = str(name or "").split(".", 1)[0]
        if top in {
            "ctypes",
            "requests",
            "paramiko",
            "websocket",
        }:
            raise ImportError(f"module '{top}' is not allowed")
        return _orig_import(name, *args, **kwargs)

    try:
        os.chdir(str(workspace_path))
        os.environ["WORKSPACE"] = str(workspace_path)
        os.environ["ARTIFACT_DIR"] = str(artifact_path)

        def _ensure_within_workspace(p: str) -> Path:
            target = Path(str(p or "")).expanduser()
            if not target.is_absolute():
                target = (workspace_path / target).resolve()
            else:
                target = target.resolve()
            try:
                target.relative_to(workspace_path)
            except Exception:
                raise ValueError("path_outside_workspace")
            return target

        def list_tabular_json_paths(pattern: str = "tabular/**/*.json", limit: int = 200) -> List[str]:
            pat = str(pattern or "tabular/**/*.json").strip() or "tabular/**/*.json"
            base = workspace_path
            paths = []
            for p in base.glob(pat):
                try:
                    rp = p.resolve().relative_to(workspace_path).as_posix()
                except Exception:
                    continue
                paths.append(rp)
                if len(paths) >= int(limit):
                    break
            return paths

        def load_tabular_json(path: str) -> Dict[str, Any]:
            import json

            p = _ensure_within_workspace(path)
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else {"value": obj}

        def load_all_tabular_json(pattern: str = "tabular/**/*.json", limit: int = 50) -> List[Dict[str, Any]]:
            out = []
            for rp in list_tabular_json_paths(pattern=pattern, limit=limit):
                try:
                    obj = load_tabular_json(rp)
                    obj["_path"] = rp
                    out.append(obj)
                except Exception:
                    continue
            return out

        import builtins

        _orig_import = builtins.__import__
        builtins.__import__ = _deny_import

        try:
            import subprocess

            def _blocked(*_a, **_k):
                raise RuntimeError("subprocess execution is blocked")

            subprocess.Popen = _blocked
            subprocess.run = _blocked
            subprocess.call = _blocked
            subprocess.check_call = _blocked
            subprocess.check_output = _blocked
        except Exception:
            pass

        try:
            import os as _os

            def _blocked_os(*_a, **_k):
                raise RuntimeError("os shell execution is blocked")

            _os.system = _blocked_os
            _os.popen = _blocked_os
        except Exception:
            pass

        try:
            import socket as _socket

            def _blocked_socket(*_a, **_k):
                raise RuntimeError("network is blocked")

            _socket.socket = _blocked_socket
            _socket.create_connection = _blocked_socket
        except Exception:
            pass

        safe_globals: Dict[str, Any] = {
            "__builtins__": builtins,
            "WORKSPACE": str(workspace_path),
            "ARTIFACT_DIR": str(artifact_path),
            "ARTIFACTS": [],
            "list_tabular_json_paths": list_tabular_json_paths,
            "load_tabular_json": load_tabular_json,
            "load_all_tabular_json": load_all_tabular_json,
        }

        with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(stderr_io):
            exec(code, safe_globals, None)

        artifacts = safe_globals.get("ARTIFACTS", [])
        if not isinstance(artifacts, list):
            artifacts = []
        artifacts = [str(x) for x in artifacts if str(x).strip()][:50]

        result_queue.put(
            PythonExecResult(
                success=True,
                stdout=stdout_io.getvalue(),
                stderr=stderr_io.getvalue(),
                artifacts=artifacts,
            ).to_dict()
        )
    except Exception:
        result_queue.put(
            PythonExecResult(
                success=False,
                stdout=stdout_io.getvalue(),
                stderr=stderr_io.getvalue() + "\n" + traceback.format_exc(),
                artifacts=[],
            ).to_dict()
        )


class PythonCodeExecutorToolGroup:
    def __init__(self, workspace: str, timeout_s: float = 45.0, artifact_dir: str = "exports"):
        self.workspace = str(workspace or "").strip()
        self.timeout_s = float(timeout_s)
        self.artifact_dir = str(artifact_dir or "exports").strip()

    def python(self, code: str) -> Dict[str, Any]:
        if not self.workspace:
            return {"success": False, "stdout": "", "stderr": "empty_workspace", "artifacts": []}
        ws = Path(self.workspace)
        if not ws.exists() or not ws.is_dir():
            return {"success": False, "stdout": "", "stderr": f"workspace_not_found:{ws}", "artifacts": []}

        src = _extract_python_fence(code)
        q: Queue = Queue()
        p = Process(target=_target, args=(q, str(ws), src, self.artifact_dir))
        p.start()
        p.join(timeout=self.timeout_s)
        if p.is_alive():
            try:
                os.kill(p.pid, signal.SIGKILL)
            except Exception:
                pass
            try:
                p.terminate()
            except Exception:
                pass
            return {
                "success": False,
                "stdout": "",
                "stderr": f"timeout_after_{self.timeout_s}s",
                "artifacts": [],
            }

        try:
            if not q.empty():
                obj = q.get_nowait()
                try:
                    arts = obj.get("artifacts", []) if isinstance(obj, dict) else []
                    if not isinstance(arts, list):
                        arts = []
                    normalized = []
                    ws_resolved = ws.resolve()
                    for a in arts[:50]:
                        p = str(a or "").strip()
                        if not p:
                            continue
                        p2 = p.replace("\\", "/")
                        try:
                            pp = Path(p2).expanduser()
                            if not pp.is_absolute():
                                pp = (ws_resolved / pp).resolve()
                            else:
                                pp = pp.resolve()
                            pp.relative_to(ws_resolved)
                            normalized.append(pp.relative_to(ws_resolved).as_posix())
                        except Exception:
                            continue
                    if isinstance(obj, dict):
                        obj["artifacts"] = normalized
                except Exception:
                    pass
                return obj
        except Exception:
            pass
        return {"success": False, "stdout": "", "stderr": "no_result", "artifacts": []}
