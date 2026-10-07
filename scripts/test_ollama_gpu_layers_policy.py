import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.ollama_local_service import _gpu_layers_ladder, _is_oom_error


def main():
    lad = _gpu_layers_ladder(32)
    assert lad[0] == 32
    assert 28 in lad and 24 in lad and 0 in lad
    assert all(lad[i] >= lad[i + 1] for i in range(len(lad) - 1))

    assert _is_oom_error("CUDA out of memory") is True
    assert _is_oom_error("API Error 500: failed to allocate") is True
    assert _is_oom_error("random network error") is False


if __name__ == "__main__":
    main()

