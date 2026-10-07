import os
import sys

if getattr(sys, 'frozen', False):
    exe_dir = os.path.dirname(sys.executable)
    internal_dir = os.path.join(exe_dir, '_internal')
    assets_dir = os.path.join(internal_dir, 'assets')
    os.environ['RSNA_DEEPSEEK_OCR_DIR'] = os.path.join(assets_dir, 'models', 'deepseek_ocr')
    os.environ['RSNA_ASSETS_DIR'] = assets_dir

    # --- PyTorch DLL search path / preload fix (WinError 1114) ---------------
    # 让 _internal 与 torch/lib 优先进入 DLL 搜索路径，并预加载
    # torch_global_deps.dll -> c10.dll，规避 "DLL initialization routine failed"。
    candidate_dirs = [
        internal_dir,
        os.path.join(internal_dir, 'torch', 'lib'),
        os.path.join(internal_dir, 'torch', 'bin'),
        exe_dir,
    ]
    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue
        try:
            os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
        except Exception:
            pass
        try:
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(d)
        except Exception:
            pass

    try:
        import ctypes
        torch_lib = os.path.join(internal_dir, 'torch', 'lib')
        for name in ('torch_global_deps.dll', 'c10.dll', 'torch_cpu.dll', 'libiomp5md.dll'):
            p = os.path.join(torch_lib, name)
            if os.path.isfile(p):
                try:
                    ctypes.WinDLL(p)
                except Exception:
                    pass
    except Exception:
        pass
