import os
import time
from pathlib import Path
from .hardware import get_cpu_fingerprint
from .license_crypto import generate_license, verify_license
from .storage import init_db, add_record

SECRET = "RSNA_MW_LICENSE_V1"

def get_base_dir():
    import sys
    if getattr(sys, 'frozen', False):
        return Path(os.path.dirname(sys.executable))
    return Path(__file__).resolve().parent.parent

def get_license_paths():
    base = get_base_dir()
    paths = []
    import sys
    if getattr(sys, "frozen", False):
        paths.append(base / "license.dat")
        try:
            meipass = getattr(sys, "_MEIPASS", "")
            if meipass:
                paths.append(Path(meipass) / "license.dat")
        except Exception:
            pass
    else:
        paths.append(base / "license.dat")
    return paths

def ensure_license(default_days=180):
    init_db()
    cpu = get_cpu_fingerprint()
    paths = get_license_paths()
    write_paths = [get_base_dir() / "license.dat"]
    token = ""
    for lp in paths:
        if lp.exists():
            try:
                token = lp.read_text(encoding="utf-8").strip()
                break
            except Exception:
                token = ""
    ok = False
    payload = {}
    if token:
        ok, payload = verify_license(token, SECRET)
    if not ok:
        exp = int(time.time()) + default_days * 86400
        token = generate_license(cpu, exp, SECRET)
        for lp in write_paths:
            try:
                lp.parent.mkdir(parents=True, exist_ok=True)
                lp.write_text(token, encoding="utf-8")
            except Exception:
                pass
        add_record(cpu, exp, token)
        ok, payload = verify_license(token, SECRET)
    remaining = max(0, int(payload.get("exp", 0)) - int(time.time()))
    return token, remaining

def validate_license_strict():
    """
    只验证当前存在的 License，不自动生成。
    如果验证失败（文件不存在、签名错误、机器码不匹配、过期），返回 (False, msg, remaining_seconds, exp_ts)
    """
    paths = get_license_paths()
    token = ""
    for lp in paths:
        if lp.exists():
            try:
                token = lp.read_text(encoding="utf-8").strip()
                break
            except Exception:
                pass
    
    if not token:
        return False, "License file not found", 0, 0
        
    ok, payload = verify_license(token, SECRET)
    if not ok:
        return False, "Invalid signature or expired", 0, int(payload.get("exp", 0) or 0)
        
    # Check machine fingerprint
    current_cpu = get_cpu_fingerprint()
    license_cpu = payload.get("cpu", "")
    
    if current_cpu != license_cpu:
        return False, f"Machine mismatch (License: {license_cpu}, Current: {current_cpu})", 0, int(payload.get("exp", 0) or 0)
        
    remaining = max(0, int(payload.get("exp", 0)) - int(time.time()))
    if remaining <= 0:
        return False, "License expired", 0, int(payload.get("exp", 0) or 0)
        
    return True, "Valid", remaining, int(payload.get("exp", 0) or 0)
