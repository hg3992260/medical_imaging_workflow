import subprocess
import sys
import os
try:
    import winreg
except Exception:
    winreg = None

def get_cpu_fingerprint():
    if sys.platform == "darwin":
        try:
            p = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=3)
            s = (p.stdout or "").strip()
            if s:
                return s
        except Exception:
            pass
        try:
            p = subprocess.run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], capture_output=True, text=True, timeout=3)
            for line in (p.stdout or "").splitlines():
                if "IOPlatformUUID" in line:
                    v = line.split("=")[-1].strip().strip('"')
                    if v:
                        return v
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            for fp in ["/etc/machine-id", "/var/lib/dbus/machine-id"]:
                if os.path.exists(fp):
                    v = open(fp, "r", encoding="utf-8").read().strip()
                    if v:
                        return v
        except Exception:
            pass
    try:
        p = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty ProcessorId"], capture_output=True, text=True, timeout=3)
        val = (p.stdout or "").strip().splitlines()
        for line in val:
            s = line.strip()
            if s:
                return s
    except Exception:
        pass
    try:
        p = subprocess.run(["wmic", "cpu", "get", "ProcessorId"], capture_output=True, text=True, timeout=3)
        lines = (p.stdout or "").splitlines()
        if len(lines) >= 2:
            s = lines[1].strip()
            if s:
                return s
    except Exception:
        pass
    if winreg is not None:
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography")
            val, _ = winreg.QueryValueEx(key, "MachineGuid")
            if val:
                return val
        except Exception:
            pass
    return os.environ.get("PROCESSOR_IDENTIFIER", "UNKNOWN")
