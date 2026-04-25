import hmac
import hashlib
import base64
import json
import time

def b64(s):
    return base64.urlsafe_b64encode(s).decode().strip("=")

def b64d(s):
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)

def generate_license(cpu_id, expires_ts, secret):
    payload = {"cpu": cpu_id, "exp": int(expires_ts)}
    data = b64(json.dumps(payload, ensure_ascii=False).encode())
    sig = b64(hmac.new(secret.encode(), data.encode(), hashlib.sha256).digest())
    return data + "." + sig

def verify_license(token, secret):
    try:
        data, sig = token.split(".", 1)
        expected = b64(hmac.new(secret.encode(), data.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return False, {}
        payload = json.loads(b64d(data))
        if int(payload.get("exp", 0)) < int(time.time()):
            return False, payload
        return True, payload
    except Exception:
        return False, {}
