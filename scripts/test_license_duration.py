import time

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from license_manage.license_crypto import generate_license, verify_license
from license_manage.integration import SECRET


def _fmt(ts: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def main():
    cpu = "TEST_CPU"
    now = int(time.time())
    for days in [30, 90, 180, 365]:
        exp = now + days * 86400
        token = generate_license(cpu, exp, SECRET)
        ok, payload = verify_license(token, SECRET)
        assert ok
        assert int(payload.get("exp", 0)) == exp
        rem_days = max(0, (exp - int(time.time())) // 86400)
        print(f"days={days} exp={_fmt(exp)} rem≈{rem_days} token_prefix={token[:24]}...")


if __name__ == "__main__":
    main()
