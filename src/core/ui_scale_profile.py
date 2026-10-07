import sys
from typing import Tuple


def is_macos() -> bool:
    return sys.platform == "darwin"


def main_window_target(avail_w: int, avail_h: int) -> Tuple[int, int, int, int, str]:
    aw = max(1, int(avail_w))
    ah = max(1, int(avail_h))
    if is_macos():
        aspect = 16 / 10
        max_w = max(1, int(aw * 0.96))
        max_h = max(1, int(ah * 0.94))
        target_w = max_w
        target_h = int(target_w / aspect)
        if target_h > max_h:
            target_h = max_h
            target_w = int(target_h * aspect)
        min_w = min(1200, max_w)
        min_h = min(780, max_h)
        target_w = min(max_w, max(min_w, target_w))
        target_h = min(max_h, max(min_h, target_h))
        return target_w, target_h, max_w, max_h, "macOS比例"
    base_w, base_h = 1920, 1080
    scale = min(aw / base_w, ah / base_h)
    if scale > 1.0:
        scale = 1.0
    target_w = int(base_w * scale)
    target_h = int(base_h * scale)
    return target_w, target_h, aw, ah, "1080比例"


def dialog_target(base_w: int, base_h: int, min_w: int, min_h: int, avail_w: int, avail_h: int) -> Tuple[int, int, int, int]:
    aw = max(1, int(avail_w))
    ah = max(1, int(avail_h))
    if not is_macos():
        return base_w, base_h, min_w, min_h
    scale = min(aw / 1728.0, ah / 1117.0)
    scale = max(0.9, min(scale, 1.35))
    target_w = int(base_w * scale)
    target_h = int(base_h * scale)
    max_w = max(1, int(aw * 0.96))
    max_h = max(1, int(ah * 0.94))
    effective_min_w = min(min_w, max_w)
    effective_min_h = min(min_h, max_h)
    target_w = min(max_w, max(effective_min_w, target_w))
    target_h = min(max_h, max(effective_min_h, target_h))
    return target_w, target_h, effective_min_w, effective_min_h


def splitter_sizes(name: str) -> Tuple[int, int]:
    key = str(name or "").strip().lower()
    if key == "step3":
        return (360, 940) if is_macos() else (400, 1000)
    if key == "step2":
        return (280, 720) if is_macos() else (300, 700)
    return (300, 700)
