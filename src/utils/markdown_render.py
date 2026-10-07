import html
import re
from pathlib import Path
from typing import Iterable, Optional


_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def md_to_html_with_local_images(md: str, base_dirs: Optional[Iterable[str]] = None) -> str:
    s = str(md or "")
    lines = s.splitlines()
    out = []
    in_list = False
    bases = []
    try:
        for b in (base_dirs or []):
            sb = str(b or "").strip()
            if sb:
                bases.append(sb)
    except Exception:
        bases = []

    def _resolve_img_src(raw_src: str) -> str:
        src0 = (raw_src or "").strip().strip("<>").strip()
        if not src0:
            return ""
        src0 = src0.replace("\\", "/")
        low = src0.lower()
        if low.startswith("http://") or low.startswith("https://") or low.startswith("file://") or low.startswith("qrc:/"):
            return src0
        for b in bases:
            try:
                p = (Path(b) / src0).expanduser().resolve()
                if p.exists() and p.is_file():
                    return p.as_uri()
            except Exception:
                continue
        return src0

    def _render_inline(text_line: str) -> str:
        s0 = str(text_line or "")
        parts = []
        last = 0
        for m in _IMG_RE.finditer(s0):
            before = s0[last : m.start()]
            if before:
                parts.append(html.escape(before))
            alt = html.escape(m.group(1) or "")
            src = html.escape(_resolve_img_src(str(m.group(2) or "")))
            parts.append(f"<img src='{src}' alt='{alt}' style='max-width:100%; height:auto;'/>")
            last = m.end()
        tail = s0[last:]
        if tail:
            parts.append(html.escape(tail))
        merged = "".join(parts) if parts else html.escape(s0)
        merged = _LINK_RE.sub(lambda mm: f"<a href='{html.escape(mm.group(2).strip())}'>{html.escape(mm.group(1))}</a>", merged)
        return merged

    def _close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw in lines:
        line = str(raw or "")
        if not line.strip():
            _close_list()
            out.append("<br/>")
            continue

        stripped_line = line.strip()
        m = _IMG_RE.search(stripped_line)
        if m and m.group(0).strip() == stripped_line:
            _close_list()
            alt = html.escape(m.group(1) or "")
            src = html.escape(_resolve_img_src(str(m.group(2) or "")))
            out.append(f"<div style='margin:8px 0'><img src='{src}' alt='{alt}' style='max-width:100%; height:auto;'/><div style='font-size:12px; opacity:0.85; margin-top:4px'>{alt}</div></div>")
            continue

        if line.startswith("### "):
            _close_list()
            out.append(f"<h3 style='margin:10px 0 6px 0'>{html.escape(line[4:].strip())}</h3>")
            continue
        if line.startswith("## "):
            _close_list()
            out.append(f"<h2 style='margin:12px 0 8px 0'>{html.escape(line[3:].strip())}</h2>")
            continue
        if line.startswith("# "):
            _close_list()
            out.append(f"<h1 style='margin:14px 0 10px 0'>{html.escape(line[2:].strip())}</h1>")
            continue

        if line.lstrip().startswith("- "):
            if not in_list:
                out.append("<ul style='margin:6px 0 6px 18px'>")
                in_list = True
            item = line.lstrip()[2:]
            item_html = _render_inline(item)
            out.append(f"<li>{item_html}</li>")
            continue

        _close_list()
        out.append(f"<div style='white-space:pre-wrap'>{_render_inline(line)}</div>")

    _close_list()
    return "<html><body style='font-family:Segoe UI, Arial; font-size:13px;'>" + "\n".join(out) + "</body></html>"
