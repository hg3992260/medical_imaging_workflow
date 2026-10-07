import os
import difflib
from typing import Dict, Any, List, Tuple, Optional


def analyze_pdf_template_layout(pdf_path: str, max_pages: int = 20) -> Dict[str, Any]:
    a4_w_pt = 595.0
    a4_h_pt = 842.0

    out: Dict[str, Any] = {
        "page_format": "A4",
        "a4_points": {"width": a4_w_pt, "height": a4_h_pt},
        "pages_analyzed": 0,
        "sections": {},
        "title": None,
        "available": False,
        "error": "",
    }

    try:
        from pdf2image import convert_from_path
    except Exception as e:
        out["error"] = f"pdf2image unavailable: {e}"
        return out

    try:
        from PIL import Image
    except Exception as e:
        out["error"] = f"PIL unavailable: {e}"
        return out

    try:
        from src.ai.pdf_template_parser import _is_heading_line
    except Exception:
        _is_heading_line = None

    def _norm_bbox(x0: float, y0: float, x1: float, y1: float, w: float, h: float) -> Dict[str, float]:
        if w <= 0 or h <= 0:
            return {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}
        return {
            "x0": max(0.0, min(1.0, x0 / w)),
            "y0": max(0.0, min(1.0, y0 / h)),
            "x1": max(0.0, min(1.0, x1 / w)),
            "y1": max(0.0, min(1.0, y1 / h)),
        }

    def _a4_bbox(norm: Dict[str, float]) -> Dict[str, float]:
        return {
            "x0": round(norm["x0"] * a4_w_pt, 2),
            "y0": round(norm["y0"] * a4_h_pt, 2),
            "x1": round(norm["x1"] * a4_w_pt, 2),
            "y1": round(norm["y1"] * a4_h_pt, 2),
        }

    poppler_path = None
    try:
        from src.services.text_service import POPPLER_PATH
        poppler_path = POPPLER_PATH
    except Exception:
        poppler_path = None

    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=max_pages, poppler_path=poppler_path)
    except Exception as e:
        out["error"] = f"convert_from_path failed: {e}"
        return out

    headings: List[Tuple[int, float, str, Dict[str, float], Dict[str, float]]] = []
    title_candidate: Optional[Tuple[int, float, str, Dict[str, float], Dict[str, float]]] = None

    for page_idx, im in enumerate(images, start=1):
        try:
            w, h = float(getattr(im, "width", 0) or 0), float(getattr(im, "height", 0) or 0)
            if w <= 0 or h <= 0:
                continue
            out["available"] = True
            out["pages_analyzed"] = page_idx
        except Exception:
            continue

    out["pages_analyzed"] = len(images)
    out["available"] = True
    return out


def analyze_pdf_template_layout_hybrid(pdf_path: str, deepseek_sections: Optional[Dict[str, str]] = None, max_pages: int = 20) -> Dict[str, Any]:
    deepseek_sections = deepseek_sections or {}
    a4_w_pt = 595.0
    a4_h_pt = 842.0
    out: Dict[str, Any] = {
        "page_format": "A4",
        "a4_points": {"width": a4_w_pt, "height": a4_h_pt},
        "pages_analyzed": 0,
        "sections": {},
        "title": None,
        "available": False,
        "error": "",
        "bbox_source": "",
    }

    try:
        from src.ai.pdf_template_parser import _is_heading_line
    except Exception:
        _is_heading_line = None

    def _norm_bbox(x0: float, y0: float, x1: float, y1: float, w: float, h: float) -> Dict[str, float]:
        if w <= 0 or h <= 0:
            return {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}
        return {
            "x0": max(0.0, min(1.0, x0 / w)),
            "y0": max(0.0, min(1.0, y0 / h)),
            "x1": max(0.0, min(1.0, x1 / w)),
            "y1": max(0.0, min(1.0, y1 / h)),
        }

    def _a4_bbox(norm: Dict[str, float]) -> Dict[str, float]:
        return {
            "x0": round(norm["x0"] * a4_w_pt, 2),
            "y0": round(norm["y0"] * a4_h_pt, 2),
            "x1": round(norm["x1"] * a4_w_pt, 2),
            "y1": round(norm["y1"] * a4_h_pt, 2),
        }

    def _union_bbox_norm(items: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if not items:
            return None
        x0 = min(b["x0"] for b in items)
        y0 = min(b["y0"] for b in items)
        x1 = max(b["x1"] for b in items)
        y1 = max(b["y1"] for b in items)
        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    def _best_title_block(blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        cand = [b for b in blocks if b.get("page") == 1]
        if not cand:
            return None
        title_txt = (deepseek_sections.get("title") or "").strip()
        if not title_txt:
            return cand[0]
        title_low = " ".join(title_txt.lower().split())
        best = None
        best_score = 0.0
        for b in cand[:80]:
            bt = " ".join((b.get("text") or "").lower().split())
            if not bt:
                continue
            score = difflib.SequenceMatcher(a=title_low, b=bt).ratio()
            if score > best_score:
                best_score = score
                best = b
        if best and best_score >= 0.55:
            return best
        return cand[0]

    def _layout_from_blocks(blocks: List[Dict[str, Any]], pages_analyzed: int, source: str) -> Dict[str, Any]:
        out2 = dict(out)
        out2["pages_analyzed"] = pages_analyzed
        out2["available"] = True
        out2["bbox_source"] = source

        title_block = _best_title_block(blocks)
        if title_block:
            out2["title"] = {
                "page": int(title_block["page"]),
                "text": (title_block.get("text") or "")[:200],
                "bbox_norm": title_block.get("bbox_norm"),
                "bbox_a4": title_block.get("bbox_a4"),
            }

        if _is_heading_line is None:
            return out2

        headings = []
        for idx, b in enumerate(blocks):
            ok, sec = _is_heading_line((b.get("text") or "").strip())
            if ok and sec:
                headings.append((idx, sec))

        if not headings:
            return out2

        seen = set()
        uniq = []
        for idx, sec in headings:
            if sec in seen:
                continue
            seen.add(sec)
            uniq.append((idx, sec))

        for j, (start_idx, sec) in enumerate(uniq):
            end_idx = uniq[j + 1][0] if j + 1 < len(uniq) else len(blocks)
            heading_block = blocks[start_idx]
            span_blocks = []
            span_bboxes = []
            for b in blocks[start_idx + 1:end_idx]:
                ok2, sec2 = _is_heading_line((b.get("text") or "").strip())
                if ok2 and sec2:
                    continue
                span_blocks.append({
                    "page": int(b.get("page") or 1),
                    "text": (b.get("text") or "")[:300],
                    "bbox_norm": b.get("bbox_norm"),
                    "bbox_a4": b.get("bbox_a4"),
                })
                if b.get("bbox_norm"):
                    span_bboxes.append(b["bbox_norm"])

            union = _union_bbox_norm(span_bboxes)
            out2["sections"][sec] = {
                "heading": {
                    "page": int(heading_block.get("page") or 1),
                    "text": (heading_block.get("text") or "")[:120],
                    "bbox_norm": heading_block.get("bbox_norm"),
                    "bbox_a4": heading_block.get("bbox_a4"),
                },
                "blocks": span_blocks[:300],
                "aggregate_bbox_norm": union,
                "aggregate_bbox_a4": _a4_bbox(union) if union else None,
                "deepseek_text_preview": (deepseek_sections.get(sec) or "")[:2000],
            }

        return out2

    try:
        import fitz
        doc = fitz.open(pdf_path)
        blocks = []
        pages = min(max_pages, len(doc))
        for pno in range(pages):
            page = doc[pno]
            rect = page.rect
            pw = float(rect.width or 0)
            ph = float(rect.height or 0)
            if pw <= 0 or ph <= 0:
                continue
            d = page.get_text("dict")
            for b in d.get("blocks", []):
                if b.get("type", 0) != 0:
                    continue
                for ln in b.get("lines", []):
                    spans = ln.get("spans", []) or []
                    text = "".join([s.get("text", "") for s in spans]).strip()
                    if not text:
                        continue
                    bbox = ln.get("bbox") or b.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    norm = _norm_bbox(x0, y0, x1, y1, pw, ph)
                    blocks.append({
                        "page": pno + 1,
                        "text": text,
                        "bbox_norm": norm,
                        "bbox_a4": _a4_bbox(norm),
                        "y0": norm["y0"],
                        "x0": norm["x0"],
                    })
        doc.close()
        blocks_sorted = sorted(blocks, key=lambda b: (b.get("page", 1), b.get("y0", 0.0), b.get("x0", 0.0)))
        if blocks_sorted:
            return _layout_from_blocks(blocks_sorted, pages_analyzed=pages, source="pymupdf")
    except Exception:
        pass

    try:
        legacy = analyze_pdf_template_layout(pdf_path, max_pages=max_pages)
        legacy["bbox_source"] = "pil"
        for sec, obj in (legacy.get("sections") or {}).items():
            if isinstance(obj, dict) and "deepseek_text_preview" not in obj:
                obj["deepseek_text_preview"] = (deepseek_sections.get(sec) or "")[:2000]
        if legacy.get("title") and isinstance(legacy["title"], dict):
            legacy["title"]["text"] = (deepseek_sections.get("title") or legacy["title"].get("text") or "")[:200]
        return legacy
    except Exception as e:
        out["available"] = False
        out["error"] = str(e)
        return out
