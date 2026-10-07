import re
from typing import Dict, List, Tuple, Any


_SECTION_ALIASES = {
    "title": ["title"],
    "abstract": ["abstract", "summary"],
    "introduction": ["introduction"],
    "methods": ["methods", "materials and methods", "materials & methods", "material and methods", "materials", "method"],
    "results": ["results", "findings"],
    "discussion": ["discussion", "interpretation"],
    "conclusion": ["conclusion", "conclusions"],
    "references": ["references", "bibliography"],
}

_CN_ALIASES = {
    "title": ["标题"],
    "abstract": ["摘要"],
    "introduction": ["引言", "简介"],
    "methods": ["方法", "材料与方法", "材料和方法", "研究方法"],
    "results": ["结果"],
    "discussion": ["讨论"],
    "conclusion": ["结论"],
    "references": ["参考文献", "文献", "引用文献"],
}


def _norm_heading(s: str) -> str:
    t = (s or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" :：.-—\t")
    return t


def _is_heading_line(line: str) -> Tuple[bool, str]:
    ln = _norm_heading(line)
    if not ln:
        return False, ""
    low = ln.lower()
    low = re.sub(r"^\d+(\.\d+)*\s+", "", low)
    low = re.sub(r"^[ivxlcdm]+\.\s+", "", low)
    low = low.strip()

    for key, aliases in _SECTION_ALIASES.items():
        for a in aliases:
            if low == a:
                return True, key
            if low.startswith(a + " "):
                return True, key
    for key, aliases in _CN_ALIASES.items():
        for a in aliases:
            if ln == a:
                return True, key
            if ln.startswith(a + " "):
                return True, key
    return False, ""


def extract_citation_style(text: str) -> str:
    """
    尝试从 PDF 文本中提取引用格式风格（例如 [1], (Author, Year), 上标等）。
    """
    import re
    if not text:
        return "Unknown"
        
    # 检查文献列表部分
    ref_match = re.search(r'(References|Bibliography|参考文献)\s*\n(.*?)(?=\n\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
    if ref_match:
        ref_text = ref_match.group(2)
        if re.search(r'^\[\d+\]', ref_text, re.MULTILINE):
            return "Vancouver (Numeric [1])"
        elif re.search(r'^\d+\.', ref_text, re.MULTILINE):
            return "AMA (Numeric 1.)"
            
    # 检查正文中的引用模式
    if re.search(r'\[\d+(?:,\s*\d+)*\]', text):
        return "Vancouver (Numeric [1])"
    elif re.search(r'\([A-Za-z]+ et al\., \d{4}\)', text):
        return "APA (Author, Year)"
        
    return "Vancouver (Numeric [1])"  # 默认后备

def parse_pdf_sections_from_text(text: str) -> Dict[str, str]:
    t = (text or "").replace("\r\n", "\n")
    raw_lines = [x.rstrip() for x in t.split("\n")]
    lines = []
    for ln in raw_lines:
        s = (ln or "").strip()
        if re.match(r"^===\s*第?\s*\d+\s*页\s*===$", s):
            continue
        lines.append(ln)
    marks: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        ok, sec = _is_heading_line(ln)
        if ok:
            marks.append((i, sec))

    if not marks:
        head = "\n".join([x for x in lines[:8] if x.strip()]).strip()
        body = "\n".join(lines).strip()
        return {
            "title": head.split("\n")[0].strip() if head else "",
            "abstract": "",
            "introduction": "",
            "methods": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
            "references": "",
            "full_text": body,
        }

    marks_sorted: List[Tuple[int, str]] = []
    seen = set()
    for idx, sec in sorted(marks, key=lambda x: x[0]):
        if sec in seen:
            continue
        seen.add(sec)
        marks_sorted.append((idx, sec))

    out: Dict[str, str] = {k: "" for k in ["title", "abstract", "introduction", "methods", "results", "discussion", "conclusion", "references"]}
    first_idx = marks_sorted[0][0]
    title_block = "\n".join([x for x in lines[:first_idx] if x.strip()]).strip()
    if title_block:
        out["title"] = title_block.split("\n")[0].strip()

    for j, (start, sec) in enumerate(marks_sorted):
        end = marks_sorted[j + 1][0] if j + 1 < len(marks_sorted) else len(lines)
        content = "\n".join([x for x in lines[start + 1 : end] if x.strip()]).strip()
        if sec in out:
            out[sec] = content

    return out


def build_style_profile(section_text: str) -> Dict[str, Any]:
    txt = (section_text or "").strip()
    if not txt:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_len_words": 0.0,
            "has_numbered_refs": False,
            "has_author_year_refs": False,
            "bullet_ratio": 0.0,
        }
    words = re.findall(r"\b\w+\b", txt)
    sentences = re.split(r"(?<=[.!?。！？])\s+", txt)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_len = 0.0
    if sentences:
        avg_len = sum(len(re.findall(r"\b\w+\b", s)) for s in sentences) / max(1, len(sentences))

    lines = [x.strip() for x in txt.split("\n") if x.strip()]
    bullet_lines = [x for x in lines if re.match(r"^(\-|\*|\u2022|\d+\.)\s+", x)]
    bullet_ratio = len(bullet_lines) / max(1, len(lines))

    has_numbered = bool(re.search(r"\[\s*\d{1,3}(\s*,\s*\d{1,3})*\s*\]", txt))
    has_author_year = bool(re.search(r"\([A-Za-z][A-Za-z\-'\s]+,\s*(19|20)\d{2}\)", txt))
    return {
        "char_count": len(txt),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_len_words": round(avg_len, 2),
        "has_numbered_refs": has_numbered,
        "has_author_year_refs": has_author_year,
        "bullet_ratio": round(bullet_ratio, 3),
    }


def summarize_pdf_template(sections: Dict[str, str]) -> Dict[str, Any]:
    order = [k for k in ["title", "abstract", "introduction", "methods", "results", "discussion", "conclusion", "references"] if (sections.get(k) or "").strip()]
    profiles = {k: build_style_profile(sections.get(k, "")) for k in order if k != "title"}
    structure = [k.capitalize() for k in order if k != "title"]
    citation_style = "Unknown"
    for k, p in profiles.items():
        if p.get("has_numbered_refs"):
            citation_style = "Numeric"
            break
        if p.get("has_author_year_refs"):
            citation_style = "AuthorYear"
            break
    constraints = {
        "source": "pdf_template",
        "detected_structure": structure,
        "style_profiles": profiles,
        "citation_style_hint": citation_style,
    }
    return constraints
