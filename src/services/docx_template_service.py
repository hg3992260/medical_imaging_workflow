"""
DocxTemplate Service — renders formatted .docx from Jinja2 templates.

Template directories:
  resources/docx_templates/journal.j2      — journal paper format
  resources/docx_templates/preprint.j2     — preprint format

Usage:
    from src.services.docx_template_service import render_docx
    path = render_docx(draft_data, template="journal", output_path="output.docx")
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("DocxTemplateService")

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "docx_templates")


def render_docx(draft_data: Dict[str, Any], template: str = "journal", output_path: str = "") -> Optional[str]:
    """Render a draft dict into a formatted .docx file using a Jinja2 template.

    draft_data expected keys:
      - title (str)
      - authors (list of dicts with name, institution, email)
      - abstract (str)
      - sections (ordered list of dicts with heading, content)
      - references (list of dicts with index, citation)
      - tables (optional list of dicts with caption, headers, rows)
      - figures (optional list of dicts with caption, path)
    """
    try:
        from docxtpl import DocxTemplate
    except ImportError:
        logger.error("docxtpl not installed; run: pip install docxtpl")
        return None

    template_path = _resolve_template(template)
    if not template_path:
        logger.error(f"Template '{template}' not found in {_TEMPLATES_DIR}")
        return None

    context = _build_context(draft_data)
    try:
        doc = DocxTemplate(template_path)
        doc.render(context)
        out_path = output_path or _default_output_path(draft_data)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        doc.save(out_path)
        logger.info(f"Rendered .docx to {out_path}")
        return out_path
    except Exception as e:
        logger.error(f"Failed to render docx: {e}")
        return None


def _resolve_template(template: str) -> Optional[str]:
    if template.endswith(".j2") or template.endswith(".docx"):
        if os.path.isfile(template):
            return template
        return None
    path = os.path.join(_TEMPLATES_DIR, f"{template}.j2")
    return path if os.path.isfile(path) else None


def _build_context(data: Dict[str, Any]) -> Dict[str, Any]:
    units = data.get("units") or {}
    meta = data.get("meta") or {}
    abstract = str(meta.get("content") if isinstance(meta, dict) else getattr(meta, "content", "") or "")
    sections = []
    label_map = {
        "intro": "Introduction", "methods": "Methods", "results": "Results",
        "discussion": "Discussion", "conclusion": "Conclusion",
    }
    for key, label in label_map.items():
        unit = units.get(key) or {}
        content = unit.get("content", "") if isinstance(unit, dict) else getattr(unit, "content", "")
        if content:
            sections.append({"heading": label, "content": str(content)})
    refs_raw = data.get("references") or []
    references = []
    for i, r in enumerate(refs_raw[:50], 1):
        citations = r.get("citation") if isinstance(r, dict) else str(r)
        references.append({"index": i, "citation": str(citations)})
    return {
        "title": str(data.get("title", data.get("project_name", "Untitled")) or ""),
        "abstract": abstract,
        "sections": sections,
        "references": references,
        "figures": data.get("figures", []),
        "tables": data.get("tables", []),
    }


def _default_output_path(data: Dict[str, Any]) -> str:
    project_name = data.get("project_name", "draft")
    import re
    safe = re.sub(r'[^\w\-_]', '_', str(project_name))[:60]
    base = os.path.join(os.getcwd(), "exports")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{safe}_formatted.docx")


_docx_templates: List[Dict[str, Any]] = []


def list_templates() -> List[Dict[str, Any]]:
    """Scan resources/docx_templates/ and return available templates."""
    if _docx_templates:
        return _docx_templates
    tdir = _TEMPLATES_DIR
    if not os.path.isdir(tdir):
        return [{"name": "journal", "path": "", "description": "Default journal format (built-in)"}]
    templates = []
    for f in sorted(os.listdir(tdir)):
        if f.endswith(".j2") or f.endswith(".docx"):
            name = f.replace(".j2", "").replace(".docx", "")
            desc = _template_description(name)
            templates.append({"name": name, "path": os.path.join(tdir, f), "description": desc})
    if not templates:
        templates.append({"name": "journal", "path": "", "description": "Default journal format (built-in)"})
    _docx_templates.extend(templates)
    return templates


def _template_description(name: str) -> str:
    descriptions = {
        "journal": "Journal paper format (IMRaD: Introduction, Methods, Results, Discussion)",
        "preprint": "Preprint format with abstract, numbered sections, references at end",
    }
    return descriptions.get(name, f"Custom template: {name}")
