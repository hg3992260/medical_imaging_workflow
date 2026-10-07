"""
Tool Router — selects optimal document processing path between text_service (full) and ppt-master (fast).

Works in two modes:
  - source code mode:  calls ppt-master scripts directly (scripts/source_to_md/*.py)
  - frozen mode:       looks for bundled scripts at _internal/ressources/ppt-master or sys._MEIPASS/ressources/ppt-master
"""

import logging
import os
import subprocess
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger("ToolRouter")


def _ppt_master_scripts_root() -> Optional[str]:
    """Locate ppt-master scripts directory (works in source and frozen modes)."""
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "skills", "ppt-master", "scripts"),
        os.path.join(os.path.dirname(sys.executable), "_internal", "ressources", "scripts"),
        os.path.join(getattr(sys, "_MEIPASS", ""), "ressources", "scripts"),
    ]
    for c in candidates:
        if os.path.isdir(c) and any(f.startswith("pdf_") for f in os.listdir(c)):
            return c
    return None


def convert_pdf_to_md(file_path: str, output_dir: Optional[str] = None) -> Optional[str]:
    """Fast path: convert PDF to Markdown via ppt-master's pdf_to_md.py."""
    scripts_root = _ppt_master_scripts_root()
    if not scripts_root:
        return None
    script = os.path.join(scripts_root, "source_to_md", "pdf_to_md.py")
    if not os.path.isfile(script):
        return None
    try:
        result = subprocess.run(
            [sys.executable, script, file_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and result.stdout.strip():
            out_path = result.stdout.strip().splitlines()[-1] if "\n" in result.stdout else result.stdout.strip()
            return out_path if os.path.isfile(out_path) else None
        return None
    except Exception:
        return None


def convert_docx_to_md(file_path: str, output_dir: Optional[str] = None) -> Optional[str]:
    """Fast path: convert DOCX to Markdown via ppt-master's doc_to_md.py."""
    scripts_root = _ppt_master_scripts_root()
    if not scripts_root:
        return None
    script = os.path.join(scripts_root, "source_to_md", "doc_to_md.py")
    if not os.path.isfile(script):
        return None
    try:
        result = subprocess.run(
            [sys.executable, script, file_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and result.stdout.strip():
            out_path = result.stdout.strip().splitlines()[-1] if "\n" in result.stdout else result.stdout.strip()
            return out_path if os.path.isfile(out_path) else None
        return None
    except Exception:
        return None


def export_svg_to_pptx(project_path: str, svg_output_dir: Optional[str] = None) -> Optional[str]:
    """Export SVG figures to PPTX via ppt-master's svg_to_pptx.py."""
    scripts_root = _ppt_master_scripts_root()
    if not scripts_root:
        return None
    script = os.path.join(scripts_root, "svg_to_pptx.py")
    if not os.path.isfile(script):
        return None
    try:
        result = subprocess.run(
            [sys.executable, script, project_path],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [l for l in result.stdout.splitlines() if ".pptx" in l]
            if lines:
                return lines[-1].strip()
        return None
    except Exception:
        return None
