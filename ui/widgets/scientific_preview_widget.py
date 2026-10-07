import json
import re
import os
import urllib.parse
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QTextBrowser,
    QFrame, QSplitter, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QMessageBox, QProgressBar, QDialog, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QTextCursor, QTextCharFormat

from src.utils.logger import get_logger
from src.services.ollama_local_service import OllamaLocalService
from src.ai.agent_skills.template_fitting import TemplateFitter
from src.utils.local_template_manager import LocalTemplateManager
from src.services.database_service import DatabaseService
from ui.widgets.refine_fragment_dialog import RefineFragmentDialog

class LLMWorker(QThread):
    """Async worker for LLM tasks"""
    finished = pyqtSignal(str) # result text
    error = pyqtSignal(str)
    
    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        try:
            import time
            start_time = time.time()
            result = self.task_func(*self.args, **self.kwargs)
            end_time = time.time()
            self.duration = end_time - start_time
            
            # If result is dict (like TemplateFitter), serialize or handle appropriately. 
            # For simplicity, we expect task_func to return the primary text or we handle it here.
            # But humanize returns str, TemplateFitter returns dict. 
            # Let's emit the raw result (which might be dict) but signal expects str? 
            # We'll use a generic object signal or just wrap in class member.
            self.result = result 
            self.finished.emit("Done") 
        except Exception as e:
            self.error.emit(str(e))

class ScientificPreviewWidget(QWidget):
    """
    Step 4: Scientific Text Preview Generation (P)
    Implements the HITL (Human-in-the-Loop) Collaboration Interface.
    """
    
    def __init__(self, config=None, parent=None, llm_service=None):
        super().__init__(parent)
        self.logger = get_logger(__name__)
        self.config = config
        self.current_result = None
        self.project_id = None
        from src.services.ollama_local_service import OllamaLocalService
        self.llm_service = llm_service if llm_service is not None else OllamaLocalService()
        self.model = "llama3"
        self.service_label = "Ollama"
        self._format_preview_enabled = False
        self._format_preview_raw_cache = ""
        self._logic_audit_state = {"conflicts": [], "resolved": set()}
        self._citation_index = {"S": {}, "C": {}, "R": []}
        self._last_citation_key = ""
        self._last_markdown_source = ""
        self.auto_approve_mode = False
        
        self.init_ui()

    # Signal to notify MainWindow that preview is done
    preview_completed = pyqtSignal(str, dict) # text, metadata

    _INLINE_FORMAT_HINT_PREFIX = "【格式阻断】"
    _MISSING_DATA_PATTERN = r"\[MISSING_DATA:[^\]]+\]"
    _DOUBLE_BRACE_PLACEHOLDER_PATTERN = r"\{\{[^}\n]{1,200}\}\}"
    _TEXT_BRACKET_PLACEHOLDER_PATTERN = r"\[(?!\d+(?:\s*,\s*\d+)*\s*\])[A-Za-z_][^\]\n]{1,80}\]"

    def set_auto_approve_mode(self, enabled: bool):
        self.auto_approve_mode = bool(enabled)

    def set_llm_service(self, svc, service_label: str = None, model: str = None):
        self.llm_service = svc
        if service_label:
            self.service_label = str(service_label)
        if model:
            self.model = str(model)

    def init_ui(self):
        try:
            from src.utils.theme_manager import ThemeManager
            is_dark = ThemeManager.get_current_theme() == "dark"
        except Exception:
            is_dark = False

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # --- Panel 1: Insight & Traceability (25%) ---
        self.panel1 = QFrame()
        self.panel1.setStyleSheet(
            f"background-color: {'#333333' if is_dark else '#F8F9FA'}; border-right: 1px solid {'#555555' if is_dark else '#E0E0E0'};"
        )
        p1_layout = QVBoxLayout(self.panel1)
        p1_layout.setContentsMargins(10, 10, 10, 10)
        
        p1_layout.addWidget(QLabel("<b>Review Critique & Issues</b>"))
        self.issue_tree = QTreeWidget()
        self.issue_tree.setHeaderHidden(True)
        self.issue_tree.itemClicked.connect(self.on_issue_clicked)
        p1_layout.addWidget(self.issue_tree)
        
        p1_layout.addWidget(QLabel("<b>Evidence & CoT</b>"))
        self.evidence_view = QTextEdit()
        self.evidence_view.setReadOnly(True)
        self.evidence_view.setPlaceholderText("Select an issue to view context...")
        p1_layout.addWidget(self.evidence_view)
        
        # --- Panel 2: Edit & Diff (50%) ---
        self.panel2 = QFrame()
        try:
            from src.utils.theme_manager import ThemeManager
            panel2_bg = "#3A3A3A" if ThemeManager.get_current_theme() == "dark" else "#FFFFFF"
        except Exception:
            panel2_bg = "#FFFFFF"
        self.panel2.setStyleSheet(f"background-color: {panel2_bg};")
        p2_layout = QVBoxLayout(self.panel2)
        p2_layout.setContentsMargins(20, 20, 20, 20)
        
        self.editor_label = QLabel("<b>Scientific Draft Editor</b>")
        self.editor_label.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
        p2_layout.addWidget(self.editor_label)
        
        self.editor = QTextEdit()
        self.editor.setStyleSheet("font-family: 'Consolas', 'Courier New'; font-size: 14px; line-height: 1.5;")
        self.editor.setAcceptRichText(True)
        self.editor.cursorPositionChanged.connect(self._on_editor_cursor_moved)
        self.editor.selectionChanged.connect(self._on_editor_selection_changed)
        # Enable CGR Context Menu
        self.editor.setContextMenuPolicy(Qt.CustomContextMenu)
        self.editor.customContextMenuRequested.connect(self.show_context_menu)
        p2_layout.addWidget(self.editor)

        # Single-view mode: do not show a second "raw/rendered" panel.
        self.render_label = QLabel()
        self.render_label.setVisible(False)
        self.render_preview = QTextBrowser()
        self.render_preview.setVisible(False)
        
        # --- Panel 3: LLM Interaction (25%) ---
        self.panel3 = QFrame()
        self.panel3.setStyleSheet(
            f"background-color: {'#333333' if is_dark else '#F8F9FA'}; border-left: 1px solid {'#555555' if is_dark else '#E0E0E0'};"
        )
        p3_layout = QVBoxLayout(self.panel3)
        p3_layout.setContentsMargins(10, 10, 10, 10)
        
        p3_layout.addWidget(QLabel("<b>Refinement Instructions</b>"))
        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("Enter refinement instructions (e.g., translate to Chinese, rewrite more formal, shorten, restructure sections)...")
        self.chat_input.setMaximumHeight(100)
        p3_layout.addWidget(self.chat_input)
        
        self.btn_rethink = QPushButton("Rethink Draft (Loop)")
        self.btn_rethink.setStyleSheet(
            f"background-color: {'#4A4A4A' if is_dark else '#2E86AB'}; color: {'#F5D0FE' if is_dark else 'white'}; padding: 8px;"
        )
        self.btn_rethink.clicked.connect(self.on_rethink_clicked)
        p3_layout.addWidget(self.btn_rethink)
        
        self.btn_template = QPushButton("TemplateFit Wizard")
        self.btn_template.setStyleSheet(
            f"background-color: {'#4A4A4A' if is_dark else '#FF9800'}; color: {'#F5D0FE' if is_dark else 'white'}; padding: 8px;"
        )
        self.btn_template.clicked.connect(self.on_template_clicked)
        self.btn_template.setVisible(False)
        
        self.btn_validate = QPushButton("Validate All (CocoIndex)")
        self.btn_validate.setStyleSheet(
            f"background-color: {'#4A4A4A' if is_dark else '#4CAF50'}; color: {'#F5D0FE' if is_dark else 'white'}; padding: 8px;"
        )
        self.btn_validate.clicked.connect(self.on_validate_clicked)
        p3_layout.addWidget(self.btn_validate)

        self.chk_format_preview = QCheckBox("格式化预览（引用美化/只读）")
        self.chk_format_preview.stateChanged.connect(self.on_format_preview_toggled)
        p3_layout.addWidget(self.chk_format_preview)
        
        p3_layout.addStretch()
        
        # --- Complete & Proceed Button ---
        self.btn_finish = QPushButton("Confirm & Proceed to Step 5")
        self.btn_finish.setStyleSheet("""
            QPushButton {
                background-color: %s; 
                color: %s; 
                padding: 12px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: %s;
            }
        """ % (("#5A5A5A" if is_dark else "#009688"), ("#F5D0FE" if is_dark else "white"), ("#6A6A6A" if is_dark else "#00796B")))
        self.btn_finish.clicked.connect(self.on_finish_clicked)
        p3_layout.addWidget(self.btn_finish)
        
        self.status_bar = QProgressBar()
        self.status_bar.setVisible(False)
        p3_layout.addWidget(self.status_bar)
        self.status_label = QLabel("Ready")
        p3_layout.addWidget(self.status_label)

        splitter.addWidget(self.panel1)
        splitter.addWidget(self.panel2)
        splitter.addWidget(self.panel3)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        
        main_layout.addWidget(splitter)

    def _strip_inline_format_hints(self, text: str) -> str:
        if not text:
            return ""
        pattern = rf"^\s*{re.escape(self._INLINE_FORMAT_HINT_PREFIX)}.*(?:\r?\n)?"
        return re.sub(pattern, "", text, flags=re.MULTILINE)

    def _inject_inline_format_hints(self, text: str, blocks: list) -> str:
        clean_text = self._strip_inline_format_hints(text or "")
        if not blocks:
            return clean_text

        result_text = clean_text
        unresolved_lines = []
        seen = set()

        for b in blocks:
            msg = str(getattr(b, "message", "") or "").strip()
            evd = str(getattr(b, "evidence", "") or "").strip()
            if not msg:
                continue
            unique_key = f"{msg}|{evd}"
            if unique_key in seen:
                continue
            seen.add(unique_key)

            inline_hint = f"\n{self._INLINE_FORMAT_HINT_PREFIX} {msg}" + (f"（{evd}）" if evd else "")

            if evd and evd in result_text:
                # Insert immediately after the first matched evidence occurrence.
                result_text = result_text.replace(evd, f"{evd}{inline_hint}", 1)
            else:
                unresolved_lines.append(f"{self._INLINE_FORMAT_HINT_PREFIX} {msg}" + (f"（{evd}）" if evd else ""))

        if unresolved_lines:
            result_text = result_text.rstrip() + "\n\n" + "\n".join(unresolved_lines) + "\n"
        return result_text

    def _set_editor_with_inline_hints_highlight(self, text_with_hints: str):
        self._set_editor_text(text_with_hints or "")
        self._rehighlight_quality_markers()

    def _set_editor_text(self, text: str):
        t = str(text or "")
        self._last_markdown_source = t
        try:
            # Convert markdown image syntax to HTML img so Qt can render local file:// images.
            # Bypasses setMarkdown's imperfect image rendering.
            import re
            def _md_images_to_html(md_text):
                def _replace(md):
                    src = (md.group(1) or "").strip()
                    if not src:
                        return ""
                    if src.lower().startswith(("http://", "https://", "file://")):
                        return f'<img src="{src}" style="max-width:95%;"/>'
                    return f'<img src="file:///{src}" style="max-width:95%;"/>'
                return re.sub(r"!\[[^\]]*\]\(([^)]+)\)", _replace, md_text)
            
            html_content = _md_images_to_html(t)
            
            # Use mistune or markdown to convert the rest of the text if possible,
            # otherwise just set the HTML (Qt will parse basic HTML tags)
            try:
                import markdown
                html_content = markdown.markdown(html_content, extensions=['tables', 'fenced_code'])
            except ImportError:
                pass
                
            self.editor.setHtml(html_content)
            if t.strip() and not self.editor.toPlainText().strip():
                self.editor.setPlainText(t)
        except Exception:
            self.editor.setPlainText(t)

    def _get_editor_text_for_export(self) -> str:
        # Export markdown so image markers are preserved into Step5.
        source_md = str(self._last_markdown_source or "")
        source_images = re.findall(r"!\[[^\]]*\]\([^)]+\)", source_md)
        if source_images:
            # Prefer original markdown source when images exist, to avoid QTextEdit markdown round-trip mutation.
            return source_md
        try:
            md = self.editor.toMarkdown() or ""
            if str(md).strip():
                md_text = str(md)
                md_images = re.findall(r"!\[[^\]]*\]\([^)]+\)", md_text)
                # QTextEdit markdown round-trip may drop some image nodes; keep full source if needed.
                if source_images and len(md_images) < len(source_images):
                    return source_md
                return md_text
        except Exception:
            pass
        if source_md.strip():
            return source_md
        return self.editor.toPlainText() or ""

    def _rehighlight_quality_markers(self):
        doc_text = self.editor.toPlainText()
        if not doc_text:
            return

        # Highlight inline format-block hints
        fmt = QTextCharFormat()
        fmt.setBackground(QColor("#FFE08A"))
        fmt.setForeground(QColor("#7A1C00"))
        for m in re.finditer(rf"{re.escape(self._INLINE_FORMAT_HINT_PREFIX)}[^\n\r]*", doc_text):
            cursor = self.editor.textCursor()
            cursor.setPosition(m.start())
            cursor.setPosition(m.end(), QTextCursor.KeepAnchor)
            cursor.mergeCharFormat(fmt)

        # Highlight all missing-data placeholders
        self._highlight_missing_data_placeholders()

    def _highlight_missing_data_placeholders(self):
        doc_text = self.editor.toPlainText() or ""
        if not doc_text:
            return
        # 1) Missing data placeholders
        fmt_missing = QTextCharFormat()
        fmt_missing.setBackground(QColor("#FFD5D5"))
        fmt_missing.setForeground(QColor("#7A0019"))
        for m in re.finditer(self._MISSING_DATA_PATTERN, doc_text):
            cursor = self.editor.textCursor()
            cursor.setPosition(m.start())
            cursor.setPosition(m.end(), QTextCursor.KeepAnchor)
            cursor.mergeCharFormat(fmt_missing)

        # 2) Double-brace placeholders like {{S12}}
        fmt_brace = QTextCharFormat()
        fmt_brace.setBackground(QColor("#FFE3C2"))
        fmt_brace.setForeground(QColor("#8A2D00"))
        for m in re.finditer(self._DOUBLE_BRACE_PLACEHOLDER_PATTERN, doc_text):
            cursor = self.editor.textCursor()
            cursor.setPosition(m.start())
            cursor.setPosition(m.end(), QTextCursor.KeepAnchor)
            cursor.mergeCharFormat(fmt_brace)

        # 3) Textual bracket placeholders like [New Citation Needed]
        fmt_text = QTextCharFormat()
        fmt_text.setBackground(QColor("#FFF3BF"))
        fmt_text.setForeground(QColor("#7A5A00"))
        for m in re.finditer(self._TEXT_BRACKET_PLACEHOLDER_PATTERN, doc_text):
            token = m.group(0)
            # Skip current standardized missing-data placeholder (already highlighted).
            if re.match(self._MISSING_DATA_PATTERN, token):
                continue
            cursor = self.editor.textCursor()
            cursor.setPosition(m.start())
            cursor.setPosition(m.end(), QTextCursor.KeepAnchor)
            cursor.mergeCharFormat(fmt_text)

    def _dedupe_section_text(self, text: str, section_name: str) -> str:
        t = str(text or "").strip()
        if not t:
            return ""
        # Remove repeated heading lines inside section body.
        pat = rf"(?im)^\s{{0,3}}(?:#{{1,6}}\s*)?{re.escape(section_name)}\b[\s:：-]*$"
        t = re.sub(pat, "", t).strip()
        # Paragraph-level dedupe.
        paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
        seen = set()
        out = []
        for p in paras:
            k = re.sub(r"\s+", " ", p).strip().lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
        return "\n\n".join(out).strip()

    def _build_paper_draft_text(self, result_dict: dict) -> str:
        units = (result_dict or {}).get("units", {}) or {}
        outline = (result_dict or {}).get("outline", {}) or {}
        references = (result_dict or {}).get("references", []) or []
        data_sources = (result_dict or {}).get("data_sources", []) or []
        sample_sources = (result_dict or {}).get("sample_sources", []) or []

        lines = []
        title_text = str(outline.get("title", "Draft") or "Draft").strip()
        lines.append("## title")
        lines.append(title_text)
        lines.append("")

        ordered_sections = [
            ("intro", "introduction"),
            ("methods", "methods"),
            ("results", "results"),
            ("discussion", "discussion"),
        ]
        for key, name in ordered_sections:
            body = self._dedupe_section_text((units.get(key) or {}).get("content", ""), name)
            body = self._sanitize_draft_text(body)
            if not body:
                body = f"[MISSING_DATA: {name}]"
            lines.append(f"## {name}")
            lines.append(body)
            lines.append("")

        # Merge/dedupe references.
        ref_seen = set()
        ref_rows = []
        for ref in references:
            if not isinstance(ref, dict):
                continue
            cit = str(ref.get("citation") or "").strip()
            if not cit:
                continue
            k = re.sub(r"\s+", " ", cit).strip().lower()
            if k in ref_seen:
                continue
            ref_seen.add(k)
            ref_rows.append(cit)
        lines.append("## reference")
        if ref_rows:
            for cit in ref_rows:
                lines.append(f"- {cit}")
        else:
            lines.append("- [MISSING_DATA: reference]")
        lines.append("")

        # Merge/dedupe data sources from sample/file lists.
        src_seen = set()
        src_rows = []
        for row in sample_sources:
            if not isinstance(row, dict):
                continue
            n = row.get("n")
            proj = str(row.get("project") or "").strip()
            sample = str(row.get("sample") or "").strip()
            sample_id = str(row.get("sample_id") or "").strip()
            if not n:
                continue
            s = f"C{n}: {proj} / {sample} ({sample_id})"
            k = re.sub(r"\s+", " ", s).strip().lower()
            if k in src_seen:
                continue
            src_seen.add(k)
            src_rows.append(s)
        for row in data_sources:
            if not isinstance(row, dict):
                continue
            n = row.get("n")
            proj = str(row.get("project") or "").strip()
            sample_n = row.get("sample_n")
            file_name = str(row.get("file") or "").strip()
            src_type = str(row.get("type") or "").strip()
            if not n:
                continue
            ctag = f"C{sample_n}" if sample_n else "C?"
            s = f"S{n}: {proj} / {ctag} / {file_name} ({src_type})"
            k = re.sub(r"\s+", " ", s).strip().lower()
            if k in src_seen:
                continue
            src_seen.add(k)
            src_rows.append(s)
        lines.append("## data source")
        if src_rows:
            for s in src_rows:
                lines.append(f"- {s}")
        else:
            lines.append("- [MISSING_DATA: data source]")
        lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _strip_conversational_preamble(self, text: str) -> str:
        t = str(text or "")
        if not t.strip():
            return t
        t = re.sub(
            r"(?im)^\s*(?:okay|ok|sure|great)[,!\.\s-]*here\s+is\s+a\s+draft\s+for\s+the\s+discussion\s+section[^\n]*\n?",
            "",
            t,
        )
        t = re.sub(
            r"(?im)^\s*(?:here\s+is|below\s+is)\s+(?:a\s+)?draft\s+for\s+the\s+discussion\s+section[^\n]*\n?",
            "",
            t,
        )
        return t.strip() + ("\n" if t.strip() else "")

    def _sanitize_draft_text(self, text: str) -> str:
        """Keep rewritten paper content only; strip thinking/prompt/chat residue."""
        t = str(text or "")
        if not t.strip():
            return t

        # Remove explicit analysis/thinking wrapper blocks.
        t = re.sub(r"(?is)<\s*analysis\s*>.*?<\s*/\s*analysis\s*>", "", t)
        t = re.sub(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>", "", t)

        drop_line_patterns = [
            r"^\s*User Instruction\s*:\s*.*$",
            r"^\s*Rewritten Text\s*:\s*.*$",
            r"^\s*Chinese Translation\s*:\s*.*$",
            r"^\s*Task\s*:\s*.*$",
            r"^\s*Rules\s*:\s*.*$",
            r"^\s*Text\s*:\s*.*$",
            r"^\s*You are\s+.*$",
            r"^\s*(?:Let's|Let us)\s+.*$",
            r"^\s*I\s+(?:will|can)\s+.*$",
            r"^\s*(?:Sure|Okay|Ok|Great)[,!\.\s-].*$",
        ]
        for p in drop_line_patterns:
            t = re.sub(rf"(?im){p}", "", t)

        # Remove common markdown code fences that accidentally wrap prompt text.
        t = re.sub(r"(?is)```(?:text|markdown)?\s*(?:User Instruction|Rewritten Text|Task)\s*:.*?```", "", t)
        # Remove assistant-style conversational/refusal paragraphs.
        cleaned_paras = []
        for para in re.split(r"\n\s*\n", t):
            p = str(para or "").strip()
            if not p:
                continue
            pl = p.lower().replace("’", "'")
            chatty_cues = [
                "i'm here to help",
                "to assist you effectively",
                "could you provide the specific text",
                "that way, i can offer you",
                "let me know what you'd like to change or improve",
            ]
            if any(c in pl for c in chatty_cues):
                continue
            cleaned_paras.append(p)
        t = "\n\n".join(cleaned_paras)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip() + ("\n" if t.strip() else "")

    def load_data(self, result_dict):
        """Load data from Step 3 result"""
        self.current_result = result_dict
        self.logger.info("ScientificPreviewWidget: Loading data...")
        
        # Update model if available
        if 'model_name' in result_dict:
            self.model = result_dict['model_name']
        if 'llm_service_name' in result_dict:
            self.service_label = str(result_dict.get('llm_service_name') or self.service_label)
        
        # 1. Populate Editor in final paper format (merged & deduped)
        full_text = self._build_paper_draft_text(result_dict)
        full_text = self._sanitize_draft_text(full_text)
        full_text = self._repair_markdown_image_paths(full_text)
        full_text = self._append_missing_export_figures(full_text)
        
        self._set_editor_text(full_text)
        self._highlight_missing_data_placeholders()
        self._format_preview_enabled = False
        self._format_preview_raw_cache = ""
        self._build_citation_index(result_dict)
        try:
            self.chk_format_preview.blockSignals(True)
            self.chk_format_preview.setChecked(False)
        finally:
            self.chk_format_preview.blockSignals(False)
        self.editor.setReadOnly(False)
        
        units = result_dict.get('units', {}) if isinstance(result_dict, dict) else {}
        self._refresh_issue_tree(units)
        
        # 3. Populate Context
        outline = result_dict.get('outline', {})
        
        # Helper to safely convert list/str to str
        def safe_str(val):
            if isinstance(val, list):
                return ", ".join(str(v) for v in val)
            return str(val) if val is not None else "N/A"

        core_claims = outline.get('core_claims', [])
        target_audience = outline.get('target_audience', 'N/A')

        context_text = f"**Core Claims:**\n{safe_str(core_claims)}\n\n"
        context_text += "**Target Audience:**\n" + safe_str(target_audience)
        
        self.evidence_view.setHtml(context_text)
        
        self.status_label.setText("Draft Loaded.")

    def _refresh_issue_tree(self, units: dict = None):
        self.issue_tree.clear()
        units = units if isinstance(units, dict) else ((self.current_result or {}).get("units", {}) or {})
        order = ["intro", "methods", "results", "discussion"]

        review_root = QTreeWidgetItem(self.issue_tree, ["Review Insights"])
        review_root.setExpanded(True)
        audit_root = QTreeWidgetItem(self.issue_tree, ["Logic Audit"])
        audit_root.setExpanded(True)
        fact_root = QTreeWidgetItem(self.issue_tree, ["Fact Matrix"])
        fact_root.setExpanded(True)

        has_issues = False
        for key in order:
            if key not in units:
                continue
            unit_data = units.get(key) or {}
            critique = unit_data.get("critique")
            if not critique:
                continue
            unit_node = QTreeWidgetItem(review_root, [f"Section: {key.capitalize()} (Score: {critique.get('score', 'N/A')})"])
            unit_node.setExpanded(True)
            issues = critique.get("issues", [])
            if issues:
                has_issues = True
                for issue in issues:
                    QTreeWidgetItem(unit_node, [f"⚠️ {issue}"])
            else:
                QTreeWidgetItem(unit_node, ["✅ No major issues found."])

        if not has_issues:
            QTreeWidgetItem(review_root, ["No critical issues detected by AI Reviewer."])

        self._populate_logic_audit(audit_root, units)
        self._populate_fact_matrix(fact_root, units, self.current_result or {})

    def _sync_current_result_from_text(self, full_text: str):
        text = str(full_text or "").strip()
        if not text:
            return
        if not isinstance(self.current_result, dict):
            self.current_result = {}

        units = dict((self.current_result.get("units") or {}))
        heading_map = [
            ("introduction", "intro"),
            ("methods", "methods"),
            ("results", "results"),
            ("discussion", "discussion"),
        ]
        for heading, key in heading_map:
            body = self._extract_section_body(text, heading)
            if not body.strip():
                continue
            unit = dict(units.get(key) or {})
            unit["content"] = body.strip()
            units[key] = unit
        self.current_result["units"] = units

        title_body = self._extract_section_body(text, "title")
        if title_body.strip():
            outline = dict((self.current_result.get("outline") or {}))
            first_line = next((ln.strip() for ln in title_body.splitlines() if ln.strip()), "")
            if first_line:
                outline["title"] = first_line
                self.current_result["outline"] = outline

        self._refresh_issue_tree(units)

    def _build_citation_index(self, result_dict: dict):
        idx_s = {}
        idx_c = {}
        idx_r = []
        try:
            for row in (result_dict.get("data_sources") or []):
                if not isinstance(row, dict):
                    continue
                n = row.get("n")
                try:
                    nn = int(n)
                except Exception:
                    continue
                idx_s[nn] = row
        except Exception:
            idx_s = {}
        try:
            for row in (result_dict.get("sample_sources") or []):
                if not isinstance(row, dict):
                    continue
                n = row.get("n")
                try:
                    nn = int(n)
                except Exception:
                    continue
                idx_c[nn] = row
        except Exception:
            idx_c = {}
        try:
            refs = result_dict.get("references") or []
            for ref in refs:
                if isinstance(ref, dict):
                    idx_r.append(ref)
        except Exception:
            idx_r = []
        self._citation_index = {"S": idx_s, "C": idx_c, "R": idx_r}

    def _on_editor_selection_changed(self):
        if self._format_preview_enabled:
            return
        cur = self.editor.textCursor()
        if not cur.hasSelection():
            return
        s = (cur.selectedText() or "").replace("\u2029", "\n")
        key = self._extract_citation_key(s)
        if key:
            self._show_citation_key(key)

    def _on_editor_cursor_moved(self):
        if self._format_preview_enabled:
            return
        cur = self.editor.textCursor()
        pos = cur.position()
        text = self.editor.toPlainText()
        key = self._extract_citation_key_from_position(text, pos)
        if key:
            self._show_citation_key(key)

    def _extract_citation_key(self, s: str) -> str:
        if not s:
            return ""
        m = re.search(r"\{\{S(\d+)\}\}", s)
        if m:
            return f"S{m.group(1)}"
        m = re.search(r"\{\{C(\d+)\}\}", s)
        if m:
            return f"C{m.group(1)}"
        m = re.search(r"\bC(\d+)\b", s)
        if m:
            return f"C{m.group(1)}"
        m = re.search(r"\[(\d+)\]", s)
        if m:
            return f"R{m.group(1)}"
        return ""

    def _extract_citation_key_from_position(self, text: str, pos: int) -> str:
        if not text:
            return ""
        p = max(0, min(len(text), int(pos)))
        win = text[max(0, p - 40): min(len(text), p + 40)]
        key = self._extract_citation_key(win)
        return key

    def _show_citation_key(self, key: str):
        if not key or key == self._last_citation_key:
            return
        self._last_citation_key = key
        html = self._render_evidence_html(key)
        if html:
            self.evidence_view.setHtml(html)
            self._highlight_citation_in_editor(key)

    def _sync_render_preview(self):
        # Kept for compatibility; single-view mode no longer uses secondary preview.
        return

    def _highlight_citation_in_editor(self, key: str):
        raw = self.editor.toPlainText()
        if not raw:
            return
        if key.startswith("S"):
            needle = f"{{{{S{key[1:]}}}}}"
        elif key.startswith("C"):
            needle = f"C{key[1:]}"
        elif key.startswith("R"):
            needle = f"[{key[1:]}]"
        else:
            return

        idx = raw.find(needle)
        if idx < 0:
            return
        cursor = self.editor.textCursor()
        cursor.setPosition(idx)
        cursor.movePosition(QTextCursor.StartOfBlock)
        start = cursor.position()
        cursor.movePosition(QTextCursor.EndOfBlock)
        end = cursor.position()

        fmt = QTextCharFormat()
        try:
            from src.utils.theme_manager import ThemeManager
            mark_bg = "#4A4A4A" if ThemeManager.get_current_theme() == "dark" else "#E3F2FD"
        except Exception:
            mark_bg = "#E3F2FD"
        fmt.setBackground(QColor(mark_bg))
        cur = self.editor.textCursor()
        cur.setPosition(start)
        cur.setPosition(end, QTextCursor.KeepAnchor)
        cur.mergeCharFormat(fmt)

    def _render_evidence_html(self, key: str) -> str:
        try:
            from src.utils.theme_manager import ThemeManager
            is_dark = ThemeManager.get_current_theme() == "dark"
        except Exception:
            is_dark = False
        panel_bg = "#3A3A3A" if is_dark else "#fff"
        panel_border = "#666666" if is_dark else "#E0E0E0"
        panel_text = "#E879F9" if is_dark else "#111111"

        if key.startswith("S"):
            try:
                n = int(key[1:])
            except Exception:
                return ""
            row = (self._citation_index.get("S") or {}).get(n)
            if not isinstance(row, dict):
                return f"<b>{key}</b><br/><i>未找到对应的数据源索引。</i>"
            fpath = self._resolve_evidence_file_path(row)
            parts = [
                f"<h3 style='margin:0'>证据 {key}</h3>",
                f"<div style='border:1px solid {panel_border}; padding:10px; border-radius:8px; background:{panel_bg}; color:{panel_text}'>",
                f"<b>类型</b>: {row.get('type') or ''}<br/>",
                f"<b>项目</b>: {row.get('project') or ''}<br/>",
                f"<b>样本</b>: C{row.get('sample_n') or '?'} / {row.get('sample') or ''} ({row.get('sample_id') or ''})<br/>",
                f"<b>文件</b>: {row.get('file') or ''}<br/>",
            ]
            if fpath:
                parts.append(f"<b>路径</b>: <span style='font-family:Consolas'>{fpath}</span><br/>")
            parts.append("</div>")
            return "".join(parts)

        if key.startswith("C"):
            try:
                n = int(key[1:])
            except Exception:
                return ""
            row = (self._citation_index.get("C") or {}).get(n)
            if not isinstance(row, dict):
                return f"<b>{key}</b><br/><i>未找到对应的样本索引。</i>"
            return (
                f"<h3 style='margin:0'>样本 {key}</h3>"
                f"<div style='border:1px solid {panel_border}; padding:10px; border-radius:8px; background:{panel_bg}; color:{panel_text}'>"
                f"<b>项目</b>: {row.get('project') or ''}<br/>"
                f"<b>样本</b>: {row.get('sample') or ''}<br/>"
                f"<b>ID</b>: {row.get('sample_id') or ''}<br/>"
                "</div>"
            )

        if key.startswith("R"):
            try:
                n = int(key[1:])
            except Exception:
                return ""
            refs = self._citation_index.get("R") or []
            if n <= 0 or n > len(refs):
                return f"<b>[{n}]</b><br/><i>未找到对应的 PubMed 引用条目。</i>"
            ref = refs[n - 1]
            cit = (ref.get("citation") or "").strip() if isinstance(ref, dict) else str(ref)
            extra = []
            if isinstance(ref, dict):
                for k in ("pmid", "doi", "url"):
                    v = (ref.get(k) or "").strip()
                    if v:
                        extra.append(f"<b>{k.upper()}</b>: {v}<br/>")
            return (
                f"<h3 style='margin:0'>文献引用 [{n}]</h3>"
                f"<div style='border:1px solid {panel_border}; padding:10px; border-radius:8px; background:{panel_bg}; color:{panel_text}'>"
                f"{cit}<br/>"
                + "".join(extra)
                + "</div>"
            )
        return ""

    def _resolve_evidence_file_path(self, row: dict) -> str:
        if not self.project_id:
            return ""
        pid = str(self.project_id or "").strip()
        sid = str(row.get("sample_id") or "").strip()
        fn = str(row.get("file") or "").strip()
        st = str(row.get("type") or "").strip()
        if not pid or not sid or not fn:
            return ""
        db = DatabaseService()
        try:
            with db.get_connection() as conn:
                cur = conn.cursor()
                if st == "ocr_image":
                    cur.execute(
                        "SELECT image_path FROM ocr_sessions WHERE project_id=? AND user_id=? AND (image_name=? OR image_path LIKE ?) ORDER BY created_at DESC LIMIT 1",
                        (pid, sid, fn, f"%{fn}%"),
                    )
                    r = cur.fetchone()
                    return str(r[0]) if r and r[0] else ""
                if st == "dicom":
                    cur.execute(
                        "SELECT file_path FROM dicom_sessions WHERE project_id=? AND user_id=? AND file_name=? ORDER BY created_at DESC LIMIT 1",
                        (pid, sid, fn),
                    )
                    r = cur.fetchone()
                    return str(r[0]) if r and r[0] else ""
                if st == "document":
                    cur.execute(
                        "SELECT d.file_path FROM documents d JOIN text_sessions s ON s.session_id=d.session_id WHERE s.project_id=? AND s.user_id=? AND d.file_name=? ORDER BY d.created_at DESC LIMIT 1",
                        (pid, sid, fn),
                    )
                    r = cur.fetchone()
                    return str(r[0]) if r and r[0] else ""
        except Exception:
            return ""
        return ""

    def set_current_project(self, project_id):
        self.project_id = project_id

    def show_context_menu(self, pos):
        """Show context menu with CGR option"""
        menu = self.editor.createStandardContextMenu()

        cursor = self.editor.textCursor()
        text = self.editor.toPlainText()
        key = ""
        try:
            key = self._extract_citation_key_from_position(text, cursor.position())
        except Exception:
            key = ""
        if key:
            menu.addSeparator()
            act = menu.addAction("🔎 显示引用证据（光标处）")
            act.triggered.connect(lambda: self._show_citation_key(key))
            if key.startswith("S"):
                try:
                    n = int(key[1:])
                except Exception:
                    n = None
                row = (self._citation_index.get("S") or {}).get(n) if n else None
                fpath = self._resolve_evidence_file_path(row) if isinstance(row, dict) else ""
                if fpath:
                    act2 = menu.addAction("📋 复制证据路径")
                    act2.triggered.connect(lambda: self._copy_to_clipboard(fpath))
        
        if cursor.hasSelection():
            menu.addSeparator()
            action = menu.addAction("✨ Refine with Data (CGR)...")
            action.triggered.connect(self.on_refine_requested)
            
        menu.exec_(self.editor.mapToGlobal(pos))

    def _copy_to_clipboard(self, s: str):
        try:
            from PyQt5.QtWidgets import QApplication
            cb = QApplication.clipboard()
            cb.setText(str(s or ""))
        except Exception:
            pass
        
    def on_refine_requested(self):
        """Handle CGR request"""
        if not self.project_id:
            QMessageBox.warning(self, "Error", "Project ID not set. Please reload data.")
            return
            
        cursor = self.editor.textCursor()
        if not cursor.hasSelection():
            return
            
        selected_text = cursor.selectedText()
        # Clean selected text (remove unicode paragraph separators often in Qt selection)
        selected_text = selected_text.replace('\u2029', '\n')
        
        # Get context (simple slice)
        pos_start = cursor.selectionStart()
        pos_end = cursor.selectionEnd()
        full_text = self.editor.toPlainText()
        
        pre_context = full_text[:pos_start]
        post_context = full_text[pos_end:]
        
        dialog = RefineFragmentDialog(
            self.project_id,
            selected_text,
            pre_context,
            post_context,
            self,
            llm_service=self.llm_service,
            model=self.model,
        )
        
        if dialog.exec_() == QDialog.Accepted:
            if dialog.refined_text:
                # Replace text
                cursor.insertText(dialog.refined_text)
                self._sync_current_result_from_text(self._get_editor_text_for_export())
                self._rehighlight_quality_markers()
                self.status_label.setText("Refinement applied.")

    def show_no_data_state(self):
        """Show warning state when no data is loaded"""
        self._set_editor_text("⚠️ 第三步（数据科学总结）尚未完成。\n\n请先在 Step 3 运行分析，或者在此处手动粘贴草稿内容。")
        self.evidence_view.setHtml("<i>No context available.</i>")
        self.issue_tree.clear()
        self.status_label.setText("Waiting for analysis result...")

    def on_issue_clicked(self, item, column):
        payload = item.data(0, Qt.UserRole)
        if not isinstance(payload, dict):
            return
        if payload.get("kind") == "logic_conflict":
            self._resolve_logic_conflict(payload)
            return
        if payload.get("kind") == "fact_conflict":
            self._resolve_fact_conflict(payload)
            return

    def on_format_preview_toggled(self, state):
        enabled = state == Qt.Checked
        if enabled:
            raw = self.editor.toPlainText()
            if not raw.strip():
                try:
                    self.chk_format_preview.blockSignals(True)
                    self.chk_format_preview.setChecked(False)
                finally:
                    self.chk_format_preview.blockSignals(False)
                return
            self._format_preview_raw_cache = raw
            formatted = self._apply_citation_beautifier(raw)
            self._format_preview_enabled = True
            self.editor.setMarkdown(formatted)
            self._rehighlight_quality_markers()
            self.editor.setReadOnly(True)
            self.status_label.setText("格式化预览已开启（只读）。")
        else:
            if self._format_preview_enabled and self._format_preview_raw_cache:
                self._set_editor_text(self._format_preview_raw_cache)
                self._rehighlight_quality_markers()
            self._format_preview_enabled = False
            self._format_preview_raw_cache = ""
            self.editor.setReadOnly(False)
            self.status_label.setText("格式化预览已关闭。")

    def _populate_logic_audit(self, audit_root: QTreeWidgetItem, units: dict):
        self._logic_audit_state = {"conflicts": [], "resolved": set()}
        methods = (units.get("methods") or {}).get("content", "") or ""
        results = (units.get("results") or {}).get("content", "") or ""
        conflicts = self._detect_methods_results_conflicts(methods, results)
        self._logic_audit_state["conflicts"] = conflicts

        if not conflicts:
            QTreeWidgetItem(audit_root, ["✅ No conflicts detected between Methods and Results."])
            return
        for c in conflicts:
            title = c.get("title") or "Methods/Results mismatch"
            node = QTreeWidgetItem(audit_root, [f"⚠️ {title}"])
            node.setData(0, Qt.UserRole, {"kind": "logic_conflict", **c})

    def _populate_fact_matrix(self, fact_root: QTreeWidgetItem, units: dict, result_dict: dict):
        hv = result_dict.get("hard_facts_vector") if isinstance(result_dict, dict) else None
        if not isinstance(hv, dict):
            QTreeWidgetItem(fact_root, ["(No Hard_Facts_Vector available)"])
            return

        methods = (units.get("methods") or {}).get("content", "") or ""
        results = (units.get("results") or {}).get("content", "") or ""

        def _contains_any(text: str, terms: list) -> bool:
            t = (text or "").lower()
            for x in terms or []:
                sx = str(x or "").strip()
                if not sx:
                    continue
                if sx.lower() in t:
                    return True
            return False

        terms = []
        try:
            mts = hv.get("materials_or_core_terms") or []
            if isinstance(mts, list):
                terms.extend([str(x) for x in mts if str(x).strip()])
        except Exception:
            pass
        try:
            ss = str(hv.get("sample_size") or "").strip()
            if ss:
                terms.append(ss)
        except Exception:
            pass
        try:
            cms = hv.get("core_metrics") or []
            if isinstance(cms, list):
                terms.extend([str(x) for x in cms if str(x).strip()])
        except Exception:
            pass
        terms = terms[:12]

        if not terms:
            QTreeWidgetItem(fact_root, ["(Hard facts extracted but empty)"])
            return

        m_ok = _contains_any(methods, terms)
        r_ok = _contains_any(results, terms)
        status = "✅" if (m_ok and r_ok) else "⚠️"
        QTreeWidgetItem(fact_root, [f"{status} Hard facts coverage (Methods={m_ok}, Results={r_ok})"])

        if not m_ok:
            node = QTreeWidgetItem(fact_root, [f"⚠️ Methods missing hard facts terms"])
            node.setData(0, Qt.UserRole, {"kind": "fact_conflict", "target_heading": "Methods", "truth_terms": terms})
        if not r_ok:
            node = QTreeWidgetItem(fact_root, [f"⚠️ Results missing hard facts terms"])
            node.setData(0, Qt.UserRole, {"kind": "fact_conflict", "target_heading": "Results", "truth_terms": terms})

        show = ", ".join(terms[:10])
        QTreeWidgetItem(fact_root, [f"Terms: {show}"])

    def _detect_methods_results_conflicts(self, methods_text: str, results_text: str) -> list:
        m = methods_text or ""
        r = results_text or ""
        if not m.strip() or not r.strip():
            return []

        def _tokens(s: str) -> set:
            words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", s)
            stop = {
                "with", "from", "that", "this", "were", "was", "and", "the", "for", "into", "using", "used",
                "were", "are", "is", "as", "in", "on", "at", "by", "to", "of",
                "methods", "results", "study", "analysis", "patient", "patients",
            }
            out = set()
            for w in words:
                lw = w.lower()
                if lw in stop:
                    continue
                if len(lw) < 3:
                    continue
                out.add(w)
            return out

        mt = _tokens(m)
        rt = _tokens(r)
        if not mt or not rt:
            return []

        common = len(mt & rt)
        denom = max(1, len(mt | rt))
        jacc = common / denom

        conflicts = []
        if jacc < 0.08:
            miss_in_r = sorted(list(mt - rt))[:8]
            miss_in_m = sorted(list(rt - mt))[:8]
            conflicts.append(
                {
                    "id": "methods_results_low_overlap",
                    "title": f"Methods/Results 关键词重叠过低（{jacc:.2f}）",
                    "methods_terms": miss_in_r,
                    "results_terms": miss_in_m,
                }
            )

        def _material_like(s: str) -> set:
            return set(re.findall(r"\b[A-Z][a-z]?[A-Z][a-z]?\b", s))

        mm = _material_like(m)
        rm = _material_like(r)
        if mm and rm and mm != rm:
            only_m = sorted(list(mm - rm))[:6]
            only_r = sorted(list(rm - mm))[:6]
            if only_m or only_r:
                conflicts.append(
                    {
                        "id": "material_mismatch",
                        "title": f"材料/缩写不一致（Methods: {', '.join(only_m) or '-'}; Results: {', '.join(only_r) or '-'}）",
                        "methods_terms": only_m,
                        "results_terms": only_r,
                    }
                )

        return conflicts

    def _resolve_logic_conflict(self, payload: dict):
        cid = payload.get("id") or ""
        if cid and cid in (self._logic_audit_state.get("resolved") or set()):
            QMessageBox.information(self, "已解决", "该冲突已处理。")
            return

        box = QMessageBox(self)
        box.setWindowTitle("Logic Audit 冲突处理")
        box.setIcon(QMessageBox.Warning)
        box.setText(payload.get("title") or "检测到 Methods 与 Results 冲突。请选择处理方式：")
        details = []
        if payload.get("methods_terms"):
            details.append("Methods 特征词: " + ", ".join(payload.get("methods_terms") or []))
        if payload.get("results_terms"):
            details.append("Results 特征词: " + ", ".join(payload.get("results_terms") or []))
        if details:
            box.setInformativeText("\n".join(details))
        btn_m = box.addButton("以 Methods 为准重写 Results", QMessageBox.AcceptRole)
        btn_r = box.addButton("以 Results 为准重写 Methods", QMessageBox.AcceptRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.exec_()

        clicked = box.clickedButton()
        if clicked not in [btn_m, btn_r]:
            return

        if clicked == btn_m:
            truth_terms = payload.get("methods_terms") or []
            self._rewrite_section_with_truth(target_heading="Results", truth_terms=truth_terms, conflict_id=cid)
        else:
            truth_terms = payload.get("results_terms") or []
            self._rewrite_section_with_truth(target_heading="Methods", truth_terms=truth_terms, conflict_id=cid)

    def _rewrite_section_with_truth(self, target_heading: str, truth_terms: list, conflict_id: str = ""):
        full = self.editor.toPlainText()
        section = self._extract_section_body(full, target_heading)
        if not section.strip():
            QMessageBox.warning(self, "无法重写", f"未找到章节: {target_heading}")
            return

        truth = ", ".join([t for t in truth_terms if t])[:400]
        prompt = (
            f"Rewrite the '{target_heading}' section to be logically consistent with the paper's established facts.\n"
            f"Hard constraints:\n"
            f"- Preserve all numbers, statistics, and study design.\n"
            f"- Do not invent new results.\n"
            f"- Ensure terminology/material names are consistent.\n"
            f"- Output ONLY the rewritten section body (no headings).\n"
            f"Truth Terms (must align with): {truth}\n\n"
            f"Section Body:\n{section}\n\n"
            f"Rewritten Section Body:\n"
        )

        self.status_bar.setVisible(True)
        self.status_bar.setRange(0, 0)
        self.status_label.setText(f"正在局部重写 {target_heading}...")
        self.set_controls_enabled(False)

        def _task(llm_service: OllamaLocalService, model: str, p: str) -> str:
            res = llm_service.generate(model, p, timeout=None, num_predict=2048, options={"num_ctx": 8192, "temperature": 0.2})
            if isinstance(res, dict) and res.get("success"):
                return (res.get("response") or "").strip()
            return ""

        self.worker = LLMWorker(_task, self.llm_service, self.model, prompt)
        self.worker.finished.connect(lambda: self._on_rewrite_section_done(target_heading, full, conflict_id))
        self.worker.error.connect(self._on_llm_error)
        self.worker.start()

    def _on_rewrite_section_done(self, target_heading: str, original_full_text: str, conflict_id: str):
        try:
            new_body = self.worker.result
            if not isinstance(new_body, str) or not new_body.strip():
                self.status_label.setText("局部重写失败：空响应。")
                return
            updated = self._replace_section_body(original_full_text, target_heading, new_body)
            if not updated.strip():
                self.status_label.setText("局部重写失败：无法拼接回正文。")
                return
            self._set_editor_text(updated)
            self._sync_current_result_from_text(updated)
            if conflict_id:
                self._logic_audit_state.setdefault("resolved", set()).add(conflict_id)
            self.status_label.setText(f"{target_heading} 已更新。")
        finally:
            self._cleanup_worker()

    def _extract_section_body(self, full_text: str, heading: str) -> str:
        t = full_text or ""
        m = re.search(rf"^##\\s+{re.escape(heading)}\\s*$", t, re.MULTILINE | re.IGNORECASE)
        if not m:
            return ""
        start = m.end()
        m2 = re.search(r"^##\\s+.+$", t[start:], re.MULTILINE)
        end = start + (m2.start() if m2 else len(t[start:]))
        body = t[start:end]
        return body.strip("\n")

    def _replace_section_body(self, full_text: str, heading: str, new_body: str) -> str:
        t = full_text or ""
        m = re.search(rf"^##\\s+{re.escape(heading)}\\s*$", t, re.MULTILINE | re.IGNORECASE)
        if not m:
            return ""
        start = m.end()
        m2 = re.search(r"^##\\s+.+$", t[start:], re.MULTILINE)
        end = start + (m2.start() if m2 else len(t[start:]))
        before = t[:start]
        after = t[end:]
        nb = (new_body or "").strip("\n")
        return before + "\n" + nb + "\n" + after.lstrip("\n")

    def _apply_citation_beautifier(self, text: str) -> str:
        t = text or ""
        t = re.sub(r"\{\{\s*S(\d+)\s*\}\}", lambda m: f"[{m.group(1)}]", t, flags=re.IGNORECASE)
        t = re.sub(r"\[\s*Source\s*:\s*project_.*?\]", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\[\s*Source\s*:\s*(S\d+)\s*\]", lambda m: f"[{m.group(1)[1:]}]", t, flags=re.IGNORECASE)
        t = re.sub(r"\(\s*Source\s*:\s*(S\d+)\s*\)", lambda m: f"[{m.group(1)[1:]}]", t, flags=re.IGNORECASE)
        t = re.sub(r"\bSource\s*:\s*S(\d+)\b", lambda m: f"[{m.group(1)}]", t, flags=re.IGNORECASE)
        t = re.sub(r"\[\s*Source\s*:\s*([^\]]+)\]", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\bsource\s*:\s*project_[^\s\n]+\b", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\bsource\s*:\s*S\d+\b", "", t, flags=re.IGNORECASE)
        t = re.sub(r"[ \\t]{2,}", " ", t)
        t = re.sub(r"\\n{3,}", "\\n\\n", t)
        return t.strip()

    def _repair_markdown_image_paths(self, text: str) -> str:
        t = str(text or "")
        if not t:
            return t

        pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

        def _to_fs_path(url: str) -> str:
            u = str(url or "").strip()
            if u.lower().startswith("file:///"):
                p = urllib.parse.urlparse(u).path
                p = urllib.parse.unquote(p)
                if re.match(r"^/[A-Za-z]:/", p):
                    p = p[1:]
                return p.replace("/", "\\")
            return u

        def _to_file_uri(fs: str) -> str:
            try:
                return Path(os.path.abspath(fs)).as_uri()
            except Exception:
                p = os.path.abspath(fs).replace("\\", "/")
                return f"file:///{p}"

        def _try_fix(fs_path: str) -> str:
            fs = os.path.abspath(fs_path)
            if os.path.exists(fs):
                return fs
            d = os.path.dirname(fs)
            b = os.path.basename(fs)
            if not os.path.isdir(d):
                return fs_path

            # Common corruption fix: accidental spaces inside filename.
            b_no_space = b.replace(" ", "")
            cand = os.path.join(d, b_no_space)
            if os.path.exists(cand):
                return cand

            # Fuzzy fallback by normalized filename.
            ext = os.path.splitext(b)[1].lower()
            for fn in os.listdir(d):
                if os.path.splitext(fn)[1].lower() != ext:
                    continue
                if fn.replace(" ", "") == b_no_space:
                    cp = os.path.join(d, fn)
                    if os.path.exists(cp):
                        return cp
            return fs_path

        def _repl(m):
            alt = m.group(1)
            url = m.group(2).strip()
            fs = _to_fs_path(url)
            fixed = _try_fix(fs)
            if fixed != fs and os.path.exists(fixed):
                return f"![{alt}]({_to_file_uri(fixed)})"
            return m.group(0)

        return pattern.sub(_repl, t)

    def _append_missing_export_figures(self, text: str) -> str:
        t = str(text or "")
        if not t or not self.project_id:
            return t
        export_dir = os.path.join("f:\\RSNA\\medical_imaging_workflow\\project_storage\\projects", self.project_id, "exports")
        if not os.path.isdir(export_dir):
            return t

        existing_urls = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", t)
        existing_basenames = set()
        for u in existing_urls:
            try:
                p = urllib.parse.urlparse(str(u)).path
                p = urllib.parse.unquote(p)
                if re.match(r"^/[A-Za-z]:/", p):
                    p = p[1:]
                existing_basenames.add(os.path.basename(p.replace("/", "\\")))
            except Exception:
                continue

        files = []
        for fn in sorted(os.listdir(export_dir)):
            low = fn.lower()
            if low.endswith((".png", ".jpg", ".jpeg", ".webp")) and ("sklearn" in low or "figure" in low):
                files.append(fn)
        if not files:
            return t

        missing = [fn for fn in files if fn not in existing_basenames]
        if not missing:
            return t

        lines = []
        for fn in missing:
            fs = os.path.join(export_dir, fn)
            try:
                uri = Path(fs).as_uri()
            except Exception:
                uri = "file:///" + os.path.abspath(fs).replace("\\", "/")
            lines.append(f"![Figure]({uri})")

        extra = "\n\n### Figure Render Preview (Auto-Completed)\n\n" + "\n\n".join(lines) + "\n"
        return self._insert_into_results_section(t, extra)

    def _insert_into_results_section(self, text: str, extra_block: str) -> str:
        t = str(text or "")
        extra = str(extra_block or "").strip()
        if not t.strip() or not extra:
            return t
        extra = "\n\n" + extra + "\n"
        # Match markdown "Results" heading and inject before next heading.
        m = re.search(r"(?im)^(#{1,6}\s*results?\s*)$", t)
        if not m:
            m2 = re.search(r"(?im)^results?\s*$", t)
            if not m2:
                return t.rstrip() + extra
            start = m2.end()
        else:
            start = m.end()
        tail = t[start:]
        n = re.search(r"(?m)^\s*#{1,6}\s+\S", tail)
        if n:
            pos = start + n.start()
            return t[:pos].rstrip() + extra + t[pos:]
        return t.rstrip() + extra

    def _resolve_fact_conflict(self, payload: dict):
        target = (payload.get("target_heading") or "").strip()
        truth_terms = payload.get("truth_terms") or []
        if not target:
            return
        self._rewrite_section_with_truth(target_heading=target, truth_terms=truth_terms, conflict_id=f"fact_{target.lower()}")

    def on_rethink_clicked(self):
        instruction = self.chat_input.toPlainText().strip()
        if not instruction:
            QMessageBox.warning(self, "Input Required", "Please enter refinement instructions.")
            return

        base_text = self._get_editor_text_for_export()
        if not base_text.strip() and self.current_result and isinstance(self.current_result, dict):
            base_text = self._build_paper_draft_text(self.current_result or {})

        if not base_text.strip():
            QMessageBox.warning(self, "Empty Draft", "No draft content available.")
            return

        self.status_bar.setVisible(True)
        self.status_bar.setRange(0, 0)
        self.status_label.setText(f"Rethinking draft... ({self.service_label})")
        self.set_controls_enabled(False)

        def _is_translate_to_zh(user_instruction: str) -> bool:
            ui = (user_instruction or "").lower()
            return any(k in ui for k in ["翻译", "译为", "中文", "汉语", "chinese"])

        def _cjk_ratio(s: str) -> float:
            if not s:
                return 0.0
            total = len(s)
            cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
            return cjk / max(1, total)

        def _split_by_headings(full: str) -> list:
            t = (full or "").strip()
            if not t:
                return []
            parts = re.split(r"\n(?=#{1,6}\s)", t)
            out = []
            for p in parts:
                p = p.strip()
                if p:
                    out.append(p)
            if not out:
                return [t]
            return out

        def _task_translate_zh(text: str, user_instruction: str, llm_service: OllamaLocalService, model: str) -> str:
            sections = _split_by_headings(text)
            out = []
            for sec in sections:
                prompt = (
                    "You are a professional scientific translator.\n"
                    "Task: translate the given text into Chinese.\n"
                    "Rules:\n"
                    "1) Keep the original structure and all headings exactly (e.g., '#', '##').\n"
                    "2) Do not summarize. Do not omit any content.\n"
                    "3) Preserve numbers, units, citations, and abbreviations.\n"
                    "4) Output ONLY Chinese translation (keep headings as-is if they are in English).\n\n"
                    f"User Instruction:\n{user_instruction}\n\n"
                    f"Text:\n{sec}\n\n"
                    "Chinese Translation:\n"
                )
                res = llm_service.generate(
                    model,
                    prompt,
                    timeout=None,
                    num_predict=4096,
                    options={"num_ctx": 8192, "temperature": 0.1},
                )
                if isinstance(res, dict) and res.get("success"):
                    piece = (res.get("response") or "").strip()
                    if piece:
                        out.append(piece)
            return "\n\n".join(out).strip()

        def _task_generic(text: str, user_instruction: str, llm_service: OllamaLocalService, model: str) -> str:
            prompt = (
                "You are a precise scientific rewriting engine.\n"
                "Your job: apply the User Instruction to the provided Text.\n"
                "Rules:\n"
                "1) Follow the instruction exactly.\n"
                "2) Do not invent new facts or numbers.\n"
                "3) Keep meaning consistent with the original unless the instruction explicitly requests content change.\n"
                "4) Output ONLY the rewritten full text (no explanations).\n\n"
                f"User Instruction:\n{user_instruction}\n\n"
                f"Text:\n{text}\n\n"
                "Rewritten Text:\n"
            )
            res = llm_service.generate(
                model,
                prompt,
                timeout=None,
                num_predict=4096,
                options={"num_ctx": 8192, "temperature": 0.2},
            )
            if isinstance(res, dict) and res.get("success"):
                return (res.get("response") or "").strip()
            return ""

        def _task(text: str, user_instruction: str, llm_service: OllamaLocalService, model: str) -> str:
            if _is_translate_to_zh(user_instruction):
                rewritten = _task_translate_zh(text, user_instruction, llm_service, model)
                if rewritten and _cjk_ratio(rewritten) < 0.2:
                    rewritten = _task_translate_zh(text, "请把全文完整翻译为中文，不要删减任何内容。", llm_service, model)
                return rewritten
            return _task_generic(text, user_instruction, llm_service, model)

        self.worker = LLMWorker(_task, base_text, instruction, self.llm_service, self.model)
        self.worker.finished.connect(self._on_rethink_done)
        self.worker.error.connect(self._on_llm_error)
        self.worker.start()

    def _on_rethink_done(self):
        try:
            new_text = self.worker.result
            duration = getattr(self.worker, "duration", 0)
            if not isinstance(new_text, str) or not new_text.strip():
                self.status_label.setText("Rethink failed: empty response.")
                return
            new_text = self._strip_conversational_preamble(new_text)
            new_text = self._sanitize_draft_text(new_text)
            new_text = self._repair_markdown_image_paths(new_text)
            new_text = self._append_missing_export_figures(new_text)
            self._set_editor_text(new_text)
            self._sync_current_result_from_text(new_text)
            self.status_label.setText(f"Rethink complete. (Time: {duration:.1f}s)")
        finally:
            self._cleanup_worker()
        
    def set_controls_enabled(self, enabled):
        """Enable/Disable all LLM-related controls to prevent conflict"""
        self.btn_template.setEnabled(enabled)
        self.btn_rethink.setEnabled(enabled)
        self.btn_validate.setEnabled(enabled)
        self.editor.setReadOnly((not enabled) or bool(self._format_preview_enabled))

    def _on_llm_error(self, err_msg):
        self.status_label.setText(f"Error: {err_msg}")
        self._cleanup_worker()
        
    def _cleanup_worker(self):
        self.status_bar.setVisible(False)
        # Re-enable all controls
        self.set_controls_enabled(True)

    def on_template_clicked(self):
        """Start TemplateFit Wizard"""
        from PyQt5.QtWidgets import QDialog, QComboBox, QDialogButtonBox, QLineEdit, QHBoxLayout, QFileDialog
        import os
        
        # 1. Dialog for Template Selection
        dialog = QDialog(self)
        dialog.setWindowTitle("TemplateFit Wizard")
        dialog.resize(600, 450)
        layout = QVBoxLayout(dialog)
        
        # Search Bar
        search_layout = QHBoxLayout()
        search_input = QLineEdit()
        search_input.setPlaceholderText("Search templates (Local or Online)...")
        search_btn = QPushButton("Search")
        search_layout.addWidget(search_input)
        search_layout.addWidget(search_btn)
        layout.addLayout(search_layout)
        
        layout.addWidget(QLabel("Select Target Journal Template:"))
        
        # Combo + Browse Button
        combo_layout = QHBoxLayout()
        combo = QComboBox()
        browse_btn = QPushButton("Open from Disk...")
        combo_layout.addWidget(combo, 1) # stretch
        combo_layout.addWidget(browse_btn)
        layout.addLayout(combo_layout)
        
        # Helper to refresh local templates
        def refresh_combo(select_item=None):
            combo.clear()
            local_mgr = LocalTemplateManager()
            local_templates = local_mgr.get_all_templates()
            if local_templates:
                combo.addItems(sorted(local_templates))
                combo.insertSeparator(len(local_templates))
            
            default_templates = ["NEJM", "JAMA", "Nature"]
            combo.addItems(default_templates)
            
            if select_item:
                combo.setCurrentText(select_item)

        refresh_combo()
        
        # Preview Area
        preview = QTextEdit()
        preview.setReadOnly(True)
        preview.setPlaceholderText("Template constraints will appear here...")
        layout.addWidget(preview)
        
        # --- Helper Functions ---
        
        def update_preview():
            template_name = combo.currentText()
            if not template_name: return
            
            fitter = TemplateFitter(None)
            # Try to get constraints. 
            # If it's a temporary imported template, we might need to handle it via LocalTemplateManager if registered,
            # or we need to pass the custom config if we implement dynamic loading.
            # Currently TemplateFitter uses LocalTemplateManager internally.
            
            # If the item is newly added via "Browse", it should be in the local registry (or temporary).
            # For now, we rely on the name matching what's in LocalTemplateManager or hardcoded list.
            constraints = fitter.get_template_constraints(template_name)
            
            preview_text = json.dumps(constraints, indent=2)
            if 'abs_path' in constraints:
                preview_text = f"📍 Local Template Found: {constraints['abs_path']}\n\n" + preview_text
            elif 'temp_import' in constraints:
                 preview_text = f"📂 Imported Template: {constraints.get('path', 'Unknown')}\n\n" + preview_text
                 
            preview.setPlainText(preview_text)

        def do_online_search():
            query = search_input.text().strip()
            if not query: return
            
            search_btn.setEnabled(False)
            search_btn.setText("Searching...")
            try:
                fitter = TemplateFitter(None)
                results = fitter.search_online_templates(query)
                if results:
                    found_count = 0
                    for name in results.keys():
                        if combo.findText(name) == -1:
                            combo.addItem(name)
                            found_count += 1
                    combo.setCurrentText(list(results.keys())[0])
                    QMessageBox.information(dialog, "Search Results", f"Found {len(results)} template(s). Added {found_count} new.")
                else:
                    QMessageBox.information(dialog, "Search Results", "No matching templates found.")
            except Exception as e:
                QMessageBox.warning(dialog, "Error", str(e))
            finally:
                search_btn.setEnabled(True)
                search_btn.setText("Search")
                
        def on_browse_clicked():
            # Allow selecting folder or .json/.cls
            path = QFileDialog.getExistingDirectory(dialog, "Select Template Folder")
            if not path: return
            
            # Simple heuristic detection
            folder_name = os.path.basename(path)
            json_path = os.path.join(path, "registry.json") # or config.json
            cls_files = [f for f in os.listdir(path) if f.endswith(".cls")]
            
            new_config = {}
            template_name = folder_name
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Assume json might be the registry entry format or list of entries
                        # We take the first key or the dict itself if it has 'journal_name'
                        if "journal_name" in data:
                            new_config = data
                            template_name = data["journal_name"]
                        elif len(data) > 0:
                            k = list(data.keys())[0]
                            new_config = data[k]
                            template_name = k
                except Exception as e:
                    QMessageBox.warning(dialog, "Error", f"Failed to parse JSON: {e}")
                    return
            elif cls_files:
                # Minimal config from CLS
                cls_name = cls_files[0]
                template_name = cls_name.replace(".cls", "")
                new_config = {
                    "journal_name": template_name,
                    "path": path, # Absolute path
                    "main_cls": cls_name,
                    "max_word_count": {"main_text": 3000}, # Default
                    "structure": ["Introduction", "Methods", "Results", "Discussion"], # Default
                    "temp_import": True
                }
            else:
                # Generic folder
                new_config = {
                    "journal_name": folder_name,
                    "path": path,
                    "max_word_count": {"main_text": 3000},
                    "structure": ["Introduction", "Methods", "Results", "Discussion"],
                    "temp_import": True
                }
            
            # Register temporarily in LocalTemplateManager
            # Since LocalTemplateManager loads from file, we need to inject it into runtime.
            # But TemplateFitter instantiates a new LocalTemplateManager every time.
            # So we need to Persist this to registry.json OR handle runtime injection.
            
            # Let's write to registry.json to persist it as "Imported"
            try:
                mgr = LocalTemplateManager()
                # Make path relative if possible, or absolute
                try:
                    rel_path = os.path.relpath(path, mgr.root_dir)
                    new_config['path'] = rel_path
                except:
                    new_config['path'] = path # Absolute
                
                # Update registry file
                if not mgr.registry: mgr.registry = {}
                mgr.registry[template_name] = new_config
                
                with open(mgr.registry_path, 'w', encoding='utf-8') as f:
                    json.dump(mgr.registry, f, indent=2)
                
                QMessageBox.information(dialog, "Success", f"Imported template: {template_name}")
                refresh_combo(template_name)
                
            except Exception as e:
                QMessageBox.warning(dialog, "Import Error", str(e))

        # Connections
        combo.currentTextChanged.connect(update_preview)
        search_btn.clicked.connect(do_online_search)
        browse_btn.clicked.connect(on_browse_clicked)
        
        update_preview() # Init
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() != QDialog.Accepted:
            return
            
        # 2. Execute Fitting Loop (Async)
        target_template = combo.currentText()
        text = self.editor.toPlainText()
        if not text: return
        
        self.status_bar.setVisible(True)
        self.status_bar.setRange(0, 0)
        self.status_label.setText(f"Fitting to {target_template}...")
        
        # Disable controls
        self.set_controls_enabled(False)
        
        def run_fit(txt, tmpl, svc, mdl):
            fitter = TemplateFitter(svc, mdl)
            return fitter.fit_draft(txt, tmpl)

        self.worker = LLMWorker(run_fit, text, target_template, self.llm_service, self.model)
        self.worker.finished.connect(self._on_template_done)
        self.worker.error.connect(self._on_llm_error)
        self.worker.start()

    def _on_template_done(self):
        try:
            result = self.worker.result
            duration = getattr(self.worker, 'duration', 0)
            
            # Apply Result
            self._set_editor_text(result["fitted_text"])
            self._sync_current_result_from_text(result["fitted_text"])
            
            # Show Plan in Issues Panel (Mockup)
            root = QTreeWidgetItem(self.issue_tree, ["TemplateFit Plan"])
            root.setExpanded(True)
            QTreeWidgetItem(root, [result["plan"][:50] + "..."])
            
            # Show Validation
            val = result["validation"]
            status_color = "green" if val["status"] == "PASS" else "orange"
            self.status_label.setText(f"Fit Complete. Validation: {val['status']} (Time: {duration:.1f}s)")
            
            if val["issues"]:
                QTreeWidgetItem(root, ["Validation Issues:"])
                for issue in val["issues"]:
                    QTreeWidgetItem(root, [issue])
        finally:
            self._cleanup_worker()

    def _cleanup_worker(self):
        self.status_bar.setVisible(False)
        # Re-enable all controls
        self.set_controls_enabled(True)

    def on_validate_clicked(self):
        """Handle manual validation check via Funnel Loop (Layer 1-3)"""
        from src.ai.funnel_loop import FunnelLoopAnalyzer
        
        text = self.editor.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "Empty Draft", "草稿为空，请先生成内容。")
            return
            
        self.status_label.setText("正在执行三层漏斗(Funnel Loop)质控检查...")
        report = FunnelLoopAnalyzer.analyze_and_fix(text)
        
        current_text = report.fixed_text if report.fixed_text != text else text
        blocks = [i for i in report.issues if i.severity == "BLOCK"]
        if blocks:
            # Keep draft body clean: do not inject block hints into正文内容.
            clean_text = self._strip_inline_format_hints(current_text)
            self._set_editor_text(clean_text)
            self._highlight_missing_data_placeholders()
            self.status_label.setText("检测到格式阻断，请在问题面板查看并手动修正。")
        else:
            clean_text = self._strip_inline_format_hints(current_text)
            if clean_text != self.editor.toPlainText():
                self._set_editor_text(clean_text)
            self._highlight_missing_data_placeholders()
            
        # Show results in Issue Tree
        self.issue_tree.clear()
        root = QTreeWidgetItem(self.issue_tree, ["Funnel Loop 审计报告"])
        root.setExpanded(True)
        
        if not report.issues:
            QTreeWidgetItem(root, ["✅ 未发现重大合规性、逻辑或声称问题。"])
            self.status_label.setText("验证通过 (PASS)")
        else:
            for issue in report.issues:
                # Color code based on severity
                item = QTreeWidgetItem(root, [f"[{issue.layer} - {issue.severity}] {issue.message}"])
                if issue.severity == "BLOCK":
                    item.setForeground(0, QColor("#FF00FF")) # Neon Magenta
                elif issue.severity == "BACKTRACK":
                    item.setForeground(0, QColor("#FF9800")) # Orange
                elif issue.severity == "DOWNGRADE":
                    item.setForeground(0, QColor("#39FF14")) # Neon Green
                
                if issue.evidence:
                    sub_item = QTreeWidgetItem(item, [f"发现问题处: {issue.evidence}"])
                    sub_item.setForeground(0, QColor("#B0B0B0"))
                    
            self.status_label.setText(f"验证完成，共发现 {len(report.issues)} 个问题需人工干预。")

    def on_finish_clicked(self):
        """Handle completion of Step 4 - Strict Funnel Loop Gate"""
        from src.ai.funnel_loop import FunnelLoopAnalyzer
        
        if self._format_preview_enabled and self._format_preview_raw_cache:
            text = str(self._format_preview_raw_cache)
        else:
            text = self._get_editor_text_for_export()
        if not text.strip():
            QMessageBox.warning(self, "Empty Draft", "The draft is empty. Please generate or paste content first.")
            return

        # Funnel Loop Layer 1 & Layer 2 Gate
        report = FunnelLoopAnalyzer.analyze_and_fix(text)
        if report.fixed_text != text:
            text = report.fixed_text

        manual_completion_notes = []
        blocks = [i for i in report.issues if i.severity == "BLOCK"]
        if blocks:
            # Do not inject "【格式阻断】" hints into final draft body.
            text = self._strip_inline_format_hints(text)
            self._set_editor_text(text)
            manual_completion_notes.append("存在格式阻断内容，需人工补充")
        else:
            text = self._strip_inline_format_hints(text)
        text = self._sanitize_draft_text(text)
            
        backtracks = [i for i in report.issues if i.severity == "BACKTRACK"]
        if backtracks:
            msg = "\n".join([f"- {b.message}" for b in backtracks])
            if not self.auto_approve_mode:
                reply = QMessageBox.warning(self, "第二层: 数据一致性回溯", f"检测到数据缺失或矛盾：\n\n{msg}\n\n是否强行进入 Step 5？建议先修改。", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    return
            else:
                self.logger.info("Auto-approve mode: BACKTRACK issues bypassed and continuing to Step 5.")

        downgrades = [i for i in report.issues if i.severity == "DOWNGRADE"]
        if downgrades:
            msg = "\n".join([f"- {d.evidence}: {d.message}" for d in downgrades])
            if not self.auto_approve_mode:
                QMessageBox.information(self, "第三层: 过度声称降级", f"注意，草稿中存在过度声称，建议在 Step 5 审稿阶段修改：\n\n{msg}")
            else:
                self.logger.info(f"Auto-approve mode: DOWNGRADE notices suppressed: {msg}")

        final_text = self._apply_citation_beautifier(text)
        final_text = self._repair_markdown_image_paths(final_text)
        final_text = self._append_missing_export_figures(final_text)
        missing_tokens = re.findall(self._MISSING_DATA_PATTERN, final_text)
        has_inline_block_hints = self._INLINE_FORMAT_HINT_PREFIX in final_text
        requires_manual_completion = bool(missing_tokens) or bool(blocks) or has_inline_block_hints
        if missing_tokens:
            manual_completion_notes.append("存在 MISSING_DATA 占位符，需人工补充")
        if has_inline_block_hints and "存在格式阻断内容，需人工补充" not in manual_completion_notes:
            manual_completion_notes.append("存在格式阻断内容，需人工补充")
            
        # Metadata to pass along
        metadata = {
            "model": self.model,
            "source": "Step 4 Preview",
            "timestamp": "now", # receiver will handle formatting
            # Pass correct title if available from Step 3
            "title": self.current_result.get('outline', {}).get('title', 'Untitled Draft') if self.current_result else "Untitled Draft",
            "requires_manual_completion": requires_manual_completion,
            "manual_completion_reason": "；".join(manual_completion_notes) if manual_completion_notes else "",
            "missing_data_count": len(missing_tokens),
            "missing_data_tokens": missing_tokens,
            "block_issue_count": len(blocks),
            "block_issues": [f"{b.message} ({b.evidence})" for b in blocks],
        }
        
        self.logger.info("Step 4 completed. Emitting signal with draft content.")
        self.preview_completed.emit(final_text, metadata)
