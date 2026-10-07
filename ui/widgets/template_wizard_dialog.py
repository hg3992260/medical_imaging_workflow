import json
import os
import sys
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QRadioButton,
    QButtonGroup,
    QDialogButtonBox,
    QFileDialog,
    QMessageBox,
    QProgressBar,
)

from src.ai.agent_skills.template_fitting import TemplateFitter
from src.services.ollama_local_service import OllamaLocalService
from src.utils.local_template_manager import LocalTemplateManager
from src.utils.logger import get_logger
from src.services.ocr_service import get_ocr_service
from src.utils.model_config_loader import get_model_paths
from src.services.text_service import TextService
from src.ai.pdf_template_parser import parse_pdf_sections_from_text, summarize_pdf_template
from src.ai.pdf_template_layout import analyze_pdf_template_layout_hybrid
from src.ai.section_progression import compute_section_progression


class PdfTemplateImportWorker(QThread):
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)

    def __init__(self, pdf_path: str, max_pages: int = 80, parent=None):
        super().__init__(parent)
        self.pdf_path = pdf_path
        self.max_pages = max_pages

    def run(self):
        try:
            self.status_signal.emit("正在解析 PDF…（先用内置解析器，处理不了再用 DeepSeek-OCR 兜底）")
            self.progress_signal.emit(0)
            ts = TextService()
            r = ts.read_pdf_document(
                self.pdf_path,
                max_pages=self.max_pages,
                use_ocr=False,
                status_callback=lambda msg: self.status_signal.emit(str(msg)),
                deepseek_mode="markdown",
                ocr_engine="deepseek_ocr",
                progress_callback=lambda p: self.progress_signal.emit(int(p)),
            )
            if not r.get("success") or not (r.get("content") or "").strip():
                self.finished_signal.emit({"success": False, "error": r.get("error") or "PDF OCR 失败"})
                return

            self.status_signal.emit("正在解析章节结构…")
            self.progress_signal.emit(85)
            txt = r.get("content") or ""
            sections = parse_pdf_sections_from_text(txt)
            constraints = summarize_pdf_template(sections)

            self.status_signal.emit("正在分析版式布局与章节推进关系…")
            self.progress_signal.emit(92)
            try:
                pages_processed = int((r.get("metadata") or {}).get("processed_pages") or 20)
            except Exception:
                pages_processed = 20
            layout = analyze_pdf_template_layout_hybrid(
                self.pdf_path,
                deepseek_sections=sections,
                max_pages=min(20, pages_processed),
            )
            progression = compute_section_progression(sections)
            constraints["pdf_extracted_sections"] = {k: (sections.get(k) or "")[:2000] for k in ["title", "introduction", "methods", "results", "discussion", "conclusion", "references"]}
            constraints["pdf_layout_a4"] = layout
            constraints["section_progression"] = progression
            constraints["draft_output_sections"] = ["Title", "Introduction", "Methods", "Results", "Discussion", "Conclusion", "References"]
            constraints["pdf_path"] = self.pdf_path
            constraints["pdf_metadata"] = r.get("metadata") or {}

            self.progress_signal.emit(100)
            self.finished_signal.emit({
                "success": True, 
                "constraints": constraints,
                "content": txt,
                "metadata": r.get("metadata") or {},
                "sections": sections
            })
        except Exception as e:
            self.finished_signal.emit({"success": False, "error": str(e)})


class TemplateWizardDialog(QDialog):
    def __init__(self, parent=None, model: str = "llama3", project_id: str = "", lancedb_uri: str = ""):
        super().__init__(parent)
        self.logger = get_logger(__name__)
        self.model = model
        self.project_id = project_id or ""
        self.lancedb_uri = lancedb_uri or ""
        self.llm_service = OllamaLocalService()
        self.fitter = TemplateFitter(self.llm_service, model=self.model)
        self.local_manager = LocalTemplateManager()

        self.selected_template_name = ""
        self.selected_constraints = {}
        self.selected_full_data = {} # To store OCR content and sections
        self.selected_pdf_path = ""
        self.cleanup_scope = "template_pdf"

        self.setWindowTitle("Template Wizard")
        self.resize(780, 520)
        self._build_ui()
        self._refresh_combo()
        self._update_preview()
        self._setup_ocr_status_timer()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("选择模板（本地/线上搜索/导入）："))

        row = QHBoxLayout()
        self.template_combo = QComboBox()
        self.template_combo.currentTextChanged.connect(lambda _: self._update_preview())
        row.addWidget(self.template_combo, 2)

        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("线上搜索关键字（例如 Radiology / IEEE / Lancet）")
        row.addWidget(self.query_edit, 3)

        self.search_btn = QPushButton("线上搜索")
        self.search_btn.clicked.connect(self._do_online_search)
        row.addWidget(self.search_btn, 1)

        self.refresh_btn = QPushButton("刷新本地")
        self.refresh_btn.clicked.connect(self._refresh_combo)
        row.addWidget(self.refresh_btn, 1)

        self.import_btn = QPushButton("从磁盘导入...")
        self.import_btn.clicked.connect(self._import_from_disk)
        row.addWidget(self.import_btn, 1)

        self.pdf_btn = QPushButton("导入PDF模板...")
        self.pdf_btn.clicked.connect(self._import_pdf_template)
        row.addWidget(self.pdf_btn, 1)

        layout.addLayout(row)

        layout.addWidget(QLabel("清理范围（更安全：默认仅清理 PDF 模板向量）："))
        cleanup_row = QHBoxLayout()
        self.rb_cleanup_template_pdf = QRadioButton("仅清理 template_pdf 向量")
        self.rb_cleanup_template_pdf.setChecked(True)
        self.rb_cleanup_all = QRadioButton("清空整个项目 project_context / meta")
        self.rb_cleanup_template_pdf.toggled.connect(lambda _: self._update_preview())
        self.rb_cleanup_all.toggled.connect(lambda _: self._update_preview())
        cleanup_row.addWidget(self.rb_cleanup_template_pdf)
        cleanup_row.addWidget(self.rb_cleanup_all)
        cleanup_row.addStretch()
        layout.addLayout(cleanup_row)

        self.cleanup_group = QButtonGroup(self)
        self.cleanup_group.addButton(self.rb_cleanup_template_pdf)
        self.cleanup_group.addButton(self.rb_cleanup_all)

        ocr_row = QHBoxLayout()
        self.ocr_status_label = QLabel("DeepSeek-OCR状态: 检测中…")
        self.ocr_status_label.setStyleSheet("font-size:11px; color:#111; font-weight:bold;")
        ocr_row.addWidget(self.ocr_status_label, 5)
        self.ocr_check_btn = QPushButton("OCR 自检")
        self.ocr_check_btn.clicked.connect(lambda: self._refresh_ocr_status(show_message=True))
        ocr_row.addWidget(self.ocr_check_btn, 1)
        layout.addLayout(ocr_row)

        self.pdf_progress_label = QLabel("")
        self.pdf_progress_label.setStyleSheet("font-size:11px; color:#333;")
        layout.addWidget(self.pdf_progress_label)
        self.pdf_progress = QProgressBar()
        self.pdf_progress.setRange(0, 100)
        self.pdf_progress.setValue(0)
        self.pdf_progress.setVisible(False)
        layout.addWidget(self.pdf_progress)

        layout.addWidget(QLabel("模板约束预览（将作为 Draft 框架注入 Step3）："))
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        layout.addWidget(self.preview, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _setup_ocr_status_timer(self):
        self._refresh_ocr_status(show_message=False)
        self._ocr_timer = QTimer(self)
        self._ocr_timer.setInterval(800)
        self._ocr_timer.timeout.connect(lambda: self._refresh_ocr_status(show_message=False))
        self._ocr_timer.start()
        QTimer.singleShot(15000, self._ocr_timer.stop)

    def _refresh_ocr_status(self, show_message: bool = False):
        try:
            s = get_ocr_service()
        except Exception as e:
            self.ocr_status_label.setText(f"DeepSeek-OCR状态: 失败（{e}）")
            return

        try:
            s.ensure_deepseek_ocr_adapter()
        except Exception:
            pass

        adapter = getattr(s, "deepseek_adapter", None)
        diag = {}
        try:
            paths = get_model_paths()
            diag["config_deepseek_ocr_path"] = paths.get("deepseek_ocr")
        except Exception as e:
            diag["config_deepseek_ocr_path"] = f"error: {e}"
        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(sys.executable)
            diag["exe_dir"] = base_dir
            diag["meipass_dir"] = getattr(sys, "_MEIPASS", None)
            diag["candidate_internal"] = os.path.join(base_dir, "_internal", "assets", "models", "deepseek_ocr")
            if diag["meipass_dir"]:
                diag["candidate_meipass"] = os.path.join(diag["meipass_dir"], "assets", "models", "deepseek_ocr")
        if adapter is None:
            init_err = getattr(s, "deepseek_ocr_init_error", None) or getattr(s, "deepseek_init_error", None)
            detail = f"原因: {init_err}" if init_err else "原因: DeepSeek-OCR 未加载（启动时应预加载；此处为状态检测）"
            msg = f"DeepSeek-OCR状态: 未启用（{detail}）\n诊断: {json.dumps(diag, ensure_ascii=False)}"
            self.ocr_status_label.setText(msg)
            if show_message:
                QMessageBox.information(self, "OCR 自检", msg)
            return

        loading = bool(getattr(adapter, "loading", False)) if adapter else False
        available = bool(getattr(s, "deepseek_available", False))
        err = getattr(adapter, "load_error", None) if adapter else None

        if available:
            msg = "DeepSeek-OCR状态: 已启用"
        elif loading:
            msg = f"DeepSeek-OCR状态: 加载中…"
        else:
            detail = f"原因: {err}" if err else "原因: 未加载到本地权重"
            msg = f"DeepSeek-OCR状态: 未启用（{detail}）\n诊断: {json.dumps(diag, ensure_ascii=False)}"
        self.ocr_status_label.setText(msg)

        if show_message:
            QMessageBox.information(self, "OCR 自检", msg)

    def _refresh_combo(self):
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        try:
            templates = self.local_manager.get_all_templates() or []
        except Exception:
            templates = []
        if templates:
            for t in templates:
                self.template_combo.addItem(str(t))
        else:
            self.template_combo.addItem("NEJM")
            self.template_combo.addItem("JAMA")
            self.template_combo.addItem("Nature")
        self.template_combo.blockSignals(False)

    def _do_online_search(self):
        q = (self.query_edit.text() or "").strip()
        if not q:
            QMessageBox.information(self, "提示", "请输入线上搜索关键字。")
            return
        try:
            results = self.fitter.search_online_templates(q) or {}
        except Exception as e:
            QMessageBox.warning(self, "搜索失败", str(e))
            return
        if not results:
            QMessageBox.information(self, "无结果", "未找到匹配模板。")
            return
        for name in results.keys():
            if self.template_combo.findText(name) < 0:
                self.template_combo.addItem(name)
        self.template_combo.setCurrentText(list(results.keys())[0])
        self._update_preview()

    def _import_from_disk(self):
        folder = QFileDialog.getExistingDirectory(self, "选择模板目录")
        if not folder:
            return
        try:
            new_config = {
                "path": folder,
                "journal_name": folder.split("/")[-1].split("\\")[-1],
            }
            registry = self.local_manager.registry or {}
            if not isinstance(registry, dict):
                registry = {}
            registry[new_config["journal_name"]] = new_config
            self.local_manager.registry = registry
            try:
                self.local_manager.registry_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.local_manager.registry_path, "w", encoding="utf-8") as f:
                    json.dump(registry, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            self._refresh_combo()
            self.template_combo.setCurrentText(new_config["journal_name"])
            self._update_preview()
            QMessageBox.information(self, "导入成功", f"已导入模板：{new_config['journal_name']}")
        except Exception as e:
            QMessageBox.warning(self, "导入失败", str(e))

    def _update_preview(self):
        name = (self.template_combo.currentText() or "").strip()
        if not name:
            self.preview.setPlainText("")
            self.selected_template_name = ""
            self.selected_constraints = {}
            return
        try:
            if getattr(self, "selected_pdf_path", None) and name == (self.selected_template_name or "") and isinstance(getattr(self, "selected_constraints", None), dict) and self.selected_constraints.get("source") == "pdf_template":
                constraints = self.selected_constraints
            else:
                constraints = self.fitter.get_template_constraints(name) or {}
        except Exception as e:
            constraints = {"error": str(e)}
        self.selected_template_name = name
        self.selected_constraints = constraints
        scope = "template_pdf"
        if getattr(self, "rb_cleanup_all", None) and self.rb_cleanup_all.isChecked():
            scope = "all_project_vectors"
        self.cleanup_scope = scope
        extra = {
            "project_id": self.project_id,
            "lancedb_uri": self.lancedb_uri,
            "cleanup_scope": scope,
            "action_on_confirm": "将更新全局模板；若选择了PDF模板，将先替换历史项目留下的 template_pdf 记录，再把当前模板风格写入 LanceDB(project_context, source_type=template_pdf)。",
        }
        self.preview.setPlainText(
            json.dumps({"template_constraints": constraints, "runtime_plan": extra}, ensure_ascii=False, indent=2)
        )

    def _import_pdf_template(self):
        pdf_path, _ = QFileDialog.getOpenFileName(self, "选择PDF模板文件", "", "PDF Files (*.pdf)")
        if not pdf_path:
            return
        self._refresh_ocr_status(show_message=False)
        if getattr(self, "_pdf_worker", None) and self._pdf_worker.isRunning():
            QMessageBox.information(self, "提示", "正在处理中，请稍候。")
            return

        self.pdf_progress.setVisible(True)
        self.pdf_progress.setValue(0)
        self.pdf_progress_label.setText("准备开始…")
        self.preview.setPlainText("正在解析PDF（使用 DeepSeek-OCR 进行 OCR）...")
        self.pdf_btn.setEnabled(False)
        self.import_btn.setEnabled(False)
        self.search_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)

        self._pdf_worker = PdfTemplateImportWorker(pdf_path=pdf_path, max_pages=80, parent=self)
        self._pdf_worker.progress_signal.connect(self._on_pdf_progress)
        self._pdf_worker.status_signal.connect(self._on_pdf_status)
        self._pdf_worker.finished_signal.connect(lambda payload: self._on_pdf_finished(payload, pdf_path))
        self._pdf_worker.start()

    def _on_pdf_progress(self, value: int):
        try:
            v = max(0, min(100, int(value)))
        except Exception:
            v = 0
        self.pdf_progress.setValue(v)

    def _on_pdf_status(self, message: str):
        text = str(message or "")
        self.pdf_progress_label.setText(text)
        win = self.window()
        if hasattr(win, "on_model_status_message"):
            try:
                win.on_model_status_message(text)
            except Exception:
                pass

    def _on_pdf_finished(self, payload: dict, pdf_path: str):
        self.pdf_btn.setEnabled(True)
        self.import_btn.setEnabled(True)
        self.search_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        
        if not isinstance(payload, dict) or not payload.get("success"):
            self.pdf_progress.setVisible(False)
            self.pdf_progress_label.setText("失败")
            QMessageBox.warning(self, "PDF解析失败", (payload or {}).get("error") or "PDF解析失败")
            return

        self.pdf_progress.setValue(100)
        self.pdf_progress_label.setText("解析完成 (等待应用)")
        
        constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
        self.selected_pdf_path = pdf_path
        self.selected_template_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.selected_constraints = constraints
        self.selected_full_data = payload # Store the whole thing including content/sections
        self.template_combo.blockSignals(True)
        if self.template_combo.findText(self.selected_template_name) < 0:
            self.template_combo.addItem(self.selected_template_name)
        self.template_combo.setCurrentText(self.selected_template_name)
        self.template_combo.blockSignals(False)
        self._update_preview()
        
        QMessageBox.information(self, "解析成功", "PDF 模板解析完成。\n\n请点击 'OK' 以应用此模板并写入数据库（这将花费一些时间，请留意主界面控制台日志）。")

    def _accept(self):
        name = (self.selected_template_name or "").strip()
        if not name:
            QMessageBox.information(self, "提示", "请选择模板。")
            return
        lines = [
            f"即将把“{name}”设为全局模板。",
            "这会覆盖所有项目当前使用的模板配置。",
        ]
        if getattr(self, "selected_pdf_path", None):
            lines.append("同时会替换历史项目留下的 template_pdf 风格记录，并写入当前模板内容。")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("确认应用全局模板")
        box.setText(f"你正在覆盖全部项目的模板配置：{name}")
        box.setInformativeText("\n".join(lines[1:]))
        box.setStandardButtons(QMessageBox.NoButton)
        apply_btn = box.addButton("覆盖全部项目", QMessageBox.AcceptRole)
        cancel_btn = box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec_()
        if box.clickedButton() != apply_btn:
            return
        self.accept()
