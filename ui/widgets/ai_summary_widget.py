from typing import List, Optional, Any
import os
import time
import json
import re
import urllib.parse
import concurrent.futures
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                             QPushButton, QTextEdit, QTextBrowser, QFrame, QSplitter,
                             QApplication, QGraphicsView, QGraphicsScene, QCheckBox,
                             QDialog, QLineEdit, QInputDialog, QMessageBox,
                             QDoubleSpinBox, QSpinBox)
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QPolygonF, QPainterPath
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QTimer

from src.utils.logger import get_logger

# Import Agent Core Loop Executor
from src.ai.agent_core_loop import AgentCoreLoopExecutor
from src.services.project_storage_service import ProjectStorageService
from src.services.llm_guided_sklearn_pipeline import run_llm_guided_sklearn_pipeline

class AgentFlowVisualizer(QGraphicsView):
    """
    A widget to visualize the Agent's thought process as a dynamic flowchart.
    Nodes: [Data Mining] -> [Planning] -> [Retrieval] -> [Drafting] -> [Review] -> [Synthesis]
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        try:
            from src.utils.theme_manager import ThemeManager
            bg = "#3A3A3A" if ThemeManager.get_current_theme() == "dark" else "#F5F5F5"
        except Exception:
            bg = "#F5F5F5"
        self.setBackgroundBrush(QBrush(QColor(bg)))
        self.nodes = {}
        self.edges = []
        self.active_node = None
        
        # Define the flow structure
        self.flow_structure = [
            ("Data Mining", (50, 50)),
            ("Planning", (200, 50)),
            ("Retrieval", (350, 50)),
            ("Drafting", (500, 50)),
            ("Review", (650, 50)),
            ("Synthesis", (800, 50))
        ]
        
        self._init_graph()

    def _is_dark_theme(self):
        try:
            from src.utils.theme_manager import ThemeManager
            return ThemeManager.get_current_theme() == "dark"
        except Exception:
            return False
        
    def _init_graph(self):
        self.scene.clear()
        self.nodes = {}
        
        # Draw explicit loop edge (Review -> Drafting)
        # Review is at (650, 50), Drafting is at (500, 50)
        # We draw a curved path below the nodes
        path = QPainterPath()
        path.moveTo(650, 70) # Bottom of Review
        path.cubicTo(650, 110, 500, 110, 500, 70) # Curve back to Drafting
        
        loop_pen = QPen(QColor("#6A6A6A" if self._is_dark_theme() else "#FF9800"))
        loop_pen.setWidth(2)
        loop_pen.setStyle(Qt.DashLine)
        self.scene.addPath(path, loop_pen)
        
        # Arrowhead for loop
        arrow = QPolygonF([QPointF(500, 70), QPointF(505, 80), QPointF(495, 80)])
        self.scene.addPolygon(arrow, QPen(Qt.NoPen), QBrush(QColor("#6A6A6A" if self._is_dark_theme() else "#FF9800")))
        
        # Draw edges
        pen = QPen(QColor("#CCCCCC"))
        pen.setWidth(2)
        for i in range(len(self.flow_structure) - 1):
            start_name, start_pos = self.flow_structure[i]
            end_name, end_pos = self.flow_structure[i+1]
            self.scene.addLine(start_pos[0] + 40, start_pos[1] + 20, 
                             end_pos[0] - 40, end_pos[1] + 20, pen)
                             
        # Draw nodes
        for name, pos in self.flow_structure:
            self._create_node(name, pos)
            
    def _create_node(self, name, pos):
        # Container group
        group = self.scene.createItemGroup([])
        
        # Circle/Rect
        base_rect = "#4A4A4A" if self._is_dark_theme() else "#E0E0E0"
        rect = self.scene.addRect(0, 0, 100, 40, QPen(Qt.NoPen), QBrush(QColor(base_rect)))
        rect.setPos(pos[0] - 50, pos[1])
        group.addToGroup(rect)
        
        # Text
        text = self.scene.addText(name)
        text.setDefaultTextColor(QColor("#E879F9" if self._is_dark_theme() else "#2E86AB"))
        # Center text
        text_width = text.boundingRect().width()
        text.setPos(pos[0] - 50 + (100 - text_width)/2, pos[1] + 10)
        group.addToGroup(text)
        
        # Loop Indicator (small circle, initially hidden)
        loop_indicator = self.scene.addEllipse(0, 0, 16, 16, QPen(Qt.NoPen), QBrush(QColor("#5A5A5A" if self._is_dark_theme() else "#FF5722")))
        loop_indicator.setPos(pos[0] + 35, pos[1] - 8)
        loop_indicator.setVisible(False)
        group.addToGroup(loop_indicator)
        
        # Loop Text (counter)
        loop_text = self.scene.addText("1")
        loop_text.setDefaultTextColor(QColor("white"))
        loop_text.setFont(QFont("Arial", 8, QFont.Bold))
        loop_text.setPos(pos[0] + 39, pos[1] - 8)
        loop_text.setVisible(False)
        group.addToGroup(loop_text)
        
        self.nodes[name] = {
            "group": group, 
            "rect": rect, 
            "text": text, 
            "base_color": base_rect,
            "loop_indicator": loop_indicator,
            "loop_text": loop_text,
            "loop_count": 0
        }


        
    def update_status(self, message: str):
        """Parse log message to highlight active node and track loops"""
        target_node = None
        is_loop = False
        
        # Keyword matching mapping
        if "Data Aggregation" in message or "DataMiner" in message:
            target_node = "Data Mining"
        elif "Planning" in message or "Generating Outline" in message:
            target_node = "Planning"
        elif "Retrieval" in message or "PubMed" in message or "CocoIndex" in message:
            target_node = "Retrieval"
        elif "Drafting" in message or "Structurizer" in message or "Novelty" in message or "Phase 2" in message:
            target_node = "Drafting"
        elif "Reviewer" in message or "Refiner" in message or "Reviewing" in message:
            target_node = "Review"
            if "Refiner" in message or "Draft rejected" in message:
                is_loop = True
        elif "Synthesis" in message or "Conclusion" in message or "Final Synthesis" in message:
            target_node = "Synthesis"
            
        if target_node:
            self._activate_node(target_node, is_loop)
            
    def _activate_node(self, node_name, is_loop=False):
        # Reset previous
        if self.active_node and self.active_node in self.nodes and self.active_node != node_name:
            prev = self.nodes[self.active_node]
            prev["rect"].setBrush(QBrush(QColor(prev["base_color"])))
            prev["text"].setDefaultTextColor(QColor("#E879F9" if self._is_dark_theme() else "#2E86AB"))
            
        # Activate new
        if node_name in self.nodes:
            curr = self.nodes[node_name]
            curr["rect"].setBrush(QBrush(QColor("#6A6A6A" if self._is_dark_theme() else "#4CAF50")))
            curr["text"].setDefaultTextColor(QColor("#F5D0FE" if self._is_dark_theme() else "white"))
            
            # Handle Loop Counter
            if is_loop:
                curr["loop_count"] += 1
                curr["loop_text"].setPlainText(str(curr["loop_count"]))
                curr["loop_indicator"].setVisible(True)
                curr["loop_text"].setVisible(True)
                
            self.active_node = node_name
            
            # Force update
            self.scene.update()
            self.viewport().update()  # Ensure viewport refreshes immediately
            QApplication.processEvents() # Force event loop to process paint events


class CoreLoopWorker(QThread):
    progress_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str)
    raw_result_signal = pyqtSignal(dict) # New signal for raw structured result
    finished_signal = pyqtSignal()
    stream_signal = pyqtSignal(str) # New signal for token streaming

    def __init__(
        self,
        raw_content,
        project_id,
        kb=None,
        model="llama3",
        db_service=None,
        template_constraints_json: str = "",
        enable_docs_db: bool = False,
        llm_service=None,
        judge_config=None,
        workspace: str = "",
        project_storage_path: str = "",
    ):
        super().__init__()
        self.raw_content = raw_content
        self.project_id = project_id
        self.model = model
        self.workspace = str(workspace or "").strip()
        self.project_storage_path = str(project_storage_path or "").strip()
        self.llm_service = llm_service
        self.judge_config = dict(judge_config or {})
        self.template_constraints_json = template_constraints_json or ""
        self.enable_docs_db = bool(enable_docs_db)
        self.global_docs_dir = r"f:\RSNA\medical_imaging_workflow\src\docs"
        self.executor = AgentCoreLoopExecutor(
            kb=kb,
            db_service=db_service,
            progress_cb=self._emit_progress,
            stream_cb=self._emit_stream,
            enable_global_docs=self.enable_docs_db,
            global_docs_dir=self.global_docs_dir,
            llm_service=llm_service,
        )
        self.executor.model = model
        self.executor.tool_model = model
        try:
            self.executor.draft_template_constraints_json = self.template_constraints_json
        except Exception:
            pass
        self._is_running = True

    def _run_pre_mining(self):
        if run_llm_guided_sklearn_pipeline is None:
            return None
        storage_path = str(self.project_storage_path or "").strip()
        workspace = str(self.workspace or storage_path).strip()
        if not storage_path:
            return None
        try:
            out_dir = os.path.join(storage_path, "exports")
            self.progress_signal.emit("Phase 0.5: Step2-style tabular mining before Step3...")
            self.stream_signal.emit("> 🧠 [Step2Mining]: running tabular mining before Step3 drafting\n")
            res = run_llm_guided_sklearn_pipeline(
                llm_service=self.llm_service,
                model=self.model,
                materials_text=str(self.raw_content or ""),
                project_storage_path=storage_path,
                workspace=workspace,
                out_dir=out_dir,
            )
            if isinstance(res, dict):
                self.executor._ml_analysis_cache[f"phase1.9:{storage_path}"] = res
                self.executor._ml_analysis_cache[f"methods:{storage_path}"] = res
                self.executor._ml_analysis_cache[f"results:{storage_path}"] = res
                if res.get("success"):
                    self.stream_signal.emit("> ✅ [Step2Mining]: tabular mining completed and injected into Step3\n")
                else:
                    warns = res.get("warnings") or []
                    srcs = res.get("sources") or []
                    warn_str = ""
                    try:
                        warn_str = ", ".join([str(x) for x in warns if str(x).strip()][:6])
                    except Exception:
                        warn_str = ""
                    src_str = ""
                    try:
                        src_str = ", ".join([str(x) for x in srcs if str(x).strip()][:3])
                    except Exception:
                        src_str = ""
                    extra = ""
                    if warn_str:
                        extra += f" warnings={warn_str}"
                    if src_str:
                        extra += f" sources={src_str}"
                    self.stream_signal.emit(f"> ⚠️ [Step2Mining]: mining returned {res.get('error', 'no_usable_result')}{extra}\n")
            return res if isinstance(res, dict) else None
        except Exception as e:
            self.stream_signal.emit(f"> ⚠️ [Step2Mining]: failed before Step3: {type(e).__name__}: {e}\n")
            return None

    def run(self):
        try:
            self.progress_signal.emit(f"启动 Agent Core Loop (Model: {self.model}): Planning -> Retrieval -> Drafting -> Compression -> Synthesis")
            if hasattr(self.llm_service, "ensure_model_ready"):
                self.progress_signal.emit(f"本地模型加载阶段: {self.model}")
                ready = self.llm_service.ensure_model_ready(
                    self.model,
                    status_callback=lambda msg: self.progress_signal.emit(str(msg)),
                    timeout=180,
                )
                if isinstance(ready, dict):
                    if ready.get("model"):
                        self.model = str(ready.get("model"))
                        self.executor.model = self.model
                        self.executor.tool_model = self.model
                    if not ready.get("success", False):
                        raise RuntimeError(ready.get("error") or f"本地模型 {self.model} 未就绪")
            pre_mining = self._run_pre_mining()
            effective_raw_content = str(self.raw_content or "")
            if isinstance(pre_mining, dict) and pre_mining.get("success"):
                pre_summary = str(pre_mining.get("summary") or "").strip()
                fig_notes = str(pre_mining.get("figure_notes") or "").strip()
                pre_artifacts = [str(x) for x in (pre_mining.get("artifacts") or []) if str(x).strip()]
                inject_parts = ["[STEP2_TABULAR_MINING_PRELOAD]"]
                if pre_summary:
                    inject_parts.append(pre_summary)
                if fig_notes:
                    inject_parts.append("[STEP2_FIGURE_NOTES]\n" + fig_notes)
                if pre_artifacts:
                    inject_parts.append("[STEP2_ARTIFACTS]\n" + "\n".join(pre_artifacts[:20]))
                effective_raw_content = "\n".join(inject_parts).strip() + "\n\n" + effective_raw_content
            
            # Execute the loop
            self.progress_signal.emit("Phase 1: Generating Outline...")
            
            result = self.executor.execute_paper_writing(
                effective_raw_content,
                self.project_id,
                judge_config=self.judge_config,
                workspace=str(self.workspace or self.project_storage_path),
            )
            if isinstance(result, dict) and isinstance(pre_mining, dict):
                result["step2_pre_mining"] = pre_mining
            
            # Emit raw result first
            self.raw_result_signal.emit(result)
            
            # Format the result into Markdown
            md = self._format_result_to_md(result)
            self.result_signal.emit(md)
            
            self.progress_signal.emit("Agent Core Loop Completed Successfully.")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.progress_signal.emit(f"Error in Agent Core Loop: {e}")
            self.result_signal.emit(f"# Error\n\n{e}\n\n```text\n{tb}\n```")
        finally:
            self.finished_signal.emit()

    def _emit_progress(self, message: str):
        try:
            self.progress_signal.emit(message)
        except Exception:
            pass
            
    def _emit_stream(self, token: str):
        try:
            self.stream_signal.emit(token)
        except Exception:
            pass
    def _format_result_to_md(self, res):
        if (res.get("status") or "") == "blocked":
            error_message = str(res.get("error") or "当前输入不足以生成论文草稿。").strip()
            title = "检测到异常输入"
            subtitle = "当前输入中包含非论文素材，系统已暂停写作，以避免生成被错误信息污染的内容。"
            next_steps = [
                "- 返回 Step1，检查 OCR 输入是否混入报错日志、调试输出或 Traceback",
                "- 仅保留论文相关素材，例如病例摘要、OCR 文本、研究说明和结果摘要",
                "- 重新执行生成前，确认输入中不再包含“missing source”或“generation skipped”类文本",
            ]
            if "HardFacts" in error_message:
                title = "缺少研究数据"
                subtitle = "当前项目还没有提取到足够的研究事实，系统已暂停写作，避免生成空洞草稿。"
                next_steps = [
                    "- 返回 Step1，确认 OCR 或导入文本中已经包含病例内容、研究对象和关键描述",
                    "- 返回 Step2，确认结构化提取结果中已有样本量、核心指标、统计结果或关键术语",
                    "- 至少补充一组可写作事实，例如样本量、P 值、噪声变化、测量指标或研究对象",
                    "- 完成补充后重新执行生成，系统识别到最小 HardFacts 后会继续写作",
                ]
            elif "污染上下文" in error_message:
                title = "检测到异常输入"
                subtitle = "系统识别到当前上下文不是论文素材，而是错误提示或调试信息，因此已主动停止生成。"
                next_steps = [
                    "- 返回 Step1，删除 OCR 结果中的错误日志、Traceback、报错说明或调试输出",
                    "- 检查输入框、项目文本或中间结果，确保只保留病例数据、研究说明和结果摘要",
                    "- 如刚执行过自检或调试，请不要把控制台输出直接作为论文输入内容",
                ]

            md = f"# {title}\n\n"
            md += f"> {subtitle}\n\n"
            md += f"**系统提示**：{error_message}\n\n"
            md += "## 建议处理步骤\n"
            md += "\n".join(next_steps) + "\n\n"
            md += "## 当前状态\n"
            md += "- 系统已主动停止本次生成，避免输出空洞或被错误信息污染的草稿\n"
            md += "- 补全 Step1/Step2 所需数据后，可重新执行写作，系统会自动恢复正常链路\n\n"
            return md

        outline = res['outline']
        units = res['units']
        meta = res['meta']
        references = res.get('references', [])
        data_sources = res.get('data_sources', []) or []
        sample_sources = res.get('sample_sources', []) or []
        
        title = (outline.get("title") or "").strip() if isinstance(outline, dict) else ""
        if not title:
            title = "Draft"
        abstract = (meta.get("content") or "").strip() if isinstance(meta, dict) else ""
        sections = []
        try:
            sk = res.get("structure_skeleton") or {}
            if isinstance(sk, dict) and isinstance(sk.get("sections"), list):
                sections = [str(x).strip() for x in (sk.get("sections") or []) if str(x).strip()]
        except Exception:
            sections = []
        if not sections:
            sections = ["Title", "Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion", "References"]

        md = ""
        for s in sections:
            if s.lower() == "title":
                md += f"# {title}\n\n"
                continue
            if s.lower() == "abstract":
                if abstract:
                    md += "## Abstract\n"
                    md += f"{abstract}\n\n"
                continue
            if s.lower() == "introduction":
                md += "## Introduction\n"
                md += f"{(units.get('intro') or {}).get('content','')}\n\n"
                continue
            if s.lower() == "methods":
                md += "## Methods\n"
                md += f"{(units.get('methods') or {}).get('content','')}\n\n"
                continue
            if s.lower() == "results":
                md += "## Results\n"
                md += f"{(units.get('results') or {}).get('content','')}\n\n"
                continue
            if s.lower() == "discussion":
                md += "## Discussion\n"
                md += f"{(units.get('discussion') or {}).get('content','')}\n\n"
                continue
            if s.lower() == "conclusion":
                md += "## Conclusion\n"
                md += f"{(units.get('conclusion') or {}).get('content','')}\n\n"
                continue
            if s.lower() == "references":
                if references or data_sources or sample_sources:
                    md += "## References\n"
                    if references:
                        md += "### PubMed\n"
                        for i, ref in enumerate(references, 1):
                            md += f"{i}. {ref['citation']}\n"
                        md += "\n"
                    if sample_sources:
                        md += "### Samples\n"
                        for row in sorted(sample_sources, key=lambda x: int(x.get('n') or 0)):
                            n = row.get("n")
                            proj = row.get("project") or ""
                            sample = row.get("sample") or ""
                            sample_id = row.get("sample_id") or ""
                            md += f"- C{n}: {proj} / {sample} ({sample_id})\n"
                        md += "\n"
                    if data_sources:
                        md += "### Data Sources\n"
                        for row in sorted(data_sources, key=lambda x: int(x.get('n') or 0)):
                            n = row.get("n")
                            proj = row.get("project") or ""
                            sample_n = row.get("sample_n") or ""
                            file_name = row.get("file") or ""
                            src_type = row.get("type") or ""
                            md += f"- S{n}: {proj} / C{sample_n} / {file_name} ({src_type})\n"
                        md += "\n"
                continue

            md += f"## {s}\n\n"

        return md

    def stop(self):
        self._is_running = False

class ProjectIndexWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)

    def __init__(self, project_id: str, project_name: str, db_service, kb, force: bool, use_temp: bool):
        super().__init__()
        self.project_id = project_id
        self.project_name = project_name
        self.db_service = db_service
        self.kb = kb
        self.force = force
        self.use_temp = use_temp

    def run(self):
        try:
            from src.ai.project_context_indexer import ProjectContextIndexer
            self.progress_signal.emit(f"ProjectContext: 开始索引项目 {self.project_name} ({self.project_id})")
            idx = ProjectContextIndexer(self.db_service, self.kb, lancedb_uri=getattr(self.kb, "db_uri", None))
            res = idx.sync_project(self.project_id, force=self.force, progress_cb=lambda m: self.progress_signal.emit(m))
            payload = {"success": True, "result": res, "use_temp": self.use_temp}
            self.finished_signal.emit(payload)
        except Exception as e:
            self.finished_signal.emit({"success": False, "error": str(e), "use_temp": self.use_temp})

class AnalysisWorker(QThread):
    progress_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    stream_signal = pyqtSignal(str)

    def __init__(self, widget, model, full_text, sections, citation_text, project_name, ocr_chunks=None, use_full_ocr=False):
        super().__init__()
        self.widget = widget
        self.model = model
        self.full_text = full_text
        self.sections = sections
        self.citation_text = citation_text
        self.project_name = project_name
        self.ocr_chunks = ocr_chunks or []
        self.use_full_ocr = use_full_ocr
        self._is_running = True
        self.start_time = None

    def run(self):
        self.start_time = time.time()
        
        # --- PubMed Retrieval ---
        self.progress_signal.emit("正在检索 PubMed 相关文献...")
        try:
            # Use project name or generic query
            query = f"{self.project_name} medical imaging analysis"
            if "默认" in query: query = "Medical imaging AI analysis"
            
            refs = self.widget.pubmed.search(query, max_results=5)
            if refs:
                ref_list = []
                for i, r in enumerate(refs, 1):
                    ref_list.append(f"{i}. {r.get('citation', 'Unknown')}")
                self.citation_text = "\n".join(ref_list)
                self.progress_signal.emit(f"检索到 {len(refs)} 篇文献")
                # Stream references to the right-side output in real time
                try:
                    self.stream_signal.emit("\n## References (PubMed)\n")
                    for line in ref_list:
                        self.stream_signal.emit(line + "\n")
                except Exception:
                    pass
            else:
                self.citation_text = "No relevant references found."
                self.progress_signal.emit("未检索到文献")
                try:
                    self.stream_signal.emit("\n## References (PubMed)\nNo relevant references found.\n")
                except Exception:
                    pass
        except Exception as e:
            self.progress_signal.emit(f"PubMed 检索失败: {e}")
            self.citation_text = "(Retrieval Failed)"

        # --- Agent Plan: generate queries for KB ---
        self.progress_signal.emit("Agent Plan: 生成检索查询...")
        plan_prompt = (
            f"Given the following project data, produce a JSON with two arrays: "
            f"style_queries and project_queries. "
            f"style_queries should contain query phrases to retrieve writing style guidance "
            f"for sections Introduction, Methods, Results, Discussion, Conclusion. "
            f"project_queries should contain query phrases to retrieve relevant project context. "
            f"Return ONLY JSON.\n\n"
            f"Project Data:\n{self.full_text[:2000]}\n"
        )
        plan_text = ""
        llm_svc = getattr(self, "llm_service", None) or getattr(self.widget, "ollama", None)
        try:
            # Enable streaming so右侧显示框看到计划生成的模型动态
            res_plan = llm_svc.generate(
                self.model,
                plan_prompt,
                timeout=120,
                stream_callback=lambda t: self.stream_signal.emit(t)
            )
            if res_plan.get("success"):
                plan_text = res_plan.get("response", "")
        except Exception as e:
            self.progress_signal.emit(f"Agent Plan 失败: {e}")
        style_queries = []
        project_queries = []
        if plan_text:
            try:
                s = plan_text.strip()
                start = s.find("{")
                end = s.rfind("}")
                if start != -1 and end != -1 and end > start:
                    s = s[start:end+1]
                data = json.loads(s)
                style_queries = data.get("style_queries", []) or []
                project_queries = data.get("project_queries", []) or []
            except Exception as e:
                style_queries = []
                project_queries = []
        if not style_queries:
            style_queries = [
                "How to write Introduction section structure vocabulary",
                "How to write Methods section structure vocabulary",
                "How to write Results section structure vocabulary",
                "How to write Discussion section structure vocabulary",
                "How to write Conclusion section structure vocabulary",
            ]
        if not project_queries:
            project_queries = [
                f"{self.project_name} medical imaging",
                "medical imaging project context",
            ]
        self.progress_signal.emit("Agent Plan: 检索查询生成完成")

        # --- CocoIndex Retrieval using Agent Plan ---
        self.progress_signal.emit("CocoIndex: 检索上下文...")
        style_guide_context = ""
        project_context_base = ""
        if self.widget.kb:
            try:
                style_parts = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                    futs = [ex.submit(self.widget.kb.query_reference, q, 2) for q in style_queries]
                    for f in futs:
                        try:
                            r = f.result(timeout=2.0)
                            if r:
                                style_parts.append(str(r))
                        except concurrent.futures.TimeoutError:
                            self.progress_signal.emit("Style 检索超时，已跳过部分查询")
                        except Exception:
                            pass
                style_guide_context = "\n".join(style_parts)
            except Exception as e:
                style_guide_context = ""
                self.progress_signal.emit(f"Style 检索失败: {e}")
            try:
                proj_parts = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                    futs = [ex.submit(self.widget.kb.query_project, q, self.widget.current_project_id) for q in project_queries]
                    for f in futs:
                        try:
                            r = f.result(timeout=2.0)
                            if r:
                                proj_parts.append(str(r))
                        except concurrent.futures.TimeoutError:
                            self.progress_signal.emit("Project 检索超时，已跳过部分查询")
                        except Exception:
                            pass
                project_context_base = "\n".join(proj_parts)
            except Exception as e:
                project_context_base = ""
                self.progress_signal.emit(f"Project 检索失败: {e}")
        if not project_context_base:
            project_context_base = self.full_text[:3000]
        self.progress_signal.emit("CocoIndex: 检索完成")

        full_report = f"# {self.project_name} - 科学分析报告\n\n"
        full_report += f"**参考文献**:\n{self.citation_text}\n\n---\n\n"
        self.result_signal.emit(full_report)
         
        generated_context = ""

        for section_name, unit_key in self.sections:
            if not self._is_running: break
            
            self.progress_signal.emit(f"正在生成章节: {section_name}...")
            section_start = time.time()
            self.stream_signal.emit(f"\n\n## {section_name}\n\n")
            
            # RAG Retrieval (Mocked via widget.kb for now, but running in thread)
            # Note: widget.kb methods are synchronous, so they block this thread, not UI
            try:
                style_guide = self.widget.kb.query_reference(f"How to write {section_name} section structure vocabulary")
                project_context = self.widget.kb.query_project(f"Project details relevant for {section_name}", self.widget.current_project_id)
            except Exception as e:
                self.progress_signal.emit(f"RAG Error: {e}")
                style_guide = ""
                project_context = ""

            if self.use_full_ocr and self.ocr_chunks:
                try:
                    queries = [f"{self.project_name} {section_name}", section_name, "medical imaging"]
                    scored = []
                    for text in self.ocr_chunks:
                        t = text.lower()
                        score = 0
                        for q in queries:
                            score += t.count(q.lower())
                        if score > 0:
                            scored.append((score, text))
                    if not scored:
                        for text in self.ocr_chunks[:20]:
                            scored.append((1, text))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    buf = []
                    total = 0
                    for _, txt in scored:
                        if total > 1800:
                            break
                        seg = txt[:600]
                        buf.append(seg)
                        total += len(seg)
                    project_context = "\n".join(buf)
                except Exception:
                    if not project_context:
                        project_context = project_context_base
            else:
                if not project_context:
                    project_context = project_context_base
            # Clamp context sizes to avoid model prompt overflow
            style_block = f"{style_guide_context[:1200]}\n{style_guide[:800]}"
            proj_block = project_context[:2000]
            prev_block = generated_context[-2000:] if generated_context else "N/A"

            prompt = f"""
            Role: Expert Medical Researcher.
            Task: Write the '{section_name}' section for a research paper based on the provided project data.
             
            [Style Guidelines]
            {style_block}
             
            [Project Data Context]
            {proj_block}
             
            [Relevant Citations]
            {self.citation_text}
             
            [Context from Previous Sections]
            {prev_block}

            [Instructions]
            - Write in Chinese, but use English for standard medical terms.
            - Strictly follow the academic structure suitable for {section_name}.
            - Cite the references where appropriate.
            - Do not output conversational filler, just the report content.
            """
            
            # This call blocks this thread, enable streaming to UI
            try:
                self.stream_signal.emit(f"> [Debug] Section {section_name}: prompt_len≈{len(prompt)}\n")
            except Exception:
                pass
            res = llm_svc.generate(self.model, prompt, timeout=300, stream_callback=lambda t: self.stream_signal.emit(t))
            
            section_end = time.time()
            duration = section_end - section_start
            
            if res.get("success"):
                content = res.get("response", "")
                if not content.strip():
                    # Retry once if empty
                    try:
                        self.progress_signal.emit(f"LLM空输出，准备重试章节: {section_name}...")
                    except Exception:
                        pass
                    time.sleep(2.0)
                    res_retry = llm_svc.generate(self.model, prompt, timeout=300, stream_callback=lambda t: self.stream_signal.emit(t))
                    if res_retry.get("success"):
                        content = res_retry.get("response", "")
                section_md = f"## {section_name}\n\n{content}\n\n"
                full_report += section_md
                generated_context += f"\n\n[{section_name}]\n{content}"
                self.progress_signal.emit(f"章节 {section_name} 生成完成 (耗时 {duration:.2f}s)")
            else:
                self.progress_signal.emit(f"章节 {section_name} 生成失败: {res.get('error')} (耗时 {duration:.2f}s)")
                full_report += f"## {section_name}\n\n(Generation Failed)\n\n"
        
        # Emit final report snapshot after all sections
        self.result_signal.emit(full_report)
        
        end_time = time.time()
        elapsed = end_time - self.start_time
        self.progress_signal.emit(f"分析完成！总耗时: {elapsed:.2f} 秒")
        self.finished_signal.emit()

    def stop(self):
        self._is_running = False

from src.services.database_service import DatabaseService
from src.services.data_cleaning_service import DataCleaningService

from src.services.ollama_local_service import OllamaLocalService, CustomLLMService
from src.services.pubmed_service import PubMedService
import cocoindex

from src.ai.knowledge_base import KnowledgeBase
from ui.widgets.template_wizard_dialog import TemplateWizardDialog
from src.ai.template_pdf_indexer import index_pdf_template_into_lancedb

class TemplateApplyWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)

    def __init__(self, project_id: str, db_uri: str, cleanup_scope: str, selected_pdf: str, template_name: str, pre_extracted_data: dict = None):
        super().__init__()
        self.project_id = project_id or ""
        self.db_uri = db_uri or ""
        self.cleanup_scope = cleanup_scope or "template_pdf"
        self.selected_pdf = selected_pdf or ""
        self.template_name = template_name or ""
        self.pre_extracted_data = pre_extracted_data or {}

    def run(self):
        payload = {"success": True, "cleared": None, "indexed": None}
        try:
            from src.ai.knowledge_base import KnowledgeBase

            kb = KnowledgeBase(db_uri=self.db_uri, init_reference=False)
            if self.cleanup_scope == "all_project_vectors":
                self.progress_signal.emit("正在清空项目向量库（project_context/project_context_meta）…")
                payload["cleared"] = kb.clear_project_vector_db(self.project_id)
            else:
                self.progress_signal.emit("正在清理历史项目的全局 template_pdf 向量…")
                payload["cleared"] = kb.clear_all_template_pdf_vectors()
        except Exception as e:
            payload["success"] = False
            payload["error"] = f"clear vectors failed: {e}"
            self.finished_signal.emit(payload)
            return

        if self.selected_pdf:
            try:
                self.progress_signal.emit("正在将 PDF 模板写入 LanceDB（template_pdf）…")
                last_p = {"v": -1}
                def _emit_progress(p):
                    try:
                        ip = int(float(p))
                    except Exception:
                        self.progress_signal.emit(str(p))
                        return
                    prev = int(last_p.get("v", -1))
                    if ip == prev:
                        return
                    if prev >= 0 and (ip - prev) < 2 and ip != 100:
                        return
                    last_p["v"] = ip
                    self.progress_signal.emit(f"PDF OCR 进度: {ip}%")
                payload["indexed"] = index_pdf_template_into_lancedb(
                    project_id=self.project_id,
                    pdf_path=self.selected_pdf,
                    template_name=self.template_name,
                    lancedb_uri=self.db_uri,
                    progress_cb=_emit_progress,
                    pre_extracted_data=self.pre_extracted_data,
                )
            except Exception as e:
                payload["success"] = False
                payload["error"] = f"index pdf template failed: {e}"
                self.finished_signal.emit(payload)
                return

        self.progress_signal.emit("模板应用完成。")
        self.finished_signal.emit(payload)

class CustomAPIDialog(QDialog):
    MODES = [
        ("Chat (对话)", "chat"),
        ("Thinking Mode (思考模式)", "thinking"),
        ("JSON Output", "json"),
        ("FIM Completion (Beta)", "fim"),
        ("Chat Prefix Completion (Beta)", "chat_prefix"),
    ]
    API_TYPES = [
        ("DeepSeek", "deepseek"),
        ("OpenAI / Other (OpenAI-compatible)", "openai"),
    ]

    def __init__(self, parent=None, existing_profile: dict = None):
        super().__init__(parent)
        self.setWindowTitle(
            "编辑 Custom API 配置" if existing_profile else "新增 Custom API 配置"
        )
        self.resize(520, 400)
        self.profile = dict(existing_profile) if existing_profile else {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── row 1: name ──────────────────────────────────────────────
        row_name = QHBoxLayout()
        row_name.addWidget(QLabel("配置名称:"))
        self.name_edit = QLineEdit()
        self.name_edit.setText(self.profile.get("name", ""))
        row_name.addWidget(self.name_edit)
        layout.addLayout(row_name)

        # ── row 2: base_url ───────────────────────────────────────────
        row_url = QHBoxLayout()
        row_url.addWidget(QLabel("Base URL:"))
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText(
            "OpenAI 兼容: https://api.deepseek.com\n"
            "Anthropic 兼容: https://api.deepseek.com/anthropic"
        )
        self.url_edit.setText(self.profile.get("base_url", "https://api.deepseek.com"))
        row_url.addWidget(self.url_edit)
        layout.addLayout(row_url)

        # ── row 3: api_key ───────────────────────────────────────────
        row_key = QHBoxLayout()
        row_key.addWidget(QLabel("API Key:"))
        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("sk-xxxxxxxxxxxxxxxxxxxxxxxx")
        self.key_edit.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        self.key_edit.setText(self.profile.get("api_key", ""))
        row_key.addWidget(self.key_edit)
        layout.addLayout(row_key)

        # ── row 4: model ─────────────────────────────────────────────
        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("模型名称:"))
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText(
            "deepseek-v4-flash / deepseek-v4-pro / deepseek-chat / deepseek-reasoner"
        )
        self.model_edit.setText(self.profile.get("model", "deepseek-v4-flash"))
        row_model.addWidget(self.model_edit)
        layout.addLayout(row_model)

        # ── row 5: mode ──────────────────────────────────────────────
        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("服务模式:"))
        self.mode_combo = QComboBox()
        for label, val in self.MODES:
            self.mode_combo.addItem(label, val)
        cur_mode = self.profile.get("mode", "chat")
        idx = self.mode_combo.findData(cur_mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        row_mode.addWidget(self.mode_combo)
        layout.addLayout(row_mode)

        # ── row 6: api_type ─────────────────────────────────────────
        row_type = QHBoxLayout()
        row_type.addWidget(QLabel("API 类型:"))
        self.type_combo = QComboBox()
        for label, val in self.API_TYPES:
            self.type_combo.addItem(label, val)
        cur_type = self.profile.get("api_type", "deepseek")
        idx = self.type_combo.findData(cur_type)
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)
        row_type.addWidget(self.type_combo)
        layout.addLayout(row_type)

        # ── row 7: temperature ───────────────────────────────────────
        row_temp = QHBoxLayout()
        row_temp.addWidget(QLabel("Temperature:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setDecimals(2)
        self.temp_spin.setValue(float(self.profile.get("temperature", 0.7)))
        self.temp_spin.setFixedWidth(80)
        self.temp_label = QLabel("0.70")
        self.temp_label.setFixedWidth(40)
        self.temp_spin.valueChanged.connect(
            lambda v: self.temp_label.setText(f"{v:.2f}")
        )
        row_temp.addWidget(self.temp_spin)
        row_temp.addWidget(self.temp_label)
        row_temp.addStretch()
        layout.addLayout(row_temp)

        # ── row 8: max_tokens ────────────────────────────────────────
        row_maxtk = QHBoxLayout()
        row_maxtk.addWidget(QLabel("Max Tokens:"))
        self.maxtk_spin = QSpinBox()
        self.maxtk_spin.setRange(64, 128000)
        self.maxtk_spin.setSingleStep(256)
        self.maxtk_spin.setValue(int(self.profile.get("max_tokens", 4096)))
        self.maxtk_spin.setFixedWidth(100)
        row_maxtk.addWidget(self.maxtk_spin)
        row_maxtk.addStretch()
        layout.addLayout(row_maxtk)

        # ── row 9: top_p ─────────────────────────────────────────────
        row_topp = QHBoxLayout()
        row_topp.addWidget(QLabel("Top P:"))
        self.topp_spin = QDoubleSpinBox()
        self.topp_spin.setRange(0.0, 1.0)
        self.topp_spin.setSingleStep(0.05)
        self.topp_spin.setDecimals(2)
        self.topp_spin.setValue(float(self.profile.get("top_p", 1.0)))
        self.topp_spin.setFixedWidth(80)
        row_topp.addWidget(self.topp_spin)
        row_topp.addStretch()
        layout.addLayout(row_topp)

        # ── mode hint ────────────────────────────────────────────────
        self.mode_hint = QLabel("提示: Thinking 模式会将 CoT 推理过程单独输出，适合复杂推理任务")
        self.mode_hint.setWordWrap(True)
        from PyQt5.QtGui import QColor
        self.mode_hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.mode_hint)

        layout.addStretch()

        # ── buttons ─────────────────────────────────────────────────
        btns = QHBoxLayout()
        btns.addStretch()
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self._on_save)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(self.save_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

    def _on_mode_changed(self, idx):
        mode = self.mode_combo.itemData(idx)
        hints = {
            "chat": "标准对话模式，适合大多数任务",
            "thinking": "思考模式：先输出 CoT 推理过程（reasoning_content），再输出最终答案",
            "json": "强制 JSON 输出模式，适合结构化结果解析",
            "fim": "FIM（Fill-in-the-Middle）补全，适合代码/文档中间部分补写",
            "chat_prefix": "对话前缀续写：需在多轮对话中给定 assistant 前缀，适合风格延续写作",
        }
        self.mode_hint.setText(f"提示: {hints.get(mode, '')}")

    def _on_save(self):
        name = self.name_edit.text().strip()
        base_url = self.url_edit.text().strip()
        api_key = self.key_edit.text().strip()
        model = self.model_edit.text().strip()
        mode = self.mode_combo.itemData(self.mode_combo.currentIndex())
        api_type = self.type_combo.itemData(self.type_combo.currentIndex())
        temperature = round(self.temp_spin.value(), 2)
        max_tokens = self.maxtk_spin.value()
        top_p = round(self.topp_spin.value(), 2)
        if not name or not base_url or not model:
            QMessageBox.warning(self, "缺少必填项", "配置名称 / Base URL / 模型名称 为必填项。")
            return
        self.profile = {
            "name": name,
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "mode": mode,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "api_type": api_type,
        }
        self.accept()


class AISummaryWidget(QWidget):
    analysis_completed = pyqtSignal(dict) # Signal to notify Main Window with result

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger(__name__)
        
        # Initialize Services
        self.db_service = DatabaseService()
        self.data_service = DataCleaningService(self.db_service)
        self.ollama = OllamaLocalService()
        self.pubmed = PubMedService()
        
        # Initialize CocoIndex
        try:
            cocoindex.init()
            self.kb_default = KnowledgeBase(init_reference=False)
            self.kb = self.kb_default
        except Exception as e:
            self.logger.error(f"Failed to init CocoIndex: {e}")
            self.kb_default = None
            self.kb = None

        self.current_project_id = None
        self.current_project_name = ""
        self._final_report_rendered = False
        self._stream_buffer = ""
        self.selected_template_name = ""
        self.selected_template_constraints = {}
        self.selected_template_constraints_json = ""
        self.models = []
        self.worker = None # Thread worker reference
        self.index_worker = None
        self.temp_kb_dir = ""
        self.temp_kb = None
        self._custom_llm_service: Optional[CustomLLMService] = None
        self._custom_api_profiles: List[dict] = []
        self._load_custom_api_profiles()
        self._custom_api_idx = len(self._custom_api_profiles)
        self._llm_service_for_call: Any = self.ollama
        self._chat_history: List[dict] = []
        self._token_count = 0
        self._tokenizer = None
        try:
            from transformers import AutoTokenizer
            tok_dir = os.path.join(os.path.dirname(__file__), "..", "..", "deepseek_v3_tokenizer", "deepseek_v3_tokenizer")
            tok_dir = os.path.abspath(tok_dir)
            if os.path.isdir(tok_dir):
                self._tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True, local_files_only=True)
                self.logger.info(f"Token计数器: DeepSeek v3 tokenizer loaded, max_length={self._tokenizer.model_max_length}")
        except Exception as e:
            self.logger.warning(f"Token计数器: tokenizer 加载失败 ({e}), 将使用估算模式")
            self._tokenizer = "estimate"

        self.init_ui()
        self.load_models()

    def init_ui(self):
        self.logger.info("AISummaryWidget.init_ui: start")
        try:
            from src.utils.theme_manager import ThemeManager
            is_dark = ThemeManager.get_current_theme() == "dark"
        except Exception:
            is_dark = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        
        top = QFrame()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(4, 4, 4, 4)
        
        self.project_label = QLabel("未选择项目")
        self.project_label.setStyleSheet(
            "font-size:11px; color:%s; font-weight:%s;" % (("#E9D5FF" if is_dark else "#444"), ("600" if is_dark else "normal"))
        )
        
        self.model_combo = QComboBox()
        self.model_combo.setFixedHeight(24)

        self.refresh_models_btn = QPushButton("⟳")
        self.refresh_models_btn.setFixedHeight(24)
        self.refresh_models_btn.setFixedWidth(32)
        self.refresh_models_btn.setToolTip("刷新模型列表")
        self.refresh_models_btn.clicked.connect(self._refresh_models)

        self.custom_api_btn = QPushButton("+ Custom API")
        self.custom_api_btn.setFixedHeight(24)
        self.custom_api_btn.setFixedWidth(90)
        self.custom_api_btn.clicked.connect(self._on_custom_api_clicked)

        self.custom_api_delete_btn = QPushButton("删除")
        self.custom_api_delete_btn.setFixedHeight(24)
        self.custom_api_delete_btn.setFixedWidth(44)
        self.custom_api_delete_btn.setEnabled(False)
        self.custom_api_delete_btn.clicked.connect(self._on_custom_api_delete_clicked)

        self.prompt_editor_btn = QPushButton("📝 提示词修改")
        self.prompt_editor_btn.setFixedHeight(24)
        self.prompt_editor_btn.setFixedWidth(100)
        self.prompt_editor_btn.setToolTip("Open flow-chart editor for agent core loop prompts")
        self.prompt_editor_btn.clicked.connect(self._on_prompt_editor_clicked)

        self.token_label = QLabel("Tok: 0")
        self.token_label.setFixedHeight(24)
        self.token_label.setStyleSheet("color:#888;font-size:11px;padding:0 4px;")
        self.token_reset_btn = QPushButton("🗑")
        self.token_reset_btn.setFixedHeight(22)
        self.token_reset_btn.setFixedWidth(28)
        self.token_reset_btn.setToolTip("清零Token计数")
        self.token_reset_btn.clicked.connect(self._on_token_reset)
        
        self.analyze_button = QPushButton("开始科学分析 (RAG)")
        self.analyze_button.setFixedHeight(26)
        self.analyze_button.clicked.connect(self.on_analyze_clicked)
        
        top_layout.addWidget(self.project_label)
        top_layout.addStretch()
        self.temp_kb_checkbox = QCheckBox("临时KB")
        self.temp_kb_checkbox.setChecked(False)
        top_layout.addWidget(self.temp_kb_checkbox)

        self.docs_kb_checkbox = QCheckBox("启用Docs库")
        self.docs_kb_checkbox.setChecked(False)
        self.docs_kb_checkbox.setToolTip("启用后，Step3 会检索 src/docs 下的 CT 设备资料库")
        top_layout.addWidget(self.docs_kb_checkbox)

        self.kb_sync_btn = QPushButton("同步KB(增量)")
        self.kb_sync_btn.setFixedHeight(24)
        self.kb_sync_btn.clicked.connect(lambda: self.start_project_index(force=False))
        top_layout.addWidget(self.kb_sync_btn)

        self.kb_rebuild_btn = QPushButton("重建KB(全量)")
        self.kb_rebuild_btn.setFixedHeight(24)
        self.kb_rebuild_btn.clicked.connect(lambda: self.start_project_index(force=True))
        top_layout.addWidget(self.kb_rebuild_btn)

        self.template_btn = QPushButton("Template Wizard")
        self.template_btn.setFixedHeight(24)
        self.template_btn.setStyleSheet(
            ("QPushButton{background-color:%s;color:%s;font-weight:bold;padding:4px 10px;border-radius:6px;}"
             "QPushButton:hover{background-color:%s;}"
             "QPushButton:pressed{background-color:%s;}")
            % (
                "#4A4A4A" if is_dark else "#FF1744",
                "#F5D0FE" if is_dark else "white",
                "#5A5A5A" if is_dark else "#F01440",
                "#3E3E3E" if is_dark else "#D91035",
            )
        )
        self.template_btn.clicked.connect(self.on_template_wizard_clicked)
        top_layout.addWidget(self.template_btn)

        self.template_label = QLabel("模板: 未选择")
        self.template_label.setStyleSheet(
            "font-size:11px; color:%s; font-weight:bold;" % ("#F5D0FE" if is_dark else "#111")
        )
        top_layout.addWidget(self.template_label)

        model_select_label = QLabel("选择模型")
        if is_dark:
            model_select_label.setStyleSheet("color:#E9D5FF; font-weight:600;")
        top_layout.addWidget(model_select_label)
        top_layout.addWidget(self.model_combo)
        self.model_combo.currentTextChanged.connect(self._on_model_combo_changed)
        top_layout.addWidget(self.refresh_models_btn)
        top_layout.addWidget(self.custom_api_btn)
        top_layout.addWidget(self.custom_api_delete_btn)
        top_layout.addWidget(self.token_label)
        top_layout.addWidget(self.token_reset_btn)
        top_layout.addWidget(self.prompt_editor_btn)
        top_layout.addWidget(self.analyze_button)
        top.setFixedHeight(32)
        
        # Add Flow Visualizer
        self.flow_visualizer = AgentFlowVisualizer()
        self.flow_visualizer.setFixedHeight(120)
        layout.addWidget(top)
        layout.addWidget(self.flow_visualizer)
        
        # Use stylesheet for robust font fallback and color emoji support
        # We prioritize standard UI font, then Emoji fonts
        font_style = """
            QTextEdit { 
                font-family: "Segoe UI Emoji", "Segoe UI"; 
                font-size: 12pt;
                color: %s;
            }
        """ % ("#F5D0FE" if is_dark else "#2E86AB")

        self.console_text = QTextEdit()
        self.console_text.setStyleSheet(font_style)
        self.console_text.setReadOnly(True)
        self.console_text.setLineWrapMode(QTextEdit.NoWrap)
        self.console_text.setPlaceholderText("系统日志...")
        
        self.output_text = QTextBrowser()
        self.output_text.setStyleSheet(font_style)
        self.output_text.setReadOnly(True)
        self.output_text.setOpenExternalLinks(True)
        self.output_text.setPlaceholderText("分析报告将在此生成...")
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.console_text)
        splitter.addWidget(self.output_text)
        splitter.setSizes([400, 1000])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStyleSheet("QSplitter::handle{background:#E0E0E0; width:4px;}")
        
        layout.addWidget(splitter)
        self.log("AISummaryWidget.init_ui: UI components constructed")
        self.logger.info("AISummaryWidget.init_ui: complete")


    def _refresh_models(self):
        self.log("正在刷新模型列表...")
        self.logger.info("AISummaryWidget._refresh_models: clearing cache and re-fetching")
        self.ollama.clear_models_cache()
        self.models = self.ollama.list_models()
        self._update_model_ui()
        self.log(f"模型列表刷新完成，检测到 {len(self.models)} 个模型")

    def load_models(self):
        self.logger.info("AISummaryWidget.load_models: listing models via OllamaLocalService")
        self.models = self.ollama.list_models()
        
        # Retry mechanism if models are empty (Ollama might be starting up)
        if not self.models:
            self.logger.warning("No models found initially. Retrying in 2 seconds...")
            # Use QTimer for non-blocking retry
            QTimer.singleShot(2000, self._retry_load_models)
        else:
            self._update_model_ui()

    def _retry_load_models(self):
        self.logger.info("Retrying model load...")
        self.models = self.ollama.list_models()
        if not self.models:
             # Try one more time with longer delay
             QTimer.singleShot(5000, self._final_load_attempt)
        else:
             self._update_model_ui()

    def _final_load_attempt(self):
        self.logger.info("Final model load attempt...")
        self.models = self.ollama.list_models()
        self._update_model_ui()

    def _update_model_ui(self):
        self.logger.info(f"AISummaryWidget.load_models: models_detected={len(self.models)} -> {self.models}")
        self.log(f"AISummaryWidget.load_models: models_detected={len(self.models)}")
        self.model_combo.clear()
        all_items = self.models[:]
        for p in self._custom_api_profiles:
            name = p.get("name", "Custom")
            all_items.append(f"__custom__{name}")
        if all_items:
            self.model_combo.addItems(all_items)
            self.log("AISummaryWidget.load_models: models loaded into combo, enabling analyze button")
            self.logger.info("AISummaryWidget.load_models: models loaded into combo, button enabled")
            self.model_combo.setEnabled(True)
            self.analyze_button.setEnabled(True)
        else:
            self.model_combo.addItem("未检测到模型 (点击重试)")
            self.model_combo.setEnabled(False)
            self.analyze_button.setEnabled(False)
            self.log("AISummaryWidget.load_models: 未检测到模型，请确认 Ollama 服务已启动且模型已下载。")
            self.logger.warning("AISummaryWidget.load_models: no models detected, disabling controls")
        self._update_custom_api_delete_btn_state()

    def _update_custom_api_delete_btn_state(self):
        if not hasattr(self, "custom_api_delete_btn"):
            return
        current = self.model_combo.currentText() if self.model_combo else ""
        enabled = bool(current) and str(current).startswith("__custom__")
        self.custom_api_delete_btn.setEnabled(enabled)

    def _on_model_combo_changed(self, *_):
        self._update_custom_api_delete_btn_state()

    def _custom_api_config_path(self) -> str:
        try:
            from src.core.path_config import get_coco_data_dir
            base = str(get_coco_data_dir())
        except Exception:
            base = os.path.join(os.path.expanduser("~"), "AppData", "Local", "MedicalImagingWorkflow", "coco_data")
        return os.path.join(base, "custom_api_profiles.json")

    def _load_custom_api_profiles(self):
        self._custom_api_profiles = []
        try:
            p = self._custom_api_config_path()
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._custom_api_profiles = data
        except Exception:
            self._custom_api_profiles = []

    def _save_custom_api_profiles(self):
        try:
            p = self._custom_api_config_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self._custom_api_profiles, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save custom API profiles: {e}")

    def _on_custom_api_clicked(self):
        dialog = CustomAPIDialog(self)
        if dialog.exec_() == CustomAPIDialog.Accepted:
            profile = dialog.profile
            name = profile.get("name", "Custom")
            existing = next((i for i, p in enumerate(self._custom_api_profiles) if p.get("name") == name), -1)
            if existing >= 0:
                self._custom_api_profiles[existing] = profile
            else:
                self._custom_api_profiles.append(profile)
            self._save_custom_api_profiles()
            self._llm_service_for_call = self.ollama
            self._update_model_ui()
            idx = self.model_combo.count() - 1
            for i in range(self.model_combo.count()):
                if self.model_combo.itemText(i) == f"__custom__{name}":
                    idx = i
                    break
            self.model_combo.setCurrentIndex(idx)
            self.log(f"Custom API profile '{name}' saved and selected.")
            self._update_custom_api_delete_btn_state()

    def _on_custom_api_delete_clicked(self):
        current = self.model_combo.currentText() if self.model_combo else ""
        if not current or not str(current).startswith("__custom__"):
            return
        name = str(current)[len("__custom__"):]
        ret = QMessageBox.question(
            self,
            "删除 Custom API",
            f"确认删除 Custom API 配置：{name}？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        self._custom_api_profiles = [p for p in (self._custom_api_profiles or []) if p.get("name") != name]
        self._save_custom_api_profiles()
        try:
            self._custom_llm_service = None
        except Exception:
            pass
        try:
            self._chat_history.clear()
        except Exception:
            pass
        self._update_model_ui()
        if self.model_combo and self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)
        self.log(f"Custom API profile '{name}' deleted.")

    def _on_prompt_editor_clicked(self):
        from ui.widgets.prompt_editor_dialog import PromptEditorDialog
        dlg = PromptEditorDialog(self)
        dlg.exec_()

    def _active_llm_service(self) -> Any:
        current = self.model_combo.currentText() or ""
        if current.startswith("__custom__"):
            name = current[len("__custom__"):]
            prof = next((p for p in self._custom_api_profiles if p.get("name") == name), None)
            if prof:
                self._custom_llm_service = CustomLLMService(
                    base_url=prof.get("base_url", ""),
                    api_key=prof.get("api_key", ""),
                    model=prof.get("model", "deepseek-chat"),
                    mode=prof.get("mode", "chat"),
                    temperature=float(prof.get("temperature", 0.7)),
                    max_tokens=int(prof.get("max_tokens", 4096)),
                    top_p=float(prof.get("top_p", 1.0)),
                    api_type=prof.get("api_type", "deepseek"),
                )
                self._chat_history.clear()
                return self._custom_llm_service
        self._llm_service_for_call = self.ollama
        self._chat_history.clear()
        return self.ollama

    def _selected_custom_profile(self) -> Optional[dict]:
        current = self.model_combo.currentText() or ""
        if not current.startswith("__custom__"):
            return None
        name = current[len("__custom__"):]
        return next((p for p in self._custom_api_profiles if p.get("name") == name), None)

    def _current_model_name(self) -> str:
        prof = self._selected_custom_profile()
        if prof:
            return str(prof.get("model") or "deepseek-v4-flash")
        return self.model_combo.currentText() if self.model_combo else "llama3"

    def _current_service_label(self) -> str:
        prof = self._selected_custom_profile()
        if prof:
            return f"Custom API: {prof.get('name', 'Custom')}"
        return "Ollama"

    def _current_judge_config(self) -> dict:
        svc = self._active_llm_service()
        model = self._current_model_name()
        base_url = str(getattr(svc, "base_url", "") or "").strip()
        api_key = str(getattr(svc, "api_key", "") or "").strip()
        cfg = {
            "enabled": True,
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "temperature": 0.0,
            "max_tokens": 600,
            "timeout_s": 120.0,
        }
        return cfg

    def _current_judge_provider_label(self) -> str:
        prof = self._selected_custom_profile()
        if prof:
            return "Custom API"
        return "Ollama"

    def _current_project_storage_path(self) -> str:
        if not self.current_project_id or not self.db_service:
            return ""
        try:
            storage = ProjectStorageService(self.db_service)
            return str(storage.get_project_storage_path(self.current_project_id))
        except Exception:
            return ""

    def set_current_project(self, project_id: str, project_name: str = ""):
        self.current_project_id = project_id
        self.current_project_name = project_name or project_id
        self.project_label.setText(f"当前项目: {self.current_project_name} ({self.current_project_id})")
        self._load_template_from_project_metadata()
        self._final_report_rendered = False
        self._stream_buffer = ""
        self._token_count = 0
        self._update_token_label()
        self.output_text.clear()

    def _global_template_path(self) -> str:
        try:
            from src.core.path_config import get_coco_data_dir
            base = str(get_coco_data_dir())
        except Exception:
            base = os.path.join(os.path.expanduser("~"), "AppData", "Local", "MedicalImagingWorkflow", "coco_data")
        return os.path.join(base, "draft_template_global.json")

    def _load_global_template_cache(self) -> dict:
        p = self._global_template_path()
        try:
            if not os.path.exists(p):
                return {}
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _save_global_template_cache(self):
        p = self._global_template_path()
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            obj = {
                "template_name": self.selected_template_name,
                "constraints": self.selected_template_constraints if isinstance(self.selected_template_constraints, dict) else {},
            }
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def on_template_wizard_clicked(self):
        if not self.current_project_id:
            self.log("请先选择项目再选择模板")
            return
        model = self.model_combo.currentText() if self.model_combo else "llama3"
        db_uri = ""
        try:
            db_uri = getattr(self.kb_default, "db_uri", "") or ""
        except Exception:
            db_uri = ""
        dlg = TemplateWizardDialog(self, model=model, project_id=self.current_project_id, lancedb_uri=db_uri)
        if dlg.exec_() != QDialog.Accepted:
            return
        self.selected_template_name = dlg.selected_template_name
        self.selected_template_constraints = dlg.selected_constraints or {}
        selected_pdf = getattr(dlg, "selected_pdf_path", "") or ""
        selected_full_data = getattr(dlg, "selected_full_data", {}) or {}
        try:
            self.selected_template_constraints_json = json.dumps(self.selected_template_constraints, ensure_ascii=False, indent=2)
        except Exception:
            self.selected_template_constraints_json = ""
        self.template_label.setText(f"模板: {self.selected_template_name}")
        self._save_template_to_project_metadata()
        self._save_global_template_cache()
        scope = getattr(dlg, "cleanup_scope", "template_pdf") or "template_pdf"
        if getattr(self, "_template_apply_worker", None) and self._template_apply_worker.isRunning():
            self.log("模板应用任务正在运行，请稍候…")
            return
        self.template_btn.setEnabled(False)
        self.kb_sync_btn.setEnabled(False)
        self.kb_rebuild_btn.setEnabled(False)
        self.analyze_button.setEnabled(False)
        
        # Add a progress bar in console for visualization
        self.log(f"开始应用模板: {self.selected_template_name} (含向量库写入)...")
        self.log("[PROGRESS_BAR_START]") 
        
        self._template_apply_worker = TemplateApplyWorker(
            project_id=self.current_project_id,
            db_uri=db_uri,
            cleanup_scope=scope,
            selected_pdf=selected_pdf,
            template_name=self.selected_template_name,
            pre_extracted_data=selected_full_data,
        )
        self._template_apply_worker.progress_signal.connect(self.log)
        self._template_apply_worker.finished_signal.connect(self._on_template_apply_finished)
        self._template_apply_worker.start()

    def _on_template_apply_finished(self, payload: dict):
        self.template_btn.setEnabled(True)
        self.kb_sync_btn.setEnabled(True)
        self.kb_rebuild_btn.setEnabled(True)
        self.analyze_button.setEnabled(True)

        if not isinstance(payload, dict) or not payload.get("success"):
            self.log(f"模板应用失败: {(payload or {}).get('error') or payload}")
            return
        cleared = payload.get("cleared") or {}
        if isinstance(cleared, dict) and cleared.get("success"):
            c = cleared.get("cleared") or {}
            if "project_context_meta" in c:
                self.log("已清空当前项目的向量库（project_context/project_context_meta）")
            else:
                self.log("已替换历史项目留下的全局 template_pdf 向量（不影响项目其他向量）")
        indexed = payload.get("indexed") or {}
        if indexed:
            if isinstance(indexed, dict) and indexed.get("success"):
                self.log(f"PDF模板已入库: sections={indexed.get('sections_indexed')} records={indexed.get('records_written')}")
            else:
                self.log(f"PDF模板入库失败: {(indexed or {}).get('error') or indexed}")

    def _load_template_from_project_metadata(self):
        if not self.current_project_id:
            return
        g = self._load_global_template_cache()
        if isinstance(g, dict) and (g.get("template_name") or "").strip():
            self.selected_template_name = (g.get("template_name") or "").strip()
            self.selected_template_constraints = g.get("constraints") if isinstance(g.get("constraints"), dict) else {}
            try:
                self.selected_template_constraints_json = json.dumps(self.selected_template_constraints, ensure_ascii=False, indent=2)
            except Exception:
                self.selected_template_constraints_json = ""
            self.template_label.setText(f"模板: {self.selected_template_name} (全局)")
            return
        meta = {}
        try:
            with self.db_service.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT metadata FROM projects WHERE project_id=?", (self.current_project_id,))
                row = cur.fetchone()
                if row and row[0]:
                    meta = json.loads(row[0]) if isinstance(row[0], str) else {}
        except Exception:
            meta = {}
        tmpl = (meta or {}).get("draft_template") if isinstance(meta, dict) else None
        if isinstance(tmpl, dict):
            self.selected_template_name = (tmpl.get("template_name") or "").strip()
            self.selected_template_constraints = tmpl.get("constraints") if isinstance(tmpl.get("constraints"), dict) else {}
            try:
                self.selected_template_constraints_json = json.dumps(self.selected_template_constraints, ensure_ascii=False, indent=2)
            except Exception:
                self.selected_template_constraints_json = ""
        else:
            self.selected_template_name = ""
            self.selected_template_constraints = {}
            self.selected_template_constraints_json = ""
        self.template_label.setText(f"模板: {self.selected_template_name or '未选择'}")

    def _save_template_to_project_metadata(self):
        if not self.current_project_id:
            return
        try:
            with self.db_service.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT project_id, metadata FROM projects")
                rows = cur.fetchall() or []
                draft_template = {
                    "template_name": self.selected_template_name,
                    "constraints": self.selected_template_constraints if isinstance(self.selected_template_constraints, dict) else {},
                }
                for pid, raw_meta in rows:
                    meta = {}
                    if raw_meta:
                        try:
                            meta = json.loads(raw_meta) if isinstance(raw_meta, str) else {}
                        except Exception:
                            meta = {}
                    if not isinstance(meta, dict):
                        meta = {}
                    meta["draft_template"] = draft_template
                    cur.execute(
                        "UPDATE projects SET metadata=?, updated_at=CURRENT_TIMESTAMP WHERE project_id=?",
                        (json.dumps(meta, ensure_ascii=False), pid),
                    )
                conn.commit()
        except Exception as e:
            self.log(f"保存模板配置失败: {e}")

    def _clear_project_vector_db(self):
        if not self.current_project_id:
            return
        payloads = []
        try:
            if self.kb_default:
                payloads.append(self.kb_default.clear_project_vector_db(self.current_project_id))
        except Exception as e:
            payloads.append({"success": False, "error": str(e)})
        try:
            if self.temp_kb:
                payloads.append(self.temp_kb.clear_project_vector_db(self.current_project_id))
        except Exception as e:
            payloads.append({"success": False, "error": str(e)})

        ok = any(isinstance(p, dict) and p.get("success") for p in payloads)
        if ok:
            self.log("已清空当前项目的向量库（project_context/project_context_meta）")
            self.log("如需重新索引项目数据，请手动点击“重建KB(全量)”")
        else:
            self.log(f"向量库清空未成功: {payloads}")

    def _clear_template_pdf_vectors(self):
        if not self.current_project_id:
            return
        payloads = []
        try:
            if self.kb_default:
                payloads.append(self.kb_default.clear_all_template_pdf_vectors())
        except Exception as e:
            payloads.append({"success": False, "error": str(e)})
        try:
            if self.temp_kb:
                payloads.append(self.temp_kb.clear_all_template_pdf_vectors())
        except Exception as e:
            payloads.append({"success": False, "error": str(e)})
        ok = any(isinstance(p, dict) and p.get("success") for p in payloads)
        if ok:
            self.log("已清理历史项目留下的全局 template_pdf 向量（不影响项目其他向量）")
        else:
            self.log(f"template_pdf 向量清理未成功: {payloads}")

    def log(self, message: str):
        import html
        try:
            win = self.window()
            if hasattr(win, "on_model_status_message"):
                win.on_model_status_message(message)
        except Exception:
            pass
        
        # Simple text progress bar simulation
        if message == "[PROGRESS_BAR_START]":
            try:
                from src.utils.theme_manager import ThemeManager
                is_dark = ThemeManager.get_current_theme() == "dark"
            except Exception:
                is_dark = False
            outer = "#4A4A4A" if is_dark else "#eee"
            inner = "#6A6A6A" if is_dark else "#2E86AB"
            self.console_text.append(f'<div style="background-color:{outer};height:4px;width:100%;margin:4px 0;"><div style="background-color:{inner};height:4px;width:0%;" id="pbar"></div></div>')
            QApplication.processEvents()
            return
            
        escaped = html.escape(message)
        # Force Segoe UI Emoji for every log line to ensure color rendering
        html_msg = f'<span style="font-family: \'Segoe UI Emoji\'; font-size: 12pt;">{escaped}</span>'
        self.console_text.append(html_msg)
        
        # Update Visualizer
        if hasattr(self, 'flow_visualizer'):
            self.flow_visualizer.update_status(message)
        # Force UI update
        QApplication.processEvents()

    def on_analyze_clicked(self):
        if not self.current_project_id:
            self.output_text.setPlainText("请先选择项目")
            return
        if not self.model_combo.count() or (
            self.model_combo.count() == 1 and "未检测到模型" in self.model_combo.itemText(0)
        ):
            self.output_text.setPlainText("未检测到可用模型")
            return
        
        # Prevent double clicking
        self.analyze_button.setEnabled(False)
        self.analyze_button.setText("模型加载中…")
        
        model = self._current_model_name()
        service_label = self._current_service_label()
        enable_docs_db = bool(self.docs_kb_checkbox.isChecked())
        full_text = self.build_project_db_context(self.current_project_id, self.current_project_name)
        
        self.output_text.clear()
        self._final_report_rendered = False
        self._token_count = 0
        self._update_token_label()
        self.log(f"开始分析项目: {self.current_project_name}")
        self.log(f"LLM服务: {service_label}")
        self.log(f"使用模型: {model}")
        self.log(f"Docs库开关: {'开启' if enable_docs_db else '关闭'}")
        
        try:
            self.start_rag_thread(model, full_text, enable_docs_db=enable_docs_db)
        except Exception as e:
            self.log(f"启动分析线程失败: {e}")
            self.output_text.append(f"\n[Error] {str(e)}")
            self.analyze_button.setEnabled(True)
            self.analyze_button.setText("开始科学分析 (RAG)")

    def start_project_index(self, force: bool):
        if not self.current_project_id:
            self.log("请先选择项目")
            return
        if not self.db_service:
            self.log("数据库服务不可用")
            return
        if self.kb_default is None:
            self.log("KB 初始化失败，无法索引")
            return

        use_temp = bool(self.temp_kb_checkbox.isChecked())
        if use_temp:
            from src.core.path_config import get_coco_data_dir
            import uuid
            base = str(get_coco_data_dir())
            self.temp_kb_dir = os.path.join(base, "lancedb_tmp", f"{self.current_project_id}_{uuid.uuid4().hex[:8]}")
            os.makedirs(self.temp_kb_dir, exist_ok=True)
            self.temp_kb = KnowledgeBase(db_uri=self.temp_kb_dir, init_reference=False)
            kb_to_use = self.temp_kb
        else:
            self._cleanup_temp_kb()
            kb_to_use = self.kb_default

        self.kb_sync_btn.setEnabled(False)
        self.kb_rebuild_btn.setEnabled(False)
        self.analyze_button.setEnabled(False)

        self.index_worker = ProjectIndexWorker(
            project_id=self.current_project_id,
            project_name=self.current_project_name,
            db_service=self.db_service,
            kb=kb_to_use,
            force=force,
            use_temp=use_temp,
        )
        self.index_worker.progress_signal.connect(self.log)
        self.index_worker.finished_signal.connect(self.on_index_finished)
        self.index_worker.start()

    def on_index_finished(self, payload: dict):
        self.kb_sync_btn.setEnabled(True)
        self.kb_rebuild_btn.setEnabled(True)
        self.analyze_button.setEnabled(True)

        if not payload.get("success"):
            self.log(f"ProjectContext: 索引失败: {payload.get('error')}")
            return

        if payload.get("use_temp") and self.temp_kb is not None:
            self.kb = self.temp_kb
            self.log(f"ProjectContext: 已切换为临时KB: {self.temp_kb_dir}")
        else:
            self.kb = self.kb_default

        res = payload.get("result") or {}
        self.log(f"ProjectContext: 索引完成 sources={res.get('sources')} updated={res.get('updated_sources')} chunks={res.get('chunks')}")

    def start_rag_thread(self, model: str, full_text: str, enable_docs_db: bool = False):
        self.log("Initializing Agent Core Loop Worker (Skill: Patient-Centric Data Mining)...")

        llm_svc = self._active_llm_service()
        judge_cfg = self._current_judge_config()
        project_storage_path = self._current_project_storage_path()
        self.log(f"Judge Provider: {self._current_judge_provider_label()}")
        self.log(f"Judge Model: {judge_cfg.get('model', model)}")
        if project_storage_path:
            self.log(f"Step2预挖掘目录: {project_storage_path}")
        self.worker = CoreLoopWorker(
            raw_content=full_text,
            project_id=self.current_project_id,
            kb=self.kb,
            model=model,
            db_service=self.db_service,
            template_constraints_json=self.selected_template_constraints_json,
            enable_docs_db=enable_docs_db,
            llm_service=llm_svc,
            judge_config=judge_cfg,
            workspace=project_storage_path,
            project_storage_path=project_storage_path,
        )
        
        # Connect signals
        self.worker.progress_signal.connect(self.log)
        self.worker.result_signal.connect(self.update_report_text)
        self.worker.raw_result_signal.connect(self.on_raw_result) # Connect raw result
        self.worker.stream_signal.connect(self.append_stream_text)
        self.worker.finished_signal.connect(self.on_analysis_finished)
        
        # Start the worker thread
        self.worker.start()


    def on_raw_result(self, result):
        """Handle raw structured result from Agent Loop"""
        self.logger.info("AISummaryWidget: Received raw analysis result.")
        # Inject model name for downstream widgets
        if isinstance(result, dict):
            result['model_name'] = self._current_model_name()
            result['llm_service_name'] = self._current_service_label()
        self.analysis_completed.emit(result)

    def update_report_text(self, text):
        t = str(text or "")
        # Only remove STANDALONE [[[IMG_TAG_N]]] tokens that appear as plain text.
        # These are bare placeholder tokens that the LLM failed to translate or restore.
        # Do NOT match tokens inside markdown image syntax ![...]([[[IMG_TAG_0]]]).
        t = re.sub(r"!\[\]\(\[\[\[IMG_TAG_\d+\]\]\]\)|\[\[\[IMG_TAG_\d+\]\]\](?!\))", "", t)
        # Convert markdown image syntax to HTML img so Qt can render it directly,
        # bypassing setMarkdown's imperfect image rendering.
        # Also captures the *Figure N. caption* line that follows the image markdown
        # and renders it below the image as centered italic caption.
        def _md_images_to_html(text):
            # First pass: match [Figure N](path) immediately followed by *Figure N. caption*
            def _replace_figure_block(m):
                src = m.group(1).strip()
                cap = m.group(2).strip()
                if not src:
                    return ""
                if src.lower().startswith(("http://", "https://", "file://")):
                    path = src
                else:
                    path = f"file:///{src}"
                return (
                    f'<center>'
                    f'<img src="{path}" style="width:600px;height:auto;display:block;margin:8px auto;" />'
                    f'<br><i>{cap}</i>'
                    f'</center>'
                )
            text = re.sub(r'!\[Figure \d+\]\(([^)]+)\)\s*\n?\s*\*(Figure \d+\..+)\*', _replace_figure_block, text)
            # Second pass: handle remaining orphan images without caption
            def _replace_orphan(m):
                src = m.group(1).strip()
                if not src:
                    return ""
                if src.lower().startswith(("http://", "https://", "file://")):
                    path = src
                else:
                    path = f"file:///{src}"
                return f'<center><img src="{path}" style="width:600px;height:auto;display:block;margin:8px auto;" /></center>'
            return re.sub(r"!\[[^\]]*\]\(([^)]+)\)", _replace_orphan, text)

        html_content = _md_images_to_html(t)
        self.output_text.clear()
        try:
            self.output_text.setHtml(html_content)
        except Exception:
            self.output_text.setPlainText(t)
        self._final_report_rendered = True
        sb = self.output_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def append_stream_text(self, token):
        if self._final_report_rendered:
            return
        token = str(token or "")
        # Keep only plain draft text in the main output area:
        # - drop hidden reasoning traces from online APIs
        # - drop system/debug progress lines (already shown in the console pane)
        token = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", token, flags=re.IGNORECASE)
        filtered_lines = []
        for line in token.splitlines(True):
            if re.match(r"^\s*>\s*(?:\[Debug\]|🤖\s*\[System\]|🧠\s*\[|🧰\s*\[|🧑‍⚖️\s*\[)", line):
                continue
            filtered_lines.append(line)
        token = "".join(filtered_lines)
        if not token.strip():
            return
        self._stream_buffer += token
        self._count_tokens(token)
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.End)
        self.output_text.setTextCursor(cursor)
        self.output_text.insertPlainText(token)
        sb = self.output_text.verticalScrollBar()
        sb.setValue(sb.maximum())
    
    def _count_tokens(self, text: str):
        if not text or not text.strip():
            return
        try:
            if self._tokenizer == "estimate":
                self._token_count += max(1, len(text) // 3)
            elif self._tokenizer is not None:
                self._token_count += len(self._tokenizer.encode(text))
        except Exception:
            self._token_count += max(1, len(text) // 3)
        self._update_token_label()

    def _update_token_label(self):
        try:
            c = self._token_count
            if c >= 1000000:
                s = f"Tok: {c/1000000:.1f}M"
            elif c >= 1000:
                s = f"Tok: {c/1000:.1f}K"
            else:
                s = f"Tok: {c}"
            self.token_label.setText(s)
        except Exception:
            pass

    def _on_token_reset(self):
        self._token_count = 0
        self._update_token_label()

    def on_analysis_finished(self):
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("开始科学分析 (RAG)")
        self.log("任务结束")
        self._cleanup_temp_kb()

    def _cleanup_temp_kb(self):
        if self.temp_kb_dir:
            try:
                import shutil
                shutil.rmtree(self.temp_kb_dir, ignore_errors=True)
            except Exception:
                pass
        self.temp_kb_dir = ""
        self.temp_kb = None
        if self.kb_default is not None:
            self.kb = self.kb_default

    def build_project_summary(self, summary: dict) -> str:
        lines = []
        lines.append(f"Project Title: {self.current_project_name}")
        lines.append(f"ID: {summary.get('project_id','')}")
        
        dd = summary.get('dicom_data', {})
        lines.append(f"DICOM Data: {dd.get('count',0)} sessions, {dd.get('total_roi_count',0)} ROIs.")
        
        td = summary.get('text_data', {})
        lines.append(f"Text Documents: {td.get('document_count',0)} docs, length {td.get('total_content_length',0)}.")
        
        od = summary.get('ocr_data', {})
        lines.append(f"OCR Results: {od.get('result_count',0)} items.")
        # Append OCR text corpus (limited for token budget)
        ocr_details = od.get('ocr_details') or []
        if ocr_details:
            lines.append("OCR Text Corpus:")
            count = 0
            for item in ocr_details:
                if count >= 20:
                    break
                txt = item.get('text_preview') or ""
                if txt:
                    # Clamp each preview to avoid overflow
                    lines.append(txt[:1200])
                    count += 1
        
        ms = summary.get('magic_seg_data', {})
        lines.append(f"Segmentation: {ms.get('total_count',0)} masks.")
        
        sd = summary.get('samples', {})
        lines.append("Sample Descriptions:")
        for s in sd.get('details', [])[:10]:
             lines.append(f"- {s.get('display_name','')} : {s.get('description','')}")
             
        return "\n".join(lines)
    
    def build_project_db_context(self, project_id: str, project_name: str) -> str:
        lines = []
        lines.append(f"Project Title: {project_name}")
        lines.append(f"ID: {project_id}")
        try:
            with self.db_service.get_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM dicom_sessions WHERE project_id=?", (project_id,))
                dicom_sessions = c.fetchone()[0]
                c.execute("""
                    SELECT COUNT(*) FROM roi_data rd 
                    JOIN dicom_sessions ds ON rd.session_id=ds.session_id
                    WHERE ds.project_id=?""", (project_id,))
                roi_rows = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM ocr_sessions WHERE project_id=?", (project_id,))
                ocr_sessions = c.fetchone()[0]
                c.execute("""
                    SELECT COUNT(*) FROM ocr_results r 
                    JOIN ocr_sessions s ON r.session_id=s.session_id
                    WHERE s.project_id=?""", (project_id,))
                ocr_results = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM text_sessions WHERE project_id=?", (project_id,))
                text_sessions = c.fetchone()[0]
                c.execute("""
                    SELECT COUNT(*) FROM documents d 
                    JOIN text_sessions ts ON d.session_id=ts.session_id
                    WHERE ts.project_id=?""", (project_id,))
                documents = c.fetchone()[0]
                lines.append(f"DICOM Data: {dicom_sessions} sessions, {roi_rows} ROIs.")
                lines.append(f"OCR: {ocr_sessions} sessions, {ocr_results} results.")
                lines.append(f"Text: {text_sessions} sessions, {documents} documents.")

                # Query DICOM metadata + ROI data for Step 3 context
                c.execute("""
                    SELECT ds.dicom_info, rd.properties, rd.roi_type
                    FROM dicom_sessions ds
                    LEFT JOIN roi_data rd ON rd.session_id=ds.session_id
                    WHERE ds.project_id=? ORDER BY ds.created_at DESC LIMIT 20
                """, (project_id,))
                dcm_rows = c.fetchall()
                if dcm_rows:
                    lines.append("DICOM Sessions & ROI Data:")
                    # Add data quality summary
                    total_rois = 0
                    total_pixels = 0
                    patients = set()
                    for row in dcm_rows:
                        dcm_json, roi_json, roi_type = row
                        if dcm_json:
                            import json
                            try:
                                di = json.loads(dcm_json) if isinstance(dcm_json, str) else dcm_json
                                pid = di.get("patient_id")
                                if pid and str(pid) not in ("", "None"):
                                    patients.add(str(pid))
                            except Exception:
                                pass
                        if roi_json:
                            import json
                            try:
                                rp = json.loads(roi_json) if isinstance(roi_json, str) else roi_json
                                total_rois += 1
                                total_pixels += int((rp.get("statistics") or {}).get("pixel_count", 0) or 0)
                            except Exception:
                                pass
                    if patients or total_rois:
                        dq_parts = []
                        if patients:
                            dq_parts.append(f"unique patients: {len(patients)}")
                        if total_rois:
                            dq_parts.append(f"ROIs: {total_rois}")
                            dq_parts.append(f"total pixels: {total_pixels}")
                        n_dcm = len({row[0] for row in dcm_rows if row[0]})
                        if n_dcm:
                            dq_parts.append(f"DICOM sessions: {n_dcm}")
                        lines.append(f"[DATA_QUALITY] {', '.join(dq_parts)}")
                    seen_sessions = set()
                    for row in dcm_rows:
                        dcm_json, roi_json, roi_type = row
                        if dcm_json and dcm_json not in seen_sessions:
                            seen_sessions.add(dcm_json)
                            import json
                            try:
                                di = json.loads(dcm_json) if isinstance(dcm_json, str) else dcm_json
                                parts = []
                                for k, lbl in [("patient_name","Patient"),("patient_id","ID"),("study_date","Date"),("modality","Modality"),("body_part_examined","BodyPart"),("study_description","Study"),("manufacturer","Scanner")]:
                                    v = di.get(k)
                                    if v not in (None, "", "None"):
                                        parts.append(f"{lbl}: {v}")
                                if parts:
                                    lines.append("  - " + " | ".join(parts))
                            except Exception:
                                pass
                        if roi_json:
                            import json
                            try:
                                rp = json.loads(roi_json) if isinstance(roi_json, str) else roi_json
                                stats = rp.get("statistics") or {}
                                roi_source = rp.get("source", roi_type or "unknown")
                                parts = [f"ROI({roi_source})"]
                                px = stats.get("pixel_count", 0)
                                if px:
                                    parts.append(f"pixels={px}")
                                mu = stats.get("mean", 0)
                                if mu:
                                    parts.append(f"mean={mu:.1f}HU")
                                sd = stats.get("std", 0)
                                if sd:
                                    parts.append(f"std={sd:.1f}HU")
                                ar = stats.get("area_mm2", 0)
                                if ar:
                                    parts.append(f"area={ar:.2f}mm2")
                                if len(parts) > 1:
                                    lines.append("    " + ", ".join(parts))
                            except Exception:
                                pass

                c.execute("""
                    SELECT r.recognized_text 
                    FROM ocr_results r JOIN ocr_sessions s ON r.session_id=s.session_id
                    WHERE s.project_id=? ORDER BY r.created_at DESC LIMIT 50
                """, (project_id,))
                rows = c.fetchall()
                if rows:
                    lines.append("OCR Text Corpus:")
                    total = 0
                    for row in rows:
                        txt = row[0] or ""
                        if not txt:
                            continue
                        seg = txt[:1200]
                        lines.append(seg)
                        total += len(seg)
                        if total > 24000:
                            break
                c.execute("""
                    SELECT d.file_name, d.content 
                    FROM documents d JOIN text_sessions ts ON d.session_id=ts.session_id
                    WHERE ts.project_id=? ORDER BY d.created_at DESC LIMIT 10
                """, (project_id,))
                docs = c.fetchall()
                if docs:
                    lines.append("Text Documents Corpus:")
                    acc = 0
                    for fn, content in docs:
                        if content:
                            seg = content[:2000]
                            lines.append(f"[{fn}]")
                            lines.append(seg)
                            acc += len(seg)
                            if acc > 20000:
                                break
        except Exception as e:
            lines.append(f"(DB read error: {e})")
        return "\n".join(lines)
