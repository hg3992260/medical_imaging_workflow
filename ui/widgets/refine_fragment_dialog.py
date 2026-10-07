import json
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QPushButton, QTreeWidget, QTreeWidgetItem, QSplitter,
    QProgressBar, QMessageBox, QWidget
)
from PyQt5.QtCore import Qt, pyqtSignal

from src.services.project_service import ProjectService
from src.services.ollama_local_service import OllamaLocalService
from src.ai.agent_skills.fragment_surgeon import fragment_surgeon_skill
from src.utils.logger import get_logger

class RefineFragmentDialog(QDialog):
    """
    Dialog for CGR (Contextual Granular Refinement) mode.
    Allows user to refine a text fragment using specific project data.
    """
    
    def __init__(self, project_id, selected_text, pre_context, post_context,
                 parent=None, llm_service=None, model="llama3"):
        super().__init__(parent)
        self.project_id = project_id
        self.selected_text = selected_text
        self.pre_context = pre_context
        self.post_context = post_context

        self.logger = get_logger(__name__)
        self.project_service = ProjectService()
        from src.services.ollama_local_service import OllamaLocalService
        self.llm_service = llm_service if llm_service is not None else OllamaLocalService()
        self.model = str(model or "llama3")
        
        self.refined_text = None
        
        self.init_ui()
        self.load_data_scope()
        
    def init_ui(self):
        self.setWindowTitle("Contextual Granular Refinement (CGR)")
        self.resize(900, 700)
        
        layout = QVBoxLayout(self)
        
        # --- Top: Context & Instruction ---
        top_splitter = QSplitter(Qt.Horizontal)
        
        # Left: Selection Context
        context_widget = QWidget()
        context_layout = QVBoxLayout(context_widget)
        context_layout.addWidget(QLabel("<b>Selected Context:</b>"))
        self.context_preview = QTextEdit()
        self.context_preview.setReadOnly(True)
        self.context_preview.setHtml(
            f"<div style='color:gray'>{self.pre_context[-200:]}</div>"
            f"<div style='background-color:#e6fffa; padding:5px; border:1px solid #4fd1c5'><b>{self.selected_text}</b></div>"
            f"<div style='color:gray'>{self.post_context[:200]}</div>"
        )
        context_layout.addWidget(self.context_preview)
        top_splitter.addWidget(context_widget)
        
        # Right: Data Scope Selector
        scope_widget = QWidget()
        scope_layout = QVBoxLayout(scope_widget)
        scope_layout.addWidget(QLabel("<b>Data Scope (Select Evidence):</b>"))
        self.data_tree = QTreeWidget()
        self.data_tree.setHeaderHidden(True)
        self.data_tree.setSelectionMode(QTreeWidget.NoSelection) # Use checkboxes
        scope_layout.addWidget(self.data_tree)
        top_splitter.addWidget(scope_widget)
        
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)
        layout.addWidget(top_splitter, 1)
        
        # --- Middle: Instruction ---
        layout.addWidget(QLabel("<b>Refinement Instruction:</b>"))
        self.instruction_input = QTextEdit()
        self.instruction_input.setPlaceholderText("E.g., 'Update the efficacy rates using Table 1 data'...")
        self.instruction_input.setMaximumHeight(80)
        layout.addWidget(self.instruction_input)
        
        # --- Action Bar ---
        action_layout = QHBoxLayout()
        self.btn_refine = QPushButton("✨ Refine with Data")
        self.btn_refine.setStyleSheet("background-color: #2E86AB; color: white; padding: 8px; font-weight: bold;")
        self.btn_refine.clicked.connect(self.on_refine_clicked)
        action_layout.addWidget(self.btn_refine)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)
        layout.addLayout(action_layout)
        
        # --- Bottom: Result & Diff ---
        layout.addWidget(QLabel("<b>Result Preview:</b>"))
        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setPlaceholderText("Refined text will appear here...")
        layout.addWidget(self.result_view, 1)
        
        # --- Footer Buttons ---
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        self.btn_apply = QPushButton("✔ Apply Change")
        self.btn_apply.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self.accept)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(self.btn_apply)
        layout.addLayout(btn_layout)
        
    def load_data_scope(self):
        """Load OCR data from project service into tree"""
        if not self.project_id:
            return
            
        data = self.project_service.get_project_ocr_data(self.project_id)
        
        for session in data:
            sess_item = QTreeWidgetItem(self.data_tree)
            sess_item.setText(0, session['name'])
            sess_item.setFlags(sess_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsTristate)
            sess_item.setCheckState(0, Qt.Unchecked)
            
            for child in session.get('children', []):
                child_item = QTreeWidgetItem(sess_item)
                child_item.setText(0, child['name'])
                child_item.setFlags(child_item.flags() | Qt.ItemIsUserCheckable)
                child_item.setCheckState(0, Qt.Unchecked)
                # Store content in user data
                child_item.setData(0, Qt.UserRole, child.get('content', ''))
                
        self.data_tree.expandAll()
        
    def on_refine_clicked(self):
        instruction = self.instruction_input.toPlainText().strip()
        if not instruction:
            QMessageBox.warning(self, "Input Required", "Please enter refinement instructions.")
            return
            
        # Collect Data Context
        data_context = ""
        iterator = QTreeWidgetItemIterator(self.data_tree)
        while iterator.value():
            item = iterator.value()
            if item.checkState(0) == Qt.Checked and item.childCount() == 0: # Leaf node
                content = item.data(0, Qt.UserRole)
                if content:
                    data_context += f"--- Data Source: {item.text(0)} ---\n{content}\n\n"
            iterator += 1
            
        if not data_context:
            reply = QMessageBox.question(self, "No Data Selected", 
                                       "You haven't selected any data sources. Continue purely based on instruction?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
            data_context = "No specific data provided."
            
        # UI State
        self.btn_refine.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.result_view.setPlainText("Processing... Please wait.")
        
        # Execute (Sync for now, should be threaded in prod)
        # Using QTimer to allow UI to update before freezing
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, lambda: self._execute_refine(instruction, data_context))
        
    def _execute_refine(self, instruction, data_context):
        try:
            new_text = fragment_surgeon_skill(
                self.selected_text,
                self.pre_context,
                self.post_context,
                instruction,
                data_context,
                self.llm_service,
                model=self.model,
            )
            
            self.refined_text = new_text
            self.result_view.setPlainText(new_text)
            self.result_view.setStyleSheet("background-color: #e8f5e9;") # Light green hint
            self.btn_apply.setEnabled(True)
            
        except Exception as e:
            self.result_view.setPlainText(f"Error: {e}")
            self.logger.error(f"Refinement error: {e}")
            
        finally:
            self.btn_refine.setEnabled(True)
            self.progress_bar.setVisible(False)

# Helper for tree iteration if needed
from PyQt5.QtWidgets import QTreeWidgetItemIterator
