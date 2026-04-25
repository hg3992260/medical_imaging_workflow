import os
import sys
import re
import qdarkstyle
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

class ThemeManager:
    _current_theme = "dark"  # default
    _orig_set_stylesheet = None

    @classmethod
    def _gray_for_bg(cls, value: str) -> str:
        v = value.strip().lower()
        if v in {"transparent", "none"}:
            return value
        if v.startswith("#") and len(v) == 7:
            try:
                r = int(v[1:3], 16)
                g = int(v[3:5], 16)
                b = int(v[5:7], 16)
                if abs(r - g) <= 8 and abs(g - b) <= 8:
                    return value
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                if lum >= 200:
                    return "#4A4A4A"
                if lum >= 130:
                    return "#5A5A5A"
                if lum >= 70:
                    return "#3E3E3E"
                return "#333333"
            except Exception:
                return "#4A4A4A"
        if "rgb" in v:
            return "#4A4A4A"
        if v in {"white", "snow", "ivory"}:
            return "#4A4A4A"
        return value

    @classmethod
    def _to_gray_blocks_css(cls, css: str) -> str:
        def _replace_bg(m):
            prop = m.group(1)
            val = m.group(2)
            return f"{prop}: {cls._gray_for_bg(val)}"
        def _replace_border(m):
            prop = m.group(1)
            val = m.group(2)
            val = re.sub(r"#[0-9A-Fa-f]{6}", lambda c: cls._gray_for_bg(c.group(0)), val)
            val = re.sub(r"\bwhite\b", "#4A4A4A", val, flags=re.IGNORECASE)
            return f"{prop}: {val}"
        css = re.sub(r"(background-color)\s*:\s*([^;]+)", _replace_bg, css, flags=re.IGNORECASE)
        css = re.sub(r"(?<!-)(background)\s*:\s*([^;]+)", _replace_bg, css, flags=re.IGNORECASE)
        css = re.sub(r"(alternate-background-color)\s*:\s*([^;]+)", _replace_bg, css, flags=re.IGNORECASE)
        css = re.sub(r"(selection-background-color)\s*:\s*([^;]+)", _replace_bg, css, flags=re.IGNORECASE)
        css = re.sub(r"(border-color)\s*:\s*([^;]+)", _replace_bg, css, flags=re.IGNORECASE)
        css = re.sub(r"(border(?:-(?:left|right|top|bottom))?)\s*:\s*([^;]+)", _replace_border, css, flags=re.IGNORECASE)
        return css

    @classmethod
    def _enable_dark_patch(cls):
        return

    @classmethod
    def _disable_dark_patch(cls):
        return

    @classmethod
    def _refresh_existing_widget_styles(cls, app: QApplication):
        """Re-apply existing widget styles so already-created pages are grayized in dark theme."""
        try:
            for w in app.allWidgets():
                ss = w.styleSheet()
                if isinstance(ss, str) and ss.strip():
                    w.setStyleSheet(ss)
        except Exception:
            pass

    @classmethod
    def apply_theme(cls, app: QApplication, theme: str):
        cls._current_theme = theme
        if theme == "dark":
            cls._enable_dark_patch()
            # Apply qdarkstyle + gray blocks + high-contrast purple text.
            dark_override_qss = """
                /* Global Font Override: High Contrast Purple */
                * { color: #E879F9 !important; font-weight: bold !important; }
                
                /* Base Window & Container Backgrounds (Gray Scale) */
                QMainWindow, QDialog, QWidget { background-color: #2B2B2B !important; }
                
                /* Override specific light backgrounds injected by code or Qt */
                #functionModulesGroup, #sampleInfoGroup { background-color: #333337 !important; border: 1px solid #555555 !important; }
                QLabel#currentSampleLabel, QLabel#sampleStatsLabel, QLabel#projectInfoLabel { background-color: transparent !important; }
                
                /* Force QListWidget inside sample info to be dark gray */
                QListWidget#sampleList { background-color: #1E1E1E !important; border: 1px solid #555555 !important; }
                QListWidget#sampleList::item { color: #E879F9 !important; }
                QListWidget#sampleList::item:selected { background-color: #4A4A4A !important; color: #F5D0FE !important; }
                QListWidget#sampleList::item:hover { background-color: #2D2D30 !important; }
                
                /* Menus and Status Bars */
                QMenuBar, QStatusBar { background-color: #2D2D30 !important; border-color: #555555 !important; }
                QMenuBar, QMenuBar::item, QMenu::item { color: #E879F9 !important; }
                QMenuBar::item:selected { background-color: #3E3E42 !important; }
                QMenu { background-color: #2D2D30 !important; border: 1px solid #555555 !important; }
                QMenu::item:selected { background-color: #3E3E42 !important; }
                
                /* Panels, Groups, ScrollAreas */
                QGroupBox, QScrollArea, QListWidget, QFrame { background-color: #333337 !important; border: 1px solid #555555 !important; }
                QGroupBox::title { color: #E879F9 !important; }
                
                /* Labels */
                QLabel { color: #E879F9 !important; background-color: transparent !important; }
                
                /* Buttons */
                QPushButton { background-color: #45454D !important; border: 1px solid #6A6A6A !important; color: #E879F9 !important; }
                QPushButton:hover { background-color: #5C5C61 !important; border-color: #E879F9 !important; }
                QPushButton:pressed { background-color: #68686D !important; }
                QPushButton:disabled { color: #777777 !important; border-color: #444444 !important; }
                
                /* Input Fields */
                QLineEdit, QTextEdit, QPlainTextEdit, QComboBox { background-color: #1E1E1E !important; color: #E879F9 !important; border: 1px solid #555555 !important; selection-background-color: #6A6A6A !important; selection-color: #F5D0FE !important; }
                QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus { border: 1px solid #E879F9 !important; }
                
                /* Project List Items (Custom Widget) */
                ProjectItemWidget { background-color: #333337 !important; border: 1px solid #555555 !important; }
                ProjectItemWidget:hover { background-color: #3E3E42 !important; border-color: #E879F9 !important; }
                
                /* Checkboxes and RadioButtons */
                QCheckBox::indicator, QRadioButton::indicator { border: 1px solid #E879F9 !important; background-color: #2B2B2B !important; }
                QCheckBox::indicator:checked, QRadioButton::indicator:checked { background-color: #6A6A6A !important; }
                
                /* Scrollbars */
                QScrollBar:vertical { background: #2B2B2B !important; width: 12px; }
                QScrollBar::handle:vertical { background: #555555 !important; border-radius: 6px; }
                QScrollBar::handle:vertical:hover { background: #6A6A6A !important; }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { background: none; }
                
                /* Progress Bar */
                QProgressBar { border: 1px solid #555555 !important; background-color: #2B2B2B !important; text-align: center; }
                QProgressBar::chunk { background-color: #6A6A6A !important; }
                
                /* Tables/Trees */
                QTableView, QTreeView { background-color: #1E1E1E !important; alternate-background-color: #252526 !important; gridline-color: #555555 !important; }
                QHeaderView::section { background-color: #333337 !important; color: #E879F9 !important; border: 1px solid #555555 !important; }
            """
            app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt5") + dark_override_qss)
            
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor("#2B2B2B"))
            palette.setColor(QPalette.WindowText, QColor("#E879F9"))
            palette.setColor(QPalette.Base, QColor("#1E1E1E"))
            palette.setColor(QPalette.AlternateBase, QColor("#252526"))
            palette.setColor(QPalette.ToolTipBase, QColor("#333337"))
            palette.setColor(QPalette.ToolTipText, QColor("#E879F9"))
            palette.setColor(QPalette.Text, QColor("#E879F9"))
            palette.setColor(QPalette.Button, QColor("#45454D"))
            palette.setColor(QPalette.ButtonText, QColor("#E879F9"))
            palette.setColor(QPalette.BrightText, QColor("#F5D0FE"))
            palette.setColor(QPalette.Link, QColor("#E879F9"))
            palette.setColor(QPalette.Highlight, QColor("#6A6A6A"))
            palette.setColor(QPalette.HighlightedText, QColor("#F5D0FE"))
            app.setPalette(palette)
            cls._refresh_existing_widget_styles(app)
            
        else:
            cls._disable_dark_patch()
            # Restore original light theme design (only dark theme keeps custom overrides).
            app.setStyleSheet("")

            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(240, 240, 240))
            palette.setColor(QPalette.WindowText, Qt.black)
            palette.setColor(QPalette.Base, Qt.white)
            palette.setColor(QPalette.AlternateBase, QColor(233, 231, 227))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.black)
            palette.setColor(QPalette.Text, Qt.black)
            palette.setColor(QPalette.Button, QColor(240, 240, 240))
            palette.setColor(QPalette.ButtonText, Qt.black)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.HighlightedText, Qt.white)
            app.setPalette(palette)

    @classmethod
    def get_current_theme(cls):
        return cls._current_theme
