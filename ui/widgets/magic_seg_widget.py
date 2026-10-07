#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Magic Seg Widget (Interactive)

Provides interactive UI for Segment Anything Model (SAM) based segmentation.
Matches the interface of sam_app.py (Open, Prev, Next, Clear) + Save.
"""

import os
import sys
import time
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QMessageBox, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush

from src.services.magic_seg_service import MagicSegService
from src.services.dicom_service import DicomService
from src.utils.logger import get_logger
from src.core.event_bus import get_event_bus, EventNames

class InteractiveCanvas(QLabel):
    """Canvas for displaying image and handling clicks"""
    point_added = pyqtSignal(int, int, int) # x, y, label (1=FG, 0=BG)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #2E86AB;")
        
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.refresh_display()

    def refresh_display(self):
        if not self.original_pixmap:
            self.setText("No Image Loaded")
            return
        
        # Scale to fit
        w_avail = self.width()
        h_avail = self.height()
        
        if w_avail <= 0 or h_avail <= 0:
            return

        scaled_pixmap = self.original_pixmap.scaled(
            w_avail, h_avail, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        self.scale_factor = scaled_pixmap.width() / self.original_pixmap.width()
        self.offset_x = (w_avail - scaled_pixmap.width()) // 2
        self.offset_y = (h_avail - scaled_pixmap.height()) // 2
        
        self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.refresh_display()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if not self.original_pixmap:
            return

        x = event.x()
        y = event.y()
        
        # Map to image coords
        img_x = int((x - self.offset_x) / self.scale_factor)
        img_y = int((y - self.offset_y) / self.scale_factor)
        
        if 0 <= img_x < self.original_pixmap.width() and 0 <= img_y < self.original_pixmap.height():
            label = 1 if event.button() == Qt.LeftButton else 0
            self.point_added.emit(img_x, img_y, label)

class MagicSegWidget(QWidget):
    """
    Magic Seg Interface Widget - Interactive Mode
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger(__name__)
        self.service = MagicSegService()
        self.dicom_service = DicomService()
        
        self.current_project_id = None
        self.current_user_id = "default_user"
        
        self.file_paths = []
        self.current_idx = 0
        self.current_image_cv2 = None # RGB
        self.current_mask = None
        self.input_points = []
        self.input_labels = []

        self.event_bus = get_event_bus()
        self.event_bus.subscribe(EventNames.USER_ID_SELECTED, self.on_user_selected)

        self.init_ui()
        self.setAcceptDrops(True)

    def on_user_selected(self, data):
        if isinstance(data, dict):
            self.current_user_id = data.get('user_id')
        else:
            self.current_user_id = data
        self.log(f"Sample selected: {self.current_user_id}")

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Toolbar ---
        toolbar_frame = QFrame()
        toolbar_frame.setStyleSheet("background-color: #f0f0f0; border-bottom: 1px solid #ccc;")
        toolbar_layout = QHBoxLayout(toolbar_frame)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)

        self.btn_open = QPushButton("Open Folder")
        self.btn_open.clicked.connect(self.open_folder)
        toolbar_layout.addWidget(self.btn_open)

        toolbar_layout.addSpacing(20)

        self.btn_prev = QPushButton("< Prev Image")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_prev.setEnabled(False)
        toolbar_layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next Image >")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.setEnabled(False)
        toolbar_layout.addWidget(self.btn_next)

        toolbar_layout.addSpacing(20)

        self.btn_clear = QPushButton("Clear Points")
        self.btn_clear.clicked.connect(self.clear_points)
        toolbar_layout.addWidget(self.btn_clear)

        toolbar_layout.addSpacing(20)
        
        self.btn_save = QPushButton("Save ROI")
        self.btn_save.setStyleSheet("background-color: #007acc; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_roi)
        self.btn_save.setEnabled(False)
        toolbar_layout.addWidget(self.btn_save)

        toolbar_layout.addStretch()
        
        self.lbl_info = QLabel("Left Click: Foreground | Right Click: Background")
        self.lbl_info.setStyleSheet("color: #666;")
        toolbar_layout.addWidget(self.lbl_info)

        main_layout.addWidget(toolbar_frame)

        # --- Canvas ---
        self.canvas = InteractiveCanvas()
        self.canvas.point_added.connect(self.on_point_added)
        main_layout.addWidget(self.canvas)

        # --- Status Bar ---
        self.lbl_status = QLabel("Ready. Please select a project and open a DICOM folder.")
        self.lbl_status.setStyleSheet("padding: 5px; background-color: #E0E0E0;")
        main_layout.addWidget(self.lbl_status)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls]
        dicom_files = set()
        for p in paths:
            if os.path.isdir(p):
                found = self.dicom_service.find_dicom_files(p, recursive=True)
                for f in found:
                    dicom_files.add(f)
            elif os.path.isfile(p):
                found = self.dicom_service.find_dicom_files(p, recursive=False)
                for f in found:
                    dicom_files.add(f)
        if dicom_files:
            self.file_paths = sorted(list(dicom_files))
            self.current_idx = 0
            self.load_current_image()
            self.update_buttons()
            self.log(f"Loaded {len(self.file_paths)} DICOM files from drop")
            event.acceptProposedAction()
        else:
            QMessageBox.warning(self, "No Files", "No DICOM files found.")
            event.ignore()

    def set_current_project(self, project_id):
        self.current_project_id = project_id
        self.log(f"Project switched to: {project_id}")

    def log(self, msg):
        self.lbl_status.setText(msg)
        self.logger.info(msg)
        win = self.window()
        if hasattr(win, "on_model_status_message"):
            try:
                win.on_model_status_message(msg)
            except Exception:
                pass

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder:
            return

        self.log(f"Scanning {folder}...")
        self.file_paths = self.dicom_service.find_dicom_files(folder, recursive=True)
        
        if not self.file_paths:
            QMessageBox.warning(self, "No Files", "No DICOM files found in folder.")
            return

        self.current_idx = 0
        self.load_current_image()
        self.update_buttons()

    def update_buttons(self):
        self.btn_prev.setEnabled(self.current_idx > 0)
        self.btn_next.setEnabled(self.current_idx < len(self.file_paths) - 1)
        self.btn_save.setEnabled(self.current_mask is not None and self.current_project_id is not None)

    def load_current_image(self):
        if not self.file_paths:
            return

        file_path = self.file_paths[self.current_idx]
        self.log(f"Loading {os.path.basename(file_path)} ({self.current_idx + 1}/{len(self.file_paths)})...")
        
        try:
            dicom_data = self.dicom_service.load_dicom_file(file_path)
            if not dicom_data:
                self.log("Failed to load DICOM.")
                return

            image_gray = self.dicom_service.convert_to_image(dicom_data)
            if image_gray is None:
                self.log("Failed to convert image.")
                return

            # Convert to RGB for SAM and display
            self.current_image_cv2 = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
            
            # Reset state
            self.input_points = []
            self.input_labels = []
            self.current_mask = None
            
            # Set image in SAM (async better, but for now sync)
            self.service.set_image(self.current_image_cv2, status_callback=self.log)
            
            self.display_image()
            self.log(f"Loaded {os.path.basename(file_path)}")

        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            self.log(f"Error: {e}")

    def display_image(self):
        if self.current_image_cv2 is None:
            return

        # Start with base image
        display_img = self.current_image_cv2.copy()

        # Overlay Mask
        if self.current_mask is not None:
            # Create colored mask (Green)
            color_mask = np.zeros_like(display_img)
            color_mask[self.current_mask] = [0, 255, 0]
            display_img = cv2.addWeighted(display_img, 1.0, color_mask, 0.5, 0)

        # Draw Points
        for i, (px, py) in enumerate(self.input_points):
            color = (0, 255, 0) if self.input_labels[i] == 1 else (255, 0, 0) # Green for FG, Red for BG
            cv2.circle(display_img, (px, py), 5, color, -1)
            cv2.circle(display_img, (px, py), 6, (255, 255, 255), 1)

        # Convert to QPixmap
        h, w, c = display_img.shape
        bytes_per_line = 3 * w
        qimg = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.canvas.set_image(QPixmap.fromImage(qimg))
        self.update_buttons()

    def on_point_added(self, x, y, label):
        self.input_points.append([x, y])
        self.input_labels.append(label)
        
        # Run prediction
        points_np = np.array(self.input_points)
        labels_np = np.array(self.input_labels)
        
        masks, scores, _ = self.service.predict_mask(points_np, labels_np, status_callback=self.log)
        
        if masks is not None:
            best_idx = np.argmax(scores)
            self.current_mask = masks[best_idx]
            self.display_image()
            self.log(f"Mask updated (Score: {scores[best_idx]:.3f})")
        else:
            self.log("Prediction failed.")

    def clear_points(self):
        self.input_points = []
        self.input_labels = []
        self.current_mask = None
        self.display_image()
        self.log("Points cleared.")

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_image()
            self.update_buttons()

    def next_image(self):
        if self.current_idx < len(self.file_paths) - 1:
            self.current_idx += 1
            self.load_current_image()
            self.update_buttons()

    def save_roi(self):
        if self.current_mask is None:
            return
        
        if not self.current_project_id:
            QMessageBox.warning(self, "Warning", "No project selected!")
            return
        
        file_path = self.file_paths[self.current_idx]
        dicom_data = None
        dicom_info = None
        stats_text = ""
        try:
            dicom_data = self.dicom_service.load_dicom_file(file_path)
            if dicom_data:
                dicom_info = self.dicom_service.extract_dicom_info(dicom_data)
            if dicom_data and hasattr(dicom_data, 'pixel_array'):
                import numpy as np
                raw = dicom_data.pixel_array.astype(np.float32)
                slope = float(getattr(dicom_data, 'RescaleSlope', 1.0))
                intercept = float(getattr(dicom_data, 'RescaleIntercept', 0.0))
                hu = raw * slope + intercept
                mask_bool = self.current_mask.astype(np.uint8) > 0
                roi_pixels = hu[mask_bool]
                if roi_pixels.size > 0:
                    mean_val = float(np.mean(roi_pixels))
                    std_val = float(np.std(roi_pixels))
                    min_val = float(np.min(roi_pixels))
                    max_val = float(np.max(roi_pixels))
                    median_val = float(np.median(roi_pixels))
                    pixel_count = int(roi_pixels.size)
                    area_mm2 = 0.0
                    spacing = getattr(dicom_data, 'PixelSpacing', None)
                    try:
                        if spacing and len(spacing) >= 2:
                            row_mm = float(spacing[0])
                            col_mm = float(spacing[1])
                            area_mm2 = float(pixel_count * row_mm * col_mm)
                    except Exception:
                        area_mm2 = 0.0
                    stats_text = (
                        f"文件: {os.path.basename(file_path)}\n"
                        f"选定像素数: {pixel_count}\n"
                        f"面积(mm²): {area_mm2:.2f}\n"
                        f"均值: {mean_val:.2f} HU\n"
                        f"标准差: {std_val:.2f} HU\n"
                        f"最小值: {min_val:.2f} HU\n"
                        f"最大值: {max_val:.2f} HU\n"
                        f"中位数: {median_val:.2f} HU"
                    )
                else:
                    stats_text = "选区为空，无法计算统计信息"
            else:
                stats_text = "无法读取DICOM像素数据"
        except Exception as e:
            stats_text = f"统计信息计算失败: {str(e)}"

        di = dicom_info or {}
        def _v(key, label, suffix=""):
            val = di.get(key)
            if val in (None, "", "None"):
                return None
            s = str(val)
            return f"  {label}: {s}{suffix}"
        dcm_lines = []
        for item in [
            ("patient_name", "患者姓名"),
            ("patient_id", "患者ID"),
            ("study_date", "检查日期"),
            ("modality", "成像模式"),
            ("body_part_examined", "检查部位"),
            ("study_description", "检查描述"),
            ("series_description", "系列描述"),
        ]:
            line = _v(item[0], item[1])
            if line:
                dcm_lines.append(line)
        mfg = di.get("manufacturer")
        mdl = di.get("manufacturer_model")
        if mfg not in (None, "", "None") or mdl not in (None, "", "None"):
            dcm_lines.append(f"  制造商: {_v('manufacturer','') or ''} {_v('manufacturer_model','') or ''}".strip())
        ps = di.get("pixel_spacing")
        if ps not in (None, "", "None"):
            if isinstance(ps, (list, tuple)):
                dcm_lines.append(f"  像素间距: {ps[0]}×{ps[1]} mm")
            else:
                dcm_lines.append(f"  像素间距: {ps} mm")
        st = di.get("slice_thickness")
        if st not in (None, "", "None"):
            dcm_lines.append(f"  层厚: {st} mm")
        kvp = di.get("kvp")
        if kvp not in (None, "", "None"):
            dcm_lines.append(f"  kVp: {kvp}")
        rows = di.get("rows")
        cols = di.get("columns")
        if rows not in (None, "", "None") or cols not in (None, "", "None"):
            dcm_lines.append(f"  图像尺寸: {rows or '?'}×{cols or '?'}")

        dcm_block = "\n".join(dcm_lines) if dcm_lines else "  (无 DICOM 元数据)"
        dcm_summary = (
            "═══ DICOM 元数据 ═══\n"
            f"{dcm_block}\n"
            "─── ROI 统计 ───\n"
            f"  {stats_text}"
        )

        reply = QMessageBox.question(
            self,
            "确认保存 ROI",
            f"\u786e\u8ba4\u5c06\u5f53\u524dROI\u53ca DICOM \u5143\u6570\u636e\u4fdd\u5b58\u5230\u9879\u76ee\uff1f\n\n{dcm_summary}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            self.log("保存已取消")
            return
        
        success = self.service.save_interactive_result(
            file_path, self.current_mask, self.current_project_id, self.current_user_id,
            dicom_info=dicom_info
        )
        
        if success:
            QMessageBox.information(self, "Success", "ROI saved successfully!")
            self.log(f"Saved ROI for {os.path.basename(file_path)}")
        else:
            QMessageBox.critical(self, "Error", "Failed to save ROI.")
            self.log("Failed to save ROI.")
