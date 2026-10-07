#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Magic Seg 服务模块

集成 Segment Anything Model (SAM) 进行医学影像分割。
"""

import os
import sys
import numpy as np
import cv2
import json
import threading
import time
from typing import List, Dict, Optional, Any, Union, Tuple
from functools import lru_cache
from src.core.path_config import get_base_dir

# Optional imports for Torch/SAM
try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    torch = None
    TORCH_AVAILABLE = False
    print(f"Warning: PyTorch not available. MagicSegService will be unavailable. Error: {e}")

# 添加 SAM 路径
# Prefer local models/segment-anything if it exists (for portability)
if not getattr(sys, 'frozen', False):
    SAM_PATH = os.path.join(str(get_base_dir()), "models", "segment-anything")

    # Fallback to dev path if local not found
    if not os.path.exists(SAM_PATH):
        dev_sam_path = r"F:\RSNA\segment-anything-main"
        if os.path.exists(dev_sam_path):
            SAM_PATH = dev_sam_path

    if os.path.exists(SAM_PATH) and SAM_PATH not in sys.path:
        sys.path.append(SAM_PATH)
else:
    # In frozen mode, segment_anything is packed as data under segment-anything-main/
    base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    SAM_PATH = base
    sam_pkg_dirs = [
        os.path.join(base, "segment-anything-main"),
    ]
    for d in sam_pkg_dirs:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)

try:
    import segment_anything
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except Exception as e:
    SAM_AVAILABLE = False
    print(f"Warning: segment_anything unavailable. MagicSegService will be unavailable. Error: {e}")

from src.utils.logger import get_logger
from src.services.dicom_service import DicomService
from src.services.database_service import DatabaseService

from src.utils.gpu_manager import gpu_manager

class MagicSegService:
    """
    Magic Seg 服务类 - 封装 SAM 模型操作
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(MagicSegService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized') and self.initialized:
            return
        
        self.logger = get_logger(__name__)
        self.dicom_service = DicomService()
        self.db_service = DatabaseService()
        
        self.available = TORCH_AVAILABLE and SAM_AVAILABLE
        self.device = "cpu"
        
        if self.available:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Register with GPU Manager
        if self.device == 'cuda':
            gpu_manager.register("magic_seg", self.release_gpu_resources)

        self.model_type = "vit_b"
        self.checkpoint_path = os.path.join(SAM_PATH, "sam_vit_b_01ec64.pth")
        
        self.sam = None
        self.predictor = None
        self.mask_generator = None
        
        self.initialized = True
        self.logger.info(f"MagicSegService initialized. Available: {self.available}, Device: {self.device}")

    def _emit_status(self, status_callback, message: str):
        if not callable(status_callback):
            return
        try:
            status_callback(str(message or ""))
        except Exception:
            pass

    def release_gpu_resources(self):
        """Callback to release GPU memory (move model to CPU)"""
        try:
            if self.sam is not None:
                self.logger.info("MagicSeg: Moving SAM model to CPU to free GPU...")
                self.sam.to(device='cpu', dtype=torch.float32)
                
                # We must clear predictor/generator because they might cache CUDA tensors
                self.predictor = None 
                self.mask_generator = None
                
                if torch and torch.cuda.is_available():
                    import time
                    for _ in range(3):
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.ipc_collect()
                        time.sleep(0.5)
                gc.collect()
                self.logger.info("MagicSeg: GPU resources released.")
        except Exception as e:
            self.logger.error(f"Error releasing GPU resources: {e}")

    def load_model(self, status_callback=None):
        """加载 SAM 模型 (优先尝试轻量级 ViT-B)"""
        if not self.available:
            self.logger.warning("MagicSegService is unavailable (missing torch or sam).")
            return

        # Request GPU access
        if self.device == 'cuda':
            self._emit_status(status_callback, "SAM 模型正在加载到显存，请稍候…")
            gpu_manager.request_gpu("magic_seg")

        # If model is loaded but on CPU (and we want CUDA), move it back
        if self.sam is not None:
            if self.device == 'cuda' and next(self.sam.parameters()).device.type == 'cpu':
                self.logger.info("MagicSeg: Moving SAM model back to CUDA...")
                self.sam.to(device='cuda', dtype=torch.float32)
                # Re-initialize wrappers
                self.predictor = SamPredictor(self.sam)
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=self.sam,
                    points_per_side=32,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100,
                )
            self._emit_status(status_callback, "SAM 模型已加载完成并可用。")
            return

        # ... (rest of loading logic) ...

        # 优先查找 ViT-B (Base) 模型，速度快且显存占用低
        vit_b_paths = [
            os.path.join(os.path.dirname(sys.executable), '_internal', 'assets', 'models', 'sam', 'sam_vit_b_01ec64.pth'),
            os.path.join(os.path.dirname(sys.executable), '_internal', 'assets', 'models', 'sam_vit_b_01ec64.pth'),
            os.path.join(os.path.dirname(sys.executable), 'models', 'sam', 'sam_vit_b_01ec64.pth'), # Standard Path
            os.path.join(os.path.dirname(sys.executable), 'models', 'sam_vit_b_01ec64.pth'),
            os.path.join(os.path.dirname(sys.executable), 'assets', 'models', 'sam_vit_b_01ec64.pth'),
            os.path.join(os.getcwd(), 'models', 'sam', 'sam_vit_b_01ec64.pth'),
            os.path.join(os.getcwd(), 'models', 'sam_vit_b_01ec64.pth'),
            os.path.join(SAM_PATH, 'models', 'sam', 'sam_vit_b_01ec64.pth'),
            os.path.join(SAM_PATH, 'assets', 'models', 'sam_vit_b_01ec64.pth'),
            os.path.join(SAM_PATH, 'models', 'sam_vit_b_01ec64.pth'),
            os.path.join(SAM_PATH, 'sam_vit_b_01ec64.pth')
        ]
        
        # 其次查找 ViT-H (Huge) 模型，精度高但极重
        vit_h_paths = [
            os.path.join(os.path.dirname(sys.executable), '_internal', 'assets', 'models', 'sam', 'sam_vit_h_4b8939.pth'),
            os.path.join(os.path.dirname(sys.executable), '_internal', 'assets', 'models', 'sam_vit_h_4b8939.pth'),
            os.path.join(os.path.dirname(sys.executable), 'models', 'sam', 'sam_vit_h_4b8939.pth'),
            os.path.join(os.path.dirname(sys.executable), 'models', 'sam_vit_h_4b8939.pth'),
            os.path.join(os.path.dirname(sys.executable), 'assets', 'models', 'sam_vit_h_4b8939.pth'),
            os.path.join(os.getcwd(), 'models', 'sam', 'sam_vit_h_4b8939.pth'),
            os.path.join(os.getcwd(), 'models', 'sam_vit_h_4b8939.pth'),
            os.path.join(SAM_PATH, 'models', 'sam', 'sam_vit_h_4b8939.pth'),
            os.path.join(SAM_PATH, 'assets', 'models', 'sam_vit_h_4b8939.pth'),
            os.path.join(SAM_PATH, 'models', 'sam_vit_h_4b8939.pth'),
            os.path.join(SAM_PATH, 'sam_vit_h_4b8939.pth')
        ]
        
        self.logger.info(f"Searching for SAM checkpoint (Priority: ViT-B > ViT-H)")
        
        found_path = None
        found_type = None
        
        # 1. Try ViT-B
        for p in vit_b_paths:
            if os.path.exists(p):
                found_path = p
                found_type = "vit_b"
                self.logger.info(f"Found ViT-B model at: {p}")
                break
        
        # 2. Try ViT-H if B not found
        if not found_path:
            for p in vit_h_paths:
                if os.path.exists(p):
                    found_path = p
                    found_type = "vit_h"
                    self.logger.info(f"Found ViT-H model at: {p}")
                    break
        
        if not found_path:
            self.logger.error("No SAM checkpoint found (checked vit_b and vit_h paths)")
            raise FileNotFoundError("SAM checkpoint not found. Please place 'sam_vit_b_01ec64.pth' or 'sam_vit_h_4b8939.pth' in assets/models/")

        self.checkpoint_path = found_path
        self.model_type = found_type

        # Validate file size
        file_size_mb = os.path.getsize(self.checkpoint_path) / (1024 * 1024)
        if self.model_type == "vit_h" and file_size_mb < 2000:
             self.logger.warning(f"SAM ViT-H checkpoint size ({file_size_mb:.1f} MB) is suspiciously small (expected > 2GB).")
        elif self.model_type == "vit_b" and file_size_mb < 300:
             self.logger.warning(f"SAM ViT-B checkpoint size ({file_size_mb:.1f} MB) is suspiciously small (expected ~375MB).")

        try:
            self.logger.info(f"Loading SAM model ({self.model_type}) from {self.checkpoint_path}...")
            # Ensure device is valid
            if self.device == 'cuda' and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = 'cpu'
                
            self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            self.sam.to(device=self.device, dtype=torch.float32)
            
            self.predictor = SamPredictor(self.sam)
            # Automatic generator is used for batch, predictor for interactive
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Minimum area
            )
            self.logger.info("SAM model loaded successfully.")
            self._emit_status(status_callback, "SAM 模型已加载完成并可用。")
        except Exception as e:
            self.logger.error(f"Failed to load SAM model: {e}", exc_info=True)
            # Try CPU fallback if CUDA failed
            if self.device == 'cuda':
                self.logger.info("Retrying with CPU...")
                try:
                    self.device = 'cpu'
                    self.sam.to(device='cpu', dtype=torch.float32)
                    self.predictor = SamPredictor(self.sam)
                    self.mask_generator = SamAutomaticMaskGenerator(model=self.sam)
                    self.logger.info("SAM model loaded successfully on CPU.")
                    self._emit_status(status_callback, "SAM 模型已在 CPU 上加载完成并可用。")
                except Exception as e2:
                    self.logger.error(f"Failed to load SAM model on CPU: {e2}")
                    raise

    def set_image(self, image_rgb, status_callback=None):
        """Set image for predictor"""
        self.load_model(status_callback=status_callback)
        self.predictor.set_image(image_rgb)

    def predict_mask(self, points, labels, status_callback=None):
        """Predict mask from points"""
        self.load_model(status_callback=status_callback)
        if not self.predictor:
            return None, None, None
            
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        return masks, scores, logits

    def process_dicom_file(self, file_path: str, project_id: str, user_id: str) -> Dict[str, Any]:
        """
        处理单个 DICOM 文件：转换 -> 分割 -> 统计 -> 存储 (Batch Mode)
        """
        try:
            # 1. 加载 DICOM
            dicom_data = self.dicom_service.load_dicom_file(file_path)
            if not dicom_data:
                return {'success': False, 'error': 'Failed to load DICOM'}

            # 2. 转换为图像 (RGB for SAM)
            image_gray = self.dicom_service.convert_to_image(dicom_data)
            if image_gray is None:
                return {'success': False, 'error': 'Failed to convert DICOM to image'}
            
            image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

            # 3. 运行分割 (自动模式)
            self.load_model()
            self.logger.info(f"Generating masks for {os.path.basename(file_path)}...")
            masks = self.mask_generator.generate(image_rgb)
            self.logger.info(f"Generated {len(masks)} masks.")

            # 4. 处理掩膜并计算统计信息
            processed_rois = []
            
            raw_pixels = dicom_data.pixel_array.astype(np.float32)
            slope = getattr(dicom_data, 'RescaleSlope', 1.0)
            intercept = getattr(dicom_data, 'RescaleIntercept', 0.0)
            hu_pixels = raw_pixels * slope + intercept

            for i, mask_data in enumerate(masks):
                binary_mask = mask_data['segmentation']
                stats = self._calculate_pixel_stats(hu_pixels, binary_mask)
                try:
                    spacing = getattr(dicom_data, 'PixelSpacing', None)
                    if spacing and len(spacing) >= 2:
                        row_mm = float(spacing[0])
                        col_mm = float(spacing[1])
                        if 'pixel_count' in stats:
                            stats['area_mm2'] = float(stats['pixel_count'] * row_mm * col_mm)
                    else:
                        stats['area_mm2'] = 0.0
                except Exception:
                    stats['area_mm2'] = 0.0
                
                contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    epsilon = 0.001 * cv2.arcLength(contour, True)
                    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    roi_info = {
                        'roi_id': f"sam_roi_{i}",
                        'source': 'magic_seg_sam',
                        'area': float(mask_data['area']),
                        'bbox': {
                            'x': int(mask_data['bbox'][0]),
                            'y': int(mask_data['bbox'][1]),
                            'width': int(mask_data['bbox'][2]),
                            'height': int(mask_data['bbox'][3])
                        },
                        'contour_points': approx_contour.reshape(-1, 2).tolist(),
                        'statistics': stats,
                        'sam_score': float(mask_data.get('predicted_iou', 0))
                    }
                    processed_rois.append(roi_info)

            # 5. 存储结果到数据库
            dicom_info = self.dicom_service.extract_dicom_info(dicom_data)
            
            session_id = self.dicom_service.create_dicom_session(
                user_id=user_id,
                file_path=file_path,
                dicom_info=dicom_info,
                roi_data=processed_rois,
                project_id=project_id
            )

            if session_id:
                return {
                    'success': True, 
                    'session_id': session_id, 
                    'roi_count': len(processed_rois),
                    'file_path': file_path
                }
            else:
                return {'success': False, 'error': 'Failed to save session to database'}

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def save_interactive_result(self, file_path, mask, project_id, user_id, dicom_info=None):
        """Save result from interactive session"""
        try:
            dicom_data = self.dicom_service.load_dicom_file(file_path)
            if not dicom_data:
                return False
            
            raw_pixels = dicom_data.pixel_array.astype(np.float32)
            slope = getattr(dicom_data, 'RescaleSlope', 1.0)
            intercept = getattr(dicom_data, 'RescaleIntercept', 0.0)
            hu_pixels = raw_pixels * slope + intercept
            
            stats = self._calculate_pixel_stats(hu_pixels, mask)
            try:
                spacing = getattr(dicom_data, 'PixelSpacing', None)
                if spacing and len(spacing) >= 2:
                    row_mm = float(spacing[0])
                    col_mm = float(spacing[1])
                    if 'pixel_count' in stats:
                        stats['area_mm2'] = float(stats['pixel_count'] * row_mm * col_mm)
                else:
                    stats['area_mm2'] = 0.0
            except Exception:
                stats['area_mm2'] = 0.0
            
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            processed_rois = []
            for i, contour in enumerate(contours):
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                x, y, w, h = cv2.boundingRect(contour)
                
                roi_info = {
                    'roi_id': f"sam_interactive_{i}_{int(time.time())}",
                    'source': 'magic_seg_interactive',
                    'area': float(cv2.contourArea(contour)),
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'contour_points': approx_contour.reshape(-1, 2).tolist(),
                    'statistics': stats
                }
                processed_rois.append(roi_info)
                
            if dicom_info is None:
                dicom_info = self.dicom_service.extract_dicom_info(dicom_data)
            session_id = self.dicom_service.create_dicom_session(
                user_id=user_id,
                file_path=file_path,
                dicom_info=dicom_info,
                roi_data=processed_rois,
                project_id=project_id
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to save interactive result: {e}")
            return False

    def _calculate_pixel_stats(self, hu_pixels: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """计算掩膜区域内的像素统计信息 (CT值)"""
        try:
            roi_pixels = hu_pixels[mask]
            
            if len(roi_pixels) == 0:
                return {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 
                    'median': 0.0, 'pixel_count': 0
                }
            
            return {
                'mean': float(np.mean(roi_pixels)),
                'std': float(np.std(roi_pixels)),
                'min': float(np.min(roi_pixels)),
                'max': float(np.max(roi_pixels)),
                'median': float(np.median(roi_pixels)),
                'pixel_count': int(len(roi_pixels))
            }
        except Exception as e:
            self.logger.error(f"Stats calculation error: {e}")
            return {}
    def release_gpu_resources(self):
        try:
            if self.predictor:
                self.predictor = None
            if self.mask_generator:
                self.mask_generator = None
            if self.sam:
                self.sam = None
            import torch
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except Exception:
            pass
