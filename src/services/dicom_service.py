#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DICOM处理服务模块

负责DICOM文件的读取、解析、ROI识别等功能。
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
from functools import lru_cache

if TYPE_CHECKING:
    import pydicom
    from pydicom.dataset import Dataset
else:
    Dataset = Any
import cv2
try:
    from skimage import filters, measure, morphology, segmentation
    from skimage.feature import graycomatrix, graycoprops
except Exception:
    filters = None
    measure = None
    morphology = None
    segmentation = None
    graycomatrix = None
    graycoprops = None

try:
    from scipy import ndimage
except Exception:
    ndimage = None

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None
from PIL import Image

from src.utils.logger import get_logger
from src.services.database_service import DatabaseService
from src.services.project_storage_service import ProjectStorageService


class DicomService:
    """
    DICOM处理服务类
    """
    
    def __init__(self, db_service: DatabaseService = None):
        """
        初始化DICOM服务
        
        Args:
            db_service: 数据库服务实例
        """
        self.db_service = db_service or DatabaseService()
        self.storage_service = ProjectStorageService(self.db_service)
        self.logger = get_logger(__name__)
        self._image_cache = {}
        self._dicom_cache = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = 50  # 最大缓存图像数量
        self._pydicom = None

    def _get_pydicom(self):
        if self._pydicom is not None:
            return self._pydicom
        try:
            import pydicom as _pydicom

            self._pydicom = _pydicom
            return _pydicom
        except Exception as e:
            self.logger.error(f"pydicom 导入失败: {e}")
            raise RuntimeError(
                "pydicom 导入失败（常见原因：pylibjpeg/libjpeg 插件缺少 _libjpeg）。"
                "如不需要 JPEG 解码，可卸载 pylibjpeg* / python-libjpeg；"
                "如需要解码，请安装与 Python 版本匹配的 _libjpeg 组件。"
            )
        
    def clear_cache(self):
        """清空所有缓存"""
        with self._cache_lock:
            self._image_cache.clear()
            self._dicom_cache.clear()
            gc.collect()
            self.logger.info("所有缓存已清空")
            
    def _get_cache_key(self, file_path: str, preprocessing: Dict = None) -> str:
        """生成缓存键"""
        import hashlib
        key_data = f"{file_path}_{str(preprocessing or {})}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def find_dicom_files(self, search_path: str, recursive: bool = True) -> List[str]:
        """
        在指定路径中查找DICOM文件
        
        Args:
            search_path: 搜索路径（文件或文件夹）
            recursive: 是否递归搜索子文件夹
        
        Returns:
            找到的DICOM文件路径列表
        """
        dicom_files = set()  # 使用集合避免重复
        search_path = os.path.abspath(search_path)
        
        try:
            # 如果是文件直接检查
            if os.path.isfile(search_path):
                if self._is_dicom_file(search_path):
                    return [search_path]
                return []
            
            # 如果是文件夹，遍历搜索
            if os.path.isdir(search_path):
                if recursive:
                    for root, _, files in os.walk(search_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # 规范化路径以确保唯一性
                            norm_path = os.path.normpath(file_path)
                            if self._is_dicom_file(norm_path):
                                dicom_files.add(norm_path)
                else:
                    for file in os.listdir(search_path):
                        file_path = os.path.join(search_path, file)
                        if os.path.isfile(file_path):
                            norm_path = os.path.normpath(file_path)
                            if self._is_dicom_file(norm_path):
                                dicom_files.add(norm_path)
            
            result_files = sorted(list(dicom_files))
            self.logger.info(f"在路径 {search_path} 中找到 {len(result_files)} 个DICOM文件")
            return result_files
            
        except Exception as e:
            self.logger.error(f"搜索DICOM文件失败: {e}")
            return []
    
    def _is_dicom_file(self, file_path: str) -> bool:
        """
        检查文件是否为DICOM文件
        
        Args:
            file_path: 文件路径
        
        Returns:
            是否为DICOM文件
        """
        try:
            # 检查文件是否存在且不为空
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return False
            
            # 首先检查文件扩展名，排除明显的非DICOM文件
            file_ext = os.path.splitext(file_path)[1].lower()
            non_dicom_extensions = ['.txt', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.pdf', '.doc', '.docx']
            if file_ext in non_dicom_extensions:
                return False
            
            # 尝试读取文件头部来判断是否为DICOM
            with open(file_path, 'rb') as f:
                # 读取前132字节
                header = f.read(132)
                
                # DICOM文件在128字节偏移处有"DICM"标识
                if len(header) >= 132 and header[128:132] == b'DICM':
                    return True
                
                # 有些DICOM文件可能没有前导码，需要使用force=True尝试读取
            f.seek(0)
            try:
                # 尝试用pydicom读取，使用force=True但只读取元数据
                pydicom = self._get_pydicom()
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
                
                # 严格检查是否包含关键DICOM标签，避免将随机二进制文件误判为DICOM
                # 检查SOPClassUID, StudyInstanceUID, PatientName, Modality等常见标签
                required_tags = ['SOPClassUID', 'StudyInstanceUID', 'Modality', 'PatientName', 'SeriesInstanceUID']
                matches = 0
                for tag in required_tags:
                    if hasattr(ds, tag):
                        matches += 1
                
                # 只要有任何一个关键标签，就认为是有效DICOM
                if matches > 0:
                    return True
                
                # 如果没有关键属性，检查是否包含Group 0002 (File Meta Information)
                if hasattr(ds, 'file_meta') and ds.file_meta:
                    return True
                    
                return False
            except:
                return False
                    
        except Exception as e:
            self.logger.debug(f"检查DICOM文件失败 {file_path}: {e}")
            return False
    
    def get_dicom_session_details(self, session_id: str) -> Optional[Dict]:
        """
        获取DICOM会话详情
        
        Args:
            session_id: 会话ID
        
        Returns:
            会话详情字典，包含dicom_info和roi_data
        """
        try:
            # 获取会话基本信息
            session_query = """
                SELECT session_id, user_id, file_path, dicom_info, created_at
                FROM dicom_sessions
                WHERE session_id = ?
            """
            
            session_results = self.db_service.execute_query(session_query, (session_id,))
            self.logger.debug(f"数据库查询结果类型: {type(session_results)}, 内容: {session_results}")
            
            if not session_results or len(session_results) == 0:
                self.logger.debug(f"未找到会话: {session_id}")
                return None
            
            session_data = session_results[0]
            self.logger.debug(f"会话数据类型: {type(session_data)}, 内容: {session_data}")
            
            # 解析DICOM信息
            dicom_info = {}
            if session_data['dicom_info']:
                try:
                    dicom_info = json.loads(session_data['dicom_info'])
                except json.JSONDecodeError:
                    pass
            
            # 获取ROI数据
            roi_data = self.get_session_roi_data(session_id)
            self.logger.debug(f"ROI数据类型: {type(roi_data)}, 长度: {len(roi_data) if isinstance(roi_data, list) else 'N/A'}")
            
            result = {
                'session_id': session_data['session_id'],
                'user_id': session_data['user_id'],
                'file_path': session_data['file_path'],
                'dicom_info': dicom_info,
                'roi_data': roi_data,
                'created_at': session_data['created_at']
            }
            self.logger.debug(f"返回结果类型: {type(result)}")
            return result
            
        except Exception as e:
            self.logger.error(f"获取DICOM会话详情失败: {e}")
            return None
    
    def load_dicom_file(self, file_path: str, use_cache: bool = True, force: bool = True) -> Optional[Dataset]:
        """
        加载DICOM文件
        
        Args:
            file_path: DICOM文件路径
            use_cache: 是否使用缓存
            force: 是否强制读取非标准DICOM文件
        
        Returns:
            DICOM数据集，如果加载失败则返回None
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                self.logger.error(f"DICOM文件不存在: {file_path}")
                return None
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                self.logger.error(f"DICOM文件为空: {file_path}")
                return None
            
            # 生成缓存键
            cache_key = f"dicom_{os.path.basename(file_path)}_{os.path.getmtime(file_path)}_{file_size}"
            
            # 检查缓存
            if use_cache and hasattr(self, '_dicom_cache') and cache_key in self._dicom_cache:
                self.logger.debug(f"从缓存获取DICOM: {cache_key}")
                return self._dicom_cache[cache_key]
            
            # 尝试不同的读取方法
            dicom_data = None
            pydicom = self._get_pydicom()
            
            # 方法1: 标准读取
            try:
                dicom_data = pydicom.dcmread(file_path, force=force)
                self.logger.debug(f"标准方法读取DICOM成功: {file_path}")
            except Exception as e:
                self.logger.debug(f"标准读取失败: {e}")
            
            # 方法2: 如果标准读取失败，尝试强制读取
            if dicom_data is None and not force:
                try:
                    dicom_data = pydicom.dcmread(file_path, force=True)
                    self.logger.debug(f"强制读取DICOM成功: {file_path}")
                except Exception as e:
                    self.logger.debug(f"强制读取失败: {e}")
            
            # 方法3: 尝试读取为文件对象
            if dicom_data is None:
                try:
                    with open(file_path, 'rb') as f:
                        dicom_data = pydicom.dcmread(f, force=True)
                    self.logger.debug(f"文件对象读取DICOM成功: {file_path}")
                except Exception as e:
                    self.logger.debug(f"文件对象读取失败: {e}")
            
            if dicom_data is None:
                self.logger.error(f"所有读取方法都失败: {file_path}")
                return None
            
            # 验证DICOM数据的基本完整性
            if not self._validate_dicom_data(dicom_data):
                self.logger.warning(f"DICOM数据验证失败: {file_path}")
                # 仍然返回数据，但记录警告
            
            # 初始化DICOM缓存（如果不存在）
            if not hasattr(self, '_dicom_cache'):
                self._dicom_cache = {}
            
            # 缓存DICOM数据（如果启用缓存）
            if use_cache:
                with self._cache_lock:
                    # 检查缓存大小，如果超过限制则清理最旧的缓存
                    if len(self._dicom_cache) >= self._max_cache_size:
                        # 移除最旧的缓存项
                        oldest_key = next(iter(self._dicom_cache))
                        del self._dicom_cache[oldest_key]
                        gc.collect()
                    
                    self._dicom_cache[cache_key] = dicom_data
                    self.logger.debug(f"DICOM已缓存: {cache_key}")
            
            self.logger.info(f"成功加载DICOM文件: {file_path}")
            return dicom_data
            
        except Exception as e:
            self.logger.error(f"加载DICOM文件失败: {e}")
            return None
    
    def extract_dicom_info(self, dicom_data: Dataset) -> Dict[str, Any]:
        """
        提取DICOM文件信息，支持多厂家设备兼容性
        
        Args:
            dicom_data: DICOM数据集
        
        Returns:
            DICOM信息字典
        """
        try:
            def convert_dicom_value(value, visited=None):
                """转换DICOM值为可JSON序列化的格式，避免循环引用"""
                if visited is None:
                    visited = set()
                
                # 检查循环引用
                if id(value) in visited:
                    return "<circular reference>"
                
                # 添加到已访问集合
                visited.add(id(value))
                
                try:
                    if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                        # 处理MultiValue或其他可迭代类型
                        try:
                            # 递归转换列表中的每个元素
                            result = []
                            for item in value:
                                # 对于列表中的元素，使用新的visited集合，避免误判
                                result.append(convert_dicom_value(item, visited.copy()))
                            visited.remove(id(value))
                            return result
                        except:
                            visited.remove(id(value))
                            return str(value)
                    elif hasattr(value, 'original_string'):
                        # 处理DS/IS/CS等DICOM特定类型
                        result = str(value)
                        visited.remove(id(value))
                        return result
                    else:
                        result = str(value) if value is not None else ''
                        visited.remove(id(value))
                        return result
                except Exception as e:
                    visited.discard(id(value))
                    return f"<conversion error: {str(e)}>"
            
            def get_pixel_spacing_multi_vendor(dicom_data):
                """获取像素间距，支持多厂家格式"""
                pixel_spacing_sources = []
                
                # 标准PixelSpacing
                if hasattr(dicom_data, 'PixelSpacing') and dicom_data.PixelSpacing:
                    pixel_spacing_sources.append(('PixelSpacing', dicom_data.PixelSpacing))
                
                # ImagerPixelSpacing（主要用于DR/CR图像）
                if hasattr(dicom_data, 'ImagerPixelSpacing') and dicom_data.ImagerPixelSpacing:
                    pixel_spacing_sources.append(('ImagerPixelSpacing', dicom_data.ImagerPixelSpacing))
                
                # NominalScannedPixelSpacing（某些厂家使用）
                if hasattr(dicom_data, 'NominalScannedPixelSpacing') and dicom_data.NominalScannedPixelSpacing:
                    pixel_spacing_sources.append(('NominalScannedPixelSpacing', dicom_data.NominalScannedPixelSpacing))
                
                # DetectorElementSpacing（某些设备使用）
                if hasattr(dicom_data, 'DetectorElementSpacing') and dicom_data.DetectorElementSpacing:
                    pixel_spacing_sources.append(('DetectorElementSpacing', dicom_data.DetectorElementSpacing))
                
                return pixel_spacing_sources
            
            def get_slice_thickness_multi_vendor(dicom_data):
                """获取层厚，支持多厂家格式"""
                thickness_sources = []
                
                # 标准SliceThickness
                if hasattr(dicom_data, 'SliceThickness') and dicom_data.SliceThickness:
                    thickness_sources.append(('SliceThickness', dicom_data.SliceThickness))
                
                # SpacingBetweenSlices（层间距）
                if hasattr(dicom_data, 'SpacingBetweenSlices') and dicom_data.SpacingBetweenSlices:
                    thickness_sources.append(('SpacingBetweenSlices', dicom_data.SpacingBetweenSlices))
                
                # ReconstructionDiameter（某些CT使用）
                if hasattr(dicom_data, 'ReconstructionDiameter') and dicom_data.ReconstructionDiameter:
                    thickness_sources.append(('ReconstructionDiameter', dicom_data.ReconstructionDiameter))
                
                return thickness_sources
            
            # 获取厂家信息
            manufacturer = convert_dicom_value(getattr(dicom_data, 'Manufacturer', ''))
            manufacturer_model = convert_dicom_value(getattr(dicom_data, 'ManufacturerModelName', ''))
            software_version = convert_dicom_value(getattr(dicom_data, 'SoftwareVersions', ''))
            
            # 获取多厂家兼容的像素间距
            pixel_spacing_sources = get_pixel_spacing_multi_vendor(dicom_data)
            # 处理pixel_spacing，确保它是列表格式
            pixel_spacing = []
            if pixel_spacing_sources:
                spacing_val = pixel_spacing_sources[0][1]
                if spacing_val:
                    # 将DSfloat/DSdecimal转换为float列表
                    try:
                        pixel_spacing = [float(x) for x in spacing_val]
                    except:
                        pixel_spacing = convert_dicom_value(spacing_val)
            
            pixel_spacing_source = pixel_spacing_sources[0][0] if pixel_spacing_sources else 'None'
            
            # 获取多厂家兼容的层厚
            thickness_sources = get_slice_thickness_multi_vendor(dicom_data)
            slice_thickness = thickness_sources[0][1] if thickness_sources else ''
            slice_thickness_source = thickness_sources[0][0] if thickness_sources else 'None'
            
            info = {
                'patient_id': convert_dicom_value(getattr(dicom_data, 'PatientID', '')),
                'patient_name': convert_dicom_value(getattr(dicom_data, 'PatientName', '')),
                'study_date': convert_dicom_value(getattr(dicom_data, 'StudyDate', '')),
                'study_time': convert_dicom_value(getattr(dicom_data, 'StudyTime', '')),
                'study_instance_uid': convert_dicom_value(getattr(dicom_data, 'StudyInstanceUID', '')),
                'series_instance_uid': convert_dicom_value(getattr(dicom_data, 'SeriesInstanceUID', '')),
                'sop_instance_uid': convert_dicom_value(getattr(dicom_data, 'SOPInstanceUID', '')),
                'modality': convert_dicom_value(getattr(dicom_data, 'Modality', '')),
                'body_part_examined': convert_dicom_value(getattr(dicom_data, 'BodyPartExamined', '')),
                'image_type': convert_dicom_value(getattr(dicom_data, 'ImageType', [])),
                'rows': int(getattr(dicom_data, 'Rows', 0)) if getattr(dicom_data, 'Rows', 0) else 0,
                'columns': int(getattr(dicom_data, 'Columns', 0)) if getattr(dicom_data, 'Columns', 0) else 0,
                'accession_number': convert_dicom_value(getattr(dicom_data, 'AccessionNumber', '')),
                'study_description': convert_dicom_value(getattr(dicom_data, 'StudyDescription', '')),
                'series_description': convert_dicom_value(getattr(dicom_data, 'SeriesDescription', '')),
                'series_number': convert_dicom_value(getattr(dicom_data, 'SeriesNumber', '')),
                'instance_number': convert_dicom_value(getattr(dicom_data, 'InstanceNumber', '')),
                
                # 多厂家兼容的像素间距
                'pixel_spacing': pixel_spacing,
                'pixel_spacing_source': pixel_spacing_source,
                
                # 多厂家兼容的层厚
                'slice_thickness': convert_dicom_value(slice_thickness),
                'slice_thickness_source': slice_thickness_source,
                
                # 厂家信息
                'manufacturer': manufacturer,
                'manufacturer_model': manufacturer_model,
                'software_version': software_version,
                
                # 其他标准信息
                'window_center': convert_dicom_value(getattr(dicom_data, 'WindowCenter', '')),
                'window_width': convert_dicom_value(getattr(dicom_data, 'WindowWidth', '')),
                'bits_allocated': int(getattr(dicom_data, 'BitsAllocated', 0)) if getattr(dicom_data, 'BitsAllocated', 0) else 0,
                'bits_stored': int(getattr(dicom_data, 'BitsStored', 0)) if getattr(dicom_data, 'BitsStored', 0) else 0,
                'high_bit': int(getattr(dicom_data, 'HighBit', 0)) if getattr(dicom_data, 'HighBit', 0) else 0,
                'pixel_representation': int(getattr(dicom_data, 'PixelRepresentation', 0)) if getattr(dicom_data, 'PixelRepresentation', 0) else 0,
                'photometric_interpretation': convert_dicom_value(getattr(dicom_data, 'PhotometricInterpretation', '')),
                'samples_per_pixel': int(getattr(dicom_data, 'SamplesPerPixel', 1)) if getattr(dicom_data, 'SamplesPerPixel', 1) else 1,
                
                # CT特有参数
                'rescale_intercept': convert_dicom_value(getattr(dicom_data, 'RescaleIntercept', '')),
                'rescale_slope': convert_dicom_value(getattr(dicom_data, 'RescaleSlope', '')),
                'kvp': convert_dicom_value(getattr(dicom_data, 'KVP', '')),
                'exposure_time': convert_dicom_value(getattr(dicom_data, 'ExposureTime', '')),
                'x_ray_tube_current': convert_dicom_value(getattr(dicom_data, 'XRayTubeCurrent', ''))
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"提取DICOM信息失败: {e}")
            return {}
    
    def convert_to_image(self, dicom_data: Dataset, use_cache: bool = True, 
                        fixed_window: bool = False, window_center: float = 120, 
                        window_width: float = 230) -> Optional[np.ndarray]:
        """
        将DICOM数据转换为图像数组
        
        Args:
            dicom_data: DICOM数据集
            use_cache: 是否使用缓存
            fixed_window: 是否使用固定窗宽窗位
            window_center: 固定窗位值
            window_width: 固定窗宽值
        
        Returns:
            图像数组，如果转换失败则返回None
        """
        try:
            # 生成缓存键
            sop_instance_uid = getattr(dicom_data, 'SOPInstanceUID', '')
            cache_key = f"image_{sop_instance_uid}_{fixed_window}_{window_center}_{window_width}"
            
            # 检查缓存
            if use_cache and cache_key in self._image_cache:
                self.logger.debug(f"从缓存获取图像: {cache_key}")
                return self._image_cache[cache_key].copy()
            
            # 检查像素数据是否存在
            if 'PixelData' not in dicom_data:
                self.logger.warning("DICOM文件缺少像素数据")
                return None
            
            # 获取像素数据
            pixel_array = dicom_data.pixel_array
            
            # 处理多帧图像（只取第一帧）
            if len(pixel_array.shape) > 2:
                pixel_array = pixel_array[0]
            
            # 应用Rescale Slope和Intercept（CT值转换）
            if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                slope = float(dicom_data.RescaleSlope)
                intercept = float(dicom_data.RescaleIntercept)
                pixel_array = pixel_array.astype(np.float32) * slope + intercept
            
            # 应用窗口/级别调整
            if fixed_window:
                # 使用固定窗宽窗位
                window_min = window_center - window_width / 2
                window_max = window_center + window_width / 2
            elif hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                # 使用DICOM文件中的窗宽窗位
                wc = dicom_data.WindowCenter
                ww = dicom_data.WindowWidth
                # 处理多值情况，取第一个值
                if isinstance(wc, (list, tuple)):
                    wc = float(wc[0])
                else:
                    wc = float(wc)
                if isinstance(ww, (list, tuple)):
                    ww = float(ww[0])
                else:
                    ww = float(ww)
                
                window_min = wc - ww / 2
                window_max = wc + ww / 2
            else:
                # 自动计算窗宽窗位
                pixel_min, pixel_max = pixel_array.min(), pixel_array.max()
                window_min = pixel_min
                window_max = pixel_max
            
            # 应用窗口调整
            pixel_array = np.clip(pixel_array, window_min, window_max)
            if window_max > window_min:
                pixel_array = ((pixel_array - window_min) / (window_max - window_min) * 255.0)
            else:
                pixel_array = np.zeros_like(pixel_array, dtype=np.float32)
            
            pixel_array = np.clip(pixel_array, 0, 255).astype(np.uint8)
            
            # 缓存图像（如果启用缓存）
            if use_cache:
                with self._cache_lock:
                    # 检查缓存大小，如果超过限制则清理最旧的缓存
                    if len(self._image_cache) >= self._max_cache_size:
                        # 移除最旧的缓存项
                        oldest_key = next(iter(self._image_cache))
                        del self._image_cache[oldest_key]
                        gc.collect()
                    
                    self._image_cache[cache_key] = pixel_array.copy()
                    self.logger.debug(f"图像已缓存: {cache_key}")
            
            return pixel_array
            
        except Exception as e:
            self.logger.error(f"DICOM转图像失败: {e}")
            return None
    
    def detect_roi_regions(self, image_array: np.ndarray, threshold: float = 0.5, 
                          method: str = 'otsu', min_area: int = 100, 
                          max_area: int = None, preprocessing: Dict = None, 
                          dicom_data = None, enable_ocr: bool = True,
                          kernel_size: int = 3, block_size: int = 9, c_value: int = 1,
                          min_circularity: float = 0.1, max_aspect_ratio: float = 5.0,
                          debug_mode: bool = False, save_intermediate: bool = False) -> List[Dict]:
        """
        检测图像中的ROI区域
        
        Args:
            image_array: 图像数组
            threshold: 检测阈值
            method: 检测方法 ('otsu', 'adaptive', 'watershed', 'edge', 'overlay')
            min_area: 最小面积阈值
            max_area: 最大面积阈值
            preprocessing: 预处理参数
            dicom_data: DICOM数据对象（当method为'overlay'时需要）
        
        Returns:
            ROI区域列表
        """
        try:
            if debug_mode:
                self.logger.info(f"开始ROI检测 - 输入图像尺寸: {image_array.shape}, 数据类型: {image_array.dtype}")
                self.logger.info(f"像素值范围: {image_array.min()} - {image_array.max()}, 平均值: {image_array.mean():.2f}")
            
            # 确保图像是灰度图
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                if debug_mode:
                    self.logger.info("图像从彩色转换为灰度")
            else:
                gray = image_array.copy()
                if debug_mode:
                    self.logger.info("图像已是灰度格式")
            
            if debug_mode:
                self.logger.info(f"灰度图像尺寸: {gray.shape}, 像素值范围: {gray.min()} - {gray.max()}")
            
            # 图像预处理
            processed_image = self._preprocess_image(gray, preprocessing or {})
            
            if debug_mode:
                self.logger.info(f"预处理后图像尺寸: {processed_image.shape}, 像素值范围: {processed_image.min()} - {processed_image.max()}")
                # 计算图像对比度
                contrast = processed_image.std()
                self.logger.info(f"图像对比度(标准差): {contrast:.2f}")
                
                # 检查图像是否过于均匀
                unique_values = len(np.unique(processed_image))
                self.logger.info(f"图像唯一像素值数量: {unique_values}")
                
                if unique_values < 10:
                    self.logger.warning("图像可能过于均匀，ROI检测可能困难")
                if contrast < 10:
                    self.logger.warning("图像对比度较低，可能影响ROI检测效果")
            
            # 优先尝试从DICOM RT tags提取ROI信息
            roi_regions = []
            if dicom_data is not None:
                # 首先尝试从RT Structure Set标签提取ROI
                roi_regions = self._extract_roi_from_rt_tags(dicom_data, min_area, max_area)
                
                if roi_regions:
                    self.logger.info(f"成功从RT tags提取到 {len(roi_regions)} 个ROI，跳过图像处理方法")
                else:
                    self.logger.info("RT tags中未找到ROI信息，尝试其他方法")
            
            # 如果RT tags没有提供ROI信息，则使用传统的图像处理方法
            if not roi_regions:
                if method == 'otsu':
                    roi_regions = self._detect_roi_otsu(processed_image, threshold, min_area, max_area, 
                                                       kernel_size, min_circularity, max_aspect_ratio, debug_mode)
                elif method == 'adaptive':
                    roi_regions = self._detect_roi_adaptive(processed_image, threshold, min_area, max_area,
                                                           kernel_size, block_size, c_value, min_circularity, 
                                                           max_aspect_ratio, debug_mode)
                elif method == 'watershed':
                    roi_regions = self._detect_roi_watershed(processed_image, threshold, min_area, max_area,
                                                            kernel_size, min_circularity, max_aspect_ratio, debug_mode)
                elif method == 'edge':
                    roi_regions = self._detect_roi_edge(processed_image, threshold, min_area, max_area,
                                                       kernel_size, min_circularity, max_aspect_ratio, debug_mode)
                elif method == 'overlay':
                    # 从DICOM覆盖层提取ROI
                    roi_regions = self._extract_roi_from_overlay(dicom_data, min_area, max_area)
                else:
                    roi_regions = self._detect_roi_otsu(processed_image, threshold, min_area, max_area,
                                                       kernel_size, min_circularity, max_aspect_ratio, debug_mode)
                
                # 为图像处理方法提取的ROI添加source标识
                for roi in roi_regions:
                    if 'source' not in roi:
                        roi['source'] = 'image_processing'
            
            # 保存中间结果（如果启用）
            if save_intermediate and debug_mode:
                try:
                    import os
                    debug_dir = 'debug_roi_detection'
                    os.makedirs(debug_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(debug_dir, 'processed_image.png'), processed_image)
                    self.logger.info(f"中间结果已保存到 {debug_dir} 目录")
                except Exception as e:
                    self.logger.warning(f"保存中间结果失败: {e}")
            
            # 为每个ROI添加像素统计信息和OCR识别
            for roi in roi_regions:
                if 'contour_points' in roi:
                    # 只对图像处理方法提取的ROI计算像素统计
                    if roi.get('source') != 'rt_tags':
                        statistics = self._calculate_roi_statistics(image_array, roi['contour_points'])
                        roi['statistics'] = statistics
                    
                    # 根据ROI来源决定是否进行OCR识别
                    if roi.get('source') == 'rt_tags':
                        # RT tags来源的ROI已包含结构化信息，不需要OCR
                        roi['text_info'] = {
                            'roi_name': roi.get('roi_name', ''),
                            'roi_description': roi.get('roi_description', ''),
                            'roi_type': roi.get('roi_type', ''),
                            'source': 'rt_structure_set'
                        }
                        self.logger.info(f"RT ROI信息: {roi.get('roi_name', 'Unknown')} ({roi.get('roi_type', 'Unknown')})")
                    elif enable_ocr:
                        # 对图像处理方法提取的ROI进行OCR识别
                        try:
                            from src.services.ocr_service import OCRService
                            ocr_service = OCRService()
                            ocr_result = ocr_service.extract_text_from_dicom_roi(
                                image_array, roi
                            )
                            roi['ocr_result'] = ocr_result
                            roi['text_info'] = {
                                'text': ocr_result.get('text', ''),
                                'confidence': ocr_result.get('confidence', 0.0),
                                'source': 'ocr_recognition'
                            }
                            self.logger.info(f"ROI区域OCR识别完成，检测到文本: {ocr_result.get('text', '无文本')[:50]}")
                        except Exception as e:
                            self.logger.warning(f"ROI区域OCR识别失败: {e}")
                            roi['ocr_result'] = {'text': '', 'confidence': 0.0, 'error': str(e)}
                            roi['text_info'] = {
                                'text': '',
                                'confidence': 0.0,
                                'source': 'ocr_failed',
                                'error': str(e)
                            }
            
            # 详细的调试信息输出
            if debug_mode:
                if not roi_regions:
                    self.logger.warning("=== ROI检测失败详细分析 ===")
                    self.logger.warning(f"检测方法: {method}")
                    self.logger.warning(f"图像尺寸: {processed_image.shape if 'processed_image' in locals() else 'N/A'}")
                    if 'processed_image' in locals():
                        self.logger.warning(f"像素值范围: [{processed_image.min()}, {processed_image.max()}]")
                        self.logger.warning(f"图像均值: {processed_image.mean():.2f}")
                        self.logger.warning(f"图像标准差: {processed_image.std():.2f}")
                    self.logger.warning(f"过滤参数: min_area={min_area}, max_area={max_area}")
                    self.logger.warning(f"形状参数: min_circularity={min_circularity}, max_aspect_ratio={max_aspect_ratio}")
                    self.logger.warning(f"阈值参数: threshold={threshold}")
                    self.logger.warning("建议解决方案:")
                    self.logger.warning("  1) 降低min_area参数 (当前: {}, 建议: 5-20)".format(min_area))
                    self.logger.warning("  2) 尝试不同的检测方法 (otsu, adaptive, watershed, edge)")
                    self.logger.warning("  3) 检查图像质量和对比度")
                    self.logger.warning("  4) 调整预处理参数")
                    self.logger.warning("=== 分析结束 ===")
                else:
                    self.logger.info(f"ROI检测成功: 找到 {len(roi_regions)} 个ROI区域")
                    for i, roi in enumerate(roi_regions):
                        self.logger.info(f"  ROI {i+1}: 面积={roi.get('area', 'N/A')}, 位置=({roi.get('bbox', {}).get('x', 'N/A')}, {roi.get('bbox', {}).get('y', 'N/A')})")
            
            self.logger.info(f"使用{method}方法检测到 {len(roi_regions)} 个ROI区域")
            return roi_regions
            
        except Exception as e:
            self.logger.error(f"ROI检测失败: {e}")
            if debug_mode:
                import traceback
                self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return []
        
    def process_dicom_batch(self, file_paths: List[str], user_id: str, 
                           roi_params: Dict = None, max_workers: int = 4) -> List[Dict]:
        """
        批量处理DICOM文件
        
        Args:
            file_paths: DICOM文件路径列表
            user_id: 用户ID
            roi_params: ROI检测参数
            max_workers: 最大并发工作线程数
        
        Returns:
            处理结果列表
        """
        import concurrent.futures
        from threading import Lock
        
        results = []
        results_lock = Lock()
        roi_params = roi_params or {}
        
        def process_single_file(file_path: str) -> Dict:
            """处理单个DICOM文件"""
            try:
                # 加载DICOM文件
                dicom_data = self.load_dicom_file(file_path)
                if not dicom_data:
                    return {'file_path': file_path, 'success': False, 'error': '无法加载DICOM文件'}
                
                # 提取DICOM信息
                dicom_info = self.extract_dicom_info(dicom_data)
                
                # 转换为图像
                image_array = self.convert_to_image(dicom_data, use_cache=True)
                if image_array is None:
                    return {'file_path': file_path, 'success': False, 'error': '图像转换失败'}
                
                # ROI检测
                roi_regions = self.detect_roi_regions(
                    image_array,
                    threshold=roi_params.get('threshold', 0.5),
                    method=roi_params.get('method', 'otsu'),
                    min_area=roi_params.get('min_area', 100),
                    max_area=roi_params.get('max_area'),
                    preprocessing=roi_params.get('preprocessing'),
                    dicom_data=dicom_data,
                    enable_ocr=roi_params.get('enable_ocr', True)
                )
                
                # 创建会话
                session_id = self.create_dicom_session(
                    user_id=user_id,
                    file_path=file_path,
                    dicom_info=dicom_info,
                    roi_data=roi_regions
                )
                
                return {
                    'file_path': file_path,
                    'success': True,
                    'session_id': session_id,
                    'dicom_info': dicom_info,
                    'roi_count': len(roi_regions)
                }
                
            except Exception as e:
                self.logger.error(f"批量处理文件失败 {file_path}: {e}")
                return {'file_path': file_path, 'success': False, 'error': str(e)}
        
        # 使用线程池并发处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_single_file, file_path): file_path 
                             for file_path in file_paths}
            
            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                with results_lock:
                    results.append(result)
        
        # 清理缓存以释放内存
        if len(file_paths) > 10:
            self.clear_cache()
        
        self.logger.info(f"批量处理完成，共处理 {len(file_paths)} 个文件")
        return results
        
    @lru_cache(maxsize=128)
    def _get_optimized_kernel(self, size: int, operation: str) -> np.ndarray:
        """
        获取优化的形态学核
        
        Args:
            size: 核大小
            operation: 操作类型
        
        Returns:
            形态学核
        """
        # 限制核大小范围，避免过大的核影响小ROI检测
        size = max(1, min(size, 7))  # 限制在1-7像素范围内
        
        if operation == 'ellipse':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        elif operation == 'cross':
            return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
        else:
            return np.ones((size, size), np.uint8)
            
    def optimize_memory_usage(self):
        """
        优化内存使用
        """
        # 清理缓存
        self.clear_cache()
        
        # 强制垃圾回收
        gc.collect()
        
        self.logger.info("内存优化完成")
    
    def _preprocess_image(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """
        图像预处理 - 优化版本，保护对比度并添加质量检查
        
        Args:
            image: 输入图像
            params: 预处理参数
        
        Returns:
            预处理后的图像
        """
        processed = image.copy()
        debug_mode = params.get('debug_mode', False)
        
        # 图像质量检查
        quality_info = self._assess_image_quality(processed, debug_mode)
        
        # 归一化 - 保护原始对比度
        if quality_info['needs_normalization']:
            processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if debug_mode:
                self.logger.info(f"应用归一化: 原始范围=[{image.min()}, {image.max()}] -> [0, 255]")
        else:
            processed = processed.astype(np.uint8)
            if debug_mode:
                self.logger.info("跳过归一化，保持原始对比度")
        
        # 自适应对比度增强
        if quality_info['low_contrast']:
            # 使用CLAHE（限制对比度自适应直方图均衡化）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            processed = clahe.apply(processed)
            if debug_mode:
                self.logger.info("应用CLAHE对比度增强")
        
        # 高斯滤波去噪 - 根据图像质量调整
        if params.get('gaussian_blur', False) or quality_info['noisy']:
            kernel_size = params.get('gaussian_kernel', 3 if quality_info['noisy'] else 5)
            processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)
            if debug_mode:
                self.logger.info(f"应用高斯滤波: 核大小={kernel_size}")
        
        # 直方图均衡化 - 仅在必要时使用
        if params.get('histogram_equalization', False) and not quality_info['low_contrast']:
            processed = cv2.equalizeHist(processed)
            if debug_mode:
                self.logger.info("应用直方图均衡化")
        
        # 对比度增强 - 自适应参数
        if params.get('contrast_enhancement', False):
            alpha = params.get('contrast_alpha', 1.2 if quality_info['low_contrast'] else 1.5)
            beta = params.get('contrast_beta', 10 if quality_info['low_contrast'] else 0)
            processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)
            if debug_mode:
                self.logger.info(f"应用对比度增强: alpha={alpha}, beta={beta}")
        
        # 边缘保持滤波
        if params.get('bilateral_filter', False):
            d = params.get('bilateral_d', 9)
            sigma_color = params.get('bilateral_sigma_color', 75)
            sigma_space = params.get('bilateral_sigma_space', 75)
            processed = cv2.bilateralFilter(processed, d, sigma_color, sigma_space)
            if debug_mode:
                self.logger.info(f"应用双边滤波: d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}")
        
        return processed
    
    def _assess_image_quality(self, image: np.ndarray, debug_mode: bool = False) -> Dict:
        """
        评估图像质量并提供预处理建议
        
        Args:
            image: 输入图像
            debug_mode: 是否启用调试模式
        
        Returns:
            图像质量评估结果
        """
        # 计算图像统计信息
        mean_val = np.mean(image)
        std_val = np.std(image)
        min_val = np.min(image)
        max_val = np.max(image)
        
        # 计算对比度
        contrast = std_val / mean_val if mean_val > 0 else 0
        
        # 计算动态范围
        dynamic_range = max_val - min_val
        
        # 计算梯度幅度（边缘强度）
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_strength = np.mean(gradient_magnitude)
        
        # 质量评估
        low_contrast = contrast < 0.3  # 对比度阈值
        needs_normalization = dynamic_range < 200 or max_val > 255 or min_val < 0
        noisy = std_val > mean_val * 0.8  # 噪声检测
        weak_edges = edge_strength < 10  # 边缘强度阈值
        
        quality_info = {
            'mean': float(mean_val),
            'std': float(std_val),
            'contrast': float(contrast),
            'dynamic_range': float(dynamic_range),
            'edge_strength': float(edge_strength),
            'low_contrast': low_contrast,
            'needs_normalization': needs_normalization,
            'noisy': noisy,
            'weak_edges': weak_edges
        }
        
        if debug_mode:
            self.logger.info(f"图像质量评估: 均值={mean_val:.1f}, 标准差={std_val:.1f}, 对比度={contrast:.3f}")
            self.logger.info(f"动态范围={dynamic_range:.1f}, 边缘强度={edge_strength:.1f}")
            self.logger.info(f"质量标志: 低对比度={low_contrast}, 需要归一化={needs_normalization}, 噪声={noisy}, 弱边缘={weak_edges}")
        
        return quality_info
    
    def _detect_roi_otsu(self, image: np.ndarray, threshold: float, 
                        min_area: int, max_area: int, kernel_size: int = 3,
                        min_circularity: float = 0.1, max_aspect_ratio: float = 5.0,
                        debug_mode: bool = False) -> List[Dict]:
        """
        使用Otsu阈值方法检测ROI
        """
        # 使用Otsu阈值进行二值化
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if debug_mode:
            self.logger.info(f"Otsu阈值检测: 图像尺寸={image.shape}, 二值化后非零像素={np.count_nonzero(binary)}")
        
        # 形态学操作去除噪声 - 使用优化的核
        kernel = self._get_optimized_kernel(kernel_size, 'ellipse')
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return self._extract_contours(binary, min_area, max_area, min_circularity, max_aspect_ratio, debug_mode)
    
    def _detect_roi_adaptive(self, image: np.ndarray, threshold: float, 
                           min_area: int, max_area: int, kernel_size: int = 3,
                           block_size: int = 9, c_value: int = 1,
                           min_circularity: float = 0.1, max_aspect_ratio: float = 5.0,
                           debug_mode: bool = False) -> List[Dict]:
        """
        使用自适应阈值方法检测ROI
        """
        # 自适应阈值 - 使用UI参数
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, block_size, c_value)
        
        if debug_mode:
            self.logger.info(f"自适应阈值检测: 块大小={block_size}, C值={c_value}, 二值化后非零像素={np.count_nonzero(binary)}")
        
        # 形态学操作 - 使用优化的核
        kernel = self._get_optimized_kernel(kernel_size, 'ellipse')
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return self._extract_contours(binary, min_area, max_area, min_circularity, max_aspect_ratio, debug_mode)
    
    def _detect_roi_watershed(self, image: np.ndarray, threshold: float, 
                            min_area: int, max_area: int, kernel_size: int = 3,
                            min_circularity: float = 0.1, max_aspect_ratio: float = 5.0,
                            debug_mode: bool = False) -> List[Dict]:
        """
        使用分水岭算法检测ROI
        """
        # 阈值处理
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 噪声去除 - 使用优化的核
        kernel = self._get_optimized_kernel(kernel_size, 'ellipse')
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        if debug_mode:
            self.logger.info(f"分水岭算法: 阈值处理后非零像素={np.count_nonzero(thresh)}, 开运算后非零像素={np.count_nonzero(opening)}")
        
        # 确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 确定前景区域
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # 找到未知区域
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 标记连通组件
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # 应用分水岭算法
        if len(image.shape) == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image
        
        markers = cv2.watershed(image_color, markers)
        
        # 创建二值图像
        binary = np.zeros_like(image, dtype=np.uint8)
        binary[markers > 1] = 255
        
        return self._extract_contours(binary, min_area, max_area, min_circularity, max_aspect_ratio, debug_mode)
    
    def _detect_roi_edge(self, image: np.ndarray, threshold: float, 
                        min_area: int, max_area: int, kernel_size: int = 3,
                        min_circularity: float = 0.1, max_aspect_ratio: float = 5.0,
                        debug_mode: bool = False) -> List[Dict]:
        """
        使用边缘检测方法检测ROI
        """
        # Canny边缘检测
        edges = cv2.Canny(image, 50, 150)
        
        if debug_mode:
            self.logger.info(f"边缘检测: 检测到边缘像素={np.count_nonzero(edges)}")
        
        # 形态学闭运算连接边缘 - 使用优化的核
        kernel = self._get_optimized_kernel(kernel_size + 2, 'ellipse')  # 稍大的核用于连接边缘
        binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 填充轮廓内部
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(binary)
        cv2.fillPoly(filled, contours, 255)
        
        return self._extract_contours(filled, min_area, max_area, min_circularity, max_aspect_ratio, debug_mode)
    
    def _validate_dicom_data(self, dicom_data: Dataset) -> bool:
        """
        验证DICOM数据的基本完整性
        
        Args:
            dicom_data: DICOM数据集
        
        Returns:
            验证是否通过
        """
        try:
            # 检查必要的标签
            required_tags = ['SOPInstanceUID', 'StudyInstanceUID', 'SeriesInstanceUID']
            for tag in required_tags:
                if not hasattr(dicom_data, tag):
                    self.logger.warning(f"缺少必要标签: {tag}")
                    return False
            
            # 检查像素数据（如果存在）
            if hasattr(dicom_data, 'PixelData'):
                try:
                    pixel_array = dicom_data.pixel_array
                    if pixel_array is None or pixel_array.size == 0:
                        self.logger.warning("像素数据为空")
                        return False
                except Exception as e:
                    self.logger.warning(f"像素数据访问失败: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"DICOM数据验证异常: {e}")
            return False
    
    def _extract_contours(self, binary_image: np.ndarray, min_area: int, max_area: int,
                         min_circularity: float = 0.1, max_aspect_ratio: float = 5.0,
                         debug_mode: bool = False) -> List[Dict]:
        """
        从二值图像中提取轮廓并计算ROI特征
        """
        # 查找轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 添加调试日志
        if debug_mode:
            self.logger.info(f"检测到 {len(contours)} 个原始轮廓")
        
        # 性能优化：预先过滤轮廓
        filtered_contours = []
        filtered_out_count = 0
        area_stats = []
        filter_reasons = {'area': 0, 'circularity': 0, 'aspect_ratio': 0}
        
        for contour in contours:
            area = cv2.contourArea(contour)
            area_stats.append(area)
            
            # 面积过滤
            if area < min_area or (max_area and area > max_area):
                filtered_out_count += 1
                filter_reasons['area'] += 1
                if debug_mode:
                    self.logger.debug(f"面积过滤: 面积={area:.1f}, 范围=[{min_area}, {max_area}]")
                continue
            
            # 计算边界框用于长宽比检查
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else float('inf')
            
            # 长宽比过滤
            if aspect_ratio > max_aspect_ratio:
                filtered_out_count += 1
                filter_reasons['aspect_ratio'] += 1
                if debug_mode:
                    self.logger.debug(f"长宽比过滤: 比例={aspect_ratio:.2f}, 最大={max_aspect_ratio}")
                continue
            
            # 圆形度过滤
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            if circularity < min_circularity:
                filtered_out_count += 1
                filter_reasons['circularity'] += 1
                if debug_mode:
                    self.logger.debug(f"圆形度过滤: 圆形度={circularity:.3f}, 最小={min_circularity}")
                continue
            
            filtered_contours.append((contour, area))
        
        # 记录过滤统计信息
        if area_stats:
            min_detected_area = min(area_stats)
            max_detected_area = max(area_stats)
            avg_detected_area = sum(area_stats) / len(area_stats)
            
            if debug_mode:
                self.logger.info(f"轮廓面积统计: 最小={min_detected_area:.1f}, 最大={max_detected_area:.1f}, 平均={avg_detected_area:.1f}")
                self.logger.info(f"过滤参数: min_area={min_area}, max_area={max_area}, min_circularity={min_circularity}, max_aspect_ratio={max_aspect_ratio}")
                self.logger.info(f"过滤统计: 面积过滤={filter_reasons['area']}, 圆形度过滤={filter_reasons['circularity']}, 长宽比过滤={filter_reasons['aspect_ratio']}")
            
            self.logger.info(f"ROI检测结果: 保留 {len(filtered_contours)} 个轮廓, 过滤掉 {filtered_out_count} 个轮廓")
        else:
            self.logger.warning("未检测到任何轮廓")
        
        roi_regions = []
        
        for i, (contour, area) in enumerate(filtered_contours):
            # 面积已经在预过滤中检查过了
            
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算轮廓特征
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 计算质心
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            # 计算椭圆拟合
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                ellipse_area = np.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)
                ellipse_ratio = area / ellipse_area if ellipse_area > 0 else 0
            else:
                ellipse = None
                ellipse_ratio = 0
            
            # 计算凸包
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # 计算长宽比
            aspect_ratio = w / h if h > 0 else 0
            
            # 计算椭圆拟合的离心率
            eccentricity = 0
            if ellipse is not None:
                a, b = ellipse[1][0] / 2, ellipse[1][1] / 2  # 长轴和短轴的半径
                if a > b and a > 0:
                    eccentricity = np.sqrt(1 - (b * b) / (a * a))
                elif b > 0:
                    eccentricity = np.sqrt(1 - (a * a) / (b * b))
            
            # 计算矩特征
            moments = cv2.moments(contour)
            
            # 计算Hu矩（形状不变特征）
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # 计算紧密度（compactness）
            compactness = (perimeter * perimeter) / area if area > 0 else 0
            
            # 计算等效直径
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            
            # 构建ROI信息
            roi_info = {
                'roi_id': f'roi_{i+1}',
                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'area': float(area),
                'perimeter': float(perimeter),
                'circularity': float(circularity),
                'centroid': {'x': int(cx), 'y': int(cy)},
                'contour_points': contour.reshape(-1, 2).tolist(),
                'min_area_rect': {
                    'center': (float(rect[0][0]), float(rect[0][1])),
                    'size': (float(rect[1][0]), float(rect[1][1])),
                    'angle': float(rect[2])
                },
                'ellipse': {
                    'center': (float(ellipse[0][0]), float(ellipse[0][1])) if ellipse else None,
                    'axes': (float(ellipse[1][0]), float(ellipse[1][1])) if ellipse else None,
                    'angle': float(ellipse[2]) if ellipse else None,
                    'ratio': float(ellipse_ratio)
                } if ellipse else None,
                'convex_hull': hull.reshape(-1, 2).tolist(),
                'solidity': float(solidity),
                'aspect_ratio': float(aspect_ratio),
                'extent': float(area / (w * h)) if w * h > 0 else 0,
                'compactness': float(compactness),
                'equivalent_diameter': float(equivalent_diameter),
                'eccentricity': float(eccentricity),
                'hu_moments': [float(h) for h in hu_moments]
            }
            
            roi_regions.append(roi_info)
        
        return roi_regions
    
    def _calculate_roi_statistics(self, image_array: np.ndarray, contour_points: List[List[int]]) -> Dict:
        """
        计算ROI区域内的像素统计信息
        
        Args:
            image_array: 原始图像数组
            contour_points: ROI轮廓点列表 [[x1, y1], [x2, y2], ...]
        
        Returns:
            包含统计信息的字典
        """
        try:
            # 确保图像是灰度图
            if len(image_array.shape) == 3:
                gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image_array.copy()
            
            # 将轮廓点转换为OpenCV格式
            contour = np.array(contour_points, dtype=np.int32).reshape(-1, 1, 2)
            
            # 创建ROI掩码
            mask = np.zeros(gray_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # 提取ROI区域内的像素值
            roi_pixels = gray_image[mask > 0]
            
            if len(roi_pixels) == 0:
                return {
                    'mean_intensity': 0.0,
                    'std_intensity': 0.0,
                    'min_intensity': 0.0,
                    'max_intensity': 0.0,
                    'pixel_count': 0
                }
            
            # 计算统计信息
            statistics = {
                'mean_intensity': float(np.mean(roi_pixels)),
                'std_intensity': float(np.std(roi_pixels)),
                'min_intensity': float(np.min(roi_pixels)),
                'max_intensity': float(np.max(roi_pixels)),
                'pixel_count': int(len(roi_pixels))
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"计算ROI统计信息失败: {e}")
            return {
                'mean_intensity': 0.0,
                'std_intensity': 0.0,
                'min_intensity': 0.0,
                'max_intensity': 0.0,
                'pixel_count': 0
            }
    
    def _extract_roi_from_rt_tags(self, dicom_data: Dataset, min_area: int = 5, max_area: int = None) -> List[Dict]:
        """
        从DICOM RT Structure Set标签中提取ROI信息
        
        Args:
            dicom_data: DICOM数据集
            min_area: 最小面积阈值
            max_area: 最大面积阈值
        
        Returns:
            ROI区域列表，包含从RT tags解析的结构信息
        """
        try:
            roi_regions = []
            
            # 添加详细的调试日志
            self.logger.info("=== 开始RT tags解析 ===")
            self.logger.info(f"DICOM数据集类型: {type(dicom_data)}")
            
            # 检查DICOM数据集的基本属性
            if hasattr(dicom_data, 'Modality'):
                modality = dicom_data.Modality
                self.logger.info(f"DICOM Modality: {modality}")
            else:
                self.logger.warning("DICOM数据集缺少Modality属性")
                modality = None
            
            # 检查是否为RT Structure Set文件
            if not hasattr(dicom_data, 'Modality') or dicom_data.Modality != 'RTSTRUCT':
                self.logger.info(f"非RT Structure Set文件 (Modality: {modality})，跳过RT tags解析")
                return []
            
            self.logger.info("开始从RT Structure Set标签解析ROI信息")
            
            # 检查DICOM数据集中的关键属性
            self.logger.info("检查DICOM数据集属性:")
            key_attributes = ['StructureSetROISequence', 'RTROIObservationsSequence', 'ROIContourSequence']
            for attr in key_attributes:
                has_attr = hasattr(dicom_data, attr)
                self.logger.info(f"  {attr}: {'存在' if has_attr else '不存在'}")
                if has_attr:
                    attr_value = getattr(dicom_data, attr)
                    self.logger.info(f"    长度: {len(attr_value) if hasattr(attr_value, '__len__') else 'N/A'}")
            
            # 获取结构集ROI序列 (Structure Set ROI Sequence)
            structure_set_rois = {}
            if hasattr(dicom_data, 'StructureSetROISequence'):
                self.logger.info("解析StructureSetROISequence...")
                for i, roi in enumerate(dicom_data.StructureSetROISequence):
                    roi_number = roi.ROINumber
                    roi_name = getattr(roi, 'ROIName', f'ROI_{roi_number}')
                    roi_description = getattr(roi, 'ROIDescription', '')
                    
                    structure_set_rois[roi_number] = {
                        'name': roi_name,
                        'description': roi_description,
                        'roi_number': roi_number
                    }
                    
                    self.logger.info(f"  ROI {i+1}: 编号={roi_number}, 名称='{roi_name}', 描述='{roi_description}'")
                    
                self.logger.info(f"找到 {len(structure_set_rois)} 个结构集ROI定义")
            else:
                self.logger.warning("未找到StructureSetROISequence")
            
            # 获取RT ROI观察序列 (RT ROI Observations Sequence)
            roi_observations = {}
            if hasattr(dicom_data, 'RTROIObservationsSequence'):
                for obs in dicom_data.RTROIObservationsSequence:
                    roi_number = obs.ReferencedROINumber
                    roi_type = getattr(obs, 'RTROIInterpretedType', 'ORGAN')
                    roi_observations[roi_number] = {
                        'type': roi_type,
                        'observation_number': getattr(obs, 'ObservationNumber', roi_number)
                    }
                    
                self.logger.info(f"找到 {len(roi_observations)} 个ROI观察定义")
            
            # 解析ROI轮廓序列 (ROI Contour Sequence)
            if hasattr(dicom_data, 'ROIContourSequence'):
                self.logger.info(f"解析ROIContourSequence，共 {len(dicom_data.ROIContourSequence)} 个轮廓组...")
                for contour_idx, roi_contour in enumerate(dicom_data.ROIContourSequence):
                    roi_number = roi_contour.ReferencedROINumber
                    self.logger.info(f"  处理轮廓组 {contour_idx+1}: ROI编号={roi_number}")
                    
                    # 获取ROI基本信息
                    roi_info = structure_set_rois.get(roi_number, {})
                    roi_obs = roi_observations.get(roi_number, {})
                    
                    # 获取轮廓颜色
                    roi_color = getattr(roi_contour, 'ROIDisplayColor', [255, 0, 0])  # 默认红色
                    self.logger.info(f"    ROI信息: 名称='{roi_info.get('name', 'Unknown')}', 颜色={roi_color}")
                    
                    if hasattr(roi_contour, 'ContourSequence'):
                        contour_count = len(roi_contour.ContourSequence)
                        self.logger.info(f"    包含 {contour_count} 个轮廓")
                        for contour_idx, contour in enumerate(roi_contour.ContourSequence):
                            if hasattr(contour, 'ContourData'):
                                # 提取轮廓点数据
                                contour_data = contour.ContourData
                                if len(contour_data) < 6:  # 至少需要3个点（每个点3个坐标）
                                    continue
                                    
                                # 转换为numpy数组并重塑为(n, 3)格式
                                points_3d = np.array(contour_data).reshape(-1, 3)
                                
                                # 只取x,y坐标（忽略z坐标）
                                points_2d = points_3d[:, :2]
                                
                                # 转换为OpenCV轮廓格式
                                cv_contour = points_2d.astype(np.int32).reshape(-1, 1, 2)
                                
                                # 计算轮廓面积
                                area = cv2.contourArea(cv_contour)
                                
                                # 应用面积过滤
                                if area < min_area:
                                    continue
                                if max_area is not None and area > max_area:
                                    continue
                                
                                # 计算边界框
                                x, y, w, h = cv2.boundingRect(cv_contour)
                                
                                # 计算轮廓周长
                                perimeter = cv2.arcLength(cv_contour, True)
                                
                                # 创建ROI信息字典
                                roi_data = {
                                    'roi_id': f'rt_roi_{roi_number}_{contour_idx}',
                                    'roi_number': roi_number,
                                    'roi_name': roi_info.get('name', f'ROI_{roi_number}'),
                                    'roi_description': roi_info.get('description', ''),
                                    'roi_type': roi_obs.get('type', 'ORGAN'),
                                    'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                                    'area': float(area),
                                    'perimeter': float(perimeter),
                                    'contour_points': cv_contour.reshape(-1, 2).tolist(),
                                    'centroid': {'x': int(x + w//2), 'y': int(y + h//2)},
                                    'color': roi_color,
                                    'z_coordinate': float(points_3d[0, 2]) if len(points_3d) > 0 else 0.0,
                                    'source': 'rt_tags',
                                    'contour_geometric_type': getattr(contour, 'ContourGeometricType', 'CLOSED_PLANAR'),
                                    'number_of_contour_points': getattr(contour, 'NumberOfContourPoints', len(points_2d))
                                }
                                
                                roi_regions.append(roi_data)
                                
                self.logger.info(f"从RT Structure Set成功解析 {len(roi_regions)} 个ROI轮廓")
                
            else:
                self.logger.warning("RT Structure Set文件中未找到ROIContourSequence")
            
            return roi_regions
            
        except Exception as e:
            self.logger.error(f"RT tags ROI解析失败: {e}")
            import traceback
            self.logger.debug(f"详细错误信息: {traceback.format_exc()}")
            return []
    
    def _extract_roi_from_overlay(self, dicom_data: Dataset, min_area: int, max_area: int) -> List[Dict]:
        """
        从DICOM覆盖层提取ROI信息
        
        Args:
            dicom_data: DICOM数据集
            min_area: 最小面积阈值
            max_area: 最大面积阈值
        
        Returns:
            ROI区域列表
        """
        try:
            roi_regions = []
            
            # 方法1: 使用pydicom内置方法提取覆盖层 (遍历所有可能的组)
            for group in range(0x6000, 0x601F, 2):
                try:
                    if hasattr(dicom_data, 'overlay_array'):
                        overlays = dicom_data.overlay_array(group)
                        if overlays is not None and overlays.size > 0:
                            # 处理覆盖层数据
                            overlay_image = (overlays * 255).astype(np.uint8)
                            new_rois = self._extract_contours(overlay_image, min_area, max_area)
                            roi_regions.extend(new_rois)
                            if new_rois:
                                self.logger.info(f"从内置覆盖层(组{group:04X})提取到 {len(new_rois)} 个ROI")
                except Exception as e:
                    self.logger.debug(f"内置覆盖层提取失败(组{group:04X}): {e}")
            
            # 方法2: 手动解析覆盖层数据 (如果方法1没有找到任何ROI，或者为了补充)
            # 注意：如果方法1已经找到ROI，是否还需要方法2取决于具体需求，这里假设如果不冲突可以都尝试
            # 但为了避免重复，通常如果内置方法成功就不需要手动解析了。不过考虑到鲁棒性，我们可以检查是否已经找到ROI
            
            if not roi_regions:
                for group in range(0x6000, 0x601F, 2):
                    try:
                        # 检查是否存在覆盖层数据
                        data_tag = (group, 0x3000)
                        if data_tag not in dicom_data:
                            continue
                            
                        # 尝试提取覆盖层数据
                        rows_tag = (group, 0x0010)
                        cols_tag = (group, 0x0011)
                        
                        overlay_rows = dicom_data[rows_tag].value if rows_tag in dicom_data else dicom_data.Rows
                        overlay_cols = dicom_data[cols_tag].value if cols_tag in dicom_data else dicom_data.Columns
                        overlay_data_elem = dicom_data[data_tag]
                        
                        if overlay_rows and overlay_cols and overlay_data_elem:
                            rows = int(overlay_rows)
                            cols = int(overlay_cols)
                            
                            # 解析覆盖层数据
                            overlay_data = overlay_data_elem.value
                            total_bits = rows * cols
                            required_bytes = (total_bits + 7) // 8
                            
                            # 处理数据长度
                            if len(overlay_data) < required_bytes:
                                overlay_data = overlay_data + b'\x00' * (required_bytes - len(overlay_data))
                            elif len(overlay_data) > required_bytes:
                                overlay_data = overlay_data[:required_bytes]
                                
                            # 位解包
                            overlay_bits = np.unpackbits(np.frombuffer(overlay_data, dtype=np.uint8))
                            overlay_bits = overlay_bits[:total_bits]
                            
                            # 重塑并转换
                            overlay_image = overlay_bits.reshape(rows, cols)
                            overlay_image = (overlay_image * 255).astype(np.uint8)
                            
                            new_rois = self._extract_contours(overlay_image, min_area, max_area)
                            roi_regions.extend(new_rois)
                            if new_rois:
                                self.logger.info(f"从手动解析覆盖层(组{group:04X})提取到 {len(new_rois)} 个ROI")
                        
                    except Exception as e:
                        self.logger.debug(f"手动覆盖层解析失败(组{group:04X}): {e}")
            
            # 方法3: 从RT Structure Set中提取ROI（如果存在）
            if not roi_regions and hasattr(dicom_data, 'Modality') and dicom_data.Modality == 'RTSTRUCT':
                try:
                    if hasattr(dicom_data, 'ROIContourSequence'):
                        for roi_contour in dicom_data.ROIContourSequence:
                            if hasattr(roi_contour, 'ContourSequence'):
                                for contour in roi_contour.ContourSequence:
                                    if hasattr(contour, 'ContourData'):
                                        # 提取轮廓点
                                        contour_data = contour.ContourData
                                        points = np.array(contour_data).reshape(-1, 3)[:, :2]  # 只取x,y坐标
                                        
                                        # 转换为OpenCV轮廓格式
                                        cv_contour = points.astype(np.int32).reshape(-1, 1, 2)
                                        
                                        # 计算面积
                                        area = cv2.contourArea(cv_contour)
                                        
                                        if min_area <= area <= max_area:
                                            x, y, w, h = cv2.boundingRect(cv_contour)
                                            roi_info = {
                                                'roi_id': f'roi_{len(roi_regions)+1}',
                                                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                                                'area': float(area),
                                                'contour_points': cv_contour.reshape(-1, 2).tolist(),
                                                'centroid': {'x': int(x + w//2), 'y': int(y + h//2)},
                                                'source': 'rtstruct'
                                            }
                                            roi_regions.append(roi_info)
                        
                        self.logger.info(f"从RT Structure Set提取到 {len(roi_regions)} 个ROI")
                        
                except Exception as e:
                    self.logger.debug(f"RT Structure Set解析失败: {e}")
            
            return roi_regions
            
        except Exception as e:
            self.logger.error(f"覆盖层ROI提取失败: {e}")
            return []
    
    def _make_json_serializable(self, data):
        """
        Recursively convert data to JSON serializable format.
        Handles bytes, numpy types, and other common non-serializable types.
        """
        if isinstance(data, bytes):
            # Try to decode as utf-8, fallback to string representation
            try:
                return data.decode('utf-8')
            except:
                return str(data)
        elif isinstance(data, (np.integer, int)):
            return int(data)
        elif isinstance(data, (np.floating, float)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple, set)):
            return [self._make_json_serializable(item) for item in data]
        elif hasattr(data, 'isoformat'):  # datetime, date
            return data.isoformat()
        else:
            return data

    def create_dicom_session(self, user_id: str, file_path: str, 
                           dicom_info: Dict, roi_data: List[Dict], project_id: str = None) -> Optional[str]:
        """
        创建DICOM处理会话
        
        Args:
            user_id: 用户ID
            file_path: DICOM文件路径
            dicom_info: DICOM信息
            roi_data: ROI数据
            project_id: 项目ID（可选，如果未提供则使用默认项目）
        
        Returns:
            会话ID，如果创建失败则返回None
        """
        try:
            # 验证user_id参数
            if not user_id or user_id.strip() == '':
                self.logger.error("创建DICOM会话失败: user_id不能为空")
                return None
            
            # 验证file_path参数
            if not file_path or file_path.strip() == '':
                self.logger.error("创建DICOM会话失败: file_path不能为空")
                return None
            
            # 验证文件是否存在
            if not os.path.exists(file_path):
                self.logger.error(f"DICOM文件不存在: {file_path}")
                return None
            
            # 生成会话ID（添加微秒和随机数确保唯一性）
            import uuid
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            unique_suffix = str(uuid.uuid4())[:8]
            session_id = f"dicom_{timestamp}_{user_id.strip()}_{unique_suffix}"
            
            # 如果没有提供project_id，使用默认项目ID（避免循环引用）
            if project_id is None:
                project_id = "default"
                self.logger.info("使用默认项目ID: default")
            
            # 确保 dicom_info 是 JSON 可序列化的
            dicom_info = self._make_json_serializable(dicom_info)

            # 使用项目存储服务存储DICOM文件
            stored_file_path = self.storage_service.store_dicom_file(
                project_id=project_id,
                file_path=file_path,
                metadata={'session_id': session_id, 'dicom_info': dicom_info}
            )
            
            if not stored_file_path:
                self.logger.error("存储DICOM文件失败")
                return None
            
            # 插入DICOM会话记录（使用存储后的文件路径）
            session_query = """
                INSERT INTO dicom_sessions (session_id, user_id, project_id, file_path, dicom_info)
                VALUES (?, ?, ?, ?, ?)
            """
            
            session_params = (
                session_id,
                user_id,
                project_id,
                stored_file_path,  # 使用存储后的文件路径
                json.dumps(dicom_info, ensure_ascii=False)
            )
            
            rows_affected = self.db_service.execute_update(session_query, session_params)
            
            if rows_affected == 0:
                self.logger.error("创建DICOM会话失败")
                return None
            
            # 插入ROI数据，为每个ROI生成全局唯一的roi_id
            for i, roi in enumerate(roi_data):
                # 生成全局唯一的roi_id，包含session_id和索引
                unique_roi_id = f"{session_id}_roi_{i+1}"
                
                # 如果ROI包含图像数据，存储ROI图像
                roi_image_path = None
                if 'image_data' in roi and roi['image_data'] is not None:
                    try:
                        # 保存图像数据到临时文件
                        import tempfile
                        import cv2
                        
                        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
                        os.close(temp_fd)
                        
                        image_data = roi['image_data']
                        if isinstance(image_data, np.ndarray):
                            cv2.imwrite(temp_path, image_data)
                        else:
                            # 假设是字节流
                            with open(temp_path, 'wb') as f:
                                f.write(image_data)
                                
                        # 存储ROI图像
                        roi_image_path = self.storage_service.store_roi_image(
                            project_id=project_id,
                            file_path=temp_path,
                            metadata={'session_id': session_id, 'roi_id': unique_roi_id}
                        )
                        
                        # 删除临时文件
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                    except Exception as e:
                        self.logger.error(f"存储ROI图像失败: {e}")
                        roi_image_path = None
                
                # 更新ROI属性，包含图像路径和OCR结果
                roi_properties = roi.copy()
                if roi_image_path:
                    roi_properties['image_path'] = roi_image_path
                
                # CRITICAL: 移除二进制数据，防止 JSON 序列化失败
                if 'image_data' in roi_properties:
                    del roi_properties['image_data']
                
                # 确保 roi_properties 是 JSON 可序列化的
                roi_properties = self._make_json_serializable(roi_properties)

                # 提取OCR结果用于单独存储
                ocr_text = ''
                ocr_confidence = 0.0
                if 'ocr_result' in roi:
                    ocr_result = roi['ocr_result']
                    ocr_text = ocr_result.get('text', '')
                    ocr_confidence = ocr_result.get('confidence', 0.0)
                
                roi_query = """
                    INSERT INTO roi_data (session_id, roi_id, roi_type, coordinates, 
                                         area, perimeter, properties, ocr_text, ocr_confidence, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                roi_params = (
                    session_id,
                    unique_roi_id,  # 使用全局唯一的roi_id
                    'detected',  # ROI类型
                    json.dumps(self._make_json_serializable(roi.get('contour_points', [])), ensure_ascii=False),
                    roi.get('area', 0),
                    roi.get('perimeter', 0),
                    json.dumps(roi_properties, ensure_ascii=False),
                    ocr_text,
                    ocr_confidence,
                    roi.get('source', 'unknown')
                )
                
                self.db_service.execute_update(roi_query, roi_params)
            
            self.logger.info(f"DICOM会话创建成功: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"创建DICOM会话失败: {e}")
            return None
    
    def get_project_dicom_file_path(self, project_id: str, session_id: str, original_filename: str = None) -> Optional[str]:
        """
        获取项目级别的DICOM文件路径
        
        Args:
            project_id: 项目ID
            session_id: 会话ID
            original_filename: 原始文件名（可选）
        
        Returns:
            DICOM文件的项目存储路径
        """
        try:
            return self.storage_service.get_dicom_file_path(
                project_id=project_id,
                session_id=session_id,
                filename=original_filename
            )
        except Exception as e:
            self.logger.error(f"获取项目DICOM文件路径失败: {e}")
            return None
    
    def load_dicom_from_project_storage(self, project_id: str, session_id: str, 
                                       use_cache: bool = True, force: bool = True) -> Optional[Dataset]:
        """
        从项目存储中加载DICOM文件
        
        Args:
            project_id: 项目ID
            session_id: 会话ID
            use_cache: 是否使用缓存
            force: 是否强制读取非标准DICOM文件
        
        Returns:
            DICOM数据集，如果加载失败则返回None
        """
        try:
            # 获取项目存储中的DICOM文件路径
            file_path = self.get_project_dicom_file_path(project_id, session_id)
            if not file_path:
                self.logger.error(f"未找到项目存储中的DICOM文件: project_id={project_id}, session_id={session_id}")
                return None
            
            # 使用标准方法加载DICOM文件
            return self.load_dicom_file(file_path, use_cache, force)
            
        except Exception as e:
            self.logger.error(f"从项目存储加载DICOM文件失败: {e}")
            return None
    
    def get_dicom_sessions(self, user_id: str = None, project_id: str = None, limit: int = 50) -> List[Dict]:
        """
        获取DICOM会话列表
        
        Args:
            user_id: 用户ID（可选）
            project_id: 项目ID（可选，如果未提供则获取当前项目）
            limit: 限制数量
        
        Returns:
            会话列表
        """
        try:
            # 如果没有提供project_id，使用默认项目ID（避免循环引用）
            if project_id is None:
                project_id = "default"
                self.logger.info("使用默认项目ID: default")
                # 注释掉循环引用代码
                # 注释掉循环引用代码
                # project_service = ProjectService()
                # project_id = project_service.get_current_project_id()
                # except Exception as e:
                #     self.logger.warning(f"获取当前项目ID失败: {e}，使用default项目")
                #     project_id = "default"
            
            # 构建查询条件
            where_conditions = ["project_id = ?"]
            params = [project_id]
            
            if user_id:
                where_conditions.append("user_id = ?")
                params.append(user_id)
            
            params.append(limit)
            
            query = f"""
                SELECT session_id, user_id, file_path, dicom_info, created_at
                FROM dicom_sessions
                WHERE {' AND '.join(where_conditions)}
                ORDER BY created_at DESC
                LIMIT ?
            """
            
            results = self.db_service.execute_query(query, tuple(params))
            
            sessions = []
            for row in results:
                dicom_info = {}
                if row['dicom_info']:
                    try:
                        dicom_info = json.loads(row['dicom_info'])
                    except json.JSONDecodeError:
                        pass
                
                sessions.append({
                    'session_id': row['session_id'],
                    'file_path': row['file_path'],
                    'dicom_info': dicom_info,
                    'created_at': row['created_at']
                })
            
            return sessions
            
        except Exception as e:
            self.logger.error(f"获取DICOM会话列表失败: {e}")
            return []
    
    def get_session_roi_data(self, session_id: str) -> List[Dict]:
        """
        获取会话的ROI数据
        
        Args:
            session_id: 会话ID
        
        Returns:
            ROI数据列表
        """
        try:
            query = """
                SELECT roi_id, roi_type, coordinates, area, perimeter, properties, ocr_text, ocr_confidence, source
                FROM roi_data
                WHERE session_id = ?
                ORDER BY created_at
            """
            
            results = self.db_service.execute_query(query, (session_id,))
            
            roi_data = []
            for row in results:
                coordinates = []
                properties = {}
                
                if row['coordinates']:
                    try:
                        coordinates = json.loads(row['coordinates'])
                    except json.JSONDecodeError:
                        pass
                
                if row['properties']:
                    try:
                        properties = json.loads(row['properties'])
                    except json.JSONDecodeError:
                        pass
                
                roi_data.append({
                    'roi_id': row['roi_id'],
                    'roi_type': row['roi_type'],
                    'coordinates': coordinates,
                    'area': row['area'],
                    'perimeter': row['perimeter'],
                    'properties': properties,
                    'ocr_text': row.get('ocr_text', ''),
                    'ocr_confidence': row.get('ocr_confidence', 0.0),
                    'source': row.get('source', 'unknown')
                })
            
            return roi_data
            
        except Exception as e:
            self.logger.error(f"获取ROI数据失败: {e}")
            return []
    
    def save_image_with_roi(self, image_array: np.ndarray, roi_data: List[Dict], 
                           output_path: str) -> bool:
        """
        保存带有ROI标注的图像
        
        Args:
            image_array: 图像数组
            roi_data: ROI数据
            output_path: 输出路径
        
        Returns:
            保存是否成功
        """
        try:
            # 创建彩色图像用于标注
            if len(image_array.shape) == 2:
                annotated_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            else:
                annotated_image = image_array.copy()
            
            # 绘制ROI
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            
            for i, roi in enumerate(roi_data):
                color = colors[i % len(colors)]
                
                # 绘制边界框
                if 'bbox' in roi:
                    bbox = roi['bbox']
                    cv2.rectangle(annotated_image, 
                                (bbox['x'], bbox['y']), 
                                (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                                color, 2)
                
                # 绘制轮廓
                if 'contour_points' in roi:
                    contour = np.array(roi['contour_points'], dtype=np.int32)
                    cv2.drawContours(annotated_image, [contour], -1, color, 2)
                
                # 标注ROI ID
                if 'centroid' in roi:
                    centroid = roi['centroid']
                    cv2.putText(annotated_image, roi['roi_id'], 
                              (centroid['x'], centroid['y']), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 保存图像
            cv2.imwrite(output_path, annotated_image)
            self.logger.info(f"ROI标注图像保存成功: {output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存ROI标注图像失败: {e}")
            return False
    
    def delete_dicom_session(self, session_id: str) -> bool:
        """
        删除指定的DICOM会话及其相关数据
        
        Args:
            session_id: 会话ID
        
        Returns:
            删除是否成功
        """
        try:
            # 首先删除ROI数据
            roi_delete_query = "DELETE FROM roi_data WHERE session_id = ?"
            roi_rows_affected = self.db_service.execute_update(roi_delete_query, (session_id,))
            
            # 然后删除会话记录
            session_delete_query = "DELETE FROM dicom_sessions WHERE session_id = ?"
            session_rows_affected = self.db_service.execute_update(session_delete_query, (session_id,))
            
            if session_rows_affected > 0:
                self.logger.info(f"DICOM会话删除成功: {session_id}, 删除了 {roi_rows_affected} 条ROI数据")
                return True
            else:
                self.logger.warning(f"未找到要删除的DICOM会话: {session_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"删除DICOM会话失败: {e}")
            return False
    
    def clear_all_dicom_sessions(self, user_id: str = None, project_id: str = None) -> bool:
        """
        清除指定用户或项目的所有DICOM会话
        
        Args:
            user_id: 用户ID（可选）
            project_id: 项目ID（可选，如果未提供则获取当前项目）
        
        Returns:
            清除是否成功
        """
        try:
            self.logger.info(f"开始清除DICOM会话 - 用户ID: {user_id}, 项目ID: {project_id}")
            
            # 如果没有提供project_id，使用默认项目ID（避免循环引用）
            if project_id is None:
                project_id = "default"
                self.logger.info("使用默认项目ID: default")
            
            # 构建查询条件
            where_conditions = ["project_id = ?"]
            params = [project_id]
            
            if user_id:
                where_conditions.append("user_id = ?")
                params.append(user_id)
            
            self.logger.info(f"查询条件: {' AND '.join(where_conditions)}, 参数: {params}")
            
            # 获取会话ID
            session_query = f"SELECT session_id FROM dicom_sessions WHERE {' AND '.join(where_conditions)}"
            self.logger.debug(f"执行查询: {session_query}")
            sessions = self.db_service.execute_query(session_query, tuple(params))
            
            self.logger.info(f"找到 {len(sessions)} 个待删除的会话")
            
            if not sessions:
                filter_desc = f"项目 {project_id}"
                if user_id:
                    filter_desc += f" 用户 {user_id}"
                self.logger.info(f"{filter_desc} 没有DICOM会话需要清除")
                return True
            
            session_ids = [session['session_id'] for session in sessions]
            self.logger.info(f"待删除的会话ID: {session_ids}")
            
            # 删除所有相关的ROI数据
            roi_delete_query = "DELETE FROM roi_data WHERE session_id IN ({})".format(
                ','.join(['?' for _ in session_ids])
            )
            self.logger.debug(f"删除ROI数据查询: {roi_delete_query}")
            roi_rows_affected = self.db_service.execute_update(roi_delete_query, session_ids)
            self.logger.info(f"删除了 {roi_rows_affected} 条ROI数据")
            
            # 删除所有会话记录
            session_delete_query = f"DELETE FROM dicom_sessions WHERE {' AND '.join(where_conditions)}"
            self.logger.debug(f"删除会话查询: {session_delete_query}")
            session_rows_affected = self.db_service.execute_update(session_delete_query, tuple(params))
            self.logger.info(f"删除了 {session_rows_affected} 个会话记录")
            
            # 验证删除结果
            verification_query = f"SELECT COUNT(*) as count FROM dicom_sessions WHERE {' AND '.join(where_conditions)}"
            verification_result = self.db_service.execute_query(verification_query, tuple(params))
            remaining_count = verification_result[0]['count'] if verification_result else 0
            
            if remaining_count > 0:
                self.logger.error(f"删除验证失败: 仍有 {remaining_count} 个会话残留")
                return False
            
            filter_desc = f"项目 {project_id}"
            if user_id:
                filter_desc += f" 用户 {user_id}"
            self.logger.info(f"{filter_desc} 的所有DICOM会话已清除: 删除了 {session_rows_affected} 个会话和 {roi_rows_affected} 条ROI数据")
            return True
            
        except Exception as e:
            self.logger.error(f"清除DICOM会话失败: {e}", exc_info=True)
            return False
