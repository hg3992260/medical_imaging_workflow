import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from src.services.database_service import DatabaseService
from src.services.project_service import ProjectService
from src.services.project_storage_service import ProjectStorageService
from src.utils.logger import get_logger


class DataCleaningService:
    """数据清洗服务类
    
    负责获取和汇总项目的样本数据，包括：
    - DICOM文件信息
    - OCR提取的文本信息
    - 文本处理结果
    - 样本数据统计
    """
    
    def __init__(self, db_service: DatabaseService = None):
        self.db_service = db_service or DatabaseService()
        self.project_service = ProjectService(self.db_service)
        self.storage_service = ProjectStorageService(self.db_service)
        self.logger = get_logger(__name__)
    
    def get_roi_details(self, project_id: str) -> Dict[str, Any]:
        """获取项目的ROI详细信息"""
        try:
            query = """
                SELECT 
                    rd.roi_id,
                    rd.session_id,
                    rd.roi_type,
                    rd.roi_name,
                    rd.coordinates,
                    rd.area,
                    rd.perimeter,
                    rd.properties,
                    rd.created_at,
                    ds.file_name,
                    ds.dicom_info
                FROM roi_data rd
                JOIN dicom_sessions ds ON rd.session_id = ds.session_id
                WHERE ds.project_id = ?
                ORDER BY rd.created_at DESC
                LIMIT 50
            """
            
            results = self.db_service.execute_query(query, (project_id,))
            
            if results:
                roi_list = []
                total_area = 0
                roi_types = {}
                
                for row in results:
                    # 解析坐标信息
                    coordinates = []
                    if row['coordinates']:
                        try:
                            coordinates = json.loads(row['coordinates'])
                        except json.JSONDecodeError:
                            pass
                    
                    # 解析属性信息
                    properties = {}
                    if row['properties']:
                        try:
                            properties = json.loads(row['properties'])
                        except json.JSONDecodeError:
                            pass
                    
                    # 统计ROI类型
                    roi_type = row['roi_type'] or 'unknown'
                    roi_types[roi_type] = roi_types.get(roi_type, 0) + 1
                    
                    roi_info = {
                        'roi_id': row['roi_id'],
                        'session_id': row['session_id'],
                        'roi_type': roi_type,
                        'roi_name': row['roi_name'] or f"ROI_{row['roi_id'][-8:]}",
                        'coordinates': coordinates,
                        'area': row['area'] or 0,
                        'perimeter': row['perimeter'] or 0,
                        'properties': properties,
                        'file_name': row['file_name'],
                        'created_at': row['created_at']
                    }
                    
                    roi_list.append(roi_info)
                    total_area += roi_info['area']
                
                return {
                    'total_count': len(roi_list),
                    'total_area': total_area,
                    'roi_types': roi_types,
                    'roi_list': roi_list,
                    'type': 'ROI识别数据'
                }
            
            return {
                'total_count': 0,
                'total_area': 0,
                'roi_types': {},
                'roi_list': [],
                'type': 'ROI识别数据'
            }
            
        except Exception as e:
            self.logger.error(f"获取ROI详细信息失败: {e}")
            return {
                'total_count': 0,
                'total_area': 0,
                'roi_types': {},
                'roi_list': [],
                'type': 'ROI识别数据'
            }

    def get_magic_seg_details(self, project_id: str) -> Dict[str, Any]:
        """获取Magic Seg详细信息"""
        try:
            # Magic Seg data is stored in roi_data with source='magic_seg_sam' or 'magic_seg_interactive'
            query = """
                SELECT 
                    rd.roi_id,
                    rd.session_id,
                    rd.source,
                    rd.roi_name,
                    rd.area,
                    rd.created_at,
                    ds.file_name
                FROM roi_data rd
                JOIN dicom_sessions ds ON rd.session_id = ds.session_id
                WHERE ds.project_id = ? AND (rd.source = 'magic_seg_sam' OR rd.source = 'magic_seg_interactive')
                ORDER BY rd.created_at DESC
                LIMIT 50
            """
            
            results = self.db_service.execute_query(query, (project_id,))
            
            if results:
                seg_list = []
                total_area = 0
                source_counts = {}
                
                for row in results:
                    source = row['source']
                    source_counts[source] = source_counts.get(source, 0) + 1
                    
                    seg_info = {
                        'roi_id': row['roi_id'],
                        'session_id': row['session_id'],
                        'source': source,
                        'roi_name': row['roi_name'] or f"Seg_{row['roi_id'][-8:]}",
                        'area': row['area'] or 0,
                        'file_name': row['file_name'],
                        'created_at': row['created_at']
                    }
                    
                    seg_list.append(seg_info)
                    total_area += seg_info['area']
                
                return {
                    'total_count': len(seg_list),
                    'total_area': total_area,
                    'source_counts': source_counts,
                    'seg_list': seg_list,
                    'type': 'Magic Seg数据'
                }
            
            return {
                'total_count': 0,
                'total_area': 0,
                'source_counts': {},
                'seg_list': [],
                'type': 'Magic Seg数据'
            }
            
        except Exception as e:
            self.logger.error(f"获取Magic Seg详细信息失败: {e}")
            return {
                'total_count': 0,
                'total_area': 0,
                'source_counts': {},
                'seg_list': [],
                'type': 'Magic Seg数据'
            }
    
    def get_project_sample_summary(self, project_id: str) -> Dict[str, Any]:
        """获取项目样本数据汇总
        
        Args:
            project_id: 项目ID
            
        Returns:
            Dict: 包含DICOM、OCR、文本数据的汇总信息
        """
        try:
            summary = {
                'project_id': project_id,
                'dicom_data': self._get_dicom_summary(project_id),
                'ocr_data': self._get_ocr_summary(project_id),
                'text_data': self._get_text_summary(project_id),
                'tabular_data': self._get_tabular_summary(project_id),
                'samples': self.get_sample_descriptions(project_id),
                'magic_seg_data': self.get_magic_seg_details(project_id),
                'storage_info': self._get_storage_summary(project_id),
                'updated_at': datetime.now().isoformat()
            }
            
            # 计算总体统计
            summary['total_files'] = (
                summary['dicom_data']['count'] + 
                summary['ocr_data']['count'] + 
                summary['text_data']['count'] +
                summary['magic_seg_data']['total_count']
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取项目样本汇总失败: {e}")
            return self._get_empty_summary(project_id)
    
    def get_sample_descriptions(self, project_id: str) -> Dict[str, Any]:
        """获取样本描述信息"""
        try:
            query = """
                SELECT user_id, display_name, description, created_at, last_used, is_active
                FROM user_ids
                WHERE project_id = ?
                ORDER BY created_at DESC
            """
            rows = self.db_service.execute_query(query, (project_id,))
            details = []
            for row in rows:
                details.append({
                    'user_id': row.get('user_id', ''),
                    'display_name': row.get('display_name', ''),
                    'description': row.get('description', ''),
                    'created_at': row.get('created_at', ''),
                    'last_used': row.get('last_used', ''),
                    'is_active': row.get('is_active', 1)
                })
            return {
                'count': len(details),
                'details': details,
                'type': '样本描述'
            }
        except Exception as e:
            self.logger.error(f"获取样本描述失败: {e}")
            return {
                'count': 0,
                'details': [],
                'type': '样本描述'
            }
    
    def _get_dicom_summary(self, project_id: str) -> Dict[str, Any]:
        """获取DICOM数据汇总"""
        try:
            # 获取DICOM会话数据和详细信息
            query = """
                SELECT 
                    ds.session_id,
                    ds.file_path,
                    ds.file_name,
                    ds.file_size,
                    ds.dicom_info,
                    ds.created_at,
                    COUNT(rd.roi_id) as roi_count
                FROM dicom_sessions ds
                LEFT JOIN roi_data rd ON ds.session_id = rd.session_id
                WHERE ds.project_id = ?
                GROUP BY ds.session_id
                ORDER BY ds.created_at DESC
                LIMIT 20
            """
            
            results = self.db_service.execute_query(query, (project_id,))
            
            if results:
                sessions_info = []
                patients_info = []
                total_roi_count = 0
                
                for row in results:
                    # 解析DICOM信息
                    dicom_info = {}
                    if row['dicom_info']:
                        try:
                            dicom_info = json.loads(row['dicom_info'])
                        except json.JSONDecodeError:
                            pass
                    
                    # 提取患者信息
                    patient_info = {
                        'patient_id': dicom_info.get('patient_id', '未知'),
                        'patient_name': dicom_info.get('patient_name', '未知'),
                        'study_date': dicom_info.get('study_date', ''),
                        'study_time': dicom_info.get('study_time', ''),
                        'modality': dicom_info.get('modality', ''),
                        'series_description': dicom_info.get('series_description', ''),
                        'institution_name': dicom_info.get('institution_name', '')
                    }
                    
                    if patient_info not in patients_info:
                        patients_info.append(patient_info)
                    
                    # 会话信息
                    session_info = {
                        'session_id': row['session_id'],
                        'file_name': row['file_name'],
                        'file_size': row['file_size'],
                        'roi_count': row['roi_count'],
                        'created_at': row['created_at'],
                        'dicom_info': dicom_info
                    }
                    sessions_info.append(session_info)
                    total_roi_count += row['roi_count']
                
                # 获取文件详细信息
                files_info = []
                for session in sessions_info[:10]:  # 限制显示前10个文件
                    file_info = self._get_file_info(session.get('file_name', ''))
                    if file_info:
                        file_info['roi_count'] = session['roi_count']
                        file_info['session_id'] = session['session_id']
                        files_info.append(file_info)
                
                return {
                    'count': len(results),
                    'file_count': len(results),
                    'files': files_info,
                    'sessions': sessions_info,
                    'patients': patients_info[:10],  # 限制显示前10个患者
                    'total_roi_count': total_roi_count,
                    'type': 'DICOM影像文件'
                }
            
            return {
                'count': 0, 
                'file_count': 0, 
                'files': [], 
                'sessions': [],
                'patients': [],
                'total_roi_count': 0,
                'type': 'DICOM影像文件'
            }
            
        except Exception as e:
            self.logger.error(f"获取DICOM汇总失败: {e}")
            return {
                'count': 0, 
                'file_count': 0, 
                'files': [], 
                'sessions': [],
                'patients': [],
                'total_roi_count': 0,
                'type': 'DICOM影像文件'
            }
    
    def _get_ocr_summary(self, project_id: str) -> Dict[str, Any]:
        """获取OCR数据汇总"""
        try:
            # 获取OCR会话信息
            sessions_query = """
                SELECT session_id, image_path, created_at
                FROM ocr_sessions 
                WHERE project_id = ?
                ORDER BY created_at DESC
                LIMIT 10
            """
            
            sessions = self.db_service.execute_query(sessions_query, (project_id,))
            
            # 获取OCR识别结果详细信息
            results_query = """
                SELECT 
                    or_.result_id,
                    or_.session_id,
                    or_.recognized_text as text_content,
                    or_.confidence,
                    or_.bounding_boxes as bounding_box,
                    or_.created_at,
                    os.image_path as file_path
                FROM ocr_results or_
                JOIN ocr_sessions os ON or_.session_id = os.session_id
                WHERE os.project_id = ?
                ORDER BY or_.created_at DESC
                LIMIT 50
            """
            
            ocr_results = self.db_service.execute_query(results_query, (project_id,))
            
            session_files = []
            ocr_details = []
            total_text_length = 0
            confidence_stats = []
            
            if sessions:
                for session in sessions:
                    image_path = session.get('image_path')
                    if image_path:
                        file_info = self._get_file_info(image_path)
                    else:
                        file_info = {'name': 'unknown_file', 'size': 0}
                    session_files.append({
                        'session_id': session['session_id'],
                        'file_name': file_info['name'],
                        'file_size': file_info['size'],
                        'created_at': session['created_at']
                    })
            
            if ocr_results:
                for result in ocr_results:
                    text_content = result['text_content'] or ''
                    confidence = result['confidence'] or 0
                    total_text_length += len(text_content)
                    
                    if confidence > 0:
                        confidence_stats.append(confidence)
                    
                    # 解析边界框信息
                    bounding_box = {}
                    if result['bounding_box']:
                        try:
                            bounding_box = json.loads(result['bounding_box'])
                        except json.JSONDecodeError:
                            pass
                    
                    # OCR结果没有metadata字段，使用空字典
                    metadata = {}
                    
                    # 提取文本预览 - 完整显示，不截断
                    text_preview = text_content
                    
                    file_path = result.get('file_path')
                    if file_path:
                        file_info = self._get_file_info(file_path)
                    else:
                        file_info = {'name': 'unknown_file', 'size': 0}
                    
                    ocr_details.append({
                        'result_id': result['result_id'],
                        'session_id': result['session_id'],
                        'file_name': file_info['name'],
                        'text_content': text_content,
                        'text_preview': text_preview,
                        'text_length': len(text_content),
                        'confidence': confidence,
                        'bounding_box': bounding_box,
                        'metadata': metadata,
                        'created_at': result['created_at']
                    })
            
            # 计算置信度统计
            avg_confidence = sum(confidence_stats) / len(confidence_stats) if confidence_stats else 0
            min_confidence = min(confidence_stats) if confidence_stats else 0
            max_confidence = max(confidence_stats) if confidence_stats else 0
            
            # 提取图像文件信息
            images = []
            if sessions:
                for session in sessions:
                    image_path = session.get('image_path')
                    if image_path:
                        file_info = self._get_file_info(image_path)
                        if file_info:
                            images.append({
                                'session_id': session['session_id'],
                                'file_name': file_info['name'],
                                'file_path': image_path,
                                'file_size': file_info['size'],
                                'created_at': session['created_at']
                            })
            
            return {
                'count': len(ocr_results) if ocr_results else 0,
                'session_count': len(sessions) if sessions else 0,
                'result_count': len(ocr_results) if ocr_results else 0,
                'total_text_length': total_text_length,
                'avg_confidence': round(avg_confidence, 2),
                'min_confidence': round(min_confidence, 2),
                'max_confidence': round(max_confidence, 2),
                'session_files': session_files,
                'ocr_details': ocr_details,
                'images': images,
                'type': 'OCR识别数据'
            }
            
        except Exception as e:
            self.logger.error(f"获取OCR数据汇总失败: {e}")
            return {
                'count': 0,
                'session_count': 0,
                'result_count': 0,
                'total_text_length': 0,
                'avg_confidence': 0,
                'min_confidence': 0,
                'max_confidence': 0,
                'session_files': [],
                'ocr_details': [],
                'images': [],
                'type': 'OCR识别数据'
            }
    
    def _get_text_summary(self, project_id: str) -> Dict[str, Any]:
        """获取文本处理数据汇总"""
        try:
            # 获取文本会话信息（移除LIMIT限制，获取所有会话）
            sessions_query = """
                SELECT session_id, operation_type, created_at, file_path, char_count, word_count, line_count
                FROM text_sessions 
                WHERE project_id = ?
                ORDER BY created_at DESC
            """
            
            sessions = self.db_service.execute_query(sessions_query, (project_id,))
            
            # 获取文档信息（移除LIMIT限制，获取所有文档，包含analysis_result和keywords）
            docs_query = """
                SELECT d.doc_id, d.session_id, d.file_name, d.content, d.file_size, d.created_at,
                       d.analysis_result, d.keywords, d.format
                FROM documents d
                JOIN text_sessions ts ON d.session_id = ts.session_id
                WHERE ts.project_id = ?
                ORDER BY d.created_at DESC
            """
            
            documents = self.db_service.execute_query(docs_query, (project_id,))
            
            session_files = []
            document_details = []
            total_length = 0
            keywords_summary = {}
            
            if sessions:
                for session in sessions:
                    # 使用operation_type作为文件名标识
                    operation_type = session.get('operation_type', 'text_processing')
                    file_path = session.get('file_path', '')
                    file_name = session.get('file_name') or (Path(file_path).name if file_path else f"{operation_type}_{session['session_id'][:8]}")
                    
                    session_files.append({
                        'session_id': session['session_id'],
                        'file_name': file_name,
                        'file_path': file_path,
                        'file_size': 0,  # 会话本身没有文件大小
                        'operation_type': operation_type,
                        'char_count': session.get('char_count', 0),
                        'word_count': session.get('word_count', 0),
                        'line_count': session.get('line_count', 0),
                        'created_at': session['created_at']
                    })
            
            if documents:
                for doc in documents:
                    content = doc['content'] or ''
                    content_length = len(content)
                    total_length += content_length
                    
                    # 解析analysis_result和keywords字段
                    analysis_results = {}
                    keywords = []
                    
                    if doc.get('analysis_result'):
                        try:
                            import json
                            analysis_results = json.loads(doc['analysis_result'])
                        except (json.JSONDecodeError, TypeError):
                            analysis_results = {}
                    
                    if doc.get('keywords'):
                        try:
                            import json
                            keywords = json.loads(doc['keywords'])
                            # 统计关键词频率
                            for keyword in keywords:
                                if isinstance(keyword, str):
                                    keywords_summary[keyword] = keywords_summary.get(keyword, 0) + 1
                        except (json.JSONDecodeError, TypeError):
                            keywords = []
                    
                    # 完整显示内容而不是截断
                    content_preview = content
                    
                    # 使用文档的文件名
                    file_name = doc['file_name'] or 'unknown_file'
                    
                    document_details.append({
                        'doc_id': doc['doc_id'],
                        'session_id': doc['session_id'],
                        'file_name': file_name,
                        'file_format': doc.get('format', ''),
                        'content_length': content_length,
                        'content': content,  # 完整内容
                        'content_preview': content_preview,
                        'keywords': keywords,
                        'analysis_results': analysis_results,
                        'created_at': doc['created_at']
                    })
            
            # 获取最常见的关键字（前10个）
            top_keywords = sorted(keywords_summary.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'count': len(documents) if documents else 0,
                'session_count': len(sessions) if sessions else 0,
                'document_count': len(documents) if documents else 0,
                'total_content_length': total_length,
                'keywords_count': len(keywords_summary),
                'top_keywords': top_keywords,
                'session_files': session_files,
                'document_details': document_details,
                'files': session_files,  # 添加files字段以保持一致性
                'analyses': document_details,  # 添加analyses字段以保持一致性
                'sessions': session_files,  # 会话信息
                'documents': document_details,  # 文档详情
                'all_keywords': list(keywords_summary.keys()),  # 所有关键词
                'type': '文本处理数据'
            }
            
        except Exception as e:
            self.logger.error(f"获取文本处理数据汇总失败: {e}")
            return {
                'count': 0,
                'session_count': 0,
                'document_count': 0,
                'total_content_length': 0,
                'keywords_count': 0,
                'top_keywords': [],
                'session_files': [],
                'document_details': [],
                'files': [],
                'analyses': [],
                'type': '文本处理数据'
            }
    
    def _get_tabular_summary(self, project_id: str) -> Dict[str, Any]:
        """获取表格数据汇总（Excel / CSV / 文档表格）"""
        try:
            from src.services.project_storage_service import ProjectStorageService
            storage = ProjectStorageService(self.db_service)
            tabular_dir = Path(str(storage.get_project_storage_path(project_id, "tabular") or ""))
            counts = {"excel": 0, "csv": 0, "word_table": 0, "sheets_excel": 0, "sheets_csv": 0, "sheets_word_table": 0, "files": []}
            if tabular_dir.is_dir():
                for jf in sorted(tabular_dir.glob("**/*.json"))[:200]:
                    try:
                        with open(jf, "r", encoding="utf-8") as f:
                            obj = json.load(f)
                        fname = obj.get("metadata", {}).get("source_file_name", jf.name)
                        tp = obj.get("sheet", {}).get("type") or obj.get("metadata", {}).get("type")
                        if tp in ("excel", "xlsx", "xls"):
                            counts["excel"] += 1
                            counts["sheets_excel"] += 1
                        elif tp in ("csv",):
                            counts["csv"] += 1
                            counts["sheets_csv"] += 1
                        elif tp in ("word_table", "word", "docx_table"):
                            counts["word_table"] += 1
                            counts["sheets_word_table"] += 1
                        else:
                            ext = Path(fname).suffix.lower() if fname else ""
                            if ext in (".xlsx", ".xls"):
                                counts["excel"] += 1
                                counts["sheets_excel"] += 1
                            elif ext == ".csv":
                                counts["csv"] += 1
                                counts["sheets_csv"] += 1
                            else:
                                counts["word_table"] += 1
                                counts["sheets_word_table"] += 1
                        counts["files"].append(fname)
                    except Exception:
                        pass
            return {
                "excel_count": counts["excel"],
                "csv_count": counts["csv"],
                "word_table_count": counts["word_table"],
                "total_sheets": counts["sheets_excel"] + counts["sheets_csv"] + counts["sheets_word_table"],
                "file_list": counts["files"],
                "type": "表格数据",
            }
        except Exception as e:
            self.logger.warning(f"获取表格汇总失败: {e}")
            return {"excel_count": 0, "csv_count": 0, "word_table_count": 0, "total_sheets": 0, "file_list": [], "type": "表格数据"}

    def _get_storage_summary(self, project_id: str) -> Dict[str, Any]:
        """获取存储信息汇总"""
        try:
            query = """
                SELECT 
                    quota_limit,
                    used_space,
                    storage_path
                FROM project_storage 
                WHERE project_id = ?
            """
            
            results = self.db_service.execute_query(query, (project_id,))
            
            if results:
                row = results[0]
                used_space = row['used_space'] or 0
                quota_limit = row['quota_limit'] or 0
                
                return {
                    'used_space': used_space,
                    'quota_limit': quota_limit,
                    'usage_percentage': (used_space / quota_limit * 100) if quota_limit > 0 else 0,
                    'storage_path': row['storage_path']
                }
            
            return {
                'used_space': 0,
                'quota_limit': 0,
                'usage_percentage': 0,
                'storage_path': ''
            }
            
        except Exception as e:
            self.logger.error(f"获取存储汇总失败: {e}")
            return {
                'used_space': 0,
                'quota_limit': 0,
                'usage_percentage': 0,
                'storage_path': ''
            }
    
    def _get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """获取文件信息"""
        # 检查file_path是否为None或空字符串
        if not file_path:
            return {
                'name': '',
                'path': '',
                'size': 0,
                'modified': '',
                'extension': '',
                'status': 'invalid_path'
            }
        
        try:
            path = Path(file_path)
            if path.exists():
                stat = path.stat()
                return {
                    'name': path.name,
                    'path': str(path),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'extension': path.suffix
                }
            else:
                # 文件不存在，返回基本信息
                return {
                    'name': Path(file_path).name,
                    'path': file_path,
                    'size': 0,
                    'modified': '',
                    'extension': Path(file_path).suffix,
                    'status': 'missing'
                }
        except Exception as e:
            self.logger.error(f"获取文件信息失败: {e}")
            return None
    
    def _get_empty_summary(self, project_id: str) -> Dict[str, Any]:
        """获取空的汇总信息"""
        return {
            'project_id': project_id,
            'dicom_data': {'count': 0, 'file_count': 0, 'files': [], 'type': 'DICOM影像文件'},
            'ocr_data': {'count': 0, 'texts': [], 'images': [], 'type': 'OCR文本提取'},
            'text_data': {'count': 0, 'files': [], 'analyses': [], 'type': '文本处理分析'},
            'magic_seg_data': {'total_count': 0, 'total_area': 0, 'source_counts': {}, 'seg_list': [], 'type': 'Magic Seg数据'},
            'storage_info': {
                'used_space': 0,
                'quota_limit': 0,
                'usage_percentage': 0,
                'storage_path': ''
            },
            'total_files': 0,
            'updated_at': datetime.now().isoformat()
        }
    
    def get_all_projects_for_cleaning(self) -> List[Dict[str, Any]]:
        """获取所有可用于数据清洗的项目列表"""
        try:
            projects = self.project_service.get_active_projects()
            project_list = []
            
            for project in projects:
                # 获取项目统计信息
                stats = self.project_service.get_project_stats(project.project_id)
                
                project_info = {
                    'project_id': project.project_id,
                    'name': project.name,
                    'description': project.description,
                    'created_at': project.created_at.isoformat(),
                    'stats': stats
                }
                
                project_list.append(project_info)
            
            return project_list
            
        except Exception as e:
            self.logger.error(f"获取项目列表失败: {e}")
            return []
