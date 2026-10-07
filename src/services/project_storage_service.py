import os
import shutil
import sqlite3
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from src.core.path_config import get_coco_data_dir

class ProjectStorageService:
    """项目存储管理服务
    
    负责管理项目级别的文件存储，包括：
    - 项目存储目录的创建和管理
    - 文件的项目隔离存储
    - 存储空间配额管理
    - 文件索引和元数据管理
    - 备份恢复功能
    """
    
    def __init__(self, database_service):
        self.db_service = database_service
        app_root = get_coco_data_dir().parent
        self.base_storage_path = app_root / "project_storage"
        self.projects_path = self.base_storage_path / "projects"
        self.backup_path = self.base_storage_path / "backups"
        
        # 确保基础目录存在
        self._ensure_base_directories()
    
    def _ensure_base_directories(self):
        """确保基础存储目录存在"""
        self.base_storage_path.mkdir(exist_ok=True)
        self.projects_path.mkdir(exist_ok=True)
        self.backup_path.mkdir(exist_ok=True)
    
    def create_project_storage(self, project_id: str, quota_limit: int = 5368709120) -> bool:
        """为项目创建存储空间
        
        Args:
            project_id: 项目ID
            quota_limit: 存储配额限制（字节），默认5GB
            
        Returns:
            bool: 创建是否成功
        """
        try:
            # 创建项目存储目录结构
            project_path = self.projects_path / project_id
            
            # 创建子目录
            subdirs = [
                "dicom",           # DICOM文件
                "dicom/cache",     # DICOM缓存
                "ocr",             # OCR结果
                "ocr/images",      # OCR图像
                "text",            # 文本处理
                "documents",       # 文档文件
                "tabular",         # Excel/CSV结构化表格
                "temp",            # 临时文件
                "exports"          # 导出文件
            ]
            
            for subdir in subdirs:
                (project_path / subdir).mkdir(parents=True, exist_ok=True)
            
            # 在数据库中记录存储信息
            storage_path = f"project_storage/projects/{project_id}"
            
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO project_storage 
                    (project_id, storage_path, quota_limit, used_space, created_at, updated_at)
                    VALUES (?, ?, ?, 0, ?, ?)
                """, (
                    project_id,
                    storage_path,
                    quota_limit,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
            return True
            
        except Exception as e:
            print(f"创建项目存储失败: {e}")
            return False
    
    def get_project_storage_path(self, project_id: str, subdir: str = "") -> Path:
        """获取项目存储路径
        
        Args:
            project_id: 项目ID
            subdir: 子目录名称
            
        Returns:
            Path: 存储路径
        """
        project_path = self.projects_path / project_id
        if subdir:
            return project_path / subdir
        return project_path
    
    def store_file(self, project_id: str, file_path: str, file_type: str, 
                   metadata: Dict = None, move_file: bool = False) -> Optional[str]:
        """存储文件到项目目录
        
        Args:
            project_id: 项目ID
            file_path: 源文件路径
            file_type: 文件类型 (dicom, ocr_image, ocr_result, text, document)
            metadata: 文件元数据
            move_file: 是否移动文件（而非复制）
            
        Returns:
            str: 存储后的文件ID，失败返回None
        """
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                print(f"源文件不存在: {file_path}")
                return None
            
            # 检查存储配额
            if not self._check_storage_quota(project_id, source_path.stat().st_size):
                print(f"存储配额不足")
                return None
            
            # 生成文件ID和目标路径
            file_id = self._generate_file_id(file_path)
            file_extension = source_path.suffix
            
            # 根据文件类型确定存储子目录
            subdir_map = {
                'dicom': 'dicom',
                'dicom_cache': 'dicom/cache',
                'ocr_image': 'ocr/images',
                'ocr_result': 'ocr',
                'text': 'text',
                'document': 'documents',
                'tabular': 'tabular',
                'temp': 'temp',
                'export': 'exports'
            }
            
            subdir = subdir_map.get(file_type, 'temp')
            target_dir = self.get_project_storage_path(project_id, subdir)
            target_path = target_dir / f"{file_id}{file_extension}"
            
            # 确保目标目录存在
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制或移动文件
            if move_file:
                shutil.move(str(source_path), str(target_path))
            else:
                shutil.copy2(str(source_path), str(target_path))
            
            # 记录文件信息到数据库
            file_size = target_path.stat().st_size
            self._record_file_storage(project_id, file_id, str(target_path), 
                                    file_type, file_size, metadata)
            
            # 更新项目存储使用量
            self._update_storage_usage(project_id, file_size)
            
            return file_id
            
        except Exception as e:
            print(f"存储文件失败: {e}")
            return None
    
    def get_file_path(self, project_id: str, file_id: str) -> Optional[str]:
        """获取文件存储路径
        
        Args:
            project_id: 项目ID
            file_id: 文件ID
            
        Returns:
            str: 文件路径，不存在返回None
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT sf.file_path
                    FROM storage_files sf
                    JOIN project_storage ps ON sf.storage_id = ps.storage_id
                    WHERE ps.project_id = ? AND sf.file_id = ?
                """, (project_id, file_id))
                
                result = cursor.fetchone()
                return result[0] if result else None
            
        except Exception as e:
            print(f"获取文件路径失败: {e}")
            return None
    
    def delete_file(self, project_id: str, file_id: str) -> bool:
        """删除项目文件
        
        Args:
            project_id: 项目ID
            file_id: 文件ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            # 获取文件信息
            file_path = self.get_file_path(project_id, file_id)
            if not file_path:
                return False
            
            # 获取文件大小
            path_obj = Path(file_path)
            file_size = path_obj.stat().st_size if path_obj.exists() else 0
            
            # 删除物理文件
            if path_obj.exists():
                path_obj.unlink()
            
            # 从数据库删除记录
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM storage_files 
                    WHERE storage_id IN (
                        SELECT storage_id FROM project_storage WHERE project_id = ?
                    ) AND file_id = ?
                """, (project_id, file_id))
                
                conn.commit()
            
            # 更新存储使用量
            if file_size > 0:
                self._update_storage_usage(project_id, -file_size)
            
            return True
            
        except Exception as e:
            print(f"删除文件失败: {e}")
            return False
    
    def get_storage_info(self, project_id: str) -> Dict:
        """获取项目存储信息
        
        Args:
            project_id: 项目ID
            
        Returns:
            Dict: 存储信息
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT storage_path, quota_limit, used_space, created_at, updated_at
                    FROM project_storage
                    WHERE project_id = ?
                """, (project_id,))
                
                result = cursor.fetchone()
                if not result:
                    return {}
                
                return {
                    'project_id': project_id,
                    'storage_path': result[0],
                    'quota_limit': result[1],
                    'used_space': result[2],
                    'available_space': result[1] - result[2],
                    'usage_percentage': (result[2] / result[1]) * 100 if result[1] > 0 else 0,
                    'created_at': result[3],
                    'updated_at': result[4]
                }
            
        except Exception as e:
            print(f"获取存储信息失败: {e}")
            return {}
    
    def get_project_storage_info(self, project_id: str) -> Dict:
        """获取项目存储信息（别名方法）
        
        Args:
            project_id: 项目ID
            
        Returns:
            Dict: 存储信息
        """
        return self.get_storage_info(project_id)
    
    def list_project_files(self, project_id: str, file_type: str = None) -> List[Dict]:
        """列出项目文件
        
        Args:
            project_id: 项目ID
            file_type: 文件类型过滤
            
        Returns:
            List[Dict]: 文件列表
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                sql = """
                    SELECT sf.file_id, sf.file_path, sf.file_type, sf.file_size, 
                           sf.metadata, sf.created_at
                    FROM storage_files sf
                    JOIN project_storage ps ON sf.storage_id = ps.storage_id
                    WHERE ps.project_id = ?
                """
                
                params = [project_id]
                if file_type:
                    sql += " AND sf.file_type = ?"
                    params.append(file_type)
                
                sql += " ORDER BY sf.created_at DESC"
                
                cursor.execute(sql, params)
                results = cursor.fetchall()
            
            files = []
            for row in results:
                metadata = json.loads(row[4]) if row[4] else {}
                files.append({
                    'file_id': row[0],
                    'file_path': row[1],
                    'file_type': row[2],
                    'file_size': row[3],
                    'metadata': metadata,
                    'created_at': row[5]
                })
            
            return files
            
        except Exception as e:
            print(f"列出项目文件失败: {e}")
            return []
    
    def cleanup_temp_files(self, project_id: str, older_than_hours: int = 24) -> int:
        """清理临时文件
        
        Args:
            project_id: 项目ID
            older_than_hours: 清理多少小时前的文件
            
        Returns:
            int: 清理的文件数量
        """
        try:
            temp_dir = self.get_project_storage_path(project_id, "temp")
            if not temp_dir.exists():
                return 0
            
            cleaned_count = 0
            cutoff_time = datetime.now().timestamp() - (older_than_hours * 3600)
            
            for file_path in temp_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception:
                        continue
            
            return cleaned_count
            
        except Exception as e:
            print(f"清理临时文件失败: {e}")
            return 0
    
    def store_dicom_file(self, project_id: str, file_path: str, metadata: Dict = None) -> Optional[str]:
        """存储DICOM文件
        
        Args:
            project_id: 项目ID
            file_path: DICOM文件路径
            metadata: 文件元数据
            
        Returns:
            str: 存储后的文件路径，失败返回None
        """
        file_id = self.store_file(project_id, file_path, 'dicom', metadata)
        if file_id:
            return self.get_file_path(project_id, file_id)
        return None
    
    def get_text_file_path(self, project_id: str, file_id: str) -> Optional[str]:
        """获取文本文件路径
        
        Args:
            project_id: 项目ID
            file_id: 文件ID
            
        Returns:
            Optional[str]: 文件路径，失败返回None
        """
        return self.get_file_path(project_id, file_id)
    
    def load_text_analysis(self, project_id: str, file_id: str) -> Optional[Dict]:
        """加载文本分析结果
        
        Args:
            project_id: 项目ID
            file_id: 文件ID
            
        Returns:
            Optional[Dict]: 分析结果，失败返回None
        """
        try:
            file_path = self.get_file_path(project_id, file_id)
            if not file_path:
                return None
            
            path_obj = Path(file_path)
            if not path_obj.exists():
                return None
            
            # 如果是JSON文件，直接读取
            if path_obj.suffix.lower() == '.json':
                with open(path_obj, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # 否则返回文件基本信息
            return {
                'file_path': str(path_obj),
                'file_size': path_obj.stat().st_size,
                'modified_time': path_obj.stat().st_mtime
            }
            
        except Exception as e:
            print(f"加载文本分析结果失败: {e}")
            return None
    
    def store_roi_image(self, project_id: str, file_path: str, metadata: Dict = None) -> Optional[str]:
        """存储ROI图像文件
        
        Args:
            project_id: 项目ID
            file_path: 图像文件路径
            metadata: 文件元数据
            
        Returns:
            str: 存储后的文件路径，失败返回None
        """
        file_id = self.store_file(project_id, file_path, 'ocr_image', metadata)
        if file_id:
            return self.get_file_path(project_id, file_id)
        return None

    def store_ocr_image(self, project_id: str, file_path: str, metadata: Dict = None) -> Optional[str]:
        """存储OCR图像文件
        
        Args:
            project_id: 项目ID
            file_path: 图像文件路径
            metadata: 文件元数据
            
        Returns:
            str: 存储后的文件路径，失败返回None
        """
        file_id = self.store_file(project_id, file_path, 'ocr_image', metadata)
        if file_id:
            return self.get_file_path(project_id, file_id)
        return None
    
    def store_ocr_result(self, project_id: str, result_data: Dict, filename: str = None) -> Optional[str]:
        """存储OCR结果
        
        Args:
            project_id: 项目ID
            result_data: OCR结果数据
            filename: 文件名（可选）
            
        Returns:
            str: 存储后的文件路径，失败返回None
        """
        try:
            # 生成文件名
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"ocr_result_{timestamp}.json"
            
            # 创建临时文件
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / filename
            
            # 写入OCR结果
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            # 存储文件
            file_id = self.store_file(project_id, str(temp_file), 'ocr_result', {'type': 'ocr_result'})
            
            # 清理临时文件
            if temp_file.exists():
                temp_file.unlink()
            
            if file_id:
                return self.get_file_path(project_id, file_id)
            return None
            
        except Exception as e:
            print(f"存储OCR结果失败: {e}")
            return None
    
    def store_text_file(self, project_id: str, file_path: str, metadata: Dict = None) -> Optional[str]:
        """存储文本文件
        
        Args:
            project_id: 项目ID
            file_path: 文本文件路径
            metadata: 文件元数据
            
        Returns:
            str: 存储后的文件路径，失败返回None
        """
        file_id = self.store_file(project_id, file_path, 'text', metadata)
        if file_id:
            return self.get_file_path(project_id, file_id)
        return None
    
    def store_text_analysis(self, project_id: str, analysis_data: Dict, filename: str = None) -> Optional[str]:
        """存储文本分析结果
        
        Args:
            project_id: 项目ID
            analysis_data: 分析结果数据
            filename: 文件名（可选）
            
        Returns:
            str: 存储后的文件路径，失败返回None
        """
        try:
            # 生成文件名
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"text_analysis_{timestamp}.json"
            
            # 创建临时文件
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / filename
            
            # 写入分析结果
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            
            # 存储文件
            file_id = self.store_file(project_id, str(temp_file), 'text', {'type': 'text_analysis'})
            
            # 清理临时文件
            if temp_file.exists():
                temp_file.unlink()
            
            if file_id:
                return self.get_file_path(project_id, file_id)
            return None
            
        except Exception as e:
            print(f"存储文本分析结果失败: {e}")
            return None

    def store_tabular_json(self, project_id: str, user_id: str, tabular_payload: Dict, metadata: Dict = None) -> List[Dict]:
        """
        将 Excel/CSV 解析后的结构化表格以“项目-样本-文件-工作表”树状结构保存到项目存储。
        返回每个叶子节点（sheet）的索引信息，便于 Step3 按分支遍历。
        """
        payload = tabular_payload or {}
        sheets = payload.get("sheets") or []
        if not isinstance(sheets, list) or not sheets:
            return []

        results: List[Dict] = []
        try:
            project_root = self.get_project_storage_path(project_id, "tabular")
            safe_user = (user_id or "unknown").strip() or "unknown"
            project_root.mkdir(parents=True, exist_ok=True)
            (project_root / safe_user).mkdir(parents=True, exist_ok=True)

            src_name = (payload.get("source_file_name") or "").strip()
            src_stem = Path(src_name).stem if src_name else "tabular"
            safe_stem = "".join([c if c.isalnum() or c in ["-", "_"] else "_" for c in src_stem])[:80] or "tabular"
            base_dir = project_root / safe_user / safe_stem
            base_dir.mkdir(parents=True, exist_ok=True)

            for s in sheets:
                if not isinstance(s, dict):
                    continue
                sheet_name = (s.get("sheet_name") or s.get("name") or "Sheet1").strip()
                safe_sheet = "".join([c if c.isalnum() or c in ["-", "_"] else "_" for c in sheet_name])[:80] or "Sheet1"
                out_path = base_dir / f"{safe_sheet}.json"

                node_meta = dict(metadata or {})
                node_meta.update(
                    {
                        "user_id": safe_user,
                        "source_file_name": src_name,
                        "sheet_name": sheet_name,
                        "branch_path": (node_meta.get("branch_path") or "").strip("/"),
                        "tabular_type": payload.get("type") or "",
                        "domain_tag": node_meta.get("domain_tag") or "Tabular",
                        "priority_weight": float(node_meta.get("priority_weight") or 2.0),
                        "headers": s.get("headers") or [],
                        "n_rows": int(s.get("n_rows") or 0),
                        "n_cols": int(s.get("n_cols") or 0),
                    }
                )
                if node_meta.get("branch_path"):
                    node_meta["branch_path"] = f"{node_meta['branch_path']}/{sheet_name}"
                else:
                    node_meta["branch_path"] = sheet_name

                node = {"metadata": node_meta, "sheet": s}
                data = json.dumps(node, ensure_ascii=False, indent=2)
                data_bytes = data.encode("utf-8", errors="ignore")
                if not self._check_storage_quota(project_id, len(data_bytes)):
                    continue

                with open(out_path, "wb") as f:
                    f.write(data_bytes)

                file_size = out_path.stat().st_size
                file_id = self._generate_file_id(f"{project_id}::{safe_user}::{src_name}::{sheet_name}")
                self._record_file_storage(project_id, file_id, str(out_path), "tabular", file_size, node_meta)
                self._update_storage_usage(project_id, file_size)
                results.append({"file_id": file_id, "file_path": str(out_path), "sheet_name": sheet_name, "user_id": safe_user})

        except Exception as e:
            print(f"存储Tabular JSON失败: {e}")
            return results
        return results

    def get_schema_cache_path(self, project_id: str) -> Path:
        return self.get_project_storage_path(project_id, "tabular") / "schema_cache.json"

    def load_schema_cache(self, project_id: str) -> Dict:
        p = self.get_schema_cache_path(project_id)
        if not p.exists():
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def store_schema_cache(self, project_id: str, schema_cache: Dict) -> Optional[str]:
        try:
            p = self.get_schema_cache_path(project_id)
            p.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(schema_cache or {}, ensure_ascii=False, indent=2).encode("utf-8", errors="ignore")
            if not self._check_storage_quota(project_id, len(data)):
                return None
            with open(p, "wb") as f:
                f.write(data)
            file_size = p.stat().st_size
            file_id = self._generate_file_id(f"{project_id}::schema_cache")
            meta = {
                "project_id": project_id,
                "type": "schema_cache",
                "created_at": datetime.now().isoformat(),
            }
            self._record_file_storage(project_id, file_id, str(p), "tabular", file_size, meta)
            self._update_storage_usage(project_id, file_size)
            return str(p)
        except Exception:
            return None

    def store_export_json(self, project_id: str, file_name: str, payload: Dict) -> Optional[str]:
        try:
            p = self.get_project_storage_path(project_id, "exports") / file_name
            p.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(payload or {}, ensure_ascii=False, indent=2).encode("utf-8", errors="ignore")
            if not self._check_storage_quota(project_id, len(data)):
                return None
            with open(p, "wb") as f:
                f.write(data)
            file_size = p.stat().st_size
            file_id = self._generate_file_id(f"{project_id}::export::{file_name}")
            meta = {
                "project_id": project_id,
                "type": "export_json",
                "export_name": file_name,
                "created_at": datetime.now().isoformat(),
            }
            self._record_file_storage(project_id, file_id, str(p), "export", file_size, meta)
            self._update_storage_usage(project_id, file_size)
            return str(p)
        except Exception:
            return None
    
    def _generate_file_id(self, file_path: str) -> str:
        """生成文件ID"""
        timestamp = str(int(datetime.now().timestamp() * 1000000))
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return f"{timestamp}_{file_hash}"
    
    def _check_storage_quota(self, project_id: str, file_size: int) -> bool:
        """检查存储配额"""
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT quota_limit, used_space
                    FROM project_storage
                    WHERE project_id = ?
                """, (project_id,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                quota_limit, used_space = result
                return (used_space + file_size) <= quota_limit
                
        except Exception:
            return False
    
    def _record_file_storage(self, project_id: str, file_id: str, file_path: str,
                           file_type: str, file_size: int, metadata: Dict = None):
        """记录文件存储信息"""
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 获取存储ID
                cursor.execute("""
                    SELECT storage_id FROM project_storage WHERE project_id = ?
                """, (project_id,))
                
                storage_result = cursor.fetchone()
                if not storage_result:
                    return
                
                storage_id = storage_result[0]
                metadata_json = json.dumps(metadata) if metadata else None
                
                cursor.execute("""
                    INSERT INTO storage_files 
                    (storage_id, file_id, file_path, file_type, file_size, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    storage_id,
                    file_id,
                    file_path,
                    file_type,
                    file_size,
                    metadata_json,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                
        except Exception as e:
            print(f"记录文件存储信息失败: {e}")
    
    def _update_storage_usage(self, project_id: str, size_delta: int):
        """更新存储使用量"""
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE project_storage 
                    SET used_space = used_space + ?, updated_at = ?
                    WHERE project_id = ?
                """, (size_delta, datetime.now().isoformat(), project_id))
                
                conn.commit()
                
        except Exception as e:
            print(f"更新存储使用量失败: {e}")
