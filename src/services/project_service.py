import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import uuid

from src.services.database_service import DatabaseService
from src.models.project import Project
from src.utils.logger import get_logger


class ProjectService:
    """项目管理服务类"""
    
    def __init__(self, db_service: DatabaseService = None):
        self.db_service = db_service or DatabaseService()
        self.logger = get_logger(__name__)
        self._ensure_default_project()
    
    def _ensure_default_project(self):
        """确保存在默认项目"""
        try:
            # 检查是否存在任何项目
            projects = self.get_all_projects()
            if not projects:
                # 创建默认项目
                default_project = Project(
                    project_id="default",
                    name="默认项目",
                    description="系统默认项目，包含所有历史数据",
                    status="active"
                )
                self.create_project(default_project)
                self.logger.info("创建默认项目")
                
        except Exception as e:
            self.logger.error(f"确保默认项目失败: {e}")
    
    def create_project(self, project: Project) -> bool:
        """创建新项目"""
        try:
            query = """
                INSERT INTO projects (project_id, name, description, status, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                project.project_id,
                project.name,
                project.description,
                project.status,
                json.dumps(project.metadata),
                project.created_at.isoformat(),
                project.updated_at.isoformat()
            )
            
            self.db_service.execute_query(query, params)
            self.logger.info(f"项目创建成功: {project.name} ({project.project_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"创建项目失败: {e}")
            return False
    
    def get_project_by_id(self, project_id: str) -> Optional[Project]:
        """根据ID获取项目"""
        try:
            query = """
                SELECT project_id, name, description, created_at, updated_at, status, metadata
                FROM projects
                WHERE project_id = ?
            """
            
            results = self.db_service.execute_query(query, (project_id,))
            if results and len(results) > 0:
                row = results[0]
                return Project(
                    project_id=row['project_id'],
                    name=row['name'],
                    description=row['description'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    status=row['status'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
            return None
                
        except Exception as e:
            self.logger.error(f"获取项目失败: {e}")
            return None
    
    def get_project_by_name(self, name: str) -> Optional[Project]:
        """根据名称获取项目"""
        try:
            sql = "SELECT * FROM projects WHERE name = ?"
            
            with self.db_service.get_connection() as conn:
                cursor = conn.execute(sql, (name,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_project(row)
                return None
                
        except Exception as e:
            self.logger.error(f"获取项目失败: {e}")
            return None
    
    def get_all_projects(self, status: str = None) -> List[Project]:
        """获取所有项目"""
        try:
            if status:
                query = """
                    SELECT project_id, name, description, created_at, updated_at, status, metadata
                    FROM projects
                    WHERE status = ?
                    ORDER BY created_at DESC
                """
                results = self.db_service.execute_query(query, (status,))
            else:
                query = """
                    SELECT project_id, name, description, created_at, updated_at, status, metadata
                    FROM projects
                    ORDER BY created_at DESC
                """
                results = self.db_service.execute_query(query)
            
            projects = []
            if results:
                for row in results:
                    projects.append(Project(
                        project_id=row['project_id'],
                        name=row['name'],
                        description=row['description'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at']),
                        status=row['status'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else {}
                    ))
            
            return projects
                
        except Exception as e:
            self.logger.error(f"获取项目列表失败: {e}")
            return []
    
    def get_active_projects(self) -> List[Project]:
        """获取活跃项目"""
        return self.get_all_projects(status='active')
    
    def update_project(self, project: Project) -> bool:
        """更新项目"""
        try:
            query = """
                UPDATE projects
                SET name = ?, description = ?, updated_at = ?, status = ?, metadata = ?
                WHERE project_id = ?
            """
            
            params = (
                project.name,
                project.description,
                project.updated_at.isoformat(),
                project.status,
                json.dumps(project.metadata),
                project.project_id
            )
            
            result = self.db_service.execute_query(query, params)
            self.logger.info(f"项目更新成功: {project.name}")
            return True
                    
        except Exception as e:
            self.logger.error(f"更新项目失败: {e}")
            return False
    
    def delete_project(self, project_id: str) -> bool:
        """删除项目（软删除）"""
        try:
            query = """
                UPDATE projects
                SET status = 'archived', updated_at = ?
                WHERE project_id = ?
            """
            
            params = (datetime.now().isoformat(), project_id)
            result = self.db_service.execute_query(query, params)
            self.logger.info(f"项目删除成功: {project_id}")
            return True
                    
        except Exception as e:
            self.logger.error(f"删除项目失败: {e}")
            return False
    
    def archive_project(self, project_id: str) -> bool:
        """归档项目"""
        try:
            project = self.get_project_by_id(project_id)
            if project:
                project.archive()
                return self.update_project(project)
            return False
        except Exception as e:
            self.logger.error(f"归档项目失败: {e}")
            return False
    
    def activate_project(self, project_id: str) -> bool:
        """激活项目"""
        try:
            project = self.get_project_by_id(project_id)
            if project:
                project.activate()
                return self.update_project(project)
            return False
        except Exception as e:
            self.logger.error(f"激活项目失败: {e}")
            return False
    
    def get_current_project_id(self) -> str:
        """获取当前活跃项目ID"""
        try:
            query = "SELECT project_id FROM projects WHERE status = 'active' ORDER BY updated_at DESC LIMIT 1"
            results = self.db_service.execute_query(query)
            if results and len(results) > 0:
                return results[0]['project_id']
            
            # 如果没有活跃项目，返回默认项目
            return "default"
            
        except Exception as e:
            self.logger.error(f"获取当前项目ID失败: {e}")
            return "default"
    

    
    def get_project_stats(self, project_id: str) -> dict:
        """获取项目统计信息"""
        try:
            # 获取DICOM会话数量
            dicom_query = "SELECT COUNT(*) as count FROM dicom_sessions WHERE project_id = ?"
            dicom_result = self.db_service.execute_query(dicom_query, (project_id,))
            dicom_count = dicom_result[0]['count'] if dicom_result else 0
            
            # 获取OCR会话数量
            ocr_query = "SELECT COUNT(*) as count FROM ocr_sessions WHERE project_id = ?"
            ocr_result = self.db_service.execute_query(ocr_query, (project_id,))
            ocr_count = ocr_result[0]['count'] if ocr_result else 0
            
            # 获取文本处理会话数量
            text_query = "SELECT COUNT(*) as count FROM text_sessions WHERE project_id = ?"
            text_result = self.db_service.execute_query(text_query, (project_id,))
            text_count = text_result[0]['count'] if text_result else 0
            
            return {
                'dicom_sessions': dicom_count,
                'ocr_sessions': ocr_count,
                'text_sessions': text_count,
                'total_sessions': dicom_count + ocr_count + text_count
            }
                
        except Exception as e:
            self.logger.error(f"获取项目统计失败: {e}")
            return {
                'dicom_sessions': 0,
                'ocr_sessions': 0,
                'text_sessions': 0,
                'total_sessions': 0
            }

    def get_project_ocr_data(self, project_id: str) -> List[Dict[str, Any]]:
        """
        获取项目的OCR数据结构（用于CGR模式的数据范围选择）
        返回结构：List[OCR Session Dict with Results]
        """
        try:
            # 获取该项目的所有OCR会话
            query = """
                SELECT 
                    s.session_id, s.image_path as image_name, s.created_at,
                    r.result_id, r.extracted_text, r.recognized_text, r.confidence
                FROM ocr_sessions s
                LEFT JOIN ocr_results r ON s.session_id = r.session_id
                WHERE s.project_id = ?
                ORDER BY s.created_at DESC
            """
            
            results = self.db_service.execute_query(query, (project_id,))
            
            # 组织数据结构
            sessions_map = {}
            for row in results:
                sess_id = row['session_id']
                if sess_id not in sessions_map:
                    sessions_map[sess_id] = {
                        'id': sess_id,
                        'name': row['image_name'] or f"Session {sess_id[:8]}",
                        'type': 'ocr_session',
                        'children': []
                    }
                
                if row['result_id']:
                    # 优先使用 extracted_text (新标准)，回退到 recognized_text
                    text_content = row['extracted_text'] or row['recognized_text']
                    sessions_map[sess_id]['children'].append({
                        'id': row['result_id'],
                        'name': f"Result (Conf: {row['confidence']:.2f})",
                        'type': 'ocr_result',
                        'content': text_content
                    })
            
            return list(sessions_map.values())
            
        except Exception as e:
            self.logger.error(f"获取项目OCR数据失败: {e}")
            return []

    # --- Draft Management (Step 5 Persistence) ---
    def save_draft(self, draft_data: dict) -> bool:
        """保存科学写作草稿"""
        try:
            # 兼容性处理：如果 draft_data 没有 'content' 但有 'units'，则合并 units 生成 content
            content = draft_data.get('content')
            if not content and 'units' in draft_data:
                # 按照标准顺序合并
                import re
                units = draft_data['units']
                sections = []
                section_labels = {
                    'intro': 'Introduction', 'methods': 'Methods', 'results': 'Results',
                    'discussion': 'Discussion', 'conclusion': 'Conclusion', 'abstract': 'Abstract', 'meta': 'Abstract',
                }
                for key in ['abstract', 'meta', 'intro', 'methods', 'results', 'discussion', 'conclusion']:
                    if key not in units:
                        continue
                    unit = units[key]
                    if isinstance(unit, dict):
                        text = str(unit.get('content', '') or '')
                    else:
                        text = str(getattr(unit, 'content', '') or '')
                    label = section_labels.get(key, key.capitalize())
                    cleaned = re.sub(rf'(?im)^##?\s*{re.escape(label)}\s*\n?', '', text, count=1).strip()
                    if cleaned and len(cleaned) > len(text) * 0.3:
                        text = cleaned
                    if text:
                        sections.append(f"## {label}\n\n{text}")
                content = "\n\n".join(sections)
                
            # 如果还是空的，使用 JSON dump 作为 fallback，避免报错
            if not content:
                content = json.dumps(draft_data, default=str)

            query = """
                INSERT OR REPLACE INTO drafts 
                (id, project_id, title, content, score, logs, model, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                draft_data['id'],
                draft_data['project_id'],
                draft_data.get('title'),
                content,
                draft_data.get('score', 0),
                draft_data.get('logs', ''),
                draft_data.get('model', 'unknown'),
                datetime.now().isoformat()
            )
            self.db_service.execute_query(query, params)
            return True
        except Exception as e:
            self.logger.error(f"保存草稿失败: {e}")
            return False

    def get_project_drafts(self, project_id: str) -> List[dict]:
        """获取项目的所有草稿"""
        try:
            query = "SELECT * FROM drafts WHERE project_id = ? ORDER BY updated_at DESC"
            results = self.db_service.execute_query(query, (project_id,))
            return [dict(row) for row in results]
        except Exception as e:
            self.logger.error(f"获取草稿列表失败: {e}")
            return []

    def purge_project(self, project_id: str) -> bool:
        """彻底清理项目的所有数据，包括物理文件和数据库记录"""
        if project_id == "default":
            self.logger.warning("不能删除默认项目")
            return False
            
        try:
            # 1. 删除相关的物理文件 (Project Storage)
            try:
                from src.services.project_storage_service import ProjectStorageService
                storage_service = ProjectStorageService(self.db_service)
                storage_service.delete_project_storage(project_id)
            except Exception as e:
                self.logger.warning(f"删除项目物理文件失败，将继续清理数据库: {e}")
                
            # 2. 删除数据库记录 (采用事务保证原子性)
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                table_names = {r[0] for r in cursor.fetchall()}
                
                # 删除 OCR 相关数据
                if "ocr_results" in table_names and "ocr_sessions" in table_names:
                    cursor.execute("DELETE FROM ocr_results WHERE session_id IN (SELECT session_id FROM ocr_sessions WHERE project_id = ?)", (project_id,))
                if "ocr_sessions" in table_names:
                    cursor.execute("DELETE FROM ocr_sessions WHERE project_id = ?", (project_id,))
                
                # 删除 DICOM 相关数据
                if "roi_data" in table_names and "dicom_sessions" in table_names:
                    cursor.execute("DELETE FROM roi_data WHERE session_id IN (SELECT session_id FROM dicom_sessions WHERE project_id = ?)", (project_id,))
                # 兼容历史表名
                if "roi_annotations" in table_names and "dicom_sessions" in table_names:
                    cursor.execute("DELETE FROM roi_annotations WHERE session_id IN (SELECT session_id FROM dicom_sessions WHERE project_id = ?)", (project_id,))
                if "dicom_sessions" in table_names:
                    cursor.execute("DELETE FROM dicom_sessions WHERE project_id = ?", (project_id,))
                
                # 删除文本处理相关数据
                if "documents" in table_names and "text_sessions" in table_names:
                    cursor.execute("DELETE FROM documents WHERE session_id IN (SELECT session_id FROM text_sessions WHERE project_id = ?)", (project_id,))
                if "text_sessions" in table_names:
                    cursor.execute("DELETE FROM text_sessions WHERE project_id = ?", (project_id,))
                if "content_chunks" in table_names:
                    cursor.execute("DELETE FROM content_chunks WHERE project_id = ?", (project_id,))
                
                # 删除草稿
                if "drafts" in table_names:
                    cursor.execute("DELETE FROM drafts WHERE project_id = ?", (project_id,))
                if "backup_records" in table_names:
                    cursor.execute("DELETE FROM backup_records WHERE project_id = ?", (project_id,))
                if "user_ids" in table_names:
                    cursor.execute("DELETE FROM user_ids WHERE project_id = ?", (project_id,))
                if "storage_files" in table_names and "project_storage" in table_names:
                    cursor.execute("DELETE FROM storage_files WHERE storage_id IN (SELECT storage_id FROM project_storage WHERE project_id = ?)", (project_id,))
                if "project_storage" in table_names:
                    cursor.execute("DELETE FROM project_storage WHERE project_id = ?", (project_id,))
                
                # 最后删除项目本身
                if "projects" in table_names:
                    cursor.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
                
                conn.commit()
                
            self.logger.info(f"项目 {project_id} 及其所有关联数据已成功清理")
            return True
            
        except Exception as e:
            self.logger.error(f"清理项目 {project_id} 失败: {e}")
            return False
