#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
样本服务模块

提供样本数据的业务逻辑处理，包括样本与项目的关联映射。
"""

import os
import sys
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.services.database_service import DatabaseService
from src.utils.logger import get_logger


class SampleService:
    """
    样本服务类
    
    提供样本数据的增删改查和项目关联功能。
    """
    
    def __init__(self):
        """
        初始化样本服务
        """
        self.db_service = DatabaseService()
        self.logger = get_logger(__name__)
        
    def get_samples_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        """
        根据项目ID获取样本列表
        
        Args:
            project_id: 项目ID
            
        Returns:
            List[Dict[str, Any]]: 样本列表
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, display_name, description, project_id,
                           created_at, last_used, metadata, is_active
                    FROM user_ids
                    WHERE project_id = ? OR (project_id IS NULL AND ? = 'default')
                    ORDER BY created_at DESC
                """, (project_id, project_id))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"获取项目样本失败: {e}")
            return []
            
    def get_sample_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        根据样本ID获取样本详情
        
        Args:
            user_id: 样本ID
            
        Returns:
            Optional[Dict[str, Any]]: 样本详情，如果不存在返回None
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, display_name, description, project_id,
                           created_at, last_used, metadata, is_active
                    FROM user_ids
                    WHERE user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except Exception as e:
            self.logger.error(f"获取样本详情失败: {e}")
            return None
            
    def create_sample(self, user_id: str, display_name: str, project_id: str,
                     description: str = "", metadata: str = "", is_active: bool = True) -> bool:
        """
        创建新样本
        
        Args:
            user_id: 样本ID
            display_name: 显示名称
            project_id: 项目ID
            description: 描述
            metadata: 元数据（JSON字符串）
            is_active: 是否激活
            
        Returns:
            bool: 创建是否成功
        """
        try:
            # 验证元数据格式
            if metadata:
                try:
                    json.loads(metadata)
                except json.JSONDecodeError:
                    self.logger.error("元数据格式无效")
                    return False
            
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 检查样本ID是否已存在
                cursor.execute("SELECT COUNT(*) FROM user_ids WHERE user_id = ?", (user_id,))
                if cursor.fetchone()[0] > 0:
                    self.logger.error(f"样本ID '{user_id}' 已存在")
                    return False
                
                # 创建样本
                cursor.execute("""
                    INSERT INTO user_ids (user_id, display_name, description, project_id,
                                        created_at, last_used, metadata, is_active)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
                """, (user_id, display_name, description, project_id, metadata, 1 if is_active else 0))
                
                conn.commit()
                
            self.logger.info(f"创建样本成功: {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"创建样本失败: {e}")
            return False
            
    def update_sample(self, user_id: str, display_name: str = None, description: str = None,
                     metadata: str = None, is_active: bool = None) -> bool:
        """
        更新样本信息
        
        Args:
            user_id: 样本ID
            display_name: 显示名称
            description: 描述
            metadata: 元数据（JSON字符串）
            is_active: 是否激活
            
        Returns:
            bool: 更新是否成功
        """
        try:
            # 验证元数据格式
            if metadata is not None:
                try:
                    json.loads(metadata)
                except json.JSONDecodeError:
                    self.logger.error("元数据格式无效")
                    return False
            
            # 构建更新语句
            update_fields = []
            params = []
            
            if display_name is not None:
                update_fields.append("display_name = ?")
                params.append(display_name)
                
            if description is not None:
                update_fields.append("description = ?")
                params.append(description)
                
            if metadata is not None:
                update_fields.append("metadata = ?")
                params.append(metadata)
                
            if is_active is not None:
                update_fields.append("is_active = ?")
                params.append(1 if is_active else 0)
                
            if not update_fields:
                return True  # 没有需要更新的字段
                
            update_fields.append("last_used = CURRENT_TIMESTAMP")
            params.append(user_id)
            
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                sql = f"UPDATE user_ids SET {', '.join(update_fields)} WHERE user_id = ?"
                cursor.execute(sql, params)
                
                conn.commit()
                
            self.logger.info(f"更新样本成功: {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"更新样本失败: {e}")
            return False
            
    def delete_sample(self, user_id: str) -> bool:
        """
        删除样本及其相关数据
        
        Args:
            user_id: 样本ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 删除相关数据（级联删除）
                # 删除文档数据
                cursor.execute("""
                    DELETE FROM documents 
                    WHERE session_id IN (
                        SELECT session_id FROM text_sessions WHERE user_id = ?
                    )
                """, (user_id,))
                
                # 删除文本处理会话
                cursor.execute("DELETE FROM text_sessions WHERE user_id = ?", (user_id,))
                
                # 删除OCR结果
                cursor.execute("""
                    DELETE FROM ocr_results 
                    WHERE session_id IN (
                        SELECT session_id FROM ocr_sessions WHERE user_id = ?
                    )
                """, (user_id,))
                
                # 删除OCR会话
                cursor.execute("DELETE FROM ocr_sessions WHERE user_id = ?", (user_id,))
                
                # 删除ROI数据
                cursor.execute("""
                    DELETE FROM roi_data 
                    WHERE session_id IN (
                        SELECT session_id FROM dicom_sessions WHERE user_id = ?
                    )
                """, (user_id,))
                
                # 删除DICOM会话
                cursor.execute("DELETE FROM dicom_sessions WHERE user_id = ?", (user_id,))
                
                # 删除样本
                cursor.execute("DELETE FROM user_ids WHERE user_id = ?", (user_id,))
                
                conn.commit()
                
            self.logger.info(f"删除样本成功: {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"删除样本失败: {e}")
            return False
            
    def migrate_samples_to_project(self, target_project_id: str, sample_ids: List[str] = None) -> Tuple[int, int]:
        """
        将样本迁移到指定项目
        
        Args:
            target_project_id: 目标项目ID
            sample_ids: 要迁移的样本ID列表，如果为None则迁移所有无项目关联的样本
            
        Returns:
            Tuple[int, int]: (成功迁移数量, 失败数量)
        """
        success_count = 0
        failed_count = 0
        
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                if sample_ids:
                    # 迁移指定样本
                    for user_id in sample_ids:
                        try:
                            cursor.execute("""
                                UPDATE user_ids 
                                SET project_id = ?, last_used = CURRENT_TIMESTAMP
                                WHERE user_id = ?
                            """, (target_project_id, user_id))
                            
                            if cursor.rowcount > 0:
                                success_count += 1
                            else:
                                failed_count += 1
                                
                        except Exception as e:
                            self.logger.error(f"迁移样本 {user_id} 失败: {e}")
                            failed_count += 1
                else:
                    # 迁移所有无项目关联的样本
                    cursor.execute("""
                        UPDATE user_ids 
                        SET project_id = ?, last_used = CURRENT_TIMESTAMP
                        WHERE project_id IS NULL
                    """)
                    
                    success_count = cursor.rowcount
                
                conn.commit()
                
            self.logger.info(f"样本迁移完成: 成功 {success_count}, 失败 {failed_count}")
            
        except Exception as e:
            self.logger.error(f"样本迁移失败: {e}")
            failed_count += len(sample_ids) if sample_ids else 0
            
        return success_count, failed_count
        
    def get_sample_statistics(self, project_id: str = None) -> Dict[str, Any]:
        """
        获取样本统计信息
        
        Args:
            project_id: 项目ID，如果为None则统计所有样本
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {
                    'total_samples': 0,
                    'active_samples': 0,
                    'inactive_samples': 0,
                    'samples_with_dicom_sessions': 0,
                    'samples_with_ocr_sessions': 0,
                    'samples_with_text_sessions': 0
                }
                
                # 基础统计
                if project_id:
                    cursor.execute("""
                        SELECT COUNT(*) as total,
                               SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active
                        FROM user_ids
                        WHERE project_id = ? OR (project_id IS NULL AND ? = 'default')
                    """, (project_id, project_id))
                else:
                    cursor.execute("""
                        SELECT COUNT(*) as total,
                               SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active
                        FROM user_ids
                    """)
                
                row = cursor.fetchone()
                if row:
                    stats['total_samples'] = row[0] or 0
                    stats['active_samples'] = row[1] or 0
                    stats['inactive_samples'] = stats['total_samples'] - stats['active_samples']
                
                # 会话统计
                if project_id:
                    # DICOM会话
                    cursor.execute("""
                        SELECT COUNT(DISTINCT u.user_id)
                        FROM user_ids u
                        INNER JOIN dicom_sessions d ON u.user_id = d.user_id
                        WHERE u.project_id = ? OR (u.project_id IS NULL AND ? = 'default')
                    """, (project_id, project_id))
                    stats['samples_with_dicom_sessions'] = cursor.fetchone()[0] or 0
                    
                    # OCR会话
                    cursor.execute("""
                        SELECT COUNT(DISTINCT u.user_id)
                        FROM user_ids u
                        INNER JOIN ocr_sessions o ON u.user_id = o.user_id
                        WHERE u.project_id = ? OR (u.project_id IS NULL AND ? = 'default')
                    """, (project_id, project_id))
                    stats['samples_with_ocr_sessions'] = cursor.fetchone()[0] or 0
                    
                    # 文本处理会话
                    cursor.execute("""
                        SELECT COUNT(DISTINCT u.user_id)
                        FROM user_ids u
                        INNER JOIN text_sessions t ON u.user_id = t.user_id
                        WHERE u.project_id = ? OR (u.project_id IS NULL AND ? = 'default')
                    """, (project_id, project_id))
                    stats['samples_with_text_sessions'] = cursor.fetchone()[0] or 0
                else:
                    # 全局统计
                    cursor.execute("""
                        SELECT COUNT(DISTINCT u.user_id)
                        FROM user_ids u
                        INNER JOIN dicom_sessions d ON u.user_id = d.user_id
                    """)
                    stats['samples_with_dicom_sessions'] = cursor.fetchone()[0] or 0
                    
                    cursor.execute("""
                        SELECT COUNT(DISTINCT u.user_id)
                        FROM user_ids u
                        INNER JOIN ocr_sessions o ON u.user_id = o.user_id
                    """)
                    stats['samples_with_ocr_sessions'] = cursor.fetchone()[0] or 0
                    
                    cursor.execute("""
                        SELECT COUNT(DISTINCT u.user_id)
                        FROM user_ids u
                        INNER JOIN text_sessions t ON u.user_id = t.user_id
                    """)
                    stats['samples_with_text_sessions'] = cursor.fetchone()[0] or 0
                
                return stats
                
        except Exception as e:
            self.logger.error(f"获取样本统计失败: {e}")
            return {
                'total_samples': 0,
                'active_samples': 0,
                'inactive_samples': 0,
                'samples_with_dicom_sessions': 0,
                'samples_with_ocr_sessions': 0,
                'samples_with_text_sessions': 0
            }
            
    def generate_sample_id(self, prefix: str = "sample") -> str:
        """
        生成唯一的样本ID
        
        Args:
            prefix: ID前缀
            
        Returns:
            str: 生成的样本ID
        """
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
        
    def validate_sample_data(self, user_id: str, display_name: str, metadata: str = "") -> Tuple[bool, str]:
        """
        验证样本数据
        
        Args:
            user_id: 样本ID
            display_name: 显示名称
            metadata: 元数据
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        if not user_id or not user_id.strip():
            return False, "样本ID不能为空"
            
        if not display_name or not display_name.strip():
            return False, "显示名称不能为空"
            
        if metadata:
            try:
                json.loads(metadata)
            except json.JSONDecodeError:
                return False, "元数据必须是有效的JSON格式"
                
        return True, ""
        
    def search_samples(self, project_id: str, keyword: str) -> List[Dict[str, Any]]:
        """
        搜索样本
        
        Args:
            project_id: 项目ID
            keyword: 搜索关键词
            
        Returns:
            List[Dict[str, Any]]: 匹配的样本列表
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                search_pattern = f"%{keyword}%"
                cursor.execute("""
                    SELECT user_id, display_name, description, project_id,
                           created_at, last_used, metadata, is_active
                    FROM user_ids
                    WHERE (project_id = ? OR (project_id IS NULL AND ? = 'default'))
                      AND (user_id LIKE ? OR display_name LIKE ? OR description LIKE ?)
                    ORDER BY created_at DESC
                """, (project_id, project_id, search_pattern, search_pattern, search_pattern))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"搜索样本失败: {e}")
            return []