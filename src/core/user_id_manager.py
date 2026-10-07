#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户ID管理器模块

负责管理用户ID列表、选择状态和相关的数据同步功能。
用户ID作为样本名称，所有分析模块的数据都与选中的用户ID关联。
"""

import json
from typing import List, Optional, Dict, Any
from PyQt5.QtCore import QObject, pyqtSignal
from datetime import datetime

from src.services.database_service import DatabaseService
from src.utils.logger import get_logger


class UserIDManager(QObject):
    """
    用户ID管理器类
    
    管理用户ID列表、当前选择状态，并提供数据同步功能。
    """
    
    # 信号定义
    user_id_selected = pyqtSignal(str)  # 用户ID选择信号
    user_id_added = pyqtSignal(str)     # 用户ID添加信号
    user_id_removed = pyqtSignal(str)   # 用户ID删除信号
    user_id_list_updated = pyqtSignal(list)  # 用户ID列表更新信号
    
    def __init__(self, db_service: DatabaseService = None):
        """
        初始化用户ID管理器
        
        Args:
            db_service: 数据库服务实例
        """
        super().__init__()
        
        self.db_service = db_service or DatabaseService()
        self.logger = get_logger(__name__)
        
        self._current_user_id: Optional[str] = None
        self._user_id_list: List[str] = []
        
        # 初始化数据库表
        self._ensure_user_ids_table()
        
        # 加载现有用户ID列表
        self.refresh_user_id_list()
    
    def _ensure_user_ids_table(self):
        """
        确保user_ids表存在
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 创建user_ids表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_ids (
                        user_id TEXT PRIMARY KEY,
                        display_name TEXT NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_used TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                # 创建索引
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_ids_last_used 
                    ON user_ids(last_used DESC)
                """)
                
                conn.commit()
                self.logger.info("user_ids表初始化完成")
                
        except Exception as e:
            self.logger.error(f"初始化user_ids表失败: {e}")
    
    def add_user_id(self, user_id: str, display_name: str = None, 
                   description: str = None, metadata: Dict[str, Any] = None) -> bool:
        """
        添加新的用户ID
        
        Args:
            user_id: 用户ID（样本名称）
            display_name: 显示名称
            description: 描述信息
            metadata: 元数据
        
        Returns:
            是否添加成功
        """
        try:
            if not user_id or not user_id.strip():
                self.logger.warning("用户ID不能为空")
                return False
            
            user_id = user_id.strip()
            display_name = display_name or user_id
            metadata_json = json.dumps(metadata or {})
            
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 检查是否已存在
                cursor.execute("SELECT user_id FROM user_ids WHERE user_id = ?", (user_id,))
                if cursor.fetchone():
                    self.logger.warning(f"用户ID '{user_id}' 已存在")
                    return False
                
                # 插入新用户ID
                cursor.execute("""
                    INSERT INTO user_ids (user_id, display_name, description, metadata)
                    VALUES (?, ?, ?, ?)
                """, (user_id, display_name, description, metadata_json))
                
                conn.commit()
                
                # 更新内存列表
                self._user_id_list.append(user_id)
                
                # 发送信号
                self.user_id_added.emit(user_id)
                self.user_id_list_updated.emit(self._user_id_list.copy())
                
                self.logger.info(f"用户ID '{user_id}' 添加成功")
                return True
                
        except Exception as e:
            self.logger.error(f"添加用户ID失败: {e}")
            return False
    
    def remove_user_id(self, user_id: str) -> bool:
        """
        删除用户ID
        
        Args:
            user_id: 要删除的用户ID
        
        Returns:
            是否删除成功
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 删除用户ID记录
                cursor.execute("DELETE FROM user_ids WHERE user_id = ?", (user_id,))
                
                if cursor.rowcount == 0:
                    self.logger.warning(f"用户ID '{user_id}' 不存在")
                    return False
                
                conn.commit()
                
                # 更新内存列表
                if user_id in self._user_id_list:
                    self._user_id_list.remove(user_id)
                
                # 如果删除的是当前选中的用户ID，清除选择
                if self._current_user_id == user_id:
                    self._current_user_id = None
                
                # 发送信号
                self.user_id_removed.emit(user_id)
                self.user_id_list_updated.emit(self._user_id_list.copy())
                
                self.logger.info(f"用户ID '{user_id}' 删除成功")
                return True
                
        except Exception as e:
            self.logger.error(f"删除用户ID失败: {e}")
            return False
    
    def select_user_id(self, user_id: str) -> bool:
        """
        选择用户ID
        
        Args:
            user_id: 要选择的用户ID
        
        Returns:
            是否选择成功
        """
        try:
            if user_id not in self._user_id_list:
                self.logger.warning(f"用户ID '{user_id}' 不存在于列表中")
                return False
            
            # 更新最后使用时间
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE user_ids SET last_used = CURRENT_TIMESTAMP 
                    WHERE user_id = ?
                """, (user_id,))
                conn.commit()
            
            # 设置当前选择
            self._current_user_id = user_id
            
            # 发送信号
            self.user_id_selected.emit(user_id)
            
            self.logger.info(f"用户ID '{user_id}' 选择成功")
            return True
            
        except Exception as e:
            self.logger.error(f"选择用户ID失败: {e}")
            return False
    
    def get_current_user_id(self) -> Optional[str]:
        """
        获取当前选中的用户ID
        
        Returns:
            当前选中的用户ID，如果没有选择则返回None
        """
        return self._current_user_id
    
    def get_user_id_list(self) -> List[str]:
        """
        获取用户ID列表
        
        Returns:
            用户ID列表
        """
        return self._user_id_list.copy()
    
    def get_user_id_details_list(self) -> List[Dict[str, Any]]:
        """
        获取用户ID详细信息列表
        
        Returns:
            用户ID详细信息列表
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, display_name, description, created_at, 
                           last_used, metadata
                    FROM user_ids 
                    ORDER BY last_used DESC, created_at DESC
                """)
                
                rows = cursor.fetchall()
                result = []
                
                for row in rows:
                    result.append({
                        'user_id': row[0],
                        'display_name': row[1] or row[0],
                        'description': row[2],
                        'created_at': row[3],
                        'last_used': row[4],
                        'metadata': json.loads(row[5] or '{}')
                    })
                
                return result
                
        except Exception as e:
            self.logger.error(f"获取用户ID详细信息列表失败: {e}")
            return []
    
    def get_user_id_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        获取用户ID详细信息
        
        Args:
            user_id: 用户ID
        
        Returns:
            用户ID信息字典，如果不存在则返回None
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, display_name, description, created_at, 
                           last_used, metadata
                    FROM user_ids WHERE user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'user_id': row[0],
                        'display_name': row[1],
                        'description': row[2],
                        'created_at': row[3],
                        'last_used': row[4],
                        'metadata': json.loads(row[5] or '{}')
                    }
                
                return None
                
        except Exception as e:
            self.logger.error(f"获取用户ID信息失败: {e}")
            return None
    
    def refresh_user_id_list(self):
        """
        刷新用户ID列表
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id FROM user_ids 
                    ORDER BY last_used DESC, created_at DESC
                """)
                
                rows = cursor.fetchall()
                self._user_id_list = [row[0] for row in rows]
                
                # 发送更新信号
                self.user_id_list_updated.emit(self._user_id_list.copy())
                
                self.logger.info(f"用户ID列表刷新完成，共 {len(self._user_id_list)} 个用户ID")
                
        except Exception as e:
            self.logger.error(f"刷新用户ID列表失败: {e}")
    
    def clear_selection(self):
        """
        清除当前选择
        """
        self._current_user_id = None
        self.logger.info("用户ID选择已清除")
    
    def get_user_id_statistics(self) -> Dict[str, Any]:
        """
        获取用户ID统计信息
        
        Returns:
            统计信息字典
        """
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 总数
                cursor.execute("SELECT COUNT(*) FROM user_ids")
                total_count = cursor.fetchone()[0]
                
                # 最近使用的
                cursor.execute("""
                    SELECT user_id FROM user_ids 
                    WHERE last_used IS NOT NULL 
                    ORDER BY last_used DESC LIMIT 1
                """)
                recent_row = cursor.fetchone()
                recent_user_id = recent_row[0] if recent_row else None
                
                return {
                    'total_count': total_count,
                    'current_selection': self._current_user_id,
                    'recent_user_id': recent_user_id
                }
                
        except Exception as e:
            self.logger.error(f"获取用户ID统计信息失败: {e}")
            return {
                'total_count': 0,
                'current_selection': self._current_user_id,
                'recent_user_id': None
            }