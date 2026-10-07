#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户服务模块

负责用户管理相关的业务逻辑，包括用户认证、创建、信息管理等。
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, Optional, List

from src.services.database_service import DatabaseService
from src.utils.logger import get_logger


class UserService:
    """
    用户服务类
    """
    
    def __init__(self, db_service: DatabaseService = None):
        """
        初始化用户服务
        
        Args:
            db_service: 数据库服务实例
        """
        self.db_service = db_service or DatabaseService()
        self.logger = get_logger(__name__)
    
    def authenticate_user(self, user_id: str, password: str = None) -> bool:
        """
        用户认证
        
        Args:
            user_id: 用户ID
            password: 密码（可选）
        
        Returns:
            认证是否成功
        """
        try:
            # 查询用户
            query = "SELECT user_id, password_hash FROM users WHERE user_id = ?"
            results = self.db_service.execute_query(query, (user_id,))
            
            if not results:
                self.logger.warning(f"用户 {user_id} 不存在")
                return False
            
            user = results[0]
            
            # 如果用户没有设置密码，则允许无密码登录
            if user['password_hash'] is None:
                self._update_last_login(user_id)
                return True
            
            # 验证密码
            if password is None:
                return False
            
            password_hash = self._hash_password(password)
            if user['password_hash'] == password_hash:
                self._update_last_login(user_id)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"用户认证失败: {e}")
            return False
    
    def create_user(self, user_id: str, user_info: Dict) -> bool:
        """
        创建用户
        
        Args:
            user_id: 用户ID
            user_info: 用户信息字典
        
        Returns:
            创建是否成功
        """
        try:
            # 检查用户是否已存在
            if self.user_exists(user_id):
                self.logger.warning(f"用户 {user_id} 已存在")
                return False
            
            # 准备用户数据
            name = user_info.get('name', user_id)
            email = user_info.get('email')
            password = user_info.get('password')
            preferences = user_info.get('preferences', {})
            
            # 密码哈希
            password_hash = None
            if password:
                password_hash = self._hash_password(password)
            
            # 插入用户
            query = """
                INSERT INTO users (user_id, name, email, password_hash, preferences)
                VALUES (?, ?, ?, ?, ?)
            """
            
            params = (
                user_id,
                name,
                email,
                password_hash,
                json.dumps(preferences, ensure_ascii=False)
            )
            
            rows_affected = self.db_service.execute_update(query, params)
            
            if rows_affected > 0:
                self.logger.info(f"用户 {user_id} 创建成功")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"创建用户失败: {e}")
            return False
    
    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """
        获取用户信息
        
        Args:
            user_id: 用户ID
        
        Returns:
            用户信息字典，如果用户不存在则返回None
        """
        try:
            query = """
                SELECT user_id, name, email, created_at, last_login, preferences
                FROM users WHERE user_id = ?
            """
            
            results = self.db_service.execute_query(query, (user_id,))
            
            if not results:
                return None
            
            user = results[0]
            
            # 解析preferences JSON
            preferences = {}
            if user['preferences']:
                try:
                    preferences = json.loads(user['preferences'])
                except json.JSONDecodeError:
                    self.logger.warning(f"用户 {user_id} 的preferences格式错误")
            
            return {
                'user_id': user['user_id'],
                'name': user['name'],
                'email': user['email'],
                'created_at': user['created_at'],
                'last_login': user['last_login'],
                'preferences': preferences
            }
            
        except Exception as e:
            self.logger.error(f"获取用户信息失败: {e}")
            return None
    
    def update_user_info(self, user_id: str, user_info: Dict) -> bool:
        """
        更新用户信息
        
        Args:
            user_id: 用户ID
            user_info: 要更新的用户信息
        
        Returns:
            更新是否成功
        """
        try:
            # 构建更新字段
            update_fields = []
            params = []
            
            if 'name' in user_info:
                update_fields.append('name = ?')
                params.append(user_info['name'])
            
            if 'email' in user_info:
                update_fields.append('email = ?')
                params.append(user_info['email'])
            
            if 'password' in user_info:
                update_fields.append('password_hash = ?')
                params.append(self._hash_password(user_info['password']))
            
            if 'preferences' in user_info:
                update_fields.append('preferences = ?')
                params.append(json.dumps(user_info['preferences'], ensure_ascii=False))
            
            if not update_fields:
                return True  # 没有需要更新的字段
            
            params.append(user_id)
            
            query = f"""
                UPDATE users SET {', '.join(update_fields)}
                WHERE user_id = ?
            """
            
            rows_affected = self.db_service.execute_update(query, tuple(params))
            
            if rows_affected > 0:
                self.logger.info(f"用户 {user_id} 信息更新成功")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"更新用户信息失败: {e}")
            return False
    
    def user_exists(self, user_id: str) -> bool:
        """
        检查用户是否存在
        
        Args:
            user_id: 用户ID
        
        Returns:
            用户是否存在
        """
        try:
            query = "SELECT COUNT(*) as count FROM users WHERE user_id = ?"
            results = self.db_service.execute_query(query, (user_id,))
            
            return results[0]['count'] > 0
            
        except Exception as e:
            self.logger.error(f"检查用户存在性失败: {e}")
            return False
    
    def list_users(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        获取用户列表
        
        Args:
            limit: 限制数量
            offset: 偏移量
        
        Returns:
            用户列表
        """
        try:
            query = """
                SELECT user_id, name, email, created_at, last_login
                FROM users
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            
            results = self.db_service.execute_query(query, (limit, offset))
            return results
            
        except Exception as e:
            self.logger.error(f"获取用户列表失败: {e}")
            return []
    
    def delete_user(self, user_id: str) -> bool:
        """
        删除用户
        
        Args:
            user_id: 用户ID
        
        Returns:
            删除是否成功
        """
        try:
            # 注意：这里应该考虑级联删除相关的会话数据
            # 为了数据完整性，可能需要先删除相关的会话记录
            
            query = "DELETE FROM users WHERE user_id = ?"
            rows_affected = self.db_service.execute_update(query, (user_id,))
            
            if rows_affected > 0:
                self.logger.info(f"用户 {user_id} 删除成功")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"删除用户失败: {e}")
            return False
    
    def _hash_password(self, password: str) -> str:
        """
        密码哈希
        
        Args:
            password: 明文密码
        
        Returns:
            哈希后的密码
        """
        return hashlib.sha256(password.encode('utf-8')).hexdigest()
    
    def _update_last_login(self, user_id: str):
        """
        更新最后登录时间
        
        Args:
            user_id: 用户ID
        """
        try:
            query = "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?"
            self.db_service.execute_update(query, (user_id,))
        except Exception as e:
            self.logger.error(f"更新最后登录时间失败: {e}")