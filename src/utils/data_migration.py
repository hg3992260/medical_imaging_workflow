#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据迁移工具模块

负责将现有的用户ID数据迁移到项目-样本映射系统中
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.services.database_service import DatabaseService
from src.services.project_service import ProjectService
from src.services.sample_service import SampleService
from src.core.user_id_manager import UserIDManager
from src.utils.logger import get_logger


class DataMigrationTool:
    """
    数据迁移工具类
    
    负责将现有的用户ID数据迁移到新的项目-样本映射系统中
    """
    
    def __init__(self):
        """
        初始化数据迁移工具
        """
        self.logger = get_logger(__name__)
        self.db_service = DatabaseService()
        self.project_service = ProjectService()
        self.sample_service = SampleService()
        self.user_id_manager = UserIDManager()
        
    def migrate_user_ids_to_default_project(self) -> bool:
        """
        将现有的用户ID数据迁移到默认项目中
        
        Returns:
            是否迁移成功
        """
        try:
            self.logger.info("开始迁移用户ID数据到默认项目")
            
            # 获取或创建默认项目
            default_project = self._get_or_create_default_project()
            if not default_project:
                self.logger.error("无法获取或创建默认项目")
                return False
            
            # 获取现有的用户ID列表
            user_id_details = self.user_id_manager.get_user_id_details_list()
            if not user_id_details:
                self.logger.info("没有找到需要迁移的用户ID数据")
                return True
            
            # 迁移每个用户ID作为样本
            migrated_count = 0
            for user_detail in user_id_details:
                if self._migrate_single_user_id(default_project['project_id'], user_detail):
                    migrated_count += 1
            
            self.logger.info(f"数据迁移完成，成功迁移 {migrated_count}/{len(user_id_details)} 个用户ID")
            return True
            
        except Exception as e:
            self.logger.error(f"数据迁移失败: {e}")
            return False
    
    def _get_or_create_default_project(self) -> Optional[Dict[str, Any]]:
        """
        获取或创建默认项目
        
        Returns:
            默认项目信息
        """
        try:
            # 查找名为"默认项目"的项目
            projects = self.project_service.get_all_projects()
            for project in projects:
                if project.name == '默认项目':
                    self.logger.info(f"找到默认项目: {project.project_id}")
                    return project.to_dict()
            
            # 如果没有找到，创建默认项目
            from src.models.project import Project
            new_project = Project.create_new(
                name='默认项目',
                description='用于存储迁移的用户ID数据的默认项目'
            )
            new_project.metadata = {'migrated': True, 'created_by': 'data_migration'}
            
            if self.project_service.create_project(new_project):
                self.logger.info(f"创建默认项目成功: {new_project.project_id}")
                return new_project.to_dict()
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取或创建默认项目失败: {e}")
            return None
    
    def _migrate_single_user_id(self, project_id: str, user_detail: Dict[str, Any]) -> bool:
        """
        迁移单个用户ID到项目中
        
        Args:
            project_id: 项目ID
            user_detail: 用户ID详细信息
            
        Returns:
            是否迁移成功
        """
        try:
            user_id = user_detail['user_id']
            
            # 检查样本是否已存在
            existing_samples = self.sample_service.get_samples_by_project(project_id)
            for sample in existing_samples:
                if sample['user_id'] == user_id:
                    self.logger.info(f"样本 {user_id} 已存在，跳过迁移")
                    return True
            
            # 创建样本数据
            sample_data = {
                'name': user_id,
                'display_name': user_detail.get('display_name', user_id),
                'description': user_detail.get('description', f'从用户ID {user_id} 迁移的样本数据'),
                'sample_type': 'migrated_user_id',
                'status': 'active',
                'metadata': {
                    'migrated_from_user_id': True,
                    'original_created_at': user_detail.get('created_at'),
                    'original_last_used': user_detail.get('last_used'),
                    'original_metadata': user_detail.get('metadata', {})
                }
            }
            
            # 添加样本到项目
            success = self.sample_service.create_sample(
                user_id=user_id,
                display_name=sample_data['display_name'],
                project_id=project_id,
                description=sample_data['description'],
                metadata=json.dumps(sample_data['metadata']),
                is_active=sample_data['status'] == 'active'
            )
            if success:
                self.logger.info(f"成功迁移用户ID {user_id} 到项目 {project_id}")
                return True
            else:
                self.logger.error(f"迁移用户ID {user_id} 失败")
                return False
                
        except Exception as e:
            self.logger.error(f"迁移用户ID {user_detail.get('user_id', 'unknown')} 失败: {e}")
            return False
    
    def get_migration_status(self) -> Dict[str, Any]:
        """
        获取迁移状态信息
        
        Returns:
            迁移状态信息
        """
        try:
            # 获取用户ID统计
            user_id_stats = self.user_id_manager.get_user_id_statistics()
            
            # 获取默认项目信息
            projects = self.project_service.get_all_projects()
            default_project = None
            for project in projects:
                if project.name == '默认项目':
                    default_project = project.to_dict()
                    break
            
            # 获取默认项目中的样本数量
            default_project_samples = 0
            if default_project:
                samples = self.sample_service.get_samples_by_project(default_project['project_id'])
                default_project_samples = len([s for s in samples if s.get('sample_type') == 'migrated_user_id'])
            
            return {
                'user_id_count': user_id_stats.get('total_count', 0),
                'default_project_exists': default_project is not None,
                'default_project_id': default_project['project_id'] if default_project else None,
                'migrated_samples_count': default_project_samples,
                'migration_needed': user_id_stats.get('total_count', 0) > default_project_samples
            }
            
        except Exception as e:
            self.logger.error(f"获取迁移状态失败: {e}")
            return {
                'user_id_count': 0,
                'default_project_exists': False,
                'default_project_id': None,
                'migrated_samples_count': 0,
                'migration_needed': False
            }
    
    def cleanup_migrated_data(self) -> bool:
        """
        清理已迁移的数据（可选操作）
        
        Returns:
            是否清理成功
        """
        try:
            self.logger.warning("开始清理已迁移的用户ID数据")
            
            # 获取迁移状态
            status = self.get_migration_status()
            if not status['migration_needed']:
                # 清理用户ID表中的数据
                with self.db_service.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM user_ids")
                    conn.commit()
                
                self.logger.info("用户ID数据清理完成")
                return True
            else:
                self.logger.warning("迁移未完成，跳过数据清理")
                return False
                
        except Exception as e:
            self.logger.error(f"数据清理失败: {e}")
            return False


def run_migration():
    """
    运行数据迁移
    """
    migration_tool = DataMigrationTool()
    
    # 检查迁移状态
    status = migration_tool.get_migration_status()
    print(f"迁移状态: {status}")
    
    if status['migration_needed']:
        print("开始数据迁移...")
        success = migration_tool.migrate_user_ids_to_default_project()
        if success:
            print("数据迁移完成")
        else:
            print("数据迁移失败")
    else:
        print("无需迁移数据")


if __name__ == '__main__':
    run_migration()