import sqlite3
import logging
from datetime import datetime
from typing import Optional

from .database_service import DatabaseService


class DatabaseMigration:
    """数据库迁移服务"""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.logger = logging.getLogger(__name__)
    
    def migrate_to_project_based_schema(self) -> bool:
        """迁移到基于项目的数据库架构"""
        try:
            self.logger.info("开始迁移到基于项目的数据库架构")
            
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 1. 检查并添加project_id字段到所有相关表
                self._add_project_id_columns(cursor)
                
                # 2. 创建默认项目（如果不存在）
                default_project_id = self._ensure_default_project(cursor)
                
                # 3. 将现有数据关联到默认项目
                self._migrate_existing_data_to_default_project(cursor, default_project_id)
                
                # 4. 更新外键约束
                self._update_foreign_key_constraints(cursor)
                
                # 5. 创建新的索引
                self._create_project_indexes(cursor)
                
                conn.commit()
                self.logger.info("数据库架构迁移完成")
                return True
                
        except Exception as e:
            self.logger.error(f"数据库迁移失败: {e}")
            return False
    
    def _add_project_id_columns(self, cursor):
        """为所有相关表添加project_id字段"""
        tables_to_migrate = [
            'user_ids',
            'dicom_sessions', 
            'ocr_sessions',
            'text_sessions'
        ]
        
        for table in tables_to_migrate:
            try:
                # 检查表是否存在project_id列
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'project_id' not in columns:
                    # 添加project_id列
                    cursor.execute(f"""
                        ALTER TABLE {table} 
                        ADD COLUMN project_id TEXT
                    """)
                    self.logger.info(f"为表 {table} 添加 project_id 列")
                else:
                    self.logger.info(f"表 {table} 已存在 project_id 列")
                    
            except Exception as e:
                self.logger.error(f"为表 {table} 添加 project_id 列失败: {e}")
                raise
    
    def _ensure_default_project(self, cursor) -> str:
        """确保存在默认项目"""
        default_project_id = "default-project-001"
        
        try:
            # 检查默认项目是否存在
            cursor.execute(
                "SELECT project_id FROM projects WHERE project_id = ?",
                (default_project_id,)
            )
            
            if not cursor.fetchone():
                # 创建默认项目
                cursor.execute("""
                    INSERT INTO projects (project_id, name, description, created_at, updated_at, status, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    default_project_id,
                    "默认项目",
                    "系统迁移时创建的默认项目，包含所有历史数据",
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    "active",
                    '{}'
                ))
                self.logger.info(f"创建默认项目: {default_project_id}")
            else:
                self.logger.info(f"默认项目已存在: {default_project_id}")
                
            return default_project_id
            
        except Exception as e:
            self.logger.error(f"创建默认项目失败: {e}")
            raise
    
    def _migrate_existing_data_to_default_project(self, cursor, default_project_id: str):
        """将现有数据迁移到默认项目"""
        tables_to_update = [
            'user_ids',
            'dicom_sessions',
            'ocr_sessions', 
            'text_sessions'
        ]
        
        for table in tables_to_update:
            try:
                # 更新所有project_id为NULL的记录
                cursor.execute(f"""
                    UPDATE {table} 
                    SET project_id = ? 
                    WHERE project_id IS NULL
                """, (default_project_id,))
                
                updated_rows = cursor.rowcount
                self.logger.info(f"表 {table} 更新了 {updated_rows} 条记录到默认项目")
                
            except Exception as e:
                self.logger.error(f"迁移表 {table} 数据失败: {e}")
                raise
    
    def _update_foreign_key_constraints(self, cursor):
        """更新外键约束（SQLite不支持直接修改外键，这里只是记录）"""
        # SQLite不支持直接添加外键约束到现有表
        # 这里记录需要的外键关系，在新建表时会包含这些约束
        foreign_keys = [
            "user_ids.project_id -> projects.project_id",
            "dicom_sessions.project_id -> projects.project_id",
            "ocr_sessions.project_id -> projects.project_id",
            "text_sessions.project_id -> projects.project_id"
        ]
        
        self.logger.info(f"记录外键关系: {foreign_keys}")
    
    def _create_project_indexes(self, cursor):
        """创建项目相关的索引"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_user_ids_project_id ON user_ids(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_dicom_sessions_project_id ON dicom_sessions(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_ocr_sessions_project_id ON ocr_sessions(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_text_sessions_project_id ON text_sessions(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_projects_status_name ON projects(status, name)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                self.logger.info(f"创建索引: {index_sql.split('ON')[1].strip()}")
            except Exception as e:
                self.logger.error(f"创建索引失败: {e}")
    
    def rollback_project_migration(self) -> bool:
        """回滚项目迁移（移除project_id列）"""
        try:
            self.logger.info("开始回滚项目迁移")
            
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # SQLite不支持直接删除列，需要重建表
                # 这里只是清空project_id字段
                tables_to_rollback = [
                    'user_ids',
                    'dicom_sessions',
                    'ocr_sessions',
                    'text_sessions'
                ]
                
                for table in tables_to_rollback:
                    try:
                        cursor.execute(f"UPDATE {table} SET project_id = NULL")
                        self.logger.info(f"清空表 {table} 的 project_id 字段")
                    except Exception as e:
                        self.logger.error(f"回滚表 {table} 失败: {e}")
                
                # 删除项目表（可选）
                # cursor.execute("DROP TABLE IF EXISTS projects")
                
                conn.commit()
                self.logger.info("项目迁移回滚完成")
                return True
                
        except Exception as e:
            self.logger.error(f"回滚项目迁移失败: {e}")
            return False
    
    def check_migration_status(self) -> dict:
        """检查迁移状态"""
        status = {
            'projects_table_exists': False,
            'project_id_columns_added': {},
            'default_project_exists': False,
            'data_migrated': {}
        }
        
        try:
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 检查projects表是否存在
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='projects'
                """)
                status['projects_table_exists'] = cursor.fetchone() is not None
                
                # 检查各表的project_id列
                tables_to_check = ['user_ids', 'dicom_sessions', 'ocr_sessions', 'text_sessions']
                for table in tables_to_check:
                    try:
                        cursor.execute(f"PRAGMA table_info({table})")
                        columns = [column[1] for column in cursor.fetchall()]
                        status['project_id_columns_added'][table] = 'project_id' in columns
                    except:
                        status['project_id_columns_added'][table] = False
                
                # 检查默认项目是否存在
                if status['projects_table_exists']:
                    cursor.execute(
                        "SELECT COUNT(*) FROM projects WHERE project_id = 'default-project-001'"
                    )
                    status['default_project_exists'] = cursor.fetchone()[0] > 0
                
                # 检查数据迁移状态
                for table in tables_to_check:
                    if status['project_id_columns_added'][table]:
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE project_id IS NOT NULL")
                            migrated_count = cursor.fetchone()[0]
                            cursor.execute(f"SELECT COUNT(*) FROM {table}")
                            total_count = cursor.fetchone()[0]
                            status['data_migrated'][table] = {
                                'migrated': migrated_count,
                                'total': total_count,
                                'percentage': (migrated_count / total_count * 100) if total_count > 0 else 100
                            }
                        except:
                            status['data_migrated'][table] = {'migrated': 0, 'total': 0, 'percentage': 0}
                
        except Exception as e:
            self.logger.error(f"检查迁移状态失败: {e}")
        
        return status