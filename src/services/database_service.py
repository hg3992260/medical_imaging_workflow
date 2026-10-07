#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库服务模块

负责管理SQLite数据库的连接、初始化和基本操作。
包括用户管理、DICOM会话、OCR会话和文本处理会话的数据存储。
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

from src.core.app_config import AppConfig
from src.utils.logger import get_logger


class DatabaseService:
    """
    数据库服务类
    """
    
    def __init__(self, config: AppConfig = None):
        """
        初始化数据库服务
        
        Args:
            config: 应用程序配置
        """
        self.config = config or AppConfig()
        self.db_path = self.config.get_database_path()
        self.logger = get_logger(__name__)
    
    @contextmanager
    def get_connection(self):
        """
        获取数据库连接的上下文管理器
        
        Yields:
            sqlite3.Connection: 数据库连接
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 使查询结果可以通过列名访问
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def initialize_database(self):
        """
        初始化数据库，创建所有必要的表
        """
        self.logger.info("开始初始化数据库")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 创建项目表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # 创建用户表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE,
                    password_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    preferences TEXT
                )
            """)
            
            # 创建用户ID表（样本名称管理）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_ids (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    description TEXT,
                    project_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    metadata TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    UNIQUE(user_id, project_id),
                    FOREIGN KEY (project_id) REFERENCES projects(project_id)
                )
            """)
            
            # 创建DICOM会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dicom_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    project_id TEXT,
                    file_path TEXT NOT NULL,
                    file_name TEXT,
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    dicom_info TEXT,
                    roi_type TEXT DEFAULT 'rectangle',
                    properties TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_ids(user_id),
                    FOREIGN KEY (project_id) REFERENCES projects(project_id)
                )
            """)
            
            # 创建ROI数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS roi_data (
                    roi_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    roi_type TEXT DEFAULT 'detected',
                    roi_name TEXT,
                    coordinates TEXT NOT NULL,
                    area REAL,
                    perimeter REAL,
                    properties TEXT,
                    pixel_data BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES dicom_sessions(session_id)
                )
            """)
            
            # 创建OCR会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ocr_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    project_id TEXT,
                    image_path TEXT NOT NULL,
                    image_name TEXT,
                    language TEXT DEFAULT 'chi_sim',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_ids(user_id),
                    FOREIGN KEY (project_id) REFERENCES projects(project_id)
                )
            """)
            
            # 创建OCR结果表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ocr_results (
                    result_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    recognized_text TEXT,
                    confidence REAL,
                    bounding_boxes TEXT,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES ocr_sessions(session_id)
                )
            """)
            
            # 创建文本处理会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    project_id TEXT,
                    operation_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_ids(user_id),
                    FOREIGN KEY (project_id) REFERENCES projects(project_id)
                )
            """)
            
            # 创建文档表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_name TEXT,
                    format TEXT,
                    content TEXT,
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES text_sessions(session_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text_content TEXT,
                    content_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建项目存储管理表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS project_storage (
                    storage_id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                    project_id TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    quota_limit BIGINT DEFAULT 5368709120,
                    used_space BIGINT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_cleanup TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(project_id),
                    UNIQUE(project_id)
                )
            """)
            
            # 创建存储文件索引表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS storage_files (
                    file_id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                    storage_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size BIGINT NOT NULL,
                    checksum TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (storage_id) REFERENCES project_storage(storage_id)
                )
            """)
            
            # 创建DICOM文件表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dicom_files (
                    file_id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                    session_id TEXT NOT NULL,
                    storage_file_id TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    dicom_metadata TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES dicom_sessions(session_id),
                    FOREIGN KEY (storage_file_id) REFERENCES storage_files(file_id)
                )
            """)
            
            # 创建备份记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backup_records (
                    backup_id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                    project_id TEXT NOT NULL,
                    backup_name TEXT NOT NULL,
                    backup_path TEXT NOT NULL,
                    backup_type TEXT DEFAULT 'incremental',
                    backup_size BIGINT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    verified BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (project_id) REFERENCES projects(project_id)
                )
            """)

            # 创建科学写作草稿表 (Drafts)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drafts (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT,
                    content TEXT NOT NULL,
                    score INTEGER DEFAULT 0,
                    logs TEXT,
                    model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(project_id)
                )
            """)
            
            # 创建索引
            self._create_indexes(cursor)
            
            # 执行数据库迁移
            self._migrate_database(cursor)
            
            # 执行项目架构迁移
            self._migrate_to_project_schema(cursor)
            
            # 添加OCR引擎支持
            self._migrate_ocr_engines_support(cursor)
            
            # 插入初始数据
            self._insert_initial_data(cursor)
            
            conn.commit()
            self.logger.info("数据库初始化完成")
    
    def _create_indexes(self, cursor):
        """
        创建数据库索引
        
        Args:
            cursor: 数据库游标
        """
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS idx_users_last_login ON users(last_login DESC)",
            "CREATE INDEX IF NOT EXISTS idx_user_ids_user_id ON user_ids(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_ids_display_name ON user_ids(display_name)",
            "CREATE INDEX IF NOT EXISTS idx_user_ids_created ON user_ids(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_user_ids_last_used ON user_ids(last_used)",
            "CREATE INDEX IF NOT EXISTS idx_dicom_sessions_user_id ON dicom_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_dicom_sessions_created_at ON dicom_sessions(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_roi_data_session_id ON roi_data(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_ocr_sessions_user_id ON ocr_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_ocr_sessions_project_id ON ocr_sessions(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_ocr_results_session_id ON ocr_results(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_text_sessions_user_id ON text_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_content_chunks_source ON content_chunks(source_type, source_id, chunk_index)",
            "CREATE INDEX IF NOT EXISTS idx_content_chunks_project ON content_chunks(project_id, source_type)",
            # 项目存储相关索引
            "CREATE INDEX IF NOT EXISTS idx_project_storage_project_id ON project_storage(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_storage_files_storage_id ON storage_files(storage_id)",
            "CREATE INDEX IF NOT EXISTS idx_storage_files_type ON storage_files(file_type)",
            "CREATE INDEX IF NOT EXISTS idx_dicom_files_session_id ON dicom_files(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_backup_records_project_id ON backup_records(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_backup_records_created_at ON backup_records(created_at DESC)",
            # 草稿表索引
            "CREATE INDEX IF NOT EXISTS idx_drafts_project_id ON drafts(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_drafts_created_at ON drafts(created_at DESC)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    def _migrate_database(self, cursor):
        """
        执行数据库迁移
        
        Args:
            cursor: 数据库游标
        """
        try:
            # 检查并添加dicom_sessions表的列
            cursor.execute("PRAGMA table_info(dicom_sessions)")
            dicom_columns = [column[1] for column in cursor.fetchall()]
            
            if 'dicom_info' not in dicom_columns:
                cursor.execute("ALTER TABLE dicom_sessions ADD COLUMN dicom_info TEXT")
                self.logger.info("添加dicom_info列")
            
            if 'roi_type' not in dicom_columns:
                cursor.execute("ALTER TABLE dicom_sessions ADD COLUMN roi_type TEXT DEFAULT 'rectangle'")
                self.logger.info("添加roi_type列")
            
            if 'properties' not in dicom_columns:
                cursor.execute("ALTER TABLE dicom_sessions ADD COLUMN properties TEXT")
                self.logger.info("添加properties列")
            
            # 检查roi_data表是否有roi_type和properties列
            cursor.execute("PRAGMA table_info(roi_data)")
            roi_columns = [column[1] for column in cursor.fetchall()]
            
            if 'roi_type' not in roi_columns:
                cursor.execute("ALTER TABLE roi_data ADD COLUMN roi_type TEXT DEFAULT 'detected'")
                self.logger.info("已为roi_data表添加roi_type列")
            
            if 'properties' not in roi_columns:
                cursor.execute("ALTER TABLE roi_data ADD COLUMN properties TEXT")
                self.logger.info("已为roi_data表添加properties列")
            
            if 'ocr_text' not in roi_columns:
                cursor.execute("ALTER TABLE roi_data ADD COLUMN ocr_text TEXT")
                self.logger.info("已为roi_data表添加ocr_text列")
            
            if 'ocr_confidence' not in roi_columns:
                cursor.execute("ALTER TABLE roi_data ADD COLUMN ocr_confidence REAL DEFAULT 0.0")
                self.logger.info("已为roi_data表添加ocr_confidence列")
            
            if 'source' not in roi_columns:
                cursor.execute("ALTER TABLE roi_data ADD COLUMN source TEXT DEFAULT 'unknown'")
                self.logger.info("已为roi_data表添加source列")
            
            # 检查并添加ocr_sessions表的列
            cursor.execute("PRAGMA table_info(ocr_sessions)")
            ocr_columns = [column[1] for column in cursor.fetchall()]
            
            if 'user_id' not in ocr_columns:
                cursor.execute("ALTER TABLE ocr_sessions ADD COLUMN user_id TEXT")
                self.logger.info("添加ocr_sessions.user_id列")
            
            if 'total_confidence' not in ocr_columns:
                cursor.execute("ALTER TABLE ocr_sessions ADD COLUMN total_confidence REAL DEFAULT 0.0")
                self.logger.info("添加ocr_sessions.total_confidence列")
            
            if 'total_words' not in ocr_columns:
                cursor.execute("ALTER TABLE ocr_sessions ADD COLUMN total_words INTEGER DEFAULT 0")
                self.logger.info("添加ocr_sessions.total_words列")
            
            if 'processing_time' not in ocr_columns:
                cursor.execute("ALTER TABLE ocr_sessions ADD COLUMN processing_time REAL DEFAULT 0.0")
                self.logger.info("添加ocr_sessions.processing_time列")
            
            # 检查并添加ocr_results表的列
            cursor.execute("PRAGMA table_info(ocr_results)")
            ocr_results_columns = [column[1] for column in cursor.fetchall()]
            
            # 确保 recognized_text 列存在（用于兼容现有代码）
            if 'recognized_text' not in ocr_results_columns:
                cursor.execute("ALTER TABLE ocr_results ADD COLUMN recognized_text TEXT")
                self.logger.info("添加ocr_results.recognized_text列")
            
            # 确保 extracted_text 列存在（作为新标准）
            if 'extracted_text' not in ocr_results_columns:
                cursor.execute("ALTER TABLE ocr_results ADD COLUMN extracted_text TEXT")
                self.logger.info("添加ocr_results.extracted_text列")

            if 'word_details' not in ocr_results_columns:
                cursor.execute("ALTER TABLE ocr_results ADD COLUMN word_details TEXT")
                self.logger.info("添加ocr_results.word_details列")
            
            if 'config_used' not in ocr_results_columns:
                cursor.execute("ALTER TABLE ocr_results ADD COLUMN config_used TEXT")
                self.logger.info("添加ocr_results.config_used列")
            
            if 'result_file_path' not in ocr_results_columns:
                cursor.execute("ALTER TABLE ocr_results ADD COLUMN result_file_path TEXT")
                self.logger.info("添加ocr_results.result_file_path列")
            
            # 检查并添加text_sessions表的列
            cursor.execute("PRAGMA table_info(text_sessions)")
            text_columns = [column[1] for column in cursor.fetchall()]
            
            if 'user_id' not in text_columns:
                cursor.execute("ALTER TABLE text_sessions ADD COLUMN user_id TEXT")
                self.logger.info("添加text_sessions.user_id列")
            
            if 'file_path' not in text_columns:
                cursor.execute("ALTER TABLE text_sessions ADD COLUMN file_path TEXT")
                self.logger.info("添加text_sessions.file_path列")
            
            if 'char_count' not in text_columns:
                cursor.execute("ALTER TABLE text_sessions ADD COLUMN char_count INTEGER DEFAULT 0")
                self.logger.info("添加text_sessions.char_count列")
            
            if 'word_count' not in text_columns:
                cursor.execute("ALTER TABLE text_sessions ADD COLUMN word_count INTEGER DEFAULT 0")
                self.logger.info("添加text_sessions.word_count列")
            
            if 'line_count' not in text_columns:
                cursor.execute("ALTER TABLE text_sessions ADD COLUMN line_count INTEGER DEFAULT 0")
                self.logger.info("添加text_sessions.line_count列")
            
            # 检查并添加documents表的列
            cursor.execute("PRAGMA table_info(documents)")
            documents_columns = [column[1] for column in cursor.fetchall()]
            
            if 'analysis_result' not in documents_columns:
                cursor.execute("ALTER TABLE documents ADD COLUMN analysis_result TEXT")
                self.logger.info("添加documents.analysis_result列")
            
            if 'keywords' not in documents_columns:
                cursor.execute("ALTER TABLE documents ADD COLUMN keywords TEXT")
                self.logger.info("添加documents.keywords列")
            
            if 'analysis_file_path' not in documents_columns:
                cursor.execute("ALTER TABLE documents ADD COLUMN analysis_file_path TEXT")
                self.logger.info("添加documents.analysis_file_path列")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text_content TEXT,
                    content_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 检查并添加project_storage表的列
            cursor.execute("PRAGMA table_info(project_storage)")
            storage_columns = [column[1] for column in cursor.fetchall()]
            
            if 'updated_at' not in storage_columns:
                cursor.execute("ALTER TABLE project_storage ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
                self.logger.info("添加project_storage.updated_at列")
            
            # 检查并添加backup_records表的backup_name列
            cursor.execute("PRAGMA table_info(backup_records)")
            backup_columns = [column[1] for column in cursor.fetchall()]
            
            if 'backup_name' not in backup_columns:
                cursor.execute("ALTER TABLE backup_records ADD COLUMN backup_name TEXT NOT NULL DEFAULT ''")
                self.logger.info("添加backup_records.backup_name列")
            
            # 检查并添加storage_files表的metadata列
            cursor.execute("PRAGMA table_info(storage_files)")
            storage_files_columns = [column[1] for column in cursor.fetchall()]
            
            if 'metadata' not in storage_files_columns:
                cursor.execute("ALTER TABLE storage_files ADD COLUMN metadata TEXT")
                self.logger.info("添加storage_files.metadata列")
            
            # 创建默认用户ID（用于兼容现有数据）
            cursor.execute("SELECT COUNT(*) FROM user_ids WHERE user_id = 'default'")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO user_ids (user_id, display_name, description, created_at, last_used)
                    VALUES ('default', '默认样本', '系统默认样本，用于兼容现有数据', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """)
                self.logger.info("创建默认用户ID")
            
            # 将现有的空user_id数据迁移到默认用户ID
            cursor.execute("UPDATE dicom_sessions SET user_id = 'default' WHERE user_id IS NULL")
            cursor.execute("UPDATE ocr_sessions SET user_id = 'default' WHERE user_id IS NULL")
            cursor.execute("UPDATE text_sessions SET user_id = 'default' WHERE user_id IS NULL")
                
        except Exception as e:
            self.logger.error(f"数据库迁移失败: {e}")
    
    def _migrate_to_project_schema(self, cursor):
        """
        迁移到基于项目的数据库架构
        
        Args:
            cursor: 数据库游标
        """
        try:
            # 检查是否需要迁移user_ids表的UNIQUE约束
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='user_ids'")
            result = cursor.fetchone()
            if result and 'user_id TEXT UNIQUE NOT NULL' in result[0]:
                # 需要重建表以修改UNIQUE约束
                self._rebuild_user_ids_table(cursor)
            
            # 检查是否需要迁移
            cursor.execute("PRAGMA table_info(user_ids)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'project_id' not in columns:
                # 添加project_id列到现有表
                tables_to_migrate = ['user_ids', 'dicom_sessions', 'ocr_sessions', 'text_sessions']
                
                for table in tables_to_migrate:
                    try:
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN project_id TEXT")
                        self.logger.info(f"为表 {table} 添加 project_id 列")
                    except Exception as e:
                        if "duplicate column name" not in str(e).lower():
                            self.logger.error(f"为表 {table} 添加 project_id 列失败: {e}")
                
                # 创建默认项目
                default_project_id = "default-project-001"
                cursor.execute("""
                    INSERT OR IGNORE INTO projects (project_id, name, description, status)
                    VALUES (?, ?, ?, ?)
                """, (
                    default_project_id,
                    "默认项目",
                    "系统迁移时创建的默认项目，包含所有历史数据",
                    "active"
                ))
                
                # 将现有数据关联到默认项目
                for table in tables_to_migrate:
                    cursor.execute(f"""
                        UPDATE {table} 
                        SET project_id = ? 
                        WHERE project_id IS NULL
                    """, (default_project_id,))
                
                # 为默认项目创建存储记录
                cursor.execute("""
                    INSERT OR IGNORE INTO project_storage (project_id, storage_path, quota_limit)
                    VALUES (?, ?, ?)
                """, (
                    default_project_id,
                    f'storage/projects/{default_project_id}',
                    5368709120  # 5GB默认配额
                ))
                
                # 创建项目相关索引
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_user_ids_project_id ON user_ids(project_id)",
                    "CREATE INDEX IF NOT EXISTS idx_dicom_sessions_project_id ON dicom_sessions(project_id)",
                    "CREATE INDEX IF NOT EXISTS idx_ocr_sessions_project_id ON ocr_sessions(project_id)",
                    "CREATE INDEX IF NOT EXISTS idx_text_sessions_project_id ON text_sessions(project_id)"
                ]
                
                for index_sql in indexes:
                    cursor.execute(index_sql)
                
                self.logger.info("项目架构迁移完成")
            else:
                self.logger.info("项目架构已存在，跳过迁移")
                
        except Exception as e:
            self.logger.error(f"项目架构迁移失败: {e}")
            raise
    
    def _migrate_ocr_engines_support(self, cursor=None):
        """
        迁移OCR引擎支持功能
        """
        try:
            if cursor is None:
                with self.get_connection() as conn:
                    inner_cursor = conn.cursor()
                    self._migrate_ocr_engines_support(inner_cursor)
                    conn.commit()
                return

            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='ocr_engines'
            """)

            if cursor.fetchone() is None:
                migrations_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'migrations')
                migration_file = os.path.join(migrations_dir, 'add_ocr_engines_table.sql')

                if os.path.exists(migration_file):
                    with open(migration_file, 'r', encoding='utf-8') as f:
                        migration_sql = f.read()

                    statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]

                    for statement in statements:
                        if statement:
                            cursor.execute(statement)

                    self.logger.info("OCR引擎支持迁移完成")
                else:
                    self.logger.warning(f"迁移文件不存在: {migration_file}")
            else:
                self.logger.info("OCR引擎表已存在，跳过迁移")
                    
        except Exception as e:
            self.logger.error(f"OCR引擎支持迁移失败: {e}")
            raise
    
    def _rebuild_user_ids_table(self, cursor):
        """
        重建user_ids表以修改UNIQUE约束
        
        Args:
            cursor: 数据库游标
        """
        try:
            self.logger.info("开始重建user_ids表以修改UNIQUE约束")
            
            # 备份现有数据
            cursor.execute("SELECT * FROM user_ids")
            existing_data = cursor.fetchall()
            
            # 删除旧表
            cursor.execute("DROP TABLE IF EXISTS user_ids")
            
            # 创建新表（使用正确的复合UNIQUE约束）
            cursor.execute("""
                CREATE TABLE user_ids (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    description TEXT,
                    project_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    metadata TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    UNIQUE(user_id, project_id),
                    FOREIGN KEY (project_id) REFERENCES projects(project_id)
                )
            """)
            
            # 恢复数据
            for row in existing_data:
                try:
                    cursor.execute("""
                        INSERT INTO user_ids 
                        (id, user_id, display_name, description, project_id, created_at, last_used, metadata, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, row)
                except Exception as e:
                    # 如果插入失败（可能是重复数据），记录警告但继续
                    self.logger.warning(f"恢复数据时跳过重复记录: {row[1]} (项目: {row[4]}) - {e}")
            
            self.logger.info("user_ids表重建完成")
            
        except Exception as e:
            self.logger.error(f"重建user_ids表失败: {e}")
            raise
    
    def _insert_initial_data(self, cursor):
        """
        插入初始数据
        
        Args:
            cursor: 数据库游标
        """
        # 检查是否已有用户数据
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        if user_count == 0:
            # 插入默认用户
            initial_users = [
                {
                    'user_id': 'admin',
                    'name': '系统管理员',
                    'email': 'admin@medical.com',
                    'preferences': json.dumps({
                        'theme': 'default',
                        'language': 'zh_CN'
                    })
                },
                {
                    'user_id': 'demo_user',
                    'name': '演示用户',
                    'email': 'demo@medical.com',
                    'preferences': json.dumps({
                        'theme': 'default',
                        'language': 'zh_CN'
                    })
                }
            ]
            
            for user in initial_users:
                cursor.execute("""
                    INSERT INTO users (user_id, name, email, preferences)
                    VALUES (?, ?, ?, ?)
                """, (user['user_id'], user['name'], user['email'], user['preferences']))
            
            self.logger.info("插入初始用户数据完成")
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """
        执行查询语句
        
        Args:
            query: SQL查询语句
            params: 查询参数
        
        Returns:
            查询结果列表
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # 检查cursor.description是否为None（INSERT、UPDATE、DELETE等语句）
                if cursor.description is None:
                    # 对于非查询语句，提交事务并返回空列表
                    conn.commit()
                    return []
                
                columns = [description[0] for description in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                
                return results
        except Exception as e:
            self.logger.error(f"数据库操作失败: {e}")
            raise
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """
        执行更新语句
        
        Args:
            query: SQL更新语句
            params: 更新参数
        
        Returns:
            影响的行数
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            conn.commit()
            return cursor.rowcount
    
    def get_cursor(self):
        """
        获取数据库游标
        
        Returns:
            sqlite3.Cursor: 数据库游标
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn.cursor()
    
    def rollback(self, conn=None):
        """
        回滚数据库事务
        
        Args:
            conn: 数据库连接（可选）
        """
        if conn:
            conn.rollback()
        else:
            # 如果没有提供连接，创建新连接并回滚
            with self.get_connection() as connection:
                connection.rollback()
    
    def commit(self, conn=None):
        """
        提交数据库事务
        
        Args:
            conn: 数据库连接（可选）
        """
        if conn:
            conn.commit()
        else:
            # 如果没有提供连接，创建新连接并提交
            with self.get_connection() as connection:
                connection.commit()
    
    def backup_database(self, backup_path: str = None) -> bool:
        """
        备份数据库
        
        Args:
            backup_path: 备份文件路径
        
        Returns:
            备份是否成功
        """
        try:
            if backup_path is None:
                backup_dir = self.config.get('database.backup_path', 'database/backup/')
                if not os.path.isabs(backup_dir):
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    backup_dir = os.path.join(project_root, backup_dir)
                
                os.makedirs(backup_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = os.path.join(backup_dir, f'medical_imaging_backup_{timestamp}.db')
            
            # 使用SQLite的备份API
            with self.get_connection() as source_conn:
                backup_conn = sqlite3.connect(backup_path)
                source_conn.backup(backup_conn)
                backup_conn.close()
            
            self.logger.info(f"数据库备份成功: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"数据库备份失败: {e}")
            return False
    
    # OCR引擎相关方法
    def get_ocr_engines(self, engine_type=None, available_only=False):
        """获取OCR引擎列表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM ocr_engines WHERE 1=1"
            params = []
            
            if engine_type:
                query += " AND engine_type = ?"
                params.append(engine_type)
            
            if available_only:
                query += " AND is_available = 1"
            
            query += " ORDER BY engine_name"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_ocr_engine(self, engine_id):
        """获取单个OCR引擎信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM ocr_engines WHERE engine_id = ?",
                (engine_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_ocr_engine(self, engine_id, **kwargs):
        """更新OCR引擎信息"""
        if not kwargs:
            return False
        
        # 添加更新时间
        kwargs['updated_at'] = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 构建更新语句
            set_clause = ", ".join([f"{key} = ?" for key in kwargs.keys()])
            values = list(kwargs.values()) + [engine_id]
            
            cursor.execute(
                f"UPDATE ocr_engines SET {set_clause} WHERE engine_id = ?",
                values
            )
            
            conn.commit()
            return cursor.rowcount > 0
    
    def create_ocr_session_with_engine(self, session_data):
        """创建包含引擎信息的OCR会话"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 生成会话ID
            session_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO ocr_sessions (
                    session_id, project_id, user_id, file_path, 
                    engine_id, engine_config, preprocessing_config,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                session_data.get('project_id'),
                session_data.get('user_id'),
                session_data.get('file_path'),
                session_data.get('engine_id'),
                session_data.get('engine_config'),
                session_data.get('preprocessing_config'),
                'pending',
                datetime.now().isoformat()
            ))
            
            conn.commit()
            return session_id
    
    def save_ocr_result_with_details(self, result_data):
        """保存包含详细信息的OCR结果"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            result_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO ocr_results (
                    result_id, session_id, engine_id, text_content, 
                    confidence, language_detected, word_count, line_count, 
                    paragraph_count, bbox_data, confidence_details, 
                    preprocessing_applied, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result_id,
                result_data.get('session_id'),
                result_data.get('engine_id'),
                result_data.get('text_content'),
                result_data.get('confidence'),
                result_data.get('language_detected'),
                result_data.get('word_count', 0),
                result_data.get('line_count', 0),
                result_data.get('paragraph_count', 0),
                result_data.get('bbox_data'),
                result_data.get('confidence_details'),
                result_data.get('preprocessing_applied'),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            return result_id
    
    def get_ocr_engine_stats(self, engine_id=None, days=30):
        """获取OCR引擎统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    e.engine_name,
                    e.engine_type,
                    COUNT(DISTINCT s.session_id) as session_count,
                    COUNT(r.result_id) as result_count,
                    AVG(r.confidence) as avg_confidence,
                    AVG(s.processing_time) as avg_processing_time,
                    SUM(r.word_count) as total_words
                FROM ocr_engines e
                LEFT JOIN ocr_sessions s ON e.engine_id = s.engine_id
                LEFT JOIN ocr_results r ON s.session_id = r.session_id
                WHERE s.created_at >= datetime('now', '-{} days')
            """.format(days)
            
            params = []
            if engine_id:
                query += " AND e.engine_id = ?"
                params.append(engine_id)
            
            query += " GROUP BY e.engine_id, e.engine_name, e.engine_type"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
