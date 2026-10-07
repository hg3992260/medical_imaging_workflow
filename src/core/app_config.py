#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用程序配置管理模块

负责管理应用程序的配置信息，包括数据库连接、界面设置、
文件路径等配置参数。
"""

import os
import sys
import json
from typing import Dict, Any


class AppConfig:
    """
    应用程序配置管理类
    """
    
    def __init__(self, config_file: str = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，默认为config/app_config.json
        """
        self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        if config_file is None:
            config_file = os.path.join(
                self.base_path,
                "config", "app_config.json"
            )
        
        self.config_file = config_file
        self.config = self._load_default_config()
        self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """
        加载默认配置
        
        Returns:
            默认配置字典
        """
        return {
            "app": {
                "name": "医疗数据科学结构化工作流系统 Medical Imaging Workflow",
                "version": "1.0.0",
                "author": "Medical Imaging Team (hg3992260)",
                "description": "面向医学科研的 DICOM 影像处理 / 数据清洗 / 大模型总结 / 科学文本生成一体化工作流"
            },
            "database": {
                "type": "sqlite",
                "path": "database/medical_imaging.db",
                "backup_path": "database/backup/",
                "auto_backup": True,
                "backup_interval": 24  # 小时
            },
            "ui": {
                "theme": "default",
                "language": "zh_CN",
                "window_size": {
                    "width": 1200,
                    "height": 800
                },
                "window_position": {
                    "x": 100,
                    "y": 100
                },
                "font": {
                    "family": "Microsoft YaHei",
                    "size": 12
                }
            },
            "dicom": {
                "supported_formats": [".dcm", ".dicom"],
                "max_file_size": 500,  # MB
                "cache_path": "assets/cache/dicom/",
                "export_formats": ["json", "csv", "xml"]
            },
            "ocr": {
                "engine": "deepseek_ocr",
                "languages": ["chi_sim", "eng"],
                "default_language": "chi_sim",
                "confidence_threshold": 0.6,
                "preprocessing": {
                    "denoise": True,
                    "enhance_contrast": True,
                    "binarize": True
                }
            },
            "text_processing": {
                "supported_formats": [".txt", ".docx", ".pdf"],
                "max_file_size": 50,  # MB
                "encoding": "utf-8",
                "backup_enabled": True
            },
            "logging": {
                "level": "INFO",
                "file_path": "logs/app.log",
                "max_file_size": 10,  # MB
                "backup_count": 5,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "security": {
                "session_timeout": 3600,  # 秒
                "password_min_length": 6,
                "max_login_attempts": 3,
                "lockout_duration": 300  # 秒
            }
        }
    
    def _load_config(self):
        """
        从文件加载配置
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self._merge_config(self.config, file_config)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
    
    def _merge_config(self, default: Dict, file_config: Dict):
        """
        合并配置
        
        Args:
            default: 默认配置
            file_config: 文件配置
        """
        for key, value in file_config.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def get(self, key: str, default=None):
        """
        获取配置值
        
        Args:
            key: 配置键，支持点分隔的嵌套键，如 'database.path'
            default: 默认值
        
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """
        保存配置到文件
        """
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get_database_path(self) -> str:
        """
        获取数据库文件路径
        
        Returns:
            数据库文件的绝对路径
        """
        db_path = self.get('database.path')
        if not os.path.isabs(db_path):
            # 相对路径
            # 在打包环境中，数据库通常应该放在可执行文件旁边，而不是临时目录中，以便持久化
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = self.base_path
                
            db_path = os.path.join(base_dir, db_path)

        if getattr(sys, "frozen", False):
            try:
                if not os.path.exists(db_path):
                    meipass_dir = getattr(sys, "_MEIPASS", "")
                    if meipass_dir:
                        bundled_db = os.path.join(meipass_dir, "database", "medical_imaging.db")
                        if os.path.isfile(bundled_db):
                            os.makedirs(os.path.dirname(db_path), exist_ok=True)
                            import shutil

                            shutil.copy2(bundled_db, db_path)
            except Exception:
                pass
        
        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        return db_path
