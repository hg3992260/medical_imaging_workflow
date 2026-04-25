#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志工具模块

提供统一的日志记录功能，支持文件日志和控制台日志。
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(name: str = None, 
                 log_file: str = None, 
                 level: str = 'INFO',
                 max_file_size: int = 10,
                 backup_count: int = 5) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别
        max_file_size: 最大文件大小(MB)
        backup_count: 备份文件数量
    
    Returns:
        配置好的日志记录器
    """
    if name is None:
        name = 'medical_imaging'
    
    if log_file is None:
        # 默认日志文件路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        log_dir = os.path.join(project_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'app.log')
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器（带轮转）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_file_size * 1024 * 1024,  # 转换为字节
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
    
    Returns:
        日志记录器
    """
    if name is None:
        name = 'medical_imaging'
    
    logger = logging.getLogger(name)
    
    # 如果日志记录器还没有配置，则使用默认配置
    if not logger.handlers:
        return setup_logger(name)
    
    return logger
