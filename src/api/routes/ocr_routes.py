#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR相关的API路由

提供OCR处理、引擎管理和状态检查的API端点。
"""

import os
import json
import time
from typing import Dict, Any
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest

from src.services.ocr_service import OCRService
from src.services.database_service import DatabaseService
from src.utils.logger import get_logger

# 创建蓝图
ocr_bp = Blueprint('ocr', __name__, url_prefix='/api/ocr')

# 初始化服务
logger = get_logger(__name__)
db_service = DatabaseService()
ocr_service = OCRService(db_service)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_engine(engine: str) -> str:
    e = (engine or "").strip().lower()
    if e in ("auto", "best", "default", ""):
        return "deepseek_ocr"
    if e in ("deepseek", "deepseek-ocr", "deepseek_ocr"):
        return "deepseek_ocr"
    if e in ("ollama", "ollama_ocr", "ollama-deepseek-ocr", "ollama_deepseek_ocr", "deepseek_ocr_ollama"):
        return "ollama_deepseek_ocr"
    return e

def validate_engine(engine: str) -> bool:
    """验证OCR引擎类型"""
    valid_engines = {
        'deepseek_ocr', 'ollama_deepseek_ocr',
        'deepseek', 'deepseek-ocr', 'ollama', 'ollama_ocr',
        'auto', 'best', 'default'
    }
    return (engine or "").strip().lower() in valid_engines

@ocr_bp.route('/engines', methods=['GET'])
def get_ocr_engines():
    """获取所有OCR引擎状态"""
    try:
        engines_status = ocr_service.get_ocr_engines_status()
        return jsonify({
            'success': True,
            'engines': engines_status,
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"获取OCR引擎状态失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get engines status',
            'message': str(e)
        }), 500

# Umi-OCR 等传统 OCR 引擎已移除，仅支持 DeepSeek-OCR
            'message': str(e)
        }), 500

@ocr_bp.route('/process', methods=['POST'])
def process_ocr():
    """处理OCR识别请求"""
    start_time = time.time()
    
    try:
        # 检查是否有文件上传
        if 'image' not in request.files:
            raise BadRequest('No image file provided')
        
        file = request.files['image']
        if file.filename == '':
            raise BadRequest('No file selected')
        
        if not allowed_file(file.filename):
            raise BadRequest('File type not allowed')
        
        # 获取引擎类型
        engine = normalize_engine(request.form.get('engine', 'deepseek_ocr'))
        if not validate_engine(engine):
            raise BadRequest(f'Invalid engine type: {engine}')
        
        # 获取配置参数
        config_str = request.form.get('config', '{}')
        try:
            config = json.loads(config_str) if config_str else {}
        except json.JSONDecodeError:
            raise BadRequest('Invalid config JSON format')
        
        # 保存临时文件
        filename = secure_filename(file.filename)
        temp_path = os.path.join('temp', filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        
        try:
            # 执行OCR识别
            config = dict(config or {})
            config["engine"] = engine
            result = ocr_service.extract_text_from_image(temp_path, config)
            
            processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            response_data = {
                'success': True,
                'text': result.get('text', ''),
                'confidence': result.get('confidence', 0.0),
                'engine_used': result.get('engine', engine),
                'processing_time': processing_time,
                'word_details': result.get('word_details', []),
                'config_used': result.get('config', config),
                'image_info': {
                    'filename': file.filename,
                    'size': len(file.read())
                }
            }
            
            return jsonify(response_data)
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except BadRequest as e:
        return jsonify({
            'success': False,
            'error': 'Bad Request',
            'message': str(e.description)
        }), 400
    except Exception as e:
        logger.error(f"OCR处理失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'OCR Processing Failed',
            'message': str(e)
        }), 500

@ocr_bp.route('/batch', methods=['POST'])
def process_batch_ocr():
    """批量OCR处理"""
    start_time = time.time()
    
    try:
        # 检查是否有文件上传
        if 'images' not in request.files:
            raise BadRequest('No image files provided')
        
        files = request.files.getlist('images')
        if not files or len(files) == 0:
            raise BadRequest('No files selected')
        
        # 获取引擎类型和配置
        engine = normalize_engine(request.form.get('engine', 'deepseek_ocr'))
        if not validate_engine(engine):
            raise BadRequest(f'Invalid engine type: {engine}')
        
        config_str = request.form.get('config', '{}')
        try:
            config = json.loads(config_str) if config_str else {}
        except json.JSONDecodeError:
            raise BadRequest('Invalid config JSON format')
        
        results = []
        temp_files = []
        
        try:
            # 处理每个文件
            for i, file in enumerate(files):
                if not allowed_file(file.filename):
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': 'File type not allowed'
                    })
                    continue
                
                # 保存临时文件
                filename = secure_filename(file.filename)
                temp_path = os.path.join('temp', f"batch_{i}_{filename}")
                os.makedirs('temp', exist_ok=True)
                file.save(temp_path)
                temp_files.append(temp_path)
                
                try:
                    # 执行OCR识别
                    cfg = dict(config or {})
                    cfg["engine"] = engine
                    result = ocr_service.extract_text_from_image(temp_path, cfg)
                    
                    results.append({
                        'filename': file.filename,
                        'success': True,
                        'text': result.get('text', ''),
                        'confidence': result.get('confidence', 0.0),
                        'engine_used': result.get('engine', engine),
                        'word_count': len(result.get('text', '').split())
                    })
                    
                except Exception as e:
                    logger.error(f"处理文件 {file.filename} 失败: {str(e)}")
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': str(e)
                    })
            
            total_processing_time = (time.time() - start_time) * 1000
            
            return jsonify({
                'success': True,
                'results': results,
                'summary': {
                    'total_files': len(files),
                    'successful_files': len([r for r in results if r.get('success', False)]),
                    'failed_files': len([r for r in results if not r.get('success', False)]),
                    'total_processing_time': total_processing_time,
                    'engine_used': engine
                }
            })
            
        finally:
            # 清理临时文件
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
    except BadRequest as e:
        return jsonify({
            'success': False,
            'error': 'Bad Request',
            'message': str(e.description)
        }), 400
    except Exception as e:
        logger.error(f"批量OCR处理失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Batch OCR Processing Failed',
            'message': str(e)
        }), 500

@ocr_bp.route('/languages', methods=['GET'])
def get_supported_languages():
    """获取支持的语言列表"""
    try:
        languages = ocr_service.get_supported_languages()
        return jsonify({
            'success': True,
            'languages': languages,
            'count': len(languages)
        })
    except Exception as e:
        logger.error(f"获取支持语言失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get supported languages',
            'message': str(e)
        }), 500

@ocr_bp.route('/cache/clear', methods=['POST'])
def clear_ocr_cache():
    """清理OCR缓存"""
    try:
        ocr_service.clear_cache()
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        logger.error(f"清理缓存失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to clear cache',
            'message': str(e)
        }), 500
