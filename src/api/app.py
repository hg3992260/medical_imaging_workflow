#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask API服务器

提供OCR处理的Web API接口，使用DeepSeek-OCR引擎。
"""

import os
import sys
import time
from flask import Flask, jsonify, render_template
try:
    from flask_cors import CORS
except Exception:
    CORS = None

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.database_service import DatabaseService
from src.utils.logger import get_logger
from src.api.routes.ocr_routes import ocr_bp
from src.api.routes.project_routes import project_bp
from src.api.routes.dicom_routes import dicom_bp
from src.api.routes.agent_routes import agent_bp

# 初始化Flask应用
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
if CORS:
    CORS(app)  # 启用跨域支持

# 初始化服务
logger = get_logger(__name__)
db_service = DatabaseService()

# 注册蓝图
app.register_blueprint(ocr_bp)
app.register_blueprint(project_bp, url_prefix='/api')
app.register_blueprint(dicom_bp, url_prefix='/api')
app.register_blueprint(agent_bp, url_prefix='/api')

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.errorhandler(400)
def bad_request(error):
    """处理400错误"""
    return jsonify({
        'success': False,
        'error': 'Bad Request',
        'message': str(error.description)
    }), 400

@app.errorhandler(500)
def internal_error(error):
    """处理500错误"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'Internal Server Error',
        'message': 'An internal error occurred'
    }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': time.time()
    })

# OCR相关端点已移至 src/api/routes/ocr_routes.py









if __name__ == '__main__':
    # 初始化数据库
    try:
        db_service.initialize_database()
        logger.info("数据库初始化成功")
    except Exception as e:
        logger.error(f"数据库初始化失败: {str(e)}")
    
    # 启动Flask应用
    logger.info("启动OCR API服务器")
    port = int(os.environ.get("PORT", "5001"))
    app.run(host='0.0.0.0', port=port, debug=True)
