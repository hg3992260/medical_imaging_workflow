#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本处理服务模块

负责文档处理、文本分析、格式转换等功能。
"""

import os
import sys
import json
import re
import gc
import threading
import uuid
import time
import csv
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from docx import Document
    from docx.shared import Inches
except ImportError:
    Document = None

try:
    import win32com.client
except ImportError:
    win32com = None

try:
    import openpyxl
    from openpyxl import Workbook
except ImportError:
    openpyxl = None
    Workbook = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pdf2image import convert_from_path
    import os
    # 设置poppler路径
    POPPLER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'poppler', 'poppler-22.04.0', 'Library', 'bin')
    # 打包环境修正
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        meipass_dir = getattr(sys, "_MEIPASS", os.path.join(base_dir, "_internal"))
        bundled_candidates = [
            os.path.join(base_dir, "_internal", "assets", "poppler_bin"),
            os.path.join(meipass_dir, 'assets', 'poppler_bin'),
            os.path.join(meipass_dir, 'assets', 'poppler', 'bin'),
            os.path.join(meipass_dir, 'poppler_bin'),
        ]
        for bundled_poppler in bundled_candidates:
            if os.path.isdir(bundled_poppler):
                POPPLER_PATH = bundled_poppler
                break
except ImportError:
    convert_from_path = None
    POPPLER_PATH = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from src.services.database_service import DatabaseService
from src.services.project_storage_service import ProjectStorageService
from src.utils.logger import get_logger


class TextService:
    """
    文本处理服务类
    """
    
    def __init__(self, db_service: DatabaseService = None, pdf_config: Dict[str, Any] = None):
        """
        初始化文本服务
        
        Args:
            db_service: 数据库服务实例
            pdf_config: PDF处理配置
        """
        self.db_service = db_service or DatabaseService()
        self.storage_service = ProjectStorageService(self.db_service)
        self.logger = get_logger(__name__)
        
        # PDF处理配置
        default_pdf_config = {
            'max_pages_per_batch': 50,  # 每批处理的最大页数
            'default_max_pages': 1000,   # 默认最大处理页数
            'enable_cache': True,       # 是否启用缓存
            'cache_ttl': 3600,         # 缓存过期时间（秒）
            'ocr_language': 'chi_sim+eng',  # OCR语言设置
            'enable_progress_callback': True  # 是否启用进度回调
        }
        self.pdf_config = {**default_pdf_config, **(pdf_config or {})}
        
        # 检查依赖库是否可用
        self.docx_available = Document is not None
        self.win32com_available = win32com is not None and sys.platform.startswith('win')
        self.excel_available = openpyxl is not None
        self.pymupdf_available = fitz is not None
        self.pdf2image_available = convert_from_path is not None
        self.pdfplumber_available = pdfplumber is not None
        
        # 性能优化：缓存机制
        self._text_cache = {}
        self._analysis_cache = {}
        self._pdf_cache = {}  # PDF专用缓存
        self._cache_lock = threading.Lock()
        self._max_cache_size = 100
        
        if not self.docx_available:
            self.logger.warning("python-docx未安装，Word文档处理功能不可用")
        
        if not self.excel_available:
            self.logger.warning("openpyxl未安装，Excel文档处理功能不可用")
        
        if not self.pymupdf_available:
            self.logger.warning("PyMuPDF未安装，PDF文本提取功能不可用")
        
        if not self.pdf2image_available:
            self.logger.warning("pdf2image未安装，PDF图像转换功能不可用")
        
        if not self.pdfplumber_available:
            self.logger.warning("pdfplumber未安装，PDF备选处理功能不可用")
    
    def read_text_file(self, file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        读取文本文件
        
        Args:
            file_path: 文件路径
            encoding: 文件编码
        
        Returns:
            文件内容和元信息
        """
        try:
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': f'文件不存在: {file_path}',
                    'content': '',
                    'metadata': {}
                }
            
            # 尝试不同编码读取文件
            encodings = [encoding, 'utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin-1', 'cp1252', 'iso-8859-1']
            content = ''
            used_encoding = encoding
            decode_errors = []
            
            for enc in encodings:
                try:
                    with open(file_path, 'r', encoding=enc, errors='strict') as f:
                        content = f.read()
                    used_encoding = enc
                    self.logger.info(f"成功使用编码 {enc} 读取文件: {file_path}")
                    break
                except UnicodeDecodeError as e:
                    decode_errors.append(f"{enc}: {str(e)}")
                    continue
                except Exception as e:
                    decode_errors.append(f"{enc}: {str(e)}")
                    continue
            
            # 如果所有编码都失败，尝试使用错误处理模式
            if not content:
                self.logger.warning(f"所有标准编码失败，尝试容错模式读取: {file_path}")
                for enc in ['utf-8', 'gbk', 'latin-1']:
                    try:
                        with open(file_path, 'r', encoding=enc, errors='replace') as f:
                            content = f.read()
                        used_encoding = f"{enc} (with errors replaced)"
                        self.logger.info(f"使用容错模式 {enc} 成功读取文件: {file_path}")
                        break
                    except Exception as e:
                        decode_errors.append(f"{enc} (replace): {str(e)}")
                        continue
            
            if not content:
                error_msg = f"无法解码文件内容，尝试的编码: {', '.join([e.split(':')[0] for e in decode_errors])}"
                self.logger.error(f"{error_msg}. 详细错误: {'; '.join(decode_errors)}")
                return {
                    'success': False,
                    'error': error_msg,
                    'content': '',
                    'metadata': {'decode_errors': decode_errors}
                }
            
            # 检查内容是否为空或只包含空白字符
            if not content.strip():
                self.logger.warning(f"文件内容为空或只包含空白字符: {file_path}")
                return {
                    'success': False,
                    'error': f"文件内容为空或只包含空白字符: {file_path}",
                    'content': content,
                    'metadata': {'encoding': used_encoding}
                }
            
            # 获取文件元信息
            file_stat = os.stat(file_path)
            metadata = {
                'file_path': file_path,
                'file_size': file_stat.st_size,
                'created_time': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'encoding': used_encoding,
                'line_count': len(content.splitlines()),
                'char_count': len(content),
                'word_count': len(content.split())
            }
            
            return {
                'success': True,
                'content': content,
                'metadata': metadata
            }
            
        except FileNotFoundError as e:
            error_msg = f"文本文件不存在: {file_path}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {'error_type': 'FileNotFoundError'}
            }
        except PermissionError as e:
            error_msg = f"文本文件权限错误: {file_path}, {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {'error_type': 'PermissionError'}
            }
        except UnicodeDecodeError as e:
            error_msg = f"文本文件编码错误: {file_path}, {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {'error_type': 'UnicodeDecodeError'}
            }
        except Exception as e:
            error_msg = f"读取文本文件时发生错误: {str(e)}"
            self.logger.error(f"{error_msg} - 文件: {file_path}")
            
            # 记录详细的错误信息用于调试
            import traceback
            self.logger.debug(f"文本文件处理异常详情: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            }
    
    def read_word_document(self, file_path: str) -> Dict[str, Any]:
        """
        读取Word文档
        
        Args:
            file_path: Word文档路径
        
        Returns:
            文档内容和元信息
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            if file_ext == '.doc':
                return self._read_legacy_doc_file(file_path)

            if not self.docx_available:
                return {
                    'success': False,
                    'error': 'python-docx未安装',
                    'content': '',
                    'metadata': {}
                }
            
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': f'文件不存在: {file_path}',
                    'content': '',
                    'metadata': {}
                }
            
            # 读取Word文档
            doc = Document(file_path)
            
            # 提取文本内容
            paragraphs = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text)
            
            # 除了段落，还需要遍历所有形状（如文本框），避免漏掉只包含文本框的文档
            try:
                # 遍历文档的所有内联形状（如果是旧版 python-docx，可能不支持部分形状）
                for shape in doc.inline_shapes:
                    if hasattr(shape, 'text') and shape.text and shape.text.strip():
                        paragraphs.append(shape.text)
            except Exception:
                pass
                
            content = '\n'.join(paragraphs)
            
            # 提取表格内容
            tables_content = []
            for table in doc.tables:
                table_data = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_data.append(row_data)
                tables_content.append(table_data)
            
            # 将表格内容也合并到文档内容中，避免只有表格时内容为空
            for table_data in tables_content:
                for row_data in table_data:
                    content += '\n' + '\t'.join(row_data)
            
            # 如果是只读文档或者是某种特殊加密格式，python-docx 有可能无法提取出任何内容。
            # 如果提取后依然为空，我们尝试利用 pywin32 的 COM 对象去兜底读取（因为它能打开更多受限文件）
            if not content.strip() and self.win32com_available:
                self.logger.info(f"python-docx 提取 {file_path} 内容为空，尝试使用 COM 对象兜底读取...")
                com_result = self._read_doc_via_word_com(file_path)
                if com_result.get('success') and com_result.get('content', '').strip():
                    return com_result
            
            # 获取文档属性
            core_props = doc.core_properties
            metadata = {
                'file_path': file_path,
                'title': core_props.title or '',
                'author': core_props.author or '',
                'subject': core_props.subject or '',
                'created': core_props.created.isoformat() if core_props.created else '',
                'modified': core_props.modified.isoformat() if core_props.modified else '',
                'paragraph_count': len(paragraphs),
                'table_count': len(tables_content),
                'char_count': len(content),
                'word_count': len(content.split())
            }
            
            return {
                'success': True,
                'content': content,
                'tables': tables_content,
                'metadata': metadata
            }
            
        except FileNotFoundError as e:
            error_msg = f"Word文档不存在: {file_path}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {'error_type': 'FileNotFoundError'}
            }
        except PermissionError as e:
            error_msg = f"Word文档权限错误: {file_path}, {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {'error_type': 'PermissionError'}
            }
        except Exception as e:
            error_msg = f"读取Word文档时发生错误: {str(e)}"
            self.logger.error(f"{error_msg} - 文件: {file_path}")
            
            # 记录详细的错误信息用于调试
            import traceback
            self.logger.debug(f"Word文档处理异常详情: {traceback.format_exc()}")
            
            # 如果使用 python-docx 失败，且是 Windows 系统，尝试回退到 COM 读取
            if self.win32com_available:
                self.logger.info(f"python-docx读取失败，尝试使用COM对象重新读取: {file_path}")
                com_result = self._read_doc_via_word_com(file_path)
                if com_result.get('success'):
                    return com_result
            
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            }

    def _read_legacy_doc_file(self, file_path: str) -> Dict[str, Any]:
        if self.win32com_available:
            com_result = self._read_doc_via_word_com(file_path)
            if com_result.get('success'):
                return com_result
        cli_result = self._read_doc_via_cli(file_path)
        if cli_result.get('success'):
            return cli_result
        return {
            'success': False,
            'error': f".doc读取失败: {cli_result.get('error', '无可用解析器')}",
            'content': '',
            'metadata': {'error_type': 'LegacyDocUnsupported'}
        }

    def _decode_bytes_best_effort(self, raw: bytes) -> str:
        raw = raw or b""
        has_utf16_bom = raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff")
        null_ratio = (raw.count(b"\x00") / max(len(raw), 1)) if raw else 0.0
        preferred = ['utf-8-sig', 'utf-8', 'gb18030', 'gbk', 'big5']
        if has_utf16_bom or null_ratio > 0.08:
            preferred = ['utf-16', 'utf-16le', 'utf-16be'] + preferred
        for enc in preferred + ['latin1']:
            try:
                text = raw.decode(enc, errors='strict')
                if text.strip():
                    return text
            except Exception:
                continue
        for enc in preferred + ['latin1']:
            try:
                text = raw.decode(enc, errors='ignore')
                if text.strip():
                    return text
            except Exception:
                continue
        return ""

    def _read_doc_via_word_com(self, file_path: str) -> Dict[str, Any]:
        word_app = None
        doc = None
        temp_txt = ""
        try:
            import pythoncom
            pythoncom.CoInitialize()
            word_app = win32com.client.DispatchEx("Word.Application")
            word_app.Visible = False
            doc = word_app.Documents.Open(os.path.abspath(file_path), ReadOnly=True)
            fd, temp_txt = tempfile.mkstemp(suffix=".txt")
            os.close(fd)
            saved = False
            for fmt in [7, 2]:
                try:
                    doc.SaveAs(os.path.abspath(temp_txt), FileFormat=fmt)
                    saved = True
                    break
                except Exception:
                    continue
            if not saved:
                return {'success': False, 'error': 'Word COM SaveAs failed', 'content': '', 'metadata': {}}
            with open(temp_txt, 'rb') as f:
                raw = f.read()
            content = self._decode_bytes_best_effort(raw)
            lines = [x for x in (content or '').splitlines() if x.strip()]
            tables = self._parse_tables_from_text(lines)
            return {
                'success': True,
                'content': content,
                'tables': [[list(r) for r in tbl] for tbl in tables] if tables else [],
                'metadata': {
                    'file_path': file_path,
                    'paragraph_count': len([x for x in (content or '').splitlines() if x.strip()]),
                    'table_count': 0,
                    'char_count': len(content),
                    'word_count': len((content or '').split()),
                    'parser': 'win32com_word'
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'content': '', 'metadata': {}}
        finally:
            try:
                if doc is not None:
                    doc.Close(False)
            except Exception:
                pass
            try:
                if word_app is not None:
                    word_app.Quit()
            except Exception:
                pass
            if temp_txt and os.path.exists(temp_txt):
                try:
                    os.remove(temp_txt)
                except Exception:
                    pass

    def _read_doc_via_cli(self, file_path: str) -> Dict[str, Any]:
        antiword_cmd = shutil.which('antiword')
        if antiword_cmd:
            try:
                p = subprocess.run(
                    [antiword_cmd, file_path],
                    capture_output=True,
                    timeout=120
                )
                if p.returncode == 0:
                    raw = p.stdout or b''
                    content = self._decode_bytes_best_effort(raw)
                    lines_list = [x for x in content.splitlines() if x.strip()]
                    tables = self._parse_tables_from_text(lines_list)
                    return {
                        'success': True,
                        'content': content,
                        'tables': [[list(r) for r in tbl] for tbl in tables] if tables else [],
                        'metadata': {
                            'file_path': file_path,
                            'paragraph_count': len([x for x in content.splitlines() if x.strip()]),
                            'table_count': 0,
                            'char_count': len(content),
                            'word_count': len(content.split()),
                            'parser': 'antiword'
                        }
                    }
            except Exception:
                pass

        soffice_cmd = shutil.which('soffice') or shutil.which('libreoffice')
        if soffice_cmd:
            try:
                with tempfile.TemporaryDirectory() as tdir:
                    p = subprocess.run(
                        [soffice_cmd, '--headless', '--convert-to', 'txt:Text', '--outdir', tdir, file_path],
                        capture_output=True,
                        timeout=180
                    )
                    if p.returncode != 0:
                        err = self._decode_bytes_best_effort(p.stderr or p.stdout or b'')
                        return {'success': False, 'error': err or 'soffice convert failed', 'content': '', 'metadata': {}}
                    txt_path = os.path.join(tdir, f"{Path(file_path).stem}.txt")
                    if not os.path.exists(txt_path):
                        return {'success': False, 'error': 'converted txt not found', 'content': '', 'metadata': {}}
                    with open(txt_path, 'rb') as f:
                        raw = f.read()
                    content = self._decode_bytes_best_effort(raw)
                    lines_list = [x for x in (content or '').splitlines() if x.strip()]
                    tables = self._parse_tables_from_text(lines_list)
                    return {
                        'success': True,
                        'content': content,
                        'tables': [[list(r) for r in tbl] for tbl in tables] if tables else [],
                        'metadata': {
                            'file_path': file_path,
                            'paragraph_count': len([x for x in content.splitlines() if x.strip()]),
                            'table_count': 0,
                            'char_count': len(content),
                            'word_count': len(content.split()),
                            'parser': 'soffice'
                        }
                    }
            except Exception as e:
                return {'success': False, 'error': str(e), 'content': '', 'metadata': {}}

        return self._read_doc_via_olefile(file_path)

    def _read_doc_via_olefile(self, file_path: str) -> Dict[str, Any]:
        try:
            import olefile
        except ImportError:
            return {'success': False, 'error': 'antiword/soffice/olefile unavailable', 'content': '', 'metadata': {}}
        try:
            ole = olefile.OleFileIO(file_path)
            if not ole.exists('WordDocument'):
                ole.close()
                return {'success': False, 'error': 'Not a valid OLE2 Word document', 'content': '', 'metadata': {}}

            raw_chunks: List[bytes] = []
            for stream_name in ole.listdir():
                try:
                    data = ole.openstream(stream_name).read()
                    if isinstance(data, bytes) and len(data) > 10:
                        raw_chunks.append(data)
                except Exception:
                    pass
            ole.close()

            if not raw_chunks:
                return {'success': False, 'error': 'No extractable text in OLE2 document', 'content': '', 'metadata': {}}

            content = ""
            for chunk in raw_chunks:
                null_ratio = chunk.count(b'\x00') / max(len(chunk), 1)
                if null_ratio > 0.3:
                    content += self._decode_bytes_best_effort(chunk) + '\n'
                elif b'\x00\x00' not in chunk[:200]:
                    content += self._decode_bytes_best_effort(chunk) + '\n'
            content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
            content = re.sub(r'\n{3,}', '\n\n', content).strip()
            if not content.strip():
                return {'success': False, 'error': 'No extractable text in OLE2 document', 'content': '', 'metadata': {}}
            lines = [x for x in (content or '').splitlines() if x.strip()]
            tables_extracted = self._parse_tables_from_text(lines)
            tbl_data = [[list(r) for r in tbl] for tbl in tables_extracted]
            return {
                'success': True,
                'content': content,
                'tables': tbl_data if tbl_data else [],
                'metadata': {
                    'file_path': file_path,
                    'paragraph_count': len(lines),
                    'table_count': len(tables_extracted),
                    'char_count': len(content),
                    'word_count': len((content or '').split()),
                    'parser': 'olefile',
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'content': '', 'metadata': {}}

    def _parse_tables_from_text(self, lines: List[str]) -> List[List[List[str]]]:
        tables = []
        current_table = []
        in_table = False
        for line in lines:
            stripped = line.strip()
            parts = stripped.split('\t')
            if len(parts) >= 2 and all(p.strip() for p in parts):
                if not in_table:
                    if current_table:
                        tables.append(current_table)
                    current_table = [parts]
                    in_table = True
                else:
                    current_table.append(parts)
            else:
                if in_table and len(stripped) < 80:
                    in_table = False
                elif not in_table and current_table:
                    tables.append(current_table)
                    current_table = []
                    in_table = False
        if current_table:
            tables.append(current_table)
        return tables

    def read_excel_file(self, file_path: str, sheet_name: str = None) -> Dict[str, Any]:
        """
        读取Excel文件
        
        Args:
            file_path: Excel文件路径
            sheet_name: 工作表名称（可选）
        
        Returns:
            Excel内容和元信息
        """
        try:
            if not self.excel_available:
                return {
                    'success': False,
                    'error': 'openpyxl未安装',
                    'content': {},
                    'metadata': {}
                }
            
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': f'文件不存在: {file_path}',
                    'content': {},
                    'metadata': {}
                }
            
            # 读取Excel文件
            workbook = openpyxl.load_workbook(file_path, data_only=True)
            
            sheets_data = {}
            total_rows = 0
            total_cells = 0
            
            # 如果指定了工作表名称
            if sheet_name:
                if sheet_name in workbook.sheetnames:
                    sheet = workbook[sheet_name]
                    sheet_data, rows, cells = self._extract_sheet_data(sheet)
                    sheets_data[sheet_name] = sheet_data
                    total_rows += rows
                    total_cells += cells
                else:
                    return {
                        'success': False,
                        'error': f'工作表不存在: {sheet_name}',
                        'content': {},
                        'metadata': {}
                    }
            else:
                # 读取所有工作表
                for sheet_name in workbook.sheetnames:
                    sheet = workbook[sheet_name]
                    sheet_data, rows, cells = self._extract_sheet_data(sheet)
                    sheets_data[sheet_name] = sheet_data
                    total_rows += rows
                    total_cells += cells
            
            # 获取文件元信息
            file_stat = os.stat(file_path)
            metadata = {
                'file_path': file_path,
                'file_size': file_stat.st_size,
                'created_time': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'sheet_count': len(workbook.sheetnames),
                'sheet_names': workbook.sheetnames,
                'total_rows': total_rows,
                'total_cells': total_cells
            }
            
            workbook.close()
            
            return {
                'success': True,
                'content': sheets_data,
                'metadata': metadata
            }
            
        except FileNotFoundError as e:
            error_msg = f"Excel文件不存在: {file_path}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'content': {},
                'metadata': {'error_type': 'FileNotFoundError'}
            }
        except PermissionError as e:
            error_msg = f"Excel文件权限错误: {file_path}, {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'content': {},
                'metadata': {'error_type': 'PermissionError'}
            }
        except Exception as e:
            error_msg = f"读取Excel文件时发生错误: {str(e)}"
            self.logger.error(f"{error_msg} - 文件: {file_path}")
            
            # 记录详细的错误信息用于调试
            import traceback
            self.logger.debug(f"Excel文件处理异常详情: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': error_msg,
                'content': {},
                'metadata': {
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            }
    
    def read_pdf_document(self, file_path: str, max_pages: int = None, 
                         use_ocr: bool = False, progress_callback=None, status_callback=None,
                         deepseek_mode: str = "markdown", ocr_engine: str = None) -> Dict[str, Any]:
        """
        读取PDF文档，支持文本型、扫描版和混合型PDF
        
        Args:
            file_path: PDF文件路径
            max_pages: 最大处理页数（None表示处理所有页）
            use_ocr: 是否强制使用OCR
            progress_callback: 进度回调函数
        
        Returns:
            PDF内容和元信息
        """
        try:
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': f'文件不存在: {file_path}',
                    'content': '',
                    'metadata': {}
                }
            
            # 检查文件大小
            try:
                file_size = os.path.getsize(file_path)
                if file_size > 200 * 1024 * 1024:  # 200MB限制
                    return {
                        'success': False,
                        'error': f'PDF文件过大 ({file_size / 1024 / 1024:.1f}MB)，请选择较小的文件',
                        'content': '',
                        'metadata': {}
                    }
            except OSError as e:
                return {
                    'success': False,
                    'error': f'无法访问文件: {str(e)}',
                    'content': '',
                    'metadata': {}
                }
            
            # 检查文件是否为有效的PDF
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(4)
                    if header != b'%PDF':
                        return {
                            'success': False,
                            'error': '文件不是有效的PDF格式',
                            'content': '',
                            'metadata': {}
                        }
            except Exception as e:
                return {
                    'success': False,
                    'error': f'无法验证PDF文件格式: {str(e)}',
                    'content': '',
                    'metadata': {}
                }
            
            # 应用默认配置
            if max_pages is None:
                max_pages = self.pdf_config.get('default_max_pages', 100)
            
            # 检查缓存
            cache_key = self._get_pdf_cache_key(file_path, max_pages)
            cached_result = self._get_pdf_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            # 获取文件元信息
            file_stat = os.stat(file_path)
            metadata = {
                'file_path': file_path,
                'file_size': file_stat.st_size,
                'created_time': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'processing_method': 'unknown',
                'page_count': 0,
                'char_count': 0,
                'word_count': 0
            }
            
            content = ""
            result = None
            
            # 如果强制使用OCR或者PyMuPDF不可用，直接使用OCR
            if use_ocr or not self.pymupdf_available:
                result = self._read_pdf_with_ocr(
                    file_path,
                    max_pages,
                    progress_callback,
                    status_callback=status_callback,
                    deepseek_mode=deepseek_mode,
                    ocr_engine=ocr_engine,
                )
                if result['success']:
                    content = result['content']
                    metadata.update(result['metadata'])
                    metadata['processing_method'] = 'ocr'
                else:
                    return result
            else:
                # 首先尝试使用PyMuPDF进行文本提取
                result = self._read_pdf_with_pymupdf(file_path, max_pages, progress_callback)
                if result['success'] and result['content'].strip():
                    content = result['content']
                    metadata.update(result['metadata'])
                    metadata['processing_method'] = 'text_extraction'
                else:
                    # 文本提取失败或内容为空，尝试OCR
                    self.logger.info(f"PDF文本提取失败或内容为空，转为OCR处理: {file_path}")
                    ocr_result = self._read_pdf_with_ocr(
                        file_path,
                        max_pages,
                        progress_callback,
                        status_callback=status_callback,
                        deepseek_mode=deepseek_mode,
                        ocr_engine=ocr_engine,
                    )
                    if ocr_result['success']:
                        content = ocr_result['content']
                        metadata.update(ocr_result['metadata'])
                        metadata['processing_method'] = 'hybrid_ocr'
                        result = ocr_result
                    else:
                        return ocr_result
            
            # 更新最终统计信息
            metadata['char_count'] = len(content)
            metadata['word_count'] = len(content.split())
            
            final_result = {
                'success': True,
                'content': content,
                'metadata': metadata
            }
            
            # 保存到缓存
            self._save_pdf_to_cache(cache_key, final_result)
            
            return final_result
            
        except FileNotFoundError as e:
            error_msg = f"PDF文件不存在: {file_path}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {'file_path': file_path, 'error_type': 'FileNotFoundError'}
            }
        except PermissionError as e:
            error_msg = f"PDF文件权限错误: {file_path}, {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {'file_path': file_path, 'error_type': 'PermissionError'}
            }
        except MemoryError as e:
            error_msg = f"PDF文件过大，内存不足: {file_path}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {'file_path': file_path, 'error_type': 'MemoryError'}
            }
        except Exception as e:
            error_msg = f"读取PDF文档时发生未知错误: {str(e)}"
            self.logger.error(f"{error_msg} - 文件: {file_path}")
            
            # 记录详细的错误信息用于调试
            import traceback
            self.logger.debug(f"PDF处理异常详情: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {
                    'file_path': file_path,
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            }
    
    def _read_pdf_with_pymupdf(self, file_path: str, max_pages: int = None, 
                              progress_callback=None) -> Dict[str, Any]:
        """
        使用PyMuPDF读取PDF文本内容
        
        Args:
            file_path: PDF文件路径
            max_pages: 最大处理页数
            progress_callback: 进度回调函数
        
        Returns:
            处理结果
        """
        import time
        start_time = time.time()
        
        try:
            if not self.pymupdf_available:
                self.logger.error(f"PyMuPDF不可用，无法处理PDF: {file_path}")
                return {
                    'success': False,
                    'error': 'PyMuPDF未安装或不可用',
                    'content': '',
                    'metadata': {}
                }
            
            self.logger.info(f"开始使用PyMuPDF处理PDF: {file_path}")
            
            # 尝试打开PDF文档
            try:
                doc = fitz.open(file_path)
            except Exception as e:
                self.logger.error(f"无法打开PDF文件 {file_path}: {str(e)}")
                return {
                    'success': False,
                    'error': f'PDF文件损坏或格式不支持: {str(e)}',
                    'content': '',
                    'metadata': {}
                }
            
            total_pages = len(doc)
            self.logger.info(f"PDF总页数: {total_pages}")
            
            # 检查页数限制
            if total_pages == 0:
                doc.close()
                self.logger.warning(f"PDF文件为空: {file_path}")
                return {
                    'success': False,
                    'error': 'PDF文件为空',
                    'content': '',
                    'metadata': {'page_count': 0}
                }
            
            # 确定处理页数
            pages_to_process = min(max_pages, total_pages) if max_pages else total_pages
            self.logger.info(f"计划处理页数: {pages_to_process}/{total_pages}")
            
            content_parts = []
            successful_pages = 0
            failed_pages = 0
            
            for page_num in range(pages_to_process):
                try:
                    page = doc[page_num]
                    text = page.get_text()
                    
                    if text.strip():
                        content_parts.append(f"=== 第{page_num + 1}页 ===\n{text}\n")
                        successful_pages += 1
                    else:
                        self.logger.debug(f"第{page_num + 1}页无文本内容")
                    
                    # 更新进度
                    if progress_callback:
                        progress = int((page_num + 1) / pages_to_process * 100)
                        progress_callback(progress)
                        
                except Exception as e:
                    failed_pages += 1
                    self.logger.warning(f"处理第{page_num + 1}页时出错: {str(e)}")
                    continue
            
            doc.close()
            
            processing_time = time.time() - start_time
            content = '\n'.join(content_parts)
            
            # 记录处理统计信息
            self.logger.info(f"PyMuPDF处理完成 - 成功页数: {successful_pages}, 失败页数: {failed_pages}, 耗时: {processing_time:.2f}秒")
            
            metadata = {
                'page_count': total_pages,
                'processed_pages': pages_to_process,
                'successful_pages': successful_pages,
                'failed_pages': failed_pages,
                'processing_time': processing_time,
                'extraction_method': 'pymupdf'
            }
            
            # 检查是否有有效内容
            if not content.strip():
                self.logger.warning(f"PyMuPDF未提取到有效文本内容: {file_path}")
                return {
                    'success': False,
                    'error': '未提取到有效文本内容',
                    'content': '',
                    'metadata': metadata
                }
            
            return {
                'success': True,
                'content': content,
                'metadata': metadata
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"PyMuPDF处理失败: {str(e)}"
            self.logger.error(f"{error_msg}, 耗时: {processing_time:.2f}秒")
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {'processing_time': processing_time}
            }
    
    def _read_pdf_with_ocr(self, file_path: str, max_pages: int = None, 
                          progress_callback=None, status_callback=None,
                          deepseek_mode: str = "markdown", ocr_engine: str = None) -> Dict[str, Any]:
        """
        使用OCR读取PDF内容
        
        Args:
            file_path: PDF文件路径
            max_pages: 最大处理页数
            progress_callback: 进度回调函数
        
        Returns:
            处理结果
        """
        import time
        start_time = time.time()
        
        try:
            if not self.pdf2image_available:
                self.logger.error(f"pdf2image不可用，无法进行OCR处理: {file_path}")
                return {
                    'success': False,
                    'error': 'pdf2image未安装或不可用',
                    'content': '',
                    'metadata': {}
                }
            
            self.logger.info(f"开始使用OCR处理PDF: {file_path}")
            if callable(status_callback):
                try:
                    status_callback("正在准备 DeepSeek-OCR 模型并协调显存…")
                except Exception:
                    pass
            
            # 检查OCR服务是否可用
            if not hasattr(self, 'ocr_service') or self.ocr_service is None:
                try:
                    from src.services.ocr_service import get_ocr_service
                    self.ocr_service = get_ocr_service(self.db_service)
                except Exception as e:
                    self.logger.error(f"OCR服务不可用，无法进行OCR处理: {file_path}, 错误: {str(e)}")
                    return {
                        'success': False,
                        'error': 'OCR服务未安装或不可用',
                        'content': '',
                        'metadata': {}
                    }
            
            # 转换PDF为图像
            try:
                self.logger.info(f"开始转换PDF为图像，最大页数: {max_pages or '全部'}")
                
                # 限制DPI以控制内存使用
                dpi = 150  # 降低DPI以减少内存使用
                
                if max_pages:
                    # 限制最大页数以避免内存不足
                    actual_max_pages = min(max_pages, 50)  # 最多处理50页
                    images = convert_from_path(file_path, last_page=actual_max_pages, dpi=dpi, poppler_path=POPPLER_PATH)
                else:
                    # 对于未指定页数的情况，先检查PDF页数
                    try:
                        import fitz
                        doc = fitz.open(file_path)
                        total_pages = len(doc)
                        doc.close()
                        
                        if total_pages > 50:
                            self.logger.warning(f"PDF页数过多({total_pages}页)，仅处理前50页")
                            images = convert_from_path(file_path, last_page=50, dpi=dpi, poppler_path=POPPLER_PATH)
                        else:
                            images = convert_from_path(file_path, dpi=dpi, poppler_path=POPPLER_PATH)
                    except:
                        # 如果无法获取页数，限制为前20页
                        images = convert_from_path(file_path, last_page=20, dpi=dpi, poppler_path=POPPLER_PATH)
                        
                self.logger.info(f"PDF转图像成功，共{len(images)}页")
                
            except MemoryError:
                error_msg = '内存不足，无法转换PDF为图像。请尝试处理较小的PDF文件或重启应用程序'
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'content': '',
                    'metadata': {}
                }
            except Exception as e:
                error_msg = f'PDF转图像失败: {str(e)}'
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'content': '',
                    'metadata': {}
                }
            
            if not images:
                self.logger.warning(f"PDF转换后无图像内容: {file_path}")
                return {
                    'success': False,
                    'error': 'PDF转换后无图像内容',
                    'content': '',
                    'metadata': {}
                }
            
            content_parts = []
            total_pages = len(images)
            successful_pages = 0
            failed_pages = 0
            
            for i, image in enumerate(images):
                try:
                    self.logger.debug(f"开始OCR处理第{i + 1}页")
                    
                    # 检查图像大小，如果过大则调整以节省内存
                    try:
                        width, height = image.size
                        max_dimension = 3000  # 最大尺寸限制
                        if width > max_dimension or height > max_dimension:
                            # 按比例缩放
                            ratio = min(max_dimension / width, max_dimension / height)
                            new_width = int(width * ratio)
                            new_height = int(height * ratio)
                            image = image.resize((new_width, new_height))
                            self.logger.debug(f"第{i + 1}页图像已缩放: {width}x{height} -> {new_width}x{new_height}")
                    except Exception as resize_e:
                        self.logger.warning(f"第{i + 1}页图像缩放失败: {str(resize_e)}")
                    
                    # 使用OCR服务提取文本，优先使用DeepSeek
                    try:
                        # 配置OCR引擎优先使用DeepSeek
                        ocr_config = {
                            'engine': (ocr_engine or 'deepseek_ocr'),
                            'language': 'auto',
                            'confidence_threshold': 0.5,
                            'deepseek_mode': deepseek_mode,
                            'status_callback': status_callback,
                        }
                        try:
                            if "OLLAMA_OCR_TIMEOUT_SECONDS" in os.environ:
                                ocr_config["ollama_timeout_seconds"] = float(os.environ.get("OLLAMA_OCR_TIMEOUT_SECONDS") or "90")
                        except Exception:
                            pass
                        
                        # 将PIL图像转换为numpy数组
                        import numpy as np
                        image_array = np.array(image)
                        
                        # 使用OCR服务进行文本提取（直接传递numpy数组给OCR引擎）
                        if hasattr(self.ocr_service, 'extract_text_from_image'):
                             ocr_result = self.ocr_service.extract_text_from_image(image_array, ocr_config)
                        else:
                            ocr_config['engine'] = 'deepseek'
                            ocr_result = self.ocr_service.extract_text_deepseek(image_array, ocr_config)
                        
                        # 转换结果格式以匹配extract_text_from_image的返回格式
                        if ocr_result:
                            # DeepSeek adapter returns 'text' and 'confidence' directly
                            # but just in case check structure
                            if 'extracted_text' not in ocr_result and 'text' in ocr_result:
                                ocr_result['extracted_text'] = ocr_result['text']
                                
                            ocr_result = {
                                'extracted_text': ocr_result.get('extracted_text', ocr_result.get('text', '')),
                                'ocr_engine': ocr_result.get('ocr_engine', ocr_result.get('engine', 'deepseek')),
                                'confidence_score': ocr_result.get('confidence_score', ocr_result.get('confidence', 0.0))
                            }
                        
                        if ocr_result and ocr_result.get('extracted_text', '').strip():
                            text = ocr_result['extracted_text']
                            content_parts.append(f"=== 第{i + 1}页 ===\n{text}\n")
                            successful_pages += 1
                            engine_used = ocr_result.get('ocr_engine', 'unknown')
                            self.logger.debug(f"第{i + 1}页OCR成功，提取{len(text)}个字符，使用引擎: {engine_used}")
                        else:
                            self.logger.debug(f"第{i + 1}页OCR无文本内容")
                            
                    except MemoryError:
                        failed_pages += 1
                        self.logger.error(f"第{i + 1}页OCR内存不足，跳过该页")
                        # 强制垃圾回收
                        import gc
                        gc.collect()
                        continue
                    except Exception as ocr_e:
                        failed_pages += 1
                        self.logger.warning(f"第{i + 1}页OCR失败: {str(ocr_e)}")
                        continue
                    
                    # 释放图像内存
                    try:
                        del image
                    except:
                        pass
                        
                except Exception as e:
                    failed_pages += 1
                    self.logger.warning(f"OCR处理第{i + 1}页时出错: {str(e)}")
                    continue
                finally:
                    if progress_callback:
                        try:
                            progress = int((i + 1) / total_pages * 100)
                            progress_callback(progress)
                        except Exception:
                            pass
            
            processing_time = time.time() - start_time
            content = '\n'.join(content_parts)
            
            # 记录处理统计信息
            self.logger.info(f"OCR处理完成 - 成功页数: {successful_pages}, 失败页数: {failed_pages}, 耗时: {processing_time:.2f}秒")
            
            metadata = {
                'page_count': total_pages,
                'processed_pages': total_pages,
                'successful_pages': successful_pages,
                'failed_pages': failed_pages,
                'processing_time': processing_time,
                'extraction_method': 'ocr'
            }
            
            # 检查是否有有效内容
            if not content.strip():
                self.logger.warning(f"OCR未提取到有效文本内容: {file_path}")
                return {
                    'success': False,
                    'error': 'OCR未提取到有效文本内容',
                    'content': '',
                    'metadata': metadata
                }
            
            return {
                'success': True,
                'content': content,
                'metadata': metadata
            }
            
        except FileNotFoundError as e:
            processing_time = time.time() - start_time
            error_msg = f"OCR处理时文件不存在: {file_path}"
            self.logger.error(f"{error_msg}, 耗时: {processing_time:.2f}秒")
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {'processing_time': processing_time, 'error_type': 'FileNotFoundError'}
            }
        except PermissionError as e:
            processing_time = time.time() - start_time
            error_msg = f"OCR处理时文件权限错误: {file_path}, {str(e)}"
            self.logger.error(f"{error_msg}, 耗时: {processing_time:.2f}秒")
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {'processing_time': processing_time, 'error_type': 'PermissionError'}
            }
        except MemoryError as e:
            processing_time = time.time() - start_time
            error_msg = f"OCR处理时内存不足: {file_path}"
            self.logger.error(f"{error_msg}, 耗时: {processing_time:.2f}秒")
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {'processing_time': processing_time, 'error_type': 'MemoryError'}
            }
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"OCR处理时发生未知错误: {str(e)}"
            self.logger.error(f"{error_msg}, 耗时: {processing_time:.2f}秒")
            
            # 记录详细的错误信息用于调试
            import traceback
            self.logger.debug(f"OCR处理异常详情: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'metadata': {
                    'processing_time': processing_time,
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            }
    
    def _read_image_with_ocr(self, file_path: str) -> str:
        """
        使用OCR读取图像文件，优先使用DeepSeek
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            提取的文本内容
        """
        try:
            # 确保OCR服务已初始化
            if not hasattr(self, 'ocr_service') or self.ocr_service is None:
                try:
                    from src.services.ocr_service import get_ocr_service
                    self.ocr_service = get_ocr_service(self.db_service)
                except Exception as e:
                    self.logger.error(f"OCR服务不可用: {e}")
                    raise RuntimeError("OCR服务不可用")
            
            # 配置OCR - 强制使用DeepSeek OCR
            ocr_config = {
                'engine': 'deepseek_ocr',
                'language': 'auto',
                'deepseek_mode': 'markdown',
            }
            
            # 调用OCR服务
            result = self.ocr_service.extract_text_from_image(file_path, ocr_config)
            
            if result.get('error'):
                # 如果DeepSeek失败，ocr_service内部逻辑会自动尝试回退，
                # 但如果返回了错误，我们记录日志
                self.logger.warning(f"OCR识别警告: {result.get('error')}")
                
            text = result.get('text', '')
            if not text.strip():
                self.logger.warning(f"OCR未从图像提取到文本: {file_path}")
                
            return text
            
        except Exception as e:
            self.logger.error(f"图像OCR处理失败: {e}")
            raise RuntimeError(f"图像OCR处理失败: {str(e)}")

    def read_file(self, file_path: str) -> str:
        """
        通用文件读取方法，根据文件扩展名自动选择合适的读取方法
        
        Args:
            file_path: 文件路径
        
        Returns:
            文件内容字符串
        """
        try:
            if not os.path.exists(file_path):
                error_msg = f"文件不存在: {file_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # 检查文件是否为空
            try:
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    error_msg = f"文件为空: {file_path}"
                    self.logger.warning(error_msg)
                    raise ValueError(error_msg)
            except OSError as e:
                error_msg = f"无法获取文件大小: {file_path}, 错误: {str(e)}"
                self.logger.error(error_msg)
                raise OSError(error_msg)
            
            # 检查文件权限
            if not os.access(file_path, os.R_OK):
                error_msg = f"文件无读取权限: {file_path}"
                self.logger.error(error_msg)
                raise PermissionError(error_msg)
            
            # 获取文件扩展名
            file_ext = Path(file_path).suffix.lower()
            self.logger.info(f"开始读取文件: {file_path}, 类型: {file_ext}, 大小: {file_size} bytes")
            
            # 根据文件类型选择读取方法
            if file_ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']:
                result = self.read_text_file(file_path)
                if result.get('success'):
                    content = result.get('content', '')
                    if not content.strip():
                        self.logger.warning(f"文本文件内容为空: {file_path}")
                        raise ValueError(f"文本文件内容为空: {file_path}")
                    return content
                else:
                    error_msg = f"读取文本文件失败: {result.get('error')}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
            
            elif file_ext in ['.docx', '.doc']:
                result = self.read_word_document(file_path)
                if result.get('success'):
                    content = result.get('content', '')
                    if not content.strip():
                        # Word 文档可能是扫描件或纯图片，尝试对原始文件进行OCR（如果支持）
                        self.logger.warning(f"Word文档提取文本为空，可能是扫描件或纯图片: {file_path}")
                        raise ValueError(f"Word文档内容为空: {file_path}")
                    return content
                else:
                    error_msg = f"读取Word文档失败: {result.get('error')}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
            
            elif file_ext in ['.xlsx', '.xls']:
                result = self.read_excel_file(file_path)
                if result.get('success'):
                    # 将Excel内容转换为文本
                    content_dict = result.get('content', {})
                    if not content_dict:
                        self.logger.warning(f"Excel文件内容为空: {file_path}")
                        raise ValueError(f"Excel文件内容为空: {file_path}")
                    
                    text_content = []
                    for sheet_name, sheet_data in content_dict.items():
                        text_content.append(f"=== {sheet_name} ===")
                        for row in sheet_data:
                            text_content.append('\t'.join(row))
                        text_content.append("")
                    
                    content = '\n'.join(text_content)
                    if not content.strip():
                        self.logger.warning(f"Excel文件转换后内容为空: {file_path}")
                        raise ValueError(f"Excel文件转换后内容为空: {file_path}")
                    return content
                else:
                    error_msg = f"读取Excel文件失败: {result.get('error')}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
            
            elif file_ext == '.pdf':
                result = self.read_pdf_document(file_path)
                if result.get('success'):
                    content = result.get('content', '')
                    if not content.strip():
                        self.logger.warning(f"PDF文件内容为空或无法提取文本: {file_path}")
                        raise ValueError(f"PDF文件内容为空或无法提取文本: {file_path}")
                    return content
                else:
                    error_msg = f"读取PDF文件失败: {result.get('error')}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
            
            elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
                return self._read_image_with_ocr(file_path)

            else:
                # 对于未知类型，尝试作为文本文件读取
                self.logger.info(f"未知文件类型 {file_ext}，尝试作为文本文件读取: {file_path}")
                result = self.read_text_file(file_path)
                if result.get('success'):
                    content = result.get('content', '')
                    if not content.strip():
                        self.logger.warning(f"未知类型文件内容为空: {file_path}")
                        # 内容为空也尝试OCR
                        try:
                            self.logger.info(f"内容为空，尝试作为图像进行OCR: {file_path}")
                            return self._read_image_with_ocr(file_path)
                        except Exception:
                            raise ValueError(f"未知类型文件内容为空: {file_path}")
                    return content
                else:
                    # 文本读取失败，尝试OCR
                    try:
                        self.logger.info(f"文本读取失败，尝试作为图像进行OCR: {file_path}")
                        return self._read_image_with_ocr(file_path)
                    except Exception as ocr_e:
                        error_msg = f"不支持的文件类型: {file_ext}, 文本读取和OCR均失败. 文本错误: {result.get('error')}, OCR错误: {str(ocr_e)}"
                        self.logger.error(error_msg)
                        raise RuntimeError(error_msg)
        
        except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as e:
            # 重新抛出已知异常
            raise e
        except Exception as e:
            error_msg = f"读取文件时发生未知错误: {file_path}, 错误: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            self.logger.debug(f"详细错误信息: {traceback.format_exc()}")
            raise RuntimeError(error_msg)

    def read_file_structured(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        读取文件并在支持时返回结构化表格数据（Excel/CSV）。
        """
        file_ext = Path(file_path).suffix.lower()
        if file_ext in ['.xlsx', '.xls']:
            result = self.read_excel_file(file_path)
            if not result.get('success'):
                raise RuntimeError(f"读取Excel文件失败: {result.get('error')}")
            tabular = self._excel_to_tabular_payload(file_path, result.get('content', {}) or {})
            return {"text": tabular.get("text", ""), "tabular": tabular, "format": file_ext}

        if file_ext == '.csv':
            tabular = self._read_csv_as_tabular_payload(file_path)
            return {"text": tabular.get("text", ""), "tabular": tabular, "format": file_ext}

        return {"text": self.read_file(file_path), "tabular": None, "format": file_ext}

    def _read_csv_as_tabular_payload(self, file_path: str) -> Dict[str, Any]:
        encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312"]
        rows: List[List[str]] = []
        last_err = ""
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc, newline="") as f:
                    reader = csv.reader(f)
                    for r in reader:
                        rows.append([c if c is not None else "" for c in r])
                break
            except Exception as e:
                last_err = str(e)
                rows = []
        if not rows:
            raise RuntimeError(f"CSV解析失败: {last_err}")

        headers = [str(x or "").strip() for x in (rows[0] or [])]
        body = rows[1:] if len(rows) > 1 else []

        records = []
        for r in body:
            obj = {}
            for i, h in enumerate(headers):
                if not h:
                    continue
                obj[h] = str(r[i]).strip() if i < len(r) and r[i] is not None else ""
            if obj:
                records.append(obj)

        kv = {}
        if len(headers) == 2 and records:
            key_h = headers[0] or "key"
            val_h = headers[1] or "value"
            ok = True
            for rec in records[:2000]:
                k = str(rec.get(key_h, "")).strip()
                v = str(rec.get(val_h, "")).strip()
                if not k:
                    ok = False
                    break
                kv[k] = v
            if not ok:
                kv = {}

        sheet = {
            "sheet_name": "CSV",
            "headers": headers,
            "rows": rows,
            "records": records[:5000],
            "kv": kv,
            "n_rows": len(body),
            "n_cols": len(headers),
        }
        text = self._tabular_sheet_to_text("CSV", rows, max_rows=120)
        return {"type": "csv", "source_file_name": os.path.basename(file_path), "sheets": [sheet], "text": text}

    def _excel_to_tabular_payload(self, file_path: str, content_dict: Dict[str, List[List[str]]]) -> Dict[str, Any]:
        sheets = []
        text_blocks = []
        for sheet_name, sheet_data in (content_dict or {}).items():
            rows = sheet_data or []
            headers = [str(x or "").strip() for x in (rows[0] if rows else [])]
            body = rows[1:] if len(rows) > 1 else []

            records = []
            for r in body:
                obj = {}
                for i, h in enumerate(headers):
                    if not h:
                        continue
                    obj[h] = str(r[i]).strip() if i < len(r) and r[i] is not None else ""
                if obj:
                    records.append(obj)

            kv = {}
            if len(headers) == 2 and records:
                key_h = headers[0] or "key"
                val_h = headers[1] or "value"
                ok = True
                for rec in records[:2000]:
                    k = str(rec.get(key_h, "")).strip()
                    v = str(rec.get(val_h, "")).strip()
                    if not k:
                        ok = False
                        break
                    kv[k] = v
                if not ok:
                    kv = {}

            sheets.append(
                {
                    "sheet_name": str(sheet_name),
                    "headers": headers,
                    "rows": rows,
                    "records": records[:5000],
                    "kv": kv,
                    "n_rows": len(body),
                    "n_cols": len(headers),
                }
            )
            text_blocks.append(self._tabular_sheet_to_text(str(sheet_name), rows, max_rows=120))

        return {"type": "excel", "source_file_name": os.path.basename(file_path), "sheets": sheets, "text": "\n\n".join([b for b in text_blocks if b])}

    def _word_table_to_tabular_payload(self, file_path: str, tables: List[List[List[str]]]) -> Optional[Dict[str, Any]]:
        if not tables:
            return None
        sheets = []
        text_blocks = []
        for ti, table_rows in enumerate(tables):
            if not table_rows or len(table_rows) < 1:
                continue
            sheet_name = f"Table{ti+1}"
            headers = [str(x or "").strip() for x in (table_rows[0] if table_rows else [])]
            body = table_rows[1:] if len(table_rows) > 1 else []
            records = []
            for r in body:
                obj = {}
                for i, h in enumerate(headers):
                    if not h:
                        continue
                    obj[h] = str(r[i]).strip() if i < len(r) and r[i] is not None else ""
                if obj:
                    records.append(obj)
            kv = {}
            if len(headers) == 2 and records:
                key_h = headers[0] or "key"
                val_h = headers[1] or "value"
                ok = True
                for rec in records[:2000]:
                    k = str(rec.get(key_h, "")).strip()
                    v = str(rec.get(val_h, "")).strip()
                    if not k:
                        ok = False
                        break
                    kv[k] = v
                if not ok:
                    kv = {}
            sheets.append({
                "sheet_name": sheet_name,
                "headers": headers,
                "rows": table_rows,
                "records": records[:5000],
                "kv": kv,
                "n_rows": len(body),
                "n_cols": len(headers),
            })
            text_blocks.append(self._tabular_sheet_to_text(sheet_name, table_rows, max_rows=120))
        return {"type": "word_table", "source_file_name": os.path.basename(file_path), "sheets": sheets, "text": "\n\n".join([b for b in text_blocks if b])}

    def _tabular_sheet_to_text(self, sheet_name: str, rows: List[List[Any]], max_rows: int = 120) -> str:
        rows = rows or []
        out = [f"=== {sheet_name} ==="]
        for r in rows[: max(1, int(max_rows))]:
            out.append("\t".join([str(x) if x is not None else "" for x in (r or [])]))
        return "\n".join(out).strip()
    
    def _extract_sheet_data(self, sheet) -> tuple:
        """
        提取工作表数据
        
        Args:
            sheet: 工作表对象
        
        Returns:
            (工作表数据, 行数, 单元格数)
        """
        data = []
        row_count = 0
        cell_count = 0
        
        for row in sheet.iter_rows(values_only=True):
            # 跳过完全空白的行
            if any(cell is not None and str(cell).strip() for cell in row):
                row_data = [str(cell) if cell is not None else '' for cell in row]
                data.append(row_data)
                row_count += 1
                cell_count += len(row_data)
        
        return data, row_count, cell_count
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        分析文本内容
        
        Args:
            text: 要分析的文本
        
        Returns:
            文本分析结果
        """
        try:
            if not text:
                return {
                    'char_count': 0,
                    'word_count': 0,
                    'line_count': 0,
                    'paragraph_count': 0,
                    'sentence_count': 0,
                    'chinese_char_count': 0,
                    'english_word_count': 0,
                    'number_count': 0,
                    'punctuation_count': 0
                }
            
            # 基本统计
            char_count = len(text)
            word_count = len(text.split())
            line_count = len(text.splitlines())
            paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
            
            # 句子统计（简单的句号、问号、感叹号分割）
            sentence_pattern = r'[.!?。！？]+'
            sentences = re.split(sentence_pattern, text)
            sentence_count = len([s for s in sentences if s.strip()])
            
            # 中文字符统计
            chinese_pattern = r'[\u4e00-\u9fff]'
            chinese_chars = re.findall(chinese_pattern, text)
            chinese_char_count = len(chinese_chars)
            
            # 英文单词统计
            english_pattern = r'\b[a-zA-Z]+\b'
            english_words = re.findall(english_pattern, text)
            english_word_count = len(english_words)
            
            # 数字统计
            number_pattern = r'\d+'
            numbers = re.findall(number_pattern, text)
            number_count = len(numbers)
            
            # 标点符号统计
            punctuation_pattern = r'[^\w\s\u4e00-\u9fff]'
            punctuations = re.findall(punctuation_pattern, text)
            punctuation_count = len(punctuations)
            
            return {
                'char_count': char_count,
                'word_count': word_count,
                'line_count': line_count,
                'paragraph_count': paragraph_count,
                'sentence_count': sentence_count,
                'chinese_char_count': chinese_char_count,
                'english_word_count': english_word_count,
                'number_count': number_count,
                'punctuation_count': punctuation_count
            }
            
        except Exception as e:
            self.logger.error(f"文本分析失败: {e}")
            return {}
    
    def extract_keywords(self, text: str, max_keywords: int = 20, 
                        method: str = 'frequency') -> List[str]:
        """
        提取关键词
        
        Args:
            text: 输入文本
            max_keywords: 最大关键词数量
            method: 提取方法 ('frequency', 'tfidf', 'textrank')
        
        Returns:
            关键词列表
        """
        try:
            if not text:
                return []
            
            if method == 'frequency':
                return self._extract_keywords_frequency(text, max_keywords)
            elif method == 'tfidf':
                return self._extract_keywords_tfidf(text, max_keywords)
            elif method == 'textrank':
                return self._extract_keywords_textrank(text, max_keywords)
            else:
                return self._extract_keywords_frequency(text, max_keywords)
            
        except Exception as e:
            self.logger.error(f"关键词提取失败: {e}")
            return []
    
    def _extract_keywords_frequency(self, text: str, max_keywords: int) -> List[str]:
        """基于词频的关键词提取"""
        # 移除标点符号和特殊字符
        cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # 分词（简单的空格分割）
        words = cleaned_text.split()
        
        # 过滤短词和常见停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 
                     'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        
        filtered_words = [word.lower() for word in words 
                        if len(word) > 1 and word.lower() not in stop_words]
        
        # 统计词频
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序并返回前N个
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:max_keywords]]
        
        return keywords
    
    def _extract_keywords_tfidf(self, text: str, max_keywords: int) -> List[str]:
        """基于TF-IDF的关键词提取（简化版）"""
        # 简化的TF-IDF实现
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return self._extract_keywords_frequency(text, max_keywords)
        
        # 计算词频
        word_freq = {}
        doc_freq = {}
        
        for sentence in sentences:
            words = re.findall(r'[\w\u4e00-\u9fff]+', sentence.lower())
            words = [w for w in words if len(w) > 1]
            
            # TF计算
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # DF计算
            unique_words = set(words)
            for word in unique_words:
                doc_freq[word] = doc_freq.get(word, 0) + 1
        
        # 计算TF-IDF分数
        import math
        tfidf_scores = {}
        total_docs = len(sentences)
        
        for word, tf in word_freq.items():
            df = doc_freq.get(word, 1)
            idf = math.log(total_docs / df)
            tfidf_scores[word] = tf * idf
        
        # 排序并返回
        sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, score in sorted_words[:max_keywords]]
        
        return keywords
    
    def _extract_keywords_textrank(self, text: str, max_keywords: int) -> List[str]:
        """基于TextRank的关键词提取（简化版）"""
        # 简化的TextRank实现
        words = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
        words = [w for w in words if len(w) > 1]
        
        if len(words) < 10:
            return self._extract_keywords_frequency(text, max_keywords)
        
        # 构建词共现矩阵
        word_list = list(set(words))
        word_index = {word: i for i, word in enumerate(word_list)}
        
        # 简化的共现窗口
        window_size = 5
        cooccur_matrix = [[0] * len(word_list) for _ in range(len(word_list))]
        
        for i in range(len(words)):
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if i != j:
                    word1_idx = word_index[words[i]]
                    word2_idx = word_index[words[j]]
                    cooccur_matrix[word1_idx][word2_idx] += 1
        
        # 简化的PageRank算法
        scores = [1.0] * len(word_list)
        damping = 0.85
        iterations = 30
        
        for _ in range(iterations):
            new_scores = [0.0] * len(word_list)
            for i in range(len(word_list)):
                for j in range(len(word_list)):
                    if cooccur_matrix[j][i] > 0:
                        out_degree = sum(cooccur_matrix[j])
                        if out_degree > 0:
                            new_scores[i] += damping * scores[j] * cooccur_matrix[j][i] / out_degree
                new_scores[i] += (1 - damping)
            scores = new_scores
        
        # 排序并返回
        word_scores = list(zip(word_list, scores))
        word_scores.sort(key=lambda x: x[1], reverse=True)
        keywords = [word for word, score in word_scores[:max_keywords]]
        
        return keywords
    
    def _get_pdf_cache_key(self, file_path: str, max_pages: int = None) -> str:
        """
        生成PDF缓存键
        
        Args:
            file_path: PDF文件路径
            max_pages: 最大页数
        
        Returns:
            缓存键
        """
        import hashlib
        
        # 获取文件修改时间和大小作为缓存键的一部分
        try:
            stat = os.stat(file_path)
            file_info = f"{file_path}_{stat.st_mtime}_{stat.st_size}_{max_pages}"
            return hashlib.md5(file_info.encode()).hexdigest()
        except Exception:
            return hashlib.md5(f"{file_path}_{max_pages}".encode()).hexdigest()
    
    def _get_pdf_from_cache(self, cache_key: str) -> Dict[str, Any]:
        """
        从缓存获取PDF处理结果
        
        Args:
            cache_key: 缓存键
        
        Returns:
            缓存的处理结果，如果不存在或过期则返回None
        """
        if not self.pdf_config.get('enable_cache', True):
            return None
        
        with self._cache_lock:
            if cache_key in self._pdf_cache:
                cache_entry = self._pdf_cache[cache_key]
                
                # 检查是否过期
                import time
                current_time = time.time()
                cache_time = cache_entry.get('timestamp', 0)
                ttl = self.pdf_config.get('cache_ttl', 3600)
                
                if current_time - cache_time < ttl:
                    self.logger.debug(f"PDF缓存命中: {cache_key}")
                    return cache_entry.get('data')
                else:
                    # 缓存过期，删除
                    del self._pdf_cache[cache_key]
                    self.logger.debug(f"PDF缓存过期: {cache_key}")
        
        return None
    
    def _save_pdf_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """
        保存PDF处理结果到缓存
        
        Args:
            cache_key: 缓存键
            data: 处理结果
        """
        if not self.pdf_config.get('enable_cache', True):
            return
        
        import time
        
        with self._cache_lock:
            # 检查缓存大小，如果超过限制则清理最旧的条目
            if len(self._pdf_cache) >= self._max_cache_size:
                # 删除最旧的缓存条目
                oldest_key = min(self._pdf_cache.keys(), 
                                key=lambda k: self._pdf_cache[k].get('timestamp', 0))
                del self._pdf_cache[oldest_key]
                self.logger.debug(f"清理PDF缓存: {oldest_key}")
            
            # 保存到缓存
            self._pdf_cache[cache_key] = {
                'data': data,
                'timestamp': time.time()
            }
            self.logger.debug(f"PDF结果已缓存: {cache_key}")
    
    def _clear_pdf_cache(self) -> None:
        """
        清理PDF缓存
        """
        with self._cache_lock:
            self._pdf_cache.clear()
            self.logger.info("PDF缓存已清理")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        with self._cache_lock:
            return {
                'text_cache_size': len(self._text_cache),
                'analysis_cache_size': len(self._analysis_cache),
                'pdf_cache_size': len(self._pdf_cache),
                'max_cache_size': self._max_cache_size,
                'pdf_cache_enabled': self.pdf_config.get('enable_cache', True)
            }
    
    def create_text_session(self, user_id: str, file_path: str,
                           content: str, analysis_result: Dict, keywords: List = None,
                           project_id: str = None, tabular_payload: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        创建文本处理会话
        
        Args:
            user_id: 用户ID
            file_path: 文件路径
            content: 文本内容
            analysis_result: 分析结果
            keywords: 关键词列表（可选）
            project_id: 项目ID（可选）
        
        Returns:
            会话ID，如果创建失败则返回None
        """
        try:
            # 生成会话ID（添加毫秒级时间戳和随机数确保唯一性）
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            milliseconds = int(time.time() * 1000) % 1000
            random_suffix = str(uuid.uuid4())[:8]
            session_id = f"text_{timestamp}_{milliseconds:03d}_{user_id}_{random_suffix}"
            
            # 获取项目ID，如果未提供则获取当前项目ID或使用默认值
            if project_id is None:
                try:
                    from .project_service import ProjectService
                    project_service = ProjectService()
                    project_id = project_service.get_current_project_id() or "default"
                except Exception:
                    project_id = "default"
            
            # 验证文件是否存在并存储到项目存储
            if not os.path.exists(file_path):
                self.logger.error(f"文件不存在: {file_path}")
                return None
            
            # 使用ProjectStorageService存储文件
            # 注意：不将全部内容作为元数据传递，避免数据库字段过大
            file_metadata = {
                'original_filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'char_count': len(content)
            }
            stored_file_path = self.storage_service.store_text_file(
                project_id, file_path, file_metadata
            )
            
            if not stored_file_path:
                self.logger.error(f"存储文件失败: {file_path}")
                return None
            
            # 插入文本处理会话记录
            session_query = """
                INSERT INTO text_sessions (session_id, user_id, project_id, operation_type, file_path, 
                                          char_count, word_count, line_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            session_params = (
                session_id,
                user_id,
                project_id,
                'text_processing',  # 添加operation_type字段值
                stored_file_path,  # 使用存储后的文件路径
                analysis_result.get('char_count', 0),
                analysis_result.get('word_count', 0),
                analysis_result.get('line_count', 0)
            )
            
            rows_affected = self.db_service.execute_update(session_query, session_params)
            
            if rows_affected == 0:
                self.logger.error("创建文本会话失败")
                return None
            
            # 存储分析结果和关键词到项目存储
            analysis_data = {
                'analysis_result': analysis_result,
                'keywords': keywords or self.extract_keywords(content),
                'content_preview': content  # 完整显示内容，不截断
            }

            if tabular_payload and isinstance(tabular_payload, dict):
                project_name = ""
                try:
                    rows = self.db_service.execute_query("SELECT name FROM projects WHERE project_id = ?", (project_id,))
                    if rows and rows[0].get("name"):
                        project_name = str(rows[0].get("name"))
                except Exception:
                    project_name = ""

                sample_name = ""
                try:
                    rows = self.db_service.execute_query(
                        "SELECT display_name FROM user_ids WHERE user_id = ? AND project_id = ?",
                        (user_id, project_id),
                    )
                    if rows and rows[0].get("display_name"):
                        sample_name = str(rows[0].get("display_name"))
                except Exception:
                    sample_name = ""

                src_name = tabular_payload.get("source_file_name") or os.path.basename(file_path)
                base_branch = f"{project_name or project_id}/{sample_name or user_id}/{src_name}".strip("/")
                tab_meta = {
                    "project_id": project_id,
                    "project_name": project_name,
                    "user_id": user_id,
                    "sample_name": sample_name,
                    "source_file_name": src_name,
                    "branch_path": base_branch,
                    "created_at": datetime.now().isoformat(),
                }
                stored_nodes = self.storage_service.store_tabular_json(project_id, user_id, tabular_payload, tab_meta)
                analysis_data["tabular_nodes"] = stored_nodes
            
            stored_analysis_path = self.storage_service.store_text_analysis(
                project_id, analysis_data
            )
            
            # 插入文档记录
            doc_query = """
                INSERT INTO documents (doc_id, session_id, file_path, file_name, format, content, 
                                     analysis_result, keywords, analysis_file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            file_name = os.path.basename(file_path)
            file_type = Path(file_path).suffix.lower()
            # 使用传入的关键词，如果没有则提取关键词
            if keywords is None:
                keywords = self.extract_keywords(content)
            
            doc_params = (
                uuid.uuid4().hex,
                session_id,
                stored_file_path,  # 使用存储后的文件路径
                file_name,
                file_type,
                content,
                json.dumps(analysis_result, ensure_ascii=False),
                json.dumps(keywords, ensure_ascii=False),
                stored_analysis_path  # 添加分析结果文件路径
            )
            
            self.db_service.execute_update(doc_query, doc_params)
            
            self.logger.info(f"文本会话创建成功: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"创建文本会话失败: {e}")
            return None
    
    def get_project_text_file_path(self, project_id: str, session_id: str) -> Optional[str]:
        """
        获取项目级文本文件路径
        
        Args:
            project_id: 项目ID
            session_id: 会话ID
        
        Returns:
            文件路径，如果不存在则返回None
        """
        try:
            return self.storage_service.get_text_file_path(project_id, session_id)
        except Exception as e:
            self.logger.error(f"获取项目文本文件路径失败: {e}")
            return None
    
    def load_text_analysis_from_project_storage(self, project_id: str, session_id: str) -> Optional[Dict]:
        """
        从项目存储加载文本分析结果
        
        Args:
            project_id: 项目ID
            session_id: 会话ID
        
        Returns:
            分析结果数据，如果加载失败则返回None
        """
        try:
            return self.storage_service.load_text_analysis(project_id, session_id)
        except Exception as e:
            self.logger.error(f"从项目存储加载文本分析结果失败: {e}")
            return None
    
    def process_text_from_project_storage(self, project_id: str, session_id: str) -> Optional[Dict]:
        """
        从项目存储的文件中处理文本
        
        Args:
            project_id: 项目ID
            session_id: 会话ID
        
        Returns:
            处理结果，如果失败则返回None
        """
        try:
            # 获取存储的文件路径
            file_path = self.get_project_text_file_path(project_id, session_id)
            if not file_path:
                return None
            
            # 读取文件内容
            content = self.read_file(file_path)
            if not content:
                return None
            
            # 分析文本
            analysis_result = self.analyze_text(content)
            keywords = self.extract_keywords(content)
            
            return {
                'content': content,
                'analysis_result': analysis_result,
                'keywords': keywords,
                'file_path': file_path
            }
        except Exception as e:
            self.logger.error(f"从项目存储处理文本失败: {e}")
            return None
    
    def get_text_sessions(self, user_id: str = None, limit: int = 50, project_id: str = None) -> List[Dict]:
        """
        获取用户的文本处理会话列表
        
        Args:
            user_id: 用户ID（可选）
            limit: 限制数量
            project_id: 项目ID（可选）
        
        Returns:
            会话列表
        """
        try:
            # 获取项目ID，如果未提供则获取当前项目ID或使用默认值
            if project_id is None:
                try:
                    from .project_service import ProjectService
                    project_service = ProjectService()
                    project_id = project_service.get_current_project_id() or "default"
                except Exception:
                    project_id = "default"
            
            # 构建查询条件
            where_conditions = ["s.project_id = ?"]
            params = [project_id]
            
            if user_id:
                where_conditions.append("s.user_id = ?")
                params.append(user_id)
            
            params.append(limit)
            
            query = f"""
                SELECT s.session_id, s.user_id, s.file_path, s.char_count, s.word_count, 
                       s.line_count, s.created_at, d.file_name, d.format as file_type
                FROM text_sessions s
                LEFT JOIN documents d ON s.session_id = d.session_id
                WHERE {' AND '.join(where_conditions)}
                ORDER BY s.created_at DESC
                LIMIT ?
            """
            
            results = self.db_service.execute_query(query, tuple(params))
            
            sessions = []
            for row in results:
                sessions.append({
                    'session_id': row['session_id'],
                    'file_path': row['file_path'],
                    'file_name': row['file_name'] or os.path.basename(row['file_path']),
                    'file_type': row['file_type'] or '',
                    'char_count': row['char_count'],
                    'word_count': row['word_count'],
                    'line_count': row['line_count'],
                    'created_at': row['created_at']
                })
            
            return sessions
            
        except Exception as e:
            self.logger.error(f"获取文本会话列表失败: {e}")
            return []
    
    def get_text_session_details(self, session_id: str) -> Optional[Dict]:
        """
        获取文本会话详情
        
        Args:
            session_id: 会话ID
        
        Returns:
            文本会话详情，包含会话信息和文档详情
        """
        try:
            # 获取会话基本信息
            session_query = """
                SELECT s.session_id, s.user_id, s.file_path, s.char_count, s.word_count, 
                       s.line_count, s.created_at, s.operation_type
                FROM text_sessions s
                WHERE s.session_id = ?
            """
            
            session_results = self.db_service.execute_query(session_query, (session_id,))
            
            if not session_results:
                return None
            
            session_row = session_results[0]
            
            # 获取文档详情
            doc_query = """
                SELECT file_name, format as file_type, content, analysis_result, keywords
                FROM documents
                WHERE session_id = ?
            """
            
            doc_results = self.db_service.execute_query(doc_query, (session_id,))
            
            # 解析文档数据
            file_name = ''
            file_type = ''
            content = ''
            analysis_result = {}
            keywords = []
            
            if doc_results:
                doc_row = doc_results[0]
                file_name = doc_row['file_name'] or ''
                file_type = doc_row['file_type'] or ''
                content = doc_row['content'] or ''
                
                if doc_row['analysis_result']:
                    try:
                        analysis_result = json.loads(doc_row['analysis_result'])
                    except json.JSONDecodeError:
                        pass
                
                if doc_row['keywords']:
                    try:
                        keywords = json.loads(doc_row['keywords'])
                    except json.JSONDecodeError:
                        pass
            
            return {
                'session_id': session_row['session_id'],
                'user_id': session_row['user_id'],
                'file_path': session_row['file_path'],
                'file_name': file_name,
                'file_type': file_type,
                'content': content,
                'analysis_result': analysis_result,
                'keywords': keywords,
                'char_count': session_row['char_count'],
                'word_count': session_row['word_count'],
                'line_count': session_row['line_count'],
                'operation_type': session_row['operation_type'],
                'created_at': session_row['created_at']
            }
            
        except Exception as e:
            self.logger.error(f"获取文本会话详情失败: {e}")
            return None
    
    def get_document_details(self, session_id: str) -> Optional[Dict]:
        """
        获取文档详情
        
        Args:
            session_id: 会话ID
        
        Returns:
            文档详情
        """
        try:
            query = """
                SELECT file_name, format as file_type, content, analysis_result, keywords
                FROM documents
                WHERE session_id = ?
            """
            
            results = self.db_service.execute_query(query, (session_id,))
            
            if not results:
                return None
            
            row = results[0]
            
            analysis_result = {}
            keywords = []
            
            if row['analysis_result']:
                try:
                    analysis_result = json.loads(row['analysis_result'])
                except json.JSONDecodeError:
                    pass
            
            if row['keywords']:
                try:
                    keywords = json.loads(row['keywords'])
                except json.JSONDecodeError:
                    pass
            
            return {
                'file_name': row['file_name'],
                'file_type': row['file_type'],
                'content': row['content'],
                'analysis_result': analysis_result,
                'keywords': keywords
            }
            
        except Exception as e:
            self.logger.error(f"获取文档详情失败: {e}")
            return None
    
    def export_to_file(self, content: str, file_path: str, format_type: str = 'txt') -> bool:
        """导出内容到文件"""
        try:
            if format_type.lower() == 'txt':
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
            elif format_type.lower() == 'docx':
                if not self.docx_available:
                    self.logger.error("python-docx库不可用")
                    return False
                    
                from docx import Document
                doc = Document()
                doc.add_paragraph(content)
                doc.save(file_path)
                
            elif format_type.lower() == 'xlsx':
                if not self.excel_available:
                    self.logger.error("openpyxl库不可用")
                    return False
                    
                from openpyxl import Workbook
                wb = Workbook()
                ws = wb.active
                ws['A1'] = content
                wb.save(file_path)
                
            else:
                self.logger.error(f"不支持的格式: {format_type}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"导出文件失败: {e}")
            return False
    
    def convert_text_format(self, input_file: str, output_file: str, 
                           input_format: str = None, output_format: str = None) -> bool:
        """文本格式转换"""
        try:
            # 自动检测输入格式
            if not input_format:
                input_format = input_file.split('.')[-1].lower()
            
            # 自动检测输出格式
            if not output_format:
                output_format = output_file.split('.')[-1].lower()
            
            # 读取输入文件
            if input_format == 'txt':
                content = self.read_text_file(input_file)
            elif input_format == 'docx':
                content = self.read_word_file(input_file)
            elif input_format in ['xlsx', 'xls']:
                result = self.read_excel_file(input_file)
                if result:
                    content = result.get('content', '')
                else:
                    return False
            else:
                self.logger.error(f"不支持的输入格式: {input_format}")
                return False
            
            if not content:
                self.logger.error("无法读取输入文件内容")
                return False
            
            # 导出到目标格式
            return self.export_to_file(content, output_file, output_format)
            
        except Exception as e:
            self.logger.error(f"格式转换失败: {e}")
            return False
    
    def batch_convert_format(self, input_dir: str, output_dir: str, 
                           input_format: str, output_format: str) -> Dict[str, bool]:
        """批量格式转换"""
        import os
        
        results = {}
        
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            for filename in os.listdir(input_dir):
                if filename.lower().endswith(f'.{input_format.lower()}'):
                    input_file = os.path.join(input_dir, filename)
                    output_filename = filename.rsplit('.', 1)[0] + f'.{output_format.lower()}'
                    output_file = os.path.join(output_dir, output_filename)
                    
                    success = self.convert_text_format(input_file, output_file, 
                                                     input_format, output_format)
                    results[filename] = success
                    
                    if success:
                        self.logger.info(f"转换成功: {filename} -> {output_filename}")
                    else:
                        self.logger.error(f"转换失败: {filename}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"批量转换失败: {e}")
            return results
    
    def clean_text(self, text: str, options: Dict = None) -> str:
        """文本清理和标准化"""
        if not text:
            return ""
        
        if not options:
            options = {}
        
        try:
            cleaned_text = text
            
            # 移除多余空白字符
            if options.get('remove_extra_whitespace', True):
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                cleaned_text = cleaned_text.strip()
            
            # 移除特殊字符
            if options.get('remove_special_chars', False):
                cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:]', '', cleaned_text)
            
            # 统一换行符
            if options.get('normalize_line_breaks', True):
                cleaned_text = re.sub(r'\r\n|\r', '\n', cleaned_text)
            
            # 移除空行
            if options.get('remove_empty_lines', False):
                lines = cleaned_text.split('\n')
                lines = [line for line in lines if line.strip()]
                cleaned_text = '\n'.join(lines)
            
            # 转换为小写
            if options.get('to_lowercase', False):
                cleaned_text = cleaned_text.lower()
            
            # 移除数字
            if options.get('remove_numbers', False):
                cleaned_text = re.sub(r'\d+', '', cleaned_text)
            
            # 移除标点符号
            if options.get('remove_punctuation', False):
                cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', cleaned_text)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"文本清理失败: {e}")
            return text
    
    def clear_cache(self):
        """清理缓存"""
        with self._cache_lock:
            self._text_cache.clear()
            self._analysis_cache.clear()
            gc.collect()
            self.logger.info("文本服务缓存已清理")
    
    def _get_cache_key(self, file_path: str, encoding: str = 'utf-8') -> str:
        """生成缓存键"""
        import hashlib
        key_str = f"{file_path}_{encoding}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def process_files_batch(self, file_paths: List[str], user_id: str, 
                           max_workers: int = 4) -> Dict[str, Any]:
        """
        批量处理文本文件
        
        Args:
            file_paths: 文件路径列表
            user_id: 用户ID
            max_workers: 最大并发数
        
        Returns:
            批量处理结果
        """
        results = {
            'success_count': 0,
            'error_count': 0,
            'session_ids': [],
            'errors': []
        }
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务
                future_to_file = {}
                for file_path in file_paths:
                    future = executor.submit(self._process_single_file, file_path, user_id)
                    future_to_file[future] = file_path
                
                # 收集结果
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result['success']:
                            results['success_count'] += 1
                            if result.get('session_id'):
                                results['session_ids'].append(result['session_id'])
                        else:
                            results['error_count'] += 1
                            results['errors'].append({
                                'file_path': file_path,
                                'error': result.get('error', '未知错误')
                            })
                    except Exception as e:
                        results['error_count'] += 1
                        results['errors'].append({
                            'file_path': file_path,
                            'error': str(e)
                        })
            
            # 清理缓存
            if len(self._text_cache) > self._max_cache_size:
                self.clear_cache()
            
            self.logger.info(f"批量处理完成: 成功{results['success_count']}个，失败{results['error_count']}个")
            return results
            
        except Exception as e:
            self.logger.error(f"批量处理失败: {e}")
            results['errors'].append({'error': str(e)})
            return results
    
    def _process_single_file(self, file_path: str, user_id: str) -> Dict[str, Any]:
        """处理单个文件"""
        try:
            # 检查缓存
            cache_key = self._get_cache_key(file_path)
            
            with self._cache_lock:
                if cache_key in self._text_cache:
                    content_result = self._text_cache[cache_key]
                    self.logger.debug(f"使用缓存内容: {file_path}")
                else:
                    content_result = None
            
            # 读取文件内容
            tabular_payload = None
            if not content_result:
                file_ext = Path(file_path).suffix.lower()
                
                if file_ext == '.txt':
                    content_result = self.read_text_file(file_path)
                elif file_ext in ['.docx', '.doc']:
                    content_result = self.read_word_document(file_path)
                    tables = (content_result.get('tables') or []) if content_result.get('success') else []
                    word_tabular = self._word_table_to_tabular_payload(file_path, tables) if tables else None
                    tabular_payload = word_tabular
                else:
                    return {
                        'success': False,
                        'error': f'不支持的文件格式: {file_ext}'
                    }
                
                # 缓存结果
                if content_result.get('success'):
                    with self._cache_lock:
                        if len(self._text_cache) < self._max_cache_size:
                            self._text_cache[cache_key] = content_result
            
            if not content_result.get('success'):
                return content_result
            
            # 分析文本
            content = content_result.get('content', '')
            if isinstance(content, dict):  # Excel文件返回字典
                # 将所有工作表内容合并
                all_text = []
                for sheet_data in content.values():
                    for row in sheet_data:
                        all_text.extend([str(cell) for cell in row if cell])
                content = ' '.join(all_text)
            
            analysis_result = self.analyze_text(content)
            
            # 创建会话
            session_id = self.create_text_session(
                user_id, file_path, content, analysis_result,
                tabular_payload=tabular_payload,
            )
            
            return {
                'success': True,
                'session_id': session_id,
                'analysis_result': analysis_result
            }
            
        except Exception as e:
            self.logger.error(f"处理文件失败 {file_path}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @lru_cache(maxsize=50)
    def _get_optimized_analysis_params(self, text_length: int, language: str = 'mixed') -> Dict:
        """根据文本特征获取优化的分析参数"""
        params = {
            'max_keywords': 20,
            'keyword_method': 'frequency',
            'enable_detailed_analysis': True
        }
        
        # 根据文本长度调整参数
        if text_length < 1000:
            params['max_keywords'] = 10
            params['keyword_method'] = 'frequency'
        elif text_length < 10000:
            params['max_keywords'] = 15
            params['keyword_method'] = 'tfidf'
        else:
            params['max_keywords'] = 25
            params['keyword_method'] = 'textrank'
            
        # 对于超长文本，禁用详细分析以提高性能
        if text_length > 100000:
            params['enable_detailed_analysis'] = False
            
        return params
    
    def optimize_memory_usage(self):
        """优化内存使用"""
        try:
            # 清理缓存
            self.clear_cache()
            
            # 强制垃圾回收
            gc.collect()
            
            self.logger.info("内存优化完成")
            
        except Exception as e:
            self.logger.error(f"内存优化失败: {e}")
    
    def delete_text_session(self, session_id: str, user_id: str) -> bool:
        """
        删除指定的文本处理会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
        
        Returns:
            删除是否成功
        """
        try:
            cursor = self.db_service.get_cursor()
            
            # 删除相关的文档记录
            cursor.execute(
                "DELETE FROM documents WHERE session_id = ?",
                (session_id,)
            )
            
            # 删除文本会话记录
            cursor.execute(
                "DELETE FROM text_sessions WHERE session_id = ? AND user_id = ?",
                (session_id, user_id)
            )
            
            deleted_rows = cursor.rowcount
            self.db_service.commit()
            
            if deleted_rows > 0:
                self.logger.info(f"成功删除文本会话: {session_id}")
                return True
            else:
                self.logger.warning(f"未找到要删除的文本会话: {session_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"删除文本会话失败: {e}")
            self.db_service.rollback()
            return False
    
    def clear_all_text_sessions(self, user_id: str = None, project_id: str = None) -> bool:
        """
        清除指定用户或项目的所有文本处理会话
        
        Args:
            user_id: 用户ID（可选）
            project_id: 项目ID（可选，如果未提供则获取当前项目）
        
        Returns:
            清除是否成功
        """
        try:
            # 如果没有提供project_id，获取当前项目ID
            if project_id is None:
                try:
                    from .project_service import ProjectService
                    project_service = ProjectService()
                    project_id = project_service.get_current_project_id()
                except Exception as e:
                    self.logger.warning(f"获取当前项目ID失败: {e}，使用default项目")
                    project_id = "default"
            
            # 构建查询条件
            where_conditions = ["project_id = ?"]
            params = [project_id]
            
            if user_id:
                where_conditions.append("user_id = ?")
                params.append(user_id)
            
            # 使用正确的数据库连接管理
            with self.db_service.get_connection() as conn:
                cursor = conn.cursor()
                
                # 获取会话ID
                session_query = f"SELECT session_id FROM text_sessions WHERE {' AND '.join(where_conditions)}"
                cursor.execute(session_query, tuple(params))
                session_ids = [row[0] for row in cursor.fetchall()]
                
                # 删除相关的文档记录
                for session_id in session_ids:
                    cursor.execute(
                        "DELETE FROM documents WHERE session_id = ?",
                        (session_id,)
                    )
                
                # 删除文本会话
                delete_query = f"DELETE FROM text_sessions WHERE {' AND '.join(where_conditions)}"
                cursor.execute(delete_query, tuple(params))
                
                deleted_rows = cursor.rowcount
                conn.commit()  # 在同一连接上提交事务
                
                filter_desc = f"项目 {project_id}"
                if user_id:
                    filter_desc += f" 用户 {user_id}"
                self.logger.info(f"成功清除{filter_desc}的 {deleted_rows} 个文本会话")
                return True
            
        except Exception as e:
            self.logger.error(f"清除文本会话失败: {e}")
            return False
