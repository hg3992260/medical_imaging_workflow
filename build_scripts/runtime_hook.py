import os
import sys

if getattr(sys, 'frozen', False):
    exe_dir = os.path.dirname(sys.executable)
    assets_dir = os.path.join(exe_dir, '_internal', 'assets')
    os.environ['RSNA_DEEPSEEK_OCR_DIR'] = os.path.join(assets_dir, 'models', 'deepseek_ocr')
    os.environ['RSNA_ASSETS_DIR'] = assets_dir
