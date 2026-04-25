import os
import json
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LocalTemplateManager:
    def __init__(self, registry_path='local_templates/registry.json'):
        """
        初始化本地模板管理器
        """
        self.root_dir = Path(os.getcwd())
        self.registry_path = self.root_dir / registry_path
        self.registry = self._load_registry()

    def _load_registry(self):
        """加载注册表配置"""
        if not self.registry_path.exists():
            logger.warning(f"[Warning] 本地模板注册表未找到: {self.registry_path}")
            return {}
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[Error] 读取注册表失败: {e}")
            return {}

    def search_local(self, journal_name):
        """
        在本地查找模板信息
        :param journal_name: 期刊名称 (模糊匹配)
        :return: 模板配置 dict 或 None
        """
        if not self.registry:
            return None
            
        # 1. 精确匹配
        if journal_name in self.registry:
            logger.info(f"[Success] 本地命中模板: {journal_name}")
            return self._enrich_path(self.registry[journal_name])

        # 2. 模糊匹配 (不区分大小写)
        key_lower = journal_name.lower()
        for key, config in self.registry.items():
            if key_lower in key.lower() or key.lower() in key_lower:
                logger.info(f"[Success] 本地模糊匹配命中: {key} (Search: {journal_name})")
                return self._enrich_path(config)

        logger.info(f"[Info] 本地未找到模板: {journal_name}")
        return None
        
    def get_all_templates(self):
        """获取所有本地模板列表"""
        return list(self.registry.keys())

    def _enrich_path(self, config):
        """将相对路径转换为绝对路径，方便后续模块调用"""
        base_path = self.root_dir / config['path']
        config['abs_path'] = str(base_path)
        if 'main_cls' in config:
            config['abs_cls_path'] = str(base_path / config['main_cls'])
        if 'template_tex' in config:
            config['abs_tex_path'] = str(base_path / config['template_tex'])
        return config

    def copy_template_to_workspace(self, template_config, target_dir):
        """
        将模板文件复制到当前工作区，准备进行 templateFit
        """
        src_dir = Path(template_config['abs_path'])
        dst_dir = Path(target_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 复制文件夹下的所有内容
            if not src_dir.exists():
                logger.error(f"源模板目录不存在: {src_dir}")
                return False
                
            for item in src_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, dst_dir / item.name)
            logger.info(f"[Action] 模板文件已加载至工作区: {target_dir}")
            return True
        except Exception as e:
            logger.error(f"[Error] 模板加载失败: {e}")
            return False
