from typing import List, Optional, Callable
import json
import os
from PyQt5.QtCore import QObject, pyqtSignal

class UserIDManager(QObject):
    """用户ID管理器"""
    
    # 信号定义
    user_id_added = pyqtSignal(str)  # 用户ID添加信号
    user_id_removed = pyqtSignal(str)  # 用户ID删除信号
    user_id_selected = pyqtSignal(str)  # 用户ID选择信号
    
    def __init__(self, config_file: str = "config/user_ids.json"):
        super().__init__()
        self.config_file = config_file
        self.user_ids: List[str] = []
        self.selected_user_id: Optional[str] = None
        self.load_user_ids()
    
    def load_user_ids(self):
        """从配置文件加载用户ID列表"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.user_ids = data.get('user_ids', [])
                    self.selected_user_id = data.get('selected_user_id')
        except Exception as e:
            print(f"加载用户ID配置失败: {e}")
            self.user_ids = []
            self.selected_user_id = None
    
    def save_user_ids(self):
        """保存用户ID列表到配置文件"""
        try:
            # 确保配置目录存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            data = {
                'user_ids': self.user_ids,
                'selected_user_id': self.selected_user_id
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存用户ID配置失败: {e}")
    
    def add_user_id(self, user_id: str) -> bool:
        """添加用户ID"""
        if user_id and user_id not in self.user_ids:
            self.user_ids.append(user_id)
            self.save_user_ids()
            self.user_id_added.emit(user_id)
            return True
        return False
    
    def remove_user_id(self, user_id: str) -> bool:
        """删除用户ID"""
        if user_id in self.user_ids:
            self.user_ids.remove(user_id)
            
            # 如果删除的是当前选中的用户ID，清空选择
            if self.selected_user_id == user_id:
                self.selected_user_id = None
            
            self.save_user_ids()
            self.user_id_removed.emit(user_id)
            return True
        return False
    
    def select_user_id(self, user_id: str) -> bool:
        """选择用户ID"""
        if user_id in self.user_ids or user_id is None:
            self.selected_user_id = user_id
            self.save_user_ids()
            if user_id:
                self.user_id_selected.emit(user_id)
            return True
        return False
    
    def get_user_ids(self) -> List[str]:
        """获取所有用户ID"""
        return self.user_ids.copy()
    
    def get_selected_user_id(self) -> Optional[str]:
        """获取当前选中的用户ID"""
        return self.selected_user_id
    
    def clear_all_user_ids(self):
        """清空所有用户ID"""
        self.user_ids.clear()
        self.selected_user_id = None
        self.save_user_ids()
    
    def user_id_exists(self, user_id: str) -> bool:
        """检查用户ID是否存在"""
        return user_id in self.user_ids