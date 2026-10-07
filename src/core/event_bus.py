
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件总线模块

实现组件间的松耦合通信，支持事件的发布、订阅和处理。
主要用于用户ID选择变化时的数据同步。
"""

from typing import Dict, List, Callable, Any, Optional
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from datetime import datetime
import weakref
import threading

from src.utils.logger import get_logger


class EventBus(QObject):
    """
    事件总线类
    
    提供事件的发布、订阅和处理功能，支持组件间的松耦合通信。
    """
    
    # 全局事件信号
    event_published = pyqtSignal(str, object)  # 事件名称, 事件数据
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """
        单例模式实现
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        初始化事件总线
        """
        super().__init__()
        
        if hasattr(self, '_initialized'):
            return
        
        self.logger = get_logger(__name__)
        self._subscribers: Dict[str, List[weakref.WeakMethod]] = {}
        self._event_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000
        
        # 连接全局事件信号
        self.event_published.connect(self._handle_event)
        
        self._initialized = True
        self.logger.info("事件总线初始化完成")
    
    def subscribe(self, event_name: str, callback: Callable[[Any], None]):
        """
        订阅事件
        
        Args:
            event_name: 事件名称
            callback: 回调函数
        """
        try:
            if event_name not in self._subscribers:
                self._subscribers[event_name] = []
            
            # 使用弱引用避免内存泄漏
            weak_callback = weakref.WeakMethod(callback)
            self._subscribers[event_name].append(weak_callback)
            
            self.logger.debug(f"订阅事件 '{event_name}'，当前订阅者数量: {len(self._subscribers[event_name])}")
            
        except Exception as e:
            self.logger.error(f"订阅事件失败: {e}")
    
    def unsubscribe(self, event_name: str, callback: Callable[[Any], None]):
        """
        取消订阅事件
        
        Args:
            event_name: 事件名称
            callback: 回调函数
        """
        try:
            if event_name not in self._subscribers:
                return
            
            # 查找并移除对应的弱引用
            subscribers = self._subscribers[event_name]
            for i, weak_callback in enumerate(subscribers):
                if weak_callback() == callback:
                    subscribers.pop(i)
                    break
            
            # 清理失效的弱引用
            self._cleanup_dead_references(event_name)
            
            self.logger.debug(f"取消订阅事件 '{event_name}'")
            
        except Exception as e:
            self.logger.error(f"取消订阅事件失败: {e}")
    
    def publish(self, event_name: str, data: Any = None):
        """
        发布事件
        
        Args:
            event_name: 事件名称
            data: 事件数据
        """
        try:
            # 记录事件历史
            self._add_to_history(event_name, data)
            
            # 发送信号
            self.event_published.emit(event_name, data)
            
            self.logger.debug(f"发布事件 '{event_name}'")
            
        except Exception as e:
            self.logger.error(f"发布事件失败: {e}")
    
    def _handle_event(self, event_name: str, data: Any):
        """
        处理事件
        
        Args:
            event_name: 事件名称
            data: 事件数据
        """
        try:
            if event_name not in self._subscribers:
                return
            
            # 清理失效的弱引用
            self._cleanup_dead_references(event_name)
            
            # 调用所有订阅者的回调函数
            subscribers = self._subscribers[event_name].copy()
            for weak_callback in subscribers:
                callback = weak_callback()
                if callback is not None:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"事件回调执行失败: {e}")
            
        except Exception as e:
            self.logger.error(f"处理事件失败: {e}")
    
    def _cleanup_dead_references(self, event_name: str):
        """
        清理失效的弱引用
        
        Args:
            event_name: 事件名称
        """
        if event_name in self._subscribers:
            self._subscribers[event_name] = [
                weak_ref for weak_ref in self._subscribers[event_name]
                if weak_ref() is not None
            ]
    
    def _add_to_history(self, event_name: str, data: Any):
        """
        添加事件到历史记录
        
        Args:
            event_name: 事件名称
            data: 事件数据
        """
        event_record = {
            'event_name': event_name,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        self._event_history.append(event_record)
        
        # 限制历史记录大小
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
    
    def get_event_history(self, event_name: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取事件历史记录
        
        Args:
            event_name: 事件名称过滤器，None表示获取所有事件
            limit: 返回记录数量限制
        
        Returns:
            事件历史记录列表
        """
        try:
            if event_name is None:
                history = self._event_history
            else:
                history = [
                    record for record in self._event_history
                    if record['event_name'] == event_name
                ]
            
            # 返回最近的记录
            return history[-limit:] if limit > 0 else history
            
        except Exception as e:
            self.logger.error(f"获取事件历史失败: {e}")
            return []
    
    def clear_history(self):
        """
        清空事件历史记录
        """
        self._event_history.clear()
        self.logger.info("事件历史记录已清空")
    
    def get_subscribers_count(self, event_name: str) -> int:
        """
        获取指定事件的订阅者数量
        
        Args:
            event_name: 事件名称
        
        Returns:
            订阅者数量
        """
        if event_name not in self._subscribers:
            return 0
        
        # 清理失效的弱引用
        self._cleanup_dead_references(event_name)
        
        return len(self._subscribers[event_name])
    
    def get_all_events(self) -> List[str]:
        """
        获取所有已注册的事件名称
        
        Returns:
            事件名称列表
        """
        return list(self._subscribers.keys())
    
    def has_subscribers(self, event_name: str) -> bool:
        """
        检查指定事件是否有订阅者
        
        Args:
            event_name: 事件名称
        
        Returns:
            是否有订阅者
        """
        return self.get_subscribers_count(event_name) > 0


# 全局事件总线实例
_global_event_bus = None


def get_event_bus() -> EventBus:
    """
    获取全局事件总线实例
    
    Returns:
        事件总线实例
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


# 事件名称常量
class EventNames:
    """
    事件名称常量类
    """
    
    # 用户ID相关事件
    USER_ID_SELECTED = "user_id_selected"
    USER_ID_ADDED = "user_id_added"
    USER_ID_REMOVED = "user_id_removed"
    USER_ID_LIST_UPDATED = "user_id_list_updated"
    
    # 数据同步事件
    DATA_SYNC_REQUESTED = "data_sync_requested"
    DATA_SYNC_COMPLETED = "data_sync_completed"
    
    # DICOM相关事件
    DICOM_SESSION_CREATED = "dicom_session_created"
    DICOM_SESSION_UPDATED = "dicom_session_updated"
    DICOM_SESSION_DELETED = "dicom_session_deleted"
    
    # OCR相关事件
    OCR_SESSION_CREATED = "ocr_session_created"
    OCR_SESSION_UPDATED = "ocr_session_updated"
    OCR_SESSION_DELETED = "ocr_session_deleted"
    
    # 文本处理相关事件
    TEXT_SESSION_CREATED = "text_session_created"
    TEXT_SESSION_UPDATED = "text_session_updated"
    TEXT_SESSION_DELETED = "text_session_deleted"
    
    # 处理完成事件
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_STARTED = "processing_started"
    PROCESSING_FAILED = "processing_failed"
