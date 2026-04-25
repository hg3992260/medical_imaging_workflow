import logging
import threading
import time

logger = logging.getLogger(__name__)

class GPUResourceManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GPUResourceManager, cls).__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self.current_owner = None
        self.callbacks = {} # { "module_name": release_callback_function }
        self.mutex = threading.Lock()

    def register(self, module_name, release_callback):
        """
        Register a module that uses GPU.
        release_callback: function that frees GPU memory (e.g. move model to CPU)
        """
        with self.mutex:
            self.callbacks[module_name] = release_callback
            logger.info(f"GPU Manager: Registered module '{module_name}'")

    def request_gpu(self, requester_name):
        """
        Request exclusive access to GPU.
        If another module owns it, trigger its release callback.
        """
        with self.mutex:
            if self.current_owner == requester_name:
                return True # Already owns it

            if self.current_owner:
                logger.info(f"GPU Manager: '{requester_name}' requesting GPU. Revoking from '{self.current_owner}'...")
                try:
                    # Call the release function of the current owner
                    callback = self.callbacks.get(self.current_owner)
                    if callback:
                        callback()
                    logger.info(f"GPU Manager: '{self.current_owner}' released GPU.")
                except Exception as e:
                    logger.error(f"GPU Manager: Error releasing GPU from '{self.current_owner}': {e}")
            
            self.current_owner = requester_name
            logger.info(f"GPU Manager: GPU granted to '{requester_name}'")
            return True

    def release(self, owner_name):
        """Optional: proactively release GPU"""
        with self.mutex:
            if self.current_owner == owner_name:
                self.current_owner = None
                logger.info(f"GPU Manager: '{owner_name}' released GPU voluntarily.")

# Global instance
gpu_manager = GPUResourceManager()