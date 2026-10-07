"""
Prompt Registry – centralized prompt template store with hot-reload and restore.

Usage:
    from src.ai.prompt_registry import PromptRegistry
    reg = PromptRegistry.instance()
    reg.get("outline_generation")
    reg.update("outline_generation", custom_text)
    reg.restore("outline_generation")
    reg.restore_all()
"""

import copy
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("PromptRegistry")

_DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "prompt_defaults.json")
_USER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "prompt_user_overrides.json")
_BACKUP_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "prompt_backup.json")


class PromptRegistry:
    _instance: Optional["PromptRegistry"] = None

    @classmethod
    def instance(cls) -> "PromptRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._defaults: Dict[str, str] = {}
        self._overrides: Dict[str, str] = {}
        self._labels: Dict[str, str] = {}
        self._stage_order: list = []
        self._load_defaults()
        self._load_overrides()
        self._ensure_backup()
        self._build_stage_order()

    def _load_defaults(self) -> None:
        try:
            if os.path.isfile(_DEFAULTS_PATH):
                with open(_DEFAULTS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                prompts = data.get("prompts", {})
                for stage_id, entry in prompts.items():
                    if isinstance(entry, dict):
                        self._defaults[stage_id] = entry.get("template", "")
                        self._labels[stage_id] = entry.get("label", stage_id)
                    else:
                        self._defaults[stage_id] = str(entry)
                        self._labels[stage_id] = stage_id
                logger.info(f"PromptRegistry: loaded %d default prompts", len(self._defaults))
            else:
                logger.warning("PromptRegistry: defaults file not found at %s", _DEFAULTS_PATH)
        except Exception as e:
            logger.warning("PromptRegistry: failed to load defaults: %s", e)

    def _load_overrides(self) -> None:
        try:
            if os.path.isfile(_USER_PATH):
                with open(_USER_PATH, "r", encoding="utf-8") as f:
                    self._overrides = json.load(f)
                logger.info("PromptRegistry: loaded %d user overrides", len(self._overrides))
        except Exception:
            self._overrides = {}

    def _save_overrides(self) -> None:
        try:
            if self._overrides:
                os.makedirs(os.path.dirname(_USER_PATH), exist_ok=True)
                with open(_USER_PATH, "w", encoding="utf-8") as f:
                    json.dump(self._overrides, f, ensure_ascii=False, indent=2)
            else:
                if os.path.isfile(_USER_PATH):
                    os.remove(_USER_PATH)
        except Exception as e:
            logger.warning("PromptRegistry: failed to save overrides: %s", e)

    def _build_stage_order(self) -> None:
        try:
            if os.path.isfile(_DEFAULTS_PATH):
                with open(_DEFAULTS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._stage_order = data.get("stage_order", [])
            else:
                self._stage_order = sorted(self._defaults.keys())
        except Exception:
            self._stage_order = sorted(self._defaults.keys())

    def get(self, stage_id: str) -> str:
        if stage_id in self._overrides:
            return self._overrides[stage_id]
        return self._defaults.get(stage_id, "")

    def get_default(self, stage_id: str) -> str:
        return self._defaults.get(stage_id, "")

    def get_label(self, stage_id: str) -> str:
        return self._labels.get(stage_id, stage_id)

    def is_overridden(self, stage_id: str) -> bool:
        return stage_id in self._overrides

    def update(self, stage_id: str, text: str) -> None:
        default = self._defaults.get(stage_id, "")
        if text.strip() == default.strip():
            self._overrides.pop(stage_id, None)
        else:
            self._overrides[stage_id] = text
        self._save_overrides()

    def restore(self, stage_id: str) -> None:
        self._overrides.pop(stage_id, None)
        self._save_overrides()

    def _ensure_backup(self) -> None:
        """On first load, snapshot current defaults as the restore point."""
        if not os.path.isfile(_BACKUP_PATH):
            self._write_backup()

    def _write_backup(self) -> None:
        try:
            data = {"prompts": {}}
            for stage_id, template in self._defaults.items():
                data["prompts"][stage_id] = {"label": self._labels.get(stage_id, stage_id), "template": template}
            data["stage_order"] = self._stage_order
            os.makedirs(os.path.dirname(_BACKUP_PATH), exist_ok=True)
            with open(_BACKUP_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("PromptRegistry: backup snapshot created")
        except Exception as e:
            logger.warning("PromptRegistry: failed to write backup: %s", e)

    def snapshot_now(self) -> None:
        """Force-refresh backup with current defaults (call after code updates)."""
        self._write_backup()

    def restore_all(self) -> None:
        """Restore all prompts from the backup snapshot."""
        self._overrides.clear()
        self._save_overrides()
        if os.path.isfile(_BACKUP_PATH):
            try:
                with open(_BACKUP_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                prompts = data.get("prompts", {})
                for stage_id, entry in prompts.items():
                    if isinstance(entry, dict):
                        self._defaults[stage_id] = entry.get("template", self._defaults.get(stage_id, ""))
                logger.info("PromptRegistry: restored %d prompts from backup snapshot", len(prompts))
            except Exception as e:
                logger.warning("PromptRegistry: backup restore failed: %s", e)

    def factory_reset(self) -> None:
        """Hard reset: restore from shipping defaults, ignoring backup."""
        self._overrides.clear()
        self._save_overrides()
        if os.path.isfile(_DEFAULTS_PATH):
            try:
                with open(_DEFAULTS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                prompts = data.get("prompts", {})
                self._defaults.clear()
                for stage_id, entry in prompts.items():
                    if isinstance(entry, dict):
                        self._defaults[stage_id] = entry.get("template", "")
                        self._labels[stage_id] = entry.get("label", stage_id)
                    else:
                        self._defaults[stage_id] = str(entry)
                        self._labels[stage_id] = stage_id
                logger.info("PromptRegistry: factory reset complete")
            except Exception as e:
                logger.warning("PromptRegistry: factory reset failed: %s", e)

    @property
    def stage_order(self) -> list:
        return list(self._stage_order)

    @property
    def all_stages(self) -> Dict[str, str]:
        return copy.deepcopy(self._defaults)
