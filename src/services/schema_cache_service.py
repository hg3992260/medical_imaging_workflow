import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.services.project_storage_service import ProjectStorageService


def _stable_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(obj)


def _schema_hash(headers: List[str], sample_rows: List[List[str]]) -> str:
    base = {"headers": headers or [], "sample_rows": sample_rows or []}
    return hashlib.sha256(_stable_json(base).encode("utf-8", errors="ignore")).hexdigest()


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return None
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("\x00", "").strip()


class SchemaCacheService:
    def __init__(self, db_service, storage_service: Optional[ProjectStorageService] = None):
        self.db = db_service
        self.storage = storage_service or ProjectStorageService(db_service)

    def ensure_schema_cache(
        self,
        project_id: str,
        llm_call: Callable[[str], str],
        max_new_branches: int = 8,
    ) -> Tuple[Dict[str, Any], str]:
        cache = self.storage.load_schema_cache(project_id) or {}
        if not isinstance(cache, dict):
            cache = {}

        project_keyword = self._get_project_name(project_id) or project_id
        now = datetime.utcnow().isoformat()
        cache.setdefault("project_id", project_id)
        cache.setdefault("project_keyword", project_keyword)
        cache.setdefault("canonical_mappings", [])
        cache.setdefault("branches", {})
        cache["updated_at"] = now

        snippets, branch_updates = self._collect_dirty_schema_snippets(project_id, cache)
        if not snippets or max_new_branches <= 0:
            preview = self._schema_cache_preview(cache, limit=24)
            return cache, preview

        snippets = snippets[: max(1, int(max_new_branches))]
        selected = set()
        for s in snippets:
            first = (s or "").splitlines()[0].strip()
            if first.startswith("[BRANCH]"):
                selected.add(first.replace("[BRANCH]", "", 1).strip())
        prompt = self._build_llm_prompt(project_keyword, "\n\n".join(snippets))
        raw = llm_call(prompt)
        obj = _try_parse_json(raw) or {}
        new_maps = obj.get("canonical_mappings")
        llm_ok = isinstance(new_maps, list)
        if llm_ok and new_maps:
            merged = self._merge_canonical_mappings(cache.get("canonical_mappings") or [], new_maps)
            cache["canonical_mappings"] = merged

        for bp, upd in branch_updates.items():
            if upd.get("dirty") and not llm_ok:
                continue
            if upd.get("dirty") and bp not in selected:
                continue
            upd = dict(upd)
            upd.pop("dirty", None)
            cache["branches"].setdefault(bp, {})
            cache["branches"][bp].update(upd)
            cache["branches"][bp]["last_seen_at"] = now

        self.storage.store_schema_cache(project_id, cache)
        preview = self._schema_cache_preview(cache, limit=24)
        return cache, preview

    def _get_project_name(self, project_id: str) -> str:
        try:
            rows = self.db.execute_query("SELECT name FROM projects WHERE project_id = ?", (project_id,))
            row = (rows or [{}])[0]
            return _safe_str(row.get("name")) if row else ""
        except Exception:
            return ""

    def _collect_dirty_schema_snippets(self, project_id: str, cache: Dict[str, Any]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        q = """
            SELECT sf.file_id, sf.file_path, sf.metadata, sf.created_at
            FROM storage_files sf
            JOIN project_storage ps ON ps.storage_id = sf.storage_id
            WHERE ps.project_id = ? AND sf.file_type = 'tabular'
            ORDER BY sf.created_at DESC
        """
        rows = self.db.execute_query(q, (project_id,)) or []
        cached_branches = cache.get("branches") or {}
        snippets = []
        updates: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            fp = _safe_str(r.get("file_path"))
            if not fp or not os.path.exists(fp):
                continue
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                continue
            meta = (obj or {}).get("metadata") or {}
            sheet = (obj or {}).get("sheet") or {}
            branch_path = _safe_str(meta.get("branch_path") or fp)
            headers = sheet.get("headers") or meta.get("headers") or []
            if not isinstance(headers, list):
                headers = []
            headers = [_safe_str(x) for x in headers if _safe_str(x)]

            sample_rows = []
            rows_grid = sheet.get("rows")
            if isinstance(rows_grid, list) and rows_grid:
                for rr in rows_grid[1:4]:
                    if isinstance(rr, list):
                        sample_rows.append([_safe_str(x)[:80] for x in rr[:12]])

            h = _schema_hash(headers[:120], sample_rows)
            prev = cached_branches.get(branch_path) or {}
            if _safe_str(prev.get("schema_hash")) == h:
                updates[branch_path] = {"schema_hash": h, "headers": headers[:120], "dirty": False}
                continue

            snippet = self._format_schema_snippet(branch_path, headers, sample_rows, sheet.get("kv") or {})
            snippets.append(snippet)
            updates[branch_path] = {"schema_hash": h, "headers": headers[:120], "dirty": True}
        return snippets, updates

    def _format_schema_snippet(self, branch_path: str, headers: List[str], sample_rows: List[List[str]], kv: Any) -> str:
        lines = []
        lines.append(f"[BRANCH] {branch_path}")
        if headers:
            lines.append("Columns: " + ", ".join([h[:60] for h in headers[:80]]))
        if sample_rows:
            lines.append("SampleRows:")
            for r in sample_rows[:3]:
                lines.append("- " + " | ".join([_safe_str(x)[:80] for x in r]))
        if isinstance(kv, dict) and kv:
            items = list(kv.items())[:10]
            lines.append("KeyValuePreview:")
            for k, v in items:
                lines.append(f"- {_safe_str(k)[:60]}: {_safe_str(v)[:80]}")
        return "\n".join(lines).strip()

    def _build_llm_prompt(self, project_keyword: str, schema_snippets: str) -> str:
        return (
            "### [SYSTEM ROLE]\n"
            "你是一个多模态科研数据专家。你的任务是审计不同来源的表格结构，并建立“语义等价类映射”。\n\n"
            "### [CONTEXT]\n"
            f"- 研究方向关键字：{project_keyword}\n"
            "- 核心任务：识别不同样本分支中，含义相同但名称不同的字段。\n\n"
            "### [INPUT DATA:待审计 Schema]\n"
            f"{schema_snippets}\n\n"
            "### [AUDIT RULES]\n"
            "1. 语义对齐：判断字段是否代表同一物理量/指标。\n"
            "2. 单位校验：识别量纲差异，并在映射中标注转换逻辑。\n"
            "3. 重要性判定：标注该列是否为核心科学指标（Core Metric）。\n\n"
            "### [OUTPUT FORMAT: JSON Schema_Cache]\n"
            "请严格返回 JSON：\n"
            "{\n"
            '  "canonical_mappings": [\n'
            "    {\n"
            '      "canonical_name": "标准化字段名",\n'
            '      "aliases": [\n'
            '        {"path": "分支A", "raw_name": "原名1"},\n'
            '        {"path": "分支B", "raw_name": "原名2"}\n'
            "      ],\n"
            '      "unit_logic": "转换公式或保持一致",\n'
            '      "scientific_role": "特征/标签/协变量"\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

    def _merge_canonical_mappings(self, base: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_name: Dict[str, Dict[str, Any]] = {}
        for m in base or []:
            cn = _safe_str(m.get("canonical_name"))
            if not cn:
                continue
            by_name[cn] = m

        for m in new or []:
            if not isinstance(m, dict):
                continue
            cn = _safe_str(m.get("canonical_name"))
            if not cn:
                continue
            cur = by_name.get(cn)
            if not cur:
                by_name[cn] = {"canonical_name": cn, "aliases": [], "unit_logic": "", "scientific_role": ""}
                cur = by_name[cn]

            if _safe_str(m.get("unit_logic")):
                cur["unit_logic"] = _safe_str(m.get("unit_logic"))
            if _safe_str(m.get("scientific_role")):
                cur["scientific_role"] = _safe_str(m.get("scientific_role"))

            aliases = cur.get("aliases") or []
            seen = set()
            for a in aliases:
                if not isinstance(a, dict):
                    continue
                seen.add((_safe_str(a.get("path")), _safe_str(a.get("raw_name"))))
            for a in m.get("aliases") or []:
                if not isinstance(a, dict):
                    continue
                key = (_safe_str(a.get("path")), _safe_str(a.get("raw_name")))
                if not key[0] or not key[1] or key in seen:
                    continue
                aliases.append({"path": key[0], "raw_name": key[1]})
                seen.add(key)
            cur["aliases"] = aliases

        merged = list(by_name.values())
        merged.sort(key=lambda x: _safe_str(x.get("canonical_name")))
        return merged

    def _schema_cache_preview(self, cache: Dict[str, Any], limit: int = 24) -> str:
        cms = cache.get("canonical_mappings") or []
        if not isinstance(cms, list) or not cms:
            return "[SCHEMA_CACHE]\ncanonical_mappings: 0"
        lines = ["[SCHEMA_CACHE]", f"canonical_mappings: {len(cms)}"]
        for m in cms[: max(0, int(limit))]:
            if not isinstance(m, dict):
                continue
            cn = _safe_str(m.get("canonical_name"))
            role = _safe_str(m.get("scientific_role"))
            unit = _safe_str(m.get("unit_logic"))
            aliases = m.get("aliases") or []
            lines.append(f"- {cn} | role={role or 'NA'} | unit={unit or 'NA'} | aliases={len(aliases) if isinstance(aliases, list) else 0}")
        return "\n".join(lines).strip()
