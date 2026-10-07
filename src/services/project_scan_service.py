import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.services.project_storage_service import ProjectStorageService


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("\x00", "").strip()


class ProjectScanService:
    def __init__(self, db_service):
        self.db = db_service
        self.storage = ProjectStorageService(db_service)

    def scan_project(self, project_id: str, schema_cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        now = datetime.utcnow().isoformat()
        storage_dir = self.storage.get_project_storage_path(project_id, "")
        indexed = self._load_indexed_storage_files(project_id)
        fs_scan = self._scan_storage_dir(storage_dir)

        tabular = self._scan_tabular_leaves(project_id, indexed, schema_cache)
        docs = self._scan_documents(project_id)
        ocr = self._scan_ocr(project_id)
        dicom = self._scan_dicom(project_id)
        roi = self._scan_roi(project_id)

        by_type = defaultdict(lambda: {"count": 0, "missing_file": 0})
        for r in indexed:
            ft = _safe_str(r.get("file_type"))
            fp = _safe_str(r.get("file_path"))
            by_type[ft]["count"] += 1
            if fp and not os.path.exists(fp):
                by_type[ft]["missing_file"] += 1

        indexed_paths = set([_safe_str(r.get("file_path")) for r in indexed if _safe_str(r.get("file_path"))])
        unindexed = [p for p in fs_scan if p not in indexed_paths]

        report = {
            "project_id": project_id,
            "generated_at": now,
            "storage_dir": str(storage_dir),
            "indexed_storage_files_total": len(indexed),
            "indexed_by_file_type": dict(by_type),
            "filesystem_files_total": len(fs_scan),
            "filesystem_unindexed_total": len(unindexed),
            "filesystem_unindexed_preview": unindexed[:200],
            "tabular": tabular,
            "documents": docs,
            "ocr": ocr,
            "dicom": dicom,
            "roi": roi,
        }
        return report

    def render_console_block(self, report: Dict[str, Any], max_lines: int = 60) -> str:
        if not isinstance(report, dict):
            return ""
        lines = []
        lines.append(f"[PROJECT_SCAN] project_id={report.get('project_id')}  indexed={report.get('indexed_storage_files_total')}  fs={report.get('filesystem_files_total')}")
        lines.append(f"- unindexed_files={report.get('filesystem_unindexed_total')}")
        by_type = report.get("indexed_by_file_type") or {}
        if isinstance(by_type, dict) and by_type:
            keys = sorted(by_type.keys())
            parts = []
            for k in keys:
                v = by_type.get(k) or {}
                parts.append(f"{k}:{v.get('count',0)}(missing={v.get('missing_file',0)})")
            lines.append("- storage_files_by_type=" + ", ".join(parts))

        tab = report.get("tabular") or {}
        if isinstance(tab, dict):
            lines.append(f"- tabular_leaves={tab.get('leaf_count',0)}  missing={tab.get('missing_file',0)}  meta_incomplete={tab.get('metadata_incomplete',0)}")
            ch = tab.get("canonical_hit_rate") or {}
            if isinstance(ch, dict) and ch:
                lines.append(f"- canonical_hit_rate_avg={ch.get('avg',0):.3f}  mapped={ch.get('mapped',0)}  headers={ch.get('headers',0)}")

        docs = report.get("documents") or {}
        if isinstance(docs, dict):
            lines.append(f"- documents={docs.get('count',0)}  missing_path={docs.get('missing_file',0)}")
        ocr = report.get("ocr") or {}
        if isinstance(ocr, dict):
            lines.append(f"- ocr_sessions={ocr.get('sessions',0)}  ocr_results={ocr.get('results',0)}")
        dicom = report.get("dicom") or {}
        if isinstance(dicom, dict):
            lines.append(f"- dicom_sessions={dicom.get('sessions',0)}  with_info={dicom.get('with_info',0)}  missing_path={dicom.get('missing_file',0)}")
        roi = report.get("roi") or {}
        if isinstance(roi, dict):
            lines.append(f"- roi_records={roi.get('count',0)}  missing_session={roi.get('missing_session',0)}")

        out = "\n".join(lines[: max(1, int(max_lines))]).strip()
        return out

    def export_report(self, project_id: str, report: Dict[str, Any], file_name: str = "project_scan_report.json") -> Optional[str]:
        return self.storage.store_export_json(project_id, file_name, report if isinstance(report, dict) else {})

    def _load_indexed_storage_files(self, project_id: str) -> List[Dict[str, Any]]:
        q = """
            SELECT sf.file_id, sf.file_path, sf.file_type, sf.metadata, sf.created_at
            FROM storage_files sf
            JOIN project_storage ps ON ps.storage_id = sf.storage_id
            WHERE ps.project_id = ?
            ORDER BY sf.created_at DESC
        """
        try:
            return self.db.execute_query(q, (project_id,)) or []
        except Exception:
            return []

    def _scan_storage_dir(self, storage_dir: Path) -> List[str]:
        out = []
        try:
            root = Path(storage_dir)
            if not root.exists():
                return []
            for p in root.rglob("*"):
                if p.is_file():
                    out.append(str(p))
        except Exception:
            return []
        out.sort()
        return out

    def _scan_tabular_leaves(self, project_id: str, indexed: List[Dict[str, Any]], schema_cache: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        leaves = [r for r in indexed if _safe_str(r.get("file_type")) == "tabular" and _safe_str(r.get("file_path")).lower().endswith(".json")]
        branch_paths = []
        missing_file = 0
        meta_incomplete = 0
        for r in leaves:
            fp = _safe_str(r.get("file_path"))
            if not fp or not os.path.exists(fp):
                missing_file += 1
                continue
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                meta_incomplete += 1
                continue
            meta = (obj or {}).get("metadata") or {}
            bp = _safe_str(meta.get("branch_path"))
            uid = _safe_str(meta.get("user_id"))
            if not bp or not uid:
                meta_incomplete += 1
            else:
                branch_paths.append(bp)

        branch_paths = list(dict.fromkeys(branch_paths))
        branch_paths.sort()

        hit = self._canonical_hit_rate(project_id, leaves, schema_cache)
        return {
            "leaf_count": len(leaves),
            "missing_file": missing_file,
            "metadata_incomplete": meta_incomplete,
            "branch_paths_total": len(branch_paths),
            "branch_paths_preview": branch_paths[:200],
            "canonical_hit_rate": hit,
        }

    def _canonical_hit_rate(self, project_id: str, leaves: List[Dict[str, Any]], schema_cache: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cache = schema_cache if isinstance(schema_cache, dict) else {}
        cms = cache.get("canonical_mappings") if isinstance(cache, dict) else None
        if not isinstance(cms, list) or not cms:
            return {}
        by_branch = defaultdict(set)
        for m in cms:
            if not isinstance(m, dict):
                continue
            canonical = _safe_str(m.get("canonical_name"))
            aliases = m.get("aliases") or []
            if not canonical or not isinstance(aliases, list):
                continue
            for a in aliases:
                if not isinstance(a, dict):
                    continue
                path = _safe_str(a.get("path"))
                raw = _safe_str(a.get("raw_name"))
                if path and raw:
                    by_branch[path].add(raw)

        total_headers = 0
        total_mapped = 0
        per_branch = []
        for r in leaves:
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
            bp = _safe_str(meta.get("branch_path"))
            headers = sheet.get("headers") or meta.get("headers") or []
            if not bp or not isinstance(headers, list) or not headers:
                continue
            headers = [_safe_str(h) for h in headers if _safe_str(h) and not _safe_str(h).startswith("__")]
            if not headers:
                continue
            mapped = 0
            known = by_branch.get(bp) or set()
            for h in headers:
                if h in known:
                    mapped += 1
            total_headers += len(headers)
            total_mapped += mapped
            per_branch.append({"branch_path": bp, "mapped": mapped, "headers": len(headers), "hit_rate": mapped / max(1, len(headers))})

        avg = float(total_mapped) / float(max(1, total_headers))
        per_branch.sort(key=lambda x: x.get("hit_rate", 0.0))
        return {"avg": avg, "mapped": total_mapped, "headers": total_headers, "worst_branches_preview": per_branch[:30]}

    def _scan_documents(self, project_id: str) -> Dict[str, Any]:
        q = """
            SELECT d.doc_id, d.file_path, d.created_at
            FROM documents d
            JOIN text_sessions ts ON ts.session_id = d.session_id
            WHERE ts.project_id = ?
            ORDER BY d.created_at DESC
        """
        try:
            rows = self.db.execute_query(q, (project_id,)) or []
        except Exception:
            rows = []
        missing = 0
        for r in rows:
            fp = _safe_str(r.get("file_path"))
            if fp and not os.path.exists(fp):
                missing += 1
        return {"count": len(rows), "missing_file": missing}

    def _scan_ocr(self, project_id: str) -> Dict[str, Any]:
        try:
            rows = self.db.execute_query("SELECT COUNT(*) AS n FROM ocr_sessions WHERE project_id = ?", (project_id,)) or []
            sessions = int((rows or [{}])[0].get("n", 0))
        except Exception:
            sessions = 0
        try:
            rows = self.db.execute_query(
                "SELECT COUNT(*) AS n FROM ocr_results r JOIN ocr_sessions s ON r.session_id=s.session_id WHERE s.project_id = ?",
                (project_id,),
            ) or []
            results = int((rows or [{}])[0].get("n", 0))
        except Exception:
            results = 0
        return {"sessions": sessions, "results": results}

    def _scan_dicom(self, project_id: str) -> Dict[str, Any]:
        q = "SELECT session_id, file_path, dicom_info FROM dicom_sessions WHERE project_id = ? ORDER BY created_at DESC"
        try:
            rows = self.db.execute_query(q, (project_id,)) or []
        except Exception:
            rows = []
        missing_file = 0
        with_info = 0
        for r in rows:
            fp = _safe_str(r.get("file_path"))
            if fp and not os.path.exists(fp):
                missing_file += 1
            if _safe_str(r.get("dicom_info")):
                with_info += 1
        return {"sessions": len(rows), "with_info": with_info, "missing_file": missing_file}

    def _scan_roi(self, project_id: str) -> Dict[str, Any]:
        q = """
            SELECT rd.roi_id, rd.session_id, rd.created_at
            FROM roi_data rd
            JOIN dicom_sessions ds ON rd.session_id = ds.session_id
            WHERE ds.project_id = ?
            ORDER BY rd.created_at DESC
        """
        try:
            rows = self.db.execute_query(q, (project_id,)) or []
        except Exception:
            rows = []
        missing_session = 0
        for r in rows:
            if not _safe_str(r.get("session_id")):
                missing_session += 1
        return {"count": len(rows), "missing_session": missing_session}

