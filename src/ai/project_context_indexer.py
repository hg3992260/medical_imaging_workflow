import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from src.ai.coco_flows import LANCEDB_URI
from src.utils.model_config_loader import get_model_paths

def _get_lancedb():
    try:
        import lancedb as _lancedb

        return _lancedb
    except Exception as e:
        raise RuntimeError(f"lancedb/pyarrow 不可用: {e}")


def _escape_sql_value(value: str) -> str:
    return (value or "").replace("'", "''")


def _stable_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.replace("\x00", "").strip()


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    paragraphs = []
    for p in text.replace("\r\n", "\n").split("\n\n"):
        p = p.strip()
        if p:
            paragraphs.append(p)
    if not paragraphs:
        paragraphs = [text]

    out: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for p in paragraphs:
        add_len = len(p) + (2 if buf else 0)
        if buf and buf_len + add_len > chunk_size:
            out.append("\n\n".join(buf))
            if overlap > 0:
                overlap_chars = 0
                overlap_buf: List[str] = []
                for prev in reversed(buf):
                    overlap_buf.insert(0, prev)
                    overlap_chars += len(prev) + 2
                    if overlap_chars >= overlap:
                        break
                buf = overlap_buf
                buf_len = sum(len(x) for x in buf) + 2 * max(0, len(buf) - 1)
            else:
                buf = []
                buf_len = 0

        buf.append(p)
        buf_len += add_len

    if buf:
        out.append("\n\n".join(buf))
    return out


def _safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _flatten_json(obj, prefix: str = "", out: Optional[Dict[str, str]] = None, max_items: int = 200) -> Dict[str, str]:
    if out is None:
        out = {}
    if len(out) >= max_items:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if len(out) >= max_items:
                break
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_json(v, key, out, max_items=max_items)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:30]):
            if len(out) >= max_items:
                break
            key = f"{prefix}[{i}]"
            _flatten_json(v, key, out, max_items=max_items)
    else:
        try:
            if obj is None:
                s = ""
            elif isinstance(obj, (int, float, bool)):
                s = str(obj)
            else:
                s = _normalize_text(obj)
            if s:
                out[prefix] = s[:500]
        except Exception:
            pass
    return out


def _select_keys(flat: Dict[str, str], keywords: List[str]) -> Dict[str, str]:
    keys = []
    for k in flat.keys():
        lk = k.lower()
        if any(kw in lk for kw in keywords):
            keys.append(k)
    keys = sorted(set(keys))[:60]
    return {k: flat[k] for k in keys}


def _format_kv_block(title: str, kv: Dict[str, str]) -> str:
    if not kv:
        return ""
    lines = [title]
    for k, v in kv.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


import threading

_GLOBAL_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()

def _load_embedding_model():
    global _GLOBAL_EMBEDDER
    with _EMBEDDER_LOCK:
        if _GLOBAL_EMBEDDER is not None:
            return _GLOBAL_EMBEDDER

        os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
        os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
        try:
            from sentence_transformers import SentenceTransformer, models
            import torch

            try:
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        pass
                        # torch.cuda.set_per_process_memory_fraction(0.4, i)
            except Exception as e:
                pass

            device = "cpu"
            paths = get_model_paths()
            model_path = paths.get("embedding_model")
            
            model = None
            if model_path and os.path.exists(model_path):
                try:
                    model = SentenceTransformer(model_path, local_files_only=True, device=device)
                except Exception:
                    try:
                        tr = models.Transformer(model_path, local_files_only=True)
                        pl = models.Pooling(tr.get_word_embedding_dimension())
                        model = SentenceTransformer(modules=[tr, pl], device=device)
                    except Exception:
                        model = None

            if model is None:
                try:
                    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True, device=device)
                except Exception:
                    model = None
            
            _GLOBAL_EMBEDDER = model
            return _GLOBAL_EMBEDDER
        except Exception:
            return None


@dataclass
class ProjectContextSourceRow:
    project_id: str
    user_id: str
    source_type: str
    source_id: str
    updated_at: str
    text: str


class ProjectContextIndexer:
    def __init__(self, db_service, kb=None, lancedb_uri: Optional[str] = None):
        self.db_service = db_service
        self.kb = kb
        self.db_uri = lancedb_uri or getattr(kb, "db_uri", None) or LANCEDB_URI
        self._embedder = None
        self._lancedb = None

    def sync_project(self, project_id: str, force: bool = False, progress_cb=None) -> Dict[str, int]:
        def _progress(msg: str):
            if progress_cb:
                try:
                    progress_cb(msg)
                except Exception:
                    pass

        _progress("ProjectContext: 开始同步向量库索引")

        sources = list(self._load_project_sources(project_id))
        if not sources:
            _progress("ProjectContext: 未发现可索引的数据 sources=0 updated=0 chunks=0")
            return {"sources": 0, "chunks": 0, "updated_sources": 0}

        if not self._embedder:
            self._embedder = _load_embedding_model()
        has_vector = self._embedder is not None

        if self._lancedb is None:
            self._lancedb = _get_lancedb()
        db = self._lancedb.connect(self.db_uri)

        meta_tbl = self._ensure_table(
            db,
            "project_context_meta",
            initial_rows=[{"key": "init", "project_id": "", "source_type": "", "source_id": "", "content_hash": "", "updated_at": ""}],
            drop_init=True,
        )

        ctx_init = {"id": "init", "project_id": "", "user_id": "", "source_type": "", "source_id": "", "chunk_index": 0, "content_hash": "", "updated_at": "", "text": ""}
        if has_vector:
            try:
                v = self._embedder.encode(["init"], show_progress_bar=False)[0].tolist()
            except Exception:
                v = [0.0] * 384
            ctx_init["vector"] = v

        ctx_tbl = self._ensure_table(db, "project_context", initial_rows=[ctx_init], drop_init=True, require_vector=has_vector)

        meta = self._load_meta(meta_tbl, project_id)
        current_keys = set()
        for r in sources:
            current_keys.add(f"{r.project_id}:{r.source_type}:{r.source_id}")

        stale_keys = set(meta.keys()) - current_keys
        if stale_keys:
            for key in stale_keys:
                try:
                    _, source_type, source_id = key.split(":", 2)
                except Exception:
                    continue
                try:
                    ctx_tbl.delete(
                        f"project_id = '{_escape_sql_value(project_id)}' AND source_type = '{_escape_sql_value(source_type)}' AND source_id = '{_escape_sql_value(source_id)}'"
                    )
                except Exception:
                    pass
                try:
                    meta_tbl.delete(f"key = '{_escape_sql_value(key)}'")
                except Exception:
                    pass
                meta.pop(key, None)

        updated_sources = 0
        total_chunks = 0
        batch_records = []
        batch_size = 256

        for row in sources:
            key = f"{row.project_id}:{row.source_type}:{row.source_id}"
            content_hash = _stable_hash(row.text)
            prev_hash = meta.get(key)

            if (not force) and prev_hash == content_hash:
                continue

            chunks = _chunk_text(row.text, chunk_size=1200, overlap=200)
            if not chunks:
                meta[key] = content_hash
                continue

            vectors = None
            if self._embedder is not None:
                try:
                    vectors = self._embedder.encode(chunks, show_progress_bar=False)
                except Exception:
                    vectors = None

            for i, chunk in enumerate(chunks):
                rec = {
                    "id": f"{key}:{content_hash[:12]}:{i}",
                    "project_id": row.project_id,
                    "user_id": row.user_id,
                    "source_type": row.source_type,
                    "source_id": row.source_id,
                    "chunk_index": i,
                    "content_hash": content_hash,
                    "updated_at": row.updated_at,
                    "text": chunk,
                }
                if vectors is not None:
                    rec["vector"] = vectors[i].tolist()
                batch_records.append(rec)
                total_chunks += 1

                if len(batch_records) >= batch_size:
                    self._upsert_context(ctx_tbl, batch_records)
                    batch_records = []

            meta[key] = content_hash
            updated_sources += 1

            if prev_hash and prev_hash != content_hash:
                try:
                    ctx_tbl.delete(
                        f"project_id = '{_escape_sql_value(project_id)}' AND source_type = '{_escape_sql_value(row.source_type)}' AND source_id = '{_escape_sql_value(row.source_id)}' AND content_hash != '{_escape_sql_value(content_hash)}'"
                    )
                except Exception:
                    pass

        if batch_records:
            self._upsert_context(ctx_tbl, batch_records)

        meta_rows = []
        for key, h in meta.items():
            try:
                proj, source_type, source_id = key.split(":", 2)
            except Exception:
                continue
            meta_rows.append(
                {
                    "key": key,
                    "project_id": proj,
                    "source_type": source_type,
                    "source_id": source_id,
                    "content_hash": h,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            )

        if meta_rows:
            self._upsert_meta(meta_tbl, meta_rows)

        _progress(
            f"ProjectContext: 同步完成 sources={len(sources)} updated={updated_sources} chunks={total_chunks}"
        )
        return {"sources": len(sources), "chunks": total_chunks, "updated_sources": updated_sources}

    def _ensure_table(self, db, name: str, initial_rows: List[Dict], drop_init: bool, require_vector: bool = False) -> "lancedb.table.LanceTable":
        names = []
        try:
            names = db.table_names()
        except Exception:
            try:
                resp = db.list_tables()
                names = getattr(resp, "tables", []) or []
            except Exception:
                names = []

        tbl = None
        try:
            if name in names:
                tbl = db.open_table(name)
        except Exception:
            # 如果打开表失败（例如底层损坏导致 Option::unwrap on None），强行删除并重建
            try:
                db.drop_table(name)
            except Exception:
                pass
            tbl = None

        if tbl is None:
            try:
                tbl = db.create_table(name, data=initial_rows)
            except Exception:
                # 依然有可能并发抢占导致表存在，再打开一次
                try:
                    tbl = db.open_table(name)
                except Exception:
                    # 最后一道防线：重新连接或者直接覆盖
                    db.drop_table(name, ignore_missing=True)
                    tbl = db.create_table(name, data=initial_rows)

        if require_vector and tbl is not None:
            try:
                if "vector" not in tbl.schema.names:
                    try:
                        db.drop_table(name)
                    except Exception:
                        pass
                    tbl = db.create_table(name, data=initial_rows)
            except Exception:
                pass

        if drop_init:
            try:
                if name == "project_context_meta":
                    tbl.delete("key = 'init'")
                else:
                    tbl.delete("id = 'init'")
            except Exception:
                pass
        return tbl

    def _load_meta(self, meta_tbl, project_id: str) -> Dict[str, str]:
        out = {}
        try:
            rows = meta_tbl.to_arrow().to_pylist()
        except Exception:
            rows = []
        for r in rows:
            if r.get("project_id") != project_id:
                continue
            key = r.get("key")
            h = r.get("content_hash")
            if key and h:
                out[str(key)] = str(h)
        return out

    def _upsert_context(self, ctx_tbl, records: List[Dict]):
        if not records:
            return
        try:
            ctx_tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(records)
        except Exception:
            try:
                ctx_tbl.add(records)
            except Exception:
                pass

    def _upsert_meta(self, meta_tbl, records: List[Dict]):
        if not records:
            return
        try:
            meta_tbl.merge_insert("key").when_matched_update_all().when_not_matched_insert_all().execute(records)
        except Exception:
            try:
                meta_tbl.add(records)
            except Exception:
                pass

    def _load_project_sources(self, project_id: str) -> Iterable[ProjectContextSourceRow]:
        yield from self._load_project_stats(project_id)
        yield from self._load_sample_descriptions(project_id)
        yield from self._load_documents(project_id)
        yield from self._load_ocr_results(project_id)
        yield from self._load_tabular_artifacts(project_id)
        yield from self._load_schema_cache(project_id)
        yield from self._load_roi_text(project_id)
        yield from self._load_dicom_info(project_id)

    def _load_schema_cache(self, project_id: str) -> Iterable[ProjectContextSourceRow]:
        now = datetime.utcnow().isoformat()
        q = """
            SELECT sf.file_id, sf.file_path, sf.metadata, sf.created_at
            FROM storage_files sf
            JOIN project_storage ps ON ps.storage_id = sf.storage_id
            WHERE ps.project_id = ? AND sf.file_type = 'tabular'
            ORDER BY sf.created_at DESC
        """
        rows = self.db_service.execute_query(q, (project_id,))
        for r in rows or []:
            file_id = _normalize_text(r.get("file_id") or "")
            file_path = _normalize_text(r.get("file_path") or "")
            if not file_id or not file_path or not os.path.exists(file_path):
                continue
            if not file_path.lower().endswith("schema_cache.json"):
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            cms = obj.get("canonical_mappings") or []
            if not isinstance(cms, list) or not cms:
                continue
            lines = []
            lines.append("[SCHEMA_CACHE]")
            lines.append(f"project_id: {project_id}")
            lines.append(f"canonical_mappings: {len(cms)}")
            for m in cms[:60]:
                if not isinstance(m, dict):
                    continue
                cn = _normalize_text(m.get("canonical_name") or "")
                role = _normalize_text(m.get("scientific_role") or "")
                unit = _normalize_text(m.get("unit_logic") or "")
                aliases = m.get("aliases") or []
                lines.append(f"- {cn} | role={role or 'NA'} | unit={unit or 'NA'} | aliases={len(aliases) if isinstance(aliases, list) else 0}")
            text = "\n".join([x for x in lines if x]).strip()
            if text:
                yield ProjectContextSourceRow(
                    project_id=project_id,
                    user_id="",
                    source_type="schema_cache",
                    source_id=file_id,
                    updated_at=now,
                    text=text,
                )

    def _load_tabular_artifacts(self, project_id: str) -> Iterable[ProjectContextSourceRow]:
        now = datetime.utcnow().isoformat()
        q = """
            SELECT sf.file_id, sf.file_path, sf.metadata, sf.created_at
            FROM storage_files sf
            JOIN project_storage ps ON ps.storage_id = sf.storage_id
            WHERE ps.project_id = ? AND sf.file_type = 'tabular'
            ORDER BY sf.created_at DESC
        """
        rows = self.db_service.execute_query(q, (project_id,))
        for r in rows or []:
            file_id = _normalize_text(r.get("file_id") or "")
            file_path = _normalize_text(r.get("file_path") or "")
            if not file_id or not file_path or not os.path.exists(file_path):
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("canonical_mappings") is not None:
                continue

            meta = (obj or {}).get("metadata") or {}
            sheet = (obj or {}).get("sheet") or {}
            user_id = _normalize_text(meta.get("user_id") or "")
            branch_path = _normalize_text(meta.get("branch_path") or "")
            domain_tag = _normalize_text(meta.get("domain_tag") or "Tabular")
            project_name = _normalize_text(meta.get("project_name") or "")
            sheet_name = _normalize_text(meta.get("sheet_name") or sheet.get("sheet_name") or "")
            headers = sheet.get("headers") or meta.get("headers") or []
            kv = sheet.get("kv") or {}

            lines = []
            lines.append("[TABULAR_BRANCH]")
            lines.append(f"project: {project_name or project_id}")
            lines.append(f"user_id: {user_id}")
            lines.append(f"domain_tag: {domain_tag}")
            if branch_path:
                lines.append(f"branch_path: {branch_path}")
            if sheet_name:
                lines.append(f"sheet_name: {sheet_name}")
            if isinstance(headers, list) and headers:
                lines.append("headers: " + ", ".join([_normalize_text(h) for h in headers[:100] if _normalize_text(h)]))
            if isinstance(kv, dict) and kv:
                lines.append("key_value_preview:")
                for k, v in list(kv.items())[:60]:
                    lines.append(f"- {_normalize_text(k)[:80]}: {_normalize_text(v)[:160]}")

            text = "\n".join(lines).strip()
            if text:
                yield ProjectContextSourceRow(
                    project_id=project_id,
                    user_id=user_id,
                    source_type="tabular",
                    source_id=file_id,
                    updated_at=now,
                    text=text,
                )

    def _load_project_stats(self, project_id: str) -> Iterable[ProjectContextSourceRow]:
        now = datetime.utcnow().isoformat()
        try:
            counts = {}
            rows = self.db_service.execute_query("SELECT COUNT(*) AS n FROM user_ids WHERE project_id = ? AND is_active = 1", (project_id,))
            counts["active_users"] = str((rows or [{}])[0].get("n", 0))

            rows = self.db_service.execute_query("SELECT COUNT(*) AS n FROM documents d JOIN text_sessions ts ON d.session_id=ts.session_id WHERE ts.project_id = ?", (project_id,))
            counts["documents"] = str((rows or [{}])[0].get("n", 0))

            rows = self.db_service.execute_query("SELECT COUNT(*) AS n FROM ocr_results r JOIN ocr_sessions s ON r.session_id=s.session_id WHERE s.project_id = ?", (project_id,))
            counts["ocr_results"] = str((rows or [{}])[0].get("n", 0))

            rows = self.db_service.execute_query("SELECT COUNT(*) AS n FROM roi_data rd JOIN dicom_sessions ds ON rd.session_id=ds.session_id WHERE ds.project_id = ?", (project_id,))
            counts["rois"] = str((rows or [{}])[0].get("n", 0))

            roi_dist = {}
            rows = self.db_service.execute_query(
                "SELECT rd.roi_type AS roi_type, COUNT(*) AS n FROM roi_data rd JOIN dicom_sessions ds ON rd.session_id=ds.session_id WHERE ds.project_id = ? GROUP BY rd.roi_type ORDER BY n DESC LIMIT 20",
                (project_id,),
            )
            for r in rows or []:
                k = _normalize_text(r.get("roi_type") or "unknown")
                roi_dist[k] = str(r.get("n", 0))

            ocr_conf = {}
            rows = self.db_service.execute_query(
                "SELECT AVG(COALESCE(r.confidence,0)) AS avg_conf, MIN(COALESCE(r.confidence,0)) AS min_conf, MAX(COALESCE(r.confidence,0)) AS max_conf FROM ocr_results r JOIN ocr_sessions s ON r.session_id=s.session_id WHERE s.project_id = ?",
                (project_id,),
            )
            if rows:
                ocr_conf = {k: str(rows[0].get(k, "")) for k in ["avg_conf", "min_conf", "max_conf"] if rows[0].get(k) is not None}

            text = "\n".join(
                [
                    "[PROJECT_STATS]",
                    f"project_id: {project_id}",
                    "",
                    _format_kv_block("Counts:", counts),
                    "",
                    _format_kv_block("ROI Type Distribution (top20):", roi_dist),
                    "",
                    _format_kv_block("OCR Confidence Summary:", ocr_conf),
                ]
            ).strip()
            if text:
                yield ProjectContextSourceRow(project_id=project_id, user_id="", source_type="stats", source_id="project", updated_at=now, text=text)
        except Exception:
            return

    def _load_sample_descriptions(self, project_id: str) -> Iterable[ProjectContextSourceRow]:
        q = """
            SELECT user_id, display_name, description, created_at
            FROM user_ids
            WHERE project_id = ? AND is_active = 1
            ORDER BY created_at DESC
        """
        rows = self.db_service.execute_query(q, (project_id,))
        for r in rows:
            user_id = r.get("user_id") or ""
            text = f"[SAMPLE]\nuser_id: {user_id}\ndisplay_name: {r.get('display_name') or ''}\ndescription: {r.get('description') or ''}\n"
            yield ProjectContextSourceRow(
                project_id=project_id,
                user_id=user_id,
                source_type="sample",
                source_id=user_id,
                updated_at=str(r.get("created_at") or ""),
                text=text,
            )

    def _load_documents(self, project_id: str) -> Iterable[ProjectContextSourceRow]:
        q = """
            SELECT d.doc_id, ts.user_id, d.file_name, d.format, d.content, d.analysis_result, d.keywords, d.created_at, d.modified_at
            FROM documents d
            JOIN text_sessions ts ON d.session_id = ts.session_id
            WHERE ts.project_id = ?
            ORDER BY d.created_at DESC
        """
        rows = self.db_service.execute_query(q, (project_id,))
        for r in rows:
            doc_id = r.get("doc_id") or ""
            user_id = r.get("user_id") or ""
            content = _normalize_text(r.get("content") or "")
            if not content:
                continue
            analysis_result = r.get("analysis_result")
            keywords = r.get("keywords")
            scope = "results"
            fmt = (r.get("format") or "").lower()
            if fmt in [".md", ".docx", ".doc", ".pdf"]:
                scope = "background"
            text = "\n".join(
                [
                    "[DOCUMENT]",
                    f"doc_id: {doc_id}",
                    f"user_id: {user_id}",
                    f"file_name: {r.get('file_name') or ''}",
                    f"format: {r.get('format') or ''}",
                    f"scope: {scope}",
                    f"keywords: {keywords or ''}",
                    f"analysis_result: {analysis_result or ''}",
                    "",
                    content,
                ]
            )
            yield ProjectContextSourceRow(
                project_id=project_id,
                user_id=user_id,
                source_type="document",
                source_id=doc_id,
                updated_at=str(r.get("modified_at") or r.get("created_at") or ""),
                text=text,
            )

    def _load_ocr_results(self, project_id: str) -> Iterable[ProjectContextSourceRow]:
        q = """
            SELECT r.result_id, s.user_id, r.recognized_text, r.confidence, r.created_at
            FROM ocr_results r
            JOIN ocr_sessions s ON r.session_id = s.session_id
            WHERE s.project_id = ?
            ORDER BY r.created_at DESC
        """
        rows = self.db_service.execute_query(q, (project_id,))
        for r in rows:
            result_id = r.get("result_id") or ""
            user_id = r.get("user_id") or ""
            text0 = _normalize_text(r.get("recognized_text") or "")
            if not text0:
                continue
            conf = r.get("confidence")
            text = "\n".join(
                [
                    "[OCR_RESULT]",
                    f"result_id: {result_id}",
                    f"user_id: {user_id}",
                    "scope: results",
                    f"confidence: {conf}",
                    "",
                    text0,
                ]
            )
            yield ProjectContextSourceRow(
                project_id=project_id,
                user_id=user_id,
                source_type="ocr",
                source_id=result_id,
                updated_at=str(r.get("created_at") or ""),
                text=text,
            )

    def _load_roi_text(self, project_id: str) -> Iterable[ProjectContextSourceRow]:
        q = """
            SELECT rd.roi_id, ds.user_id, rd.roi_type, rd.source, rd.area, rd.perimeter, rd.ocr_text, rd.ocr_confidence, rd.properties, rd.created_at
            FROM roi_data rd
            JOIN dicom_sessions ds ON rd.session_id = ds.session_id
            WHERE ds.project_id = ?
            ORDER BY rd.created_at DESC
        """
        rows = self.db_service.execute_query(q, (project_id,))
        for r in rows:
            roi_id = r.get("roi_id") or ""
            user_id = r.get("user_id") or ""
            ocr_text = _normalize_text(r.get("ocr_text") or "")
            props = r.get("properties") or ""
            props_text = _normalize_text(props)
            props_obj = _safe_json_loads(props_text) if props_text else None
            props_keys = {}
            if props_obj is not None:
                flat = _flatten_json(props_obj, max_items=120)
                props_keys = _select_keys(
                    flat,
                    keywords=[
                        "label",
                        "class",
                        "category",
                        "score",
                        "prob",
                        "name",
                        "diam",
                        "area",
                        "volume",
                        "long",
                        "short",
                        "length",
                        "width",
                        "height",
                        "mm",
                        "cm",
                    ],
                )
            text = "\n".join(
                [
                    "[ROI]",
                    f"roi_id: {roi_id}",
                    f"user_id: {user_id}",
                    f"roi_type: {r.get('roi_type') or ''}",
                    f"source: {r.get('source') or ''}",
                    "scope: results",
                    f"area: {r.get('area')}",
                    f"perimeter: {r.get('perimeter')}",
                    f"ocr_confidence: {r.get('ocr_confidence')}",
                    f"properties: {props_text[:2000]}",
                    _format_kv_block("properties_key_fields:", props_keys),
                    "",
                    ocr_text,
                ]
            )
            if not ocr_text and not props_text:
                continue
            yield ProjectContextSourceRow(
                project_id=project_id,
                user_id=user_id,
                source_type="roi",
                source_id=roi_id,
                updated_at=str(r.get("created_at") or ""),
                text=text,
            )

    def _load_dicom_info(self, project_id: str) -> Iterable[ProjectContextSourceRow]:
        q = """
            SELECT session_id, user_id, dicom_info, file_path, created_at
            FROM dicom_sessions
            WHERE project_id = ?
            ORDER BY created_at DESC
        """
        rows = self.db_service.execute_query(q, (project_id,))
        for r in rows:
            session_id = r.get("session_id") or ""
            user_id = r.get("user_id") or ""
            dicom_info = r.get("dicom_info") or ""
            dicom_info_text = _normalize_text(dicom_info)
            if not dicom_info_text:
                continue
            key_fields = {}
            try:
                obj = json.loads(dicom_info_text)
                flat = _flatten_json(obj, max_items=200)
                key_fields = _select_keys(
                    flat,
                    keywords=[
                        "modality",
                        "study",
                        "series",
                        "manufacturer",
                        "model",
                        "magneticfieldstrength",
                        "fieldstrength",
                        "slice",
                        "thickness",
                        "spacing",
                        "pixelspacing",
                        "rows",
                        "columns",
                        "kvp",
                        "ma",
                        "mas",
                        "exposure",
                        "tr",
                        "te",
                        "flip",
                        "kernel",
                        "convolution",
                        "window",
                        "repetition",
                        "echo",
                        "contrast",
                        "dose",
                    ],
                )
                dicom_info_text = json.dumps(obj, ensure_ascii=False)
            except Exception:
                pass
            text = "\n".join(
                [
                    "[DICOM_SESSION]",
                    f"session_id: {session_id}",
                    f"user_id: {user_id}",
                    f"file_path: {r.get('file_path') or ''}",
                    "scope: method",
                    _format_kv_block("key_dicom_fields:", key_fields),
                    "",
                    dicom_info_text,
                ]
            )
            yield ProjectContextSourceRow(
                project_id=project_id,
                user_id=user_id,
                source_type="dicom",
                source_id=session_id,
                updated_at=str(r.get("created_at") or ""),
                text=text,
            )
