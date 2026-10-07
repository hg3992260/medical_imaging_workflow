import os
import shutil
import json
import logging
import re
import math
import hashlib
import datetime
from typing import Dict, List, Tuple, Any
from pypdf import PdfReader
from .coco_flows import REF_SOURCE_DIR, PROJ_SOURCE_DIR, LANCEDB_URI, text_to_embedding
from src.utils.model_config_loader import get_model_paths
from src.core.path_config import get_coco_data_dir

logger = logging.getLogger(__name__)

def _get_lancedb():
    try:
        import lancedb as _lancedb

        return _lancedb
    except Exception as e:
        raise RuntimeError(f"lancedb/pyarrow 不可用: {e}")

class KnowledgeBase:
    def __init__(self, db_uri: str = "", init_reference: bool = True, project_id: str = None):
        self.ref_dir = REF_SOURCE_DIR
        self.proj_dir = PROJ_SOURCE_DIR
        
        # 物理隔离: 动态路径索引
        if project_id:
            # Create a project-specific database directory
            self.db_uri = os.path.join(LANCEDB_URI, project_id)
            os.makedirs(self.db_uri, exist_ok=True)
            logger.info(f"KnowledgeBase configured for project-specific isolation: {self.db_uri}")
        else:
            self.db_uri = db_uri or LANCEDB_URI
            
        self.project_id = project_id
        self._reference_ready = False
        self._reference_init_attempted = False
        
        # Default PDF path relative to coco_data or use a placeholder
        # Ideally, the user should provide this or it should be in the ref_dir
        self.pdf_source = os.path.join(str(get_coco_data_dir()), "sources", "references", "writing_guide.pdf")
        
        # If the hardcoded path exists, use it as fallback (for dev environment)
        # dev_pdf_path = r"F:\RSNA\Science Research Writing_ For Native And Non-native Speakers Of English, Second Edition - PDF Room.pdf"
        # if not os.path.exists(self.pdf_source) and os.path.exists(dev_pdf_path):
        #      self.pdf_source = dev_pdf_path
        
        if init_reference:
            try:
                self._init_reference()
            finally:
                self._reference_ready = self._reference_table_exists()
                self._reference_init_attempted = True

    def _reference_table_exists(self) -> bool:
        try:
            lancedb = _get_lancedb()
            db = lancedb.connect(self.db_uri)
            try:
                names = db.table_names()
            except Exception:
                names = []
            return "reference_knowledge" in (names or [])
        except Exception:
            return False

    def _init_reference(self):
        """
        Reference Knowledge Guide (writing_guide.pdf) has been deprecated 
        in favor of Template PDF RAG from Step 3.
        This is now a no-op to avoid missing file warnings.
        """
        pass

    def _persist_to_lancedb(self, db, table_name, text):
        """Chunk text, embed, and write to LanceDB"""
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer, models
            # Load local or download
            try:
                paths = get_model_paths()
                model_path = paths.get("embedding_model")
                if model_path and os.path.exists(model_path):
                     try:
                         # Force device=cuda if available
                         import torch
                         try:
                             if torch.cuda.is_available():
                                 for i in range(torch.cuda.device_count()):
                                     pass
                                     # torch.cuda.set_per_process_memory_fraction(0.4, i)
                         except Exception as e:
                             pass
                         device = "cpu"
                         logger.info(f"Loading SentenceTransformer on device: {device}")
                         model = SentenceTransformer(model_path, local_files_only=True, device=device)
                     except Exception:
                         try:
                             tr = models.Transformer(model_path, local_files_only=True)
                             pl = models.Pooling(tr.get_word_embedding_dimension())
                             model = SentenceTransformer(modules=[tr, pl], device=device)
                         except Exception:
                             raise
                else:
                     import torch
                     try:
                         if torch.cuda.is_available():
                             for i in range(torch.cuda.device_count()):
                                 pass
                                 # torch.cuda.set_per_process_memory_fraction(0.4, i)
                     except Exception as e:
                         pass
                     device = "cpu"
                     logger.info(f"Loading SentenceTransformer (default) on device: {device}")
                     model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True, device=device)
                logger.info("Loaded embedding model from local cache.")
            except Exception:
                logger.warning("Local embedding model not available. Skipping persistent indexing.")
                return
            
            # Simple chunking
            chunks = []
            chunk_size = 1000
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                if len(chunk.strip()) > 50:
                    chunks.append(chunk)
            
            logger.info(f"Embedding {len(chunks)} chunks...")
            embeddings = model.encode(chunks)
            
            data = []
            for i, chunk in enumerate(chunks):
                data.append({
                    "vector": embeddings[i].tolist(),
                    "text": chunk,
                    "id": f"ref_{i}"
                })
            
            if table_name in db.table_names():
                db.drop_table(table_name)
                
            db.create_table(table_name, data=data)
            logger.info(f"Successfully persisted {len(data)} chunks to {table_name}.")
            
        except ImportError:
            logger.warning("sentence-transformers not installed. Skipping persistent indexing.")
        except Exception as e:
            logger.error(f"Error during embedding/persistence: {e}")

    def update_project_context(self, project_id: str, summary_text: str):
        """
        Update the project context by writing the summary to the watch folder.
        """
        filename = f"project_{project_id}.txt"
        path = os.path.join(self.proj_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        logger.info(f"Project context updated for {project_id}")

    def clear_project_vector_db(self, project_id: str) -> Dict[str, Any]:
        pid = str(project_id or "").strip()
        if not pid:
            return {"success": False, "error": "empty project_id"}
        pid_sql = pid.replace("'", "''")
        cleared = {"project_context": False, "project_context_meta": False, "proj_file": False}
        try:
            lancedb = _get_lancedb()
            db = lancedb.connect(self.db_uri)
            names = []
            try:
                names = db.table_names()
            except Exception:
                names = []
            if "project_context" in names:
                try:
                    tbl = db.open_table("project_context")
                    tbl.delete(f"project_id = '{pid_sql}'")
                    cleared["project_context"] = True
                except Exception:
                    pass
            if "project_context_meta" in names:
                try:
                    mt = db.open_table("project_context_meta")
                    mt.delete(f"project_id = '{pid_sql}'")
                    cleared["project_context_meta"] = True
                except Exception:
                    pass
        except Exception as e:
            return {"success": False, "error": str(e), "cleared": cleared}

        try:
            filename = f"project_{pid}.txt"
            path = os.path.join(self.proj_dir, filename)
            if os.path.exists(path):
                os.remove(path)
                cleared["proj_file"] = True
        except Exception:
            pass

        return {"success": True, "cleared": cleared}

    def clear_project_template_pdf_vectors(self, project_id: str) -> Dict[str, Any]:
        pid = str(project_id or "").strip()
        if not pid:
            return {"success": False, "error": "empty project_id"}
        pid_sql = pid.replace("'", "''")
        cleared = {"project_context": False}
        try:
            lancedb = _get_lancedb()
            db = lancedb.connect(self.db_uri)
            names = []
            try:
                names = db.table_names()
            except Exception:
                names = []
            if "project_context" in names:
                try:
                    tbl = db.open_table("project_context")
                    tbl.delete(f"project_id = '{pid_sql}' AND source_type = 'template_pdf'")
                    cleared["project_context"] = True
                except Exception:
                    pass
        except Exception as e:
            return {"success": False, "error": str(e), "cleared": cleared}
        return {"success": True, "cleared": cleared}

    def clear_all_template_pdf_vectors(self) -> Dict[str, Any]:
        cleared = {"project_context": False}
        try:
            lancedb = _get_lancedb()
            db = lancedb.connect(self.db_uri)
            names = []
            try:
                names = db.table_names()
            except Exception:
                names = []
            if "project_context" in names:
                try:
                    tbl = db.open_table("project_context")
                    tbl.delete("source_type = 'template_pdf'")
                    cleared["project_context"] = True
                except Exception:
                    pass
        except Exception as e:
            return {"success": False, "error": str(e), "cleared": cleared}
        return {"success": True, "cleared": cleared}

    def query_reference(self, query: str, limit: int = 3) -> str:
        """
        Query the reference knowledge base.
        """
        if not self._reference_ready and not self._reference_init_attempted:
            self._reference_init_attempted = True
            if not self._reference_table_exists():
                try:
                    self._init_reference()
                except Exception:
                    pass
            self._reference_ready = self._reference_table_exists()
        return self._query("reference_knowledge", query, limit)

    def query_project(self, query: str, project_id: str = "", limit: int = 5) -> str:
        """
        Query the project context.
        """
        filters = {}
        if project_id:
            filters["project_id"] = project_id
        return self._query("project_context", query, limit, filters=filters or None)

    def query_project_records(self, query: str, project_id: str = "", limit: int = 12) -> List[dict]:
        filters = {}
        if project_id:
            filters["project_id"] = project_id
        return self._query_records("project_context", query, limit, filters=filters or None)

    def sync_global_docs_database(self, docs_dir: str, force: bool = False, progress_cb=None) -> Dict[str, Any]:
        """
        Build a dedicated LanceDB table from local radiology docs (PDFs).
        """
        def _progress(msg: str):
            if progress_cb:
                try:
                    progress_cb(msg)
                except Exception:
                    pass

        docs_dir = str(docs_dir or "").strip()
        if not docs_dir or not os.path.isdir(docs_dir):
            return {"success": False, "error": f"invalid docs_dir: {docs_dir}"}

        table_name = "global_docs_context"
        meta_name = "global_docs_context_meta"
        lancedb = _get_lancedb()
        db = lancedb.connect(self.db_uri)

        pdf_files = []
        for root, _dirs, files in os.walk(docs_dir):
            for fn in files:
                if str(fn).lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, fn))
        pdf_files = sorted(list(dict.fromkeys([os.path.abspath(p) for p in pdf_files])))
        if not pdf_files:
            return {"success": False, "error": "no pdf found in docs_dir"}

        # Fast path: already indexed and no force (only if sources unchanged)
        if not force:
            try:
                if table_name in (db.table_names() or []):
                    tbl = db.open_table(table_name)
                    cnt = len(tbl.to_arrow())
                    if cnt > 0 and meta_name in (db.table_names() or []):
                        meta_tbl = db.open_table(meta_name)
                        meta_rows = meta_tbl.to_arrow().to_pylist()
                        indexed = {}
                        for r in meta_rows or []:
                            if isinstance(r, dict) and r.get("source_id") and r.get("sha256"):
                                indexed[str(r["source_id"])] = str(r["sha256"])

                        current = {}
                        for p in pdf_files:
                            try:
                                with open(p, "rb") as fh:
                                    digest = hashlib.sha256(fh.read()).hexdigest()
                                rel = os.path.relpath(p, docs_dir).replace("\\", "/")
                                current[rel] = digest
                            except Exception:
                                continue

                        if current and indexed == current:
                            _progress(f"DocsKB: 索引未变化 chunks={cnt} sources={len(indexed)}，跳过重建")
                            return {"success": True, "chunks": cnt, "reused": True, "sources": len(indexed)}
            except Exception:
                pass

        _progress(f"DocsKB: 扫描到 PDF={len(pdf_files)}")

        docs = []
        hash_rows = []
        for p in pdf_files:
            try:
                source_id = os.path.relpath(p, docs_dir).replace("\\", "/")
                with open(p, "rb") as fh:
                    digest = hashlib.sha256(fh.read()).hexdigest()
                reader = PdfReader(p)
                parts = []
                for page in reader.pages:
                    try:
                        t = page.extract_text() or ""
                    except Exception:
                        t = ""
                    if t.strip():
                        parts.append(t.strip())
                text = "\n\n".join(parts).strip()
                if not text:
                    continue
                docs.append((source_id, text))
                hash_rows.append(
                    {
                        "source_id": source_id,
                        "sha256": digest,
                        "updated_at": datetime.datetime.utcnow().isoformat(),
                    }
                )
            except Exception as e:
                logger.warning(f"DocsKB: parse failed {p}: {e}")

        if not docs:
            return {"success": False, "error": "no extractable text from docs pdfs"}

        # Build chunks
        chunks: List[Dict[str, Any]] = []
        chunk_size = 1200
        overlap = 200
        seq = 0
        for source_id, text in docs:
            start = 0
            n = len(text)
            while start < n:
                end = min(n, start + chunk_size)
                c = text[start:end].strip()
                if len(c) > 80:
                    chunks.append(
                        {
                            "id": f"gd_{seq}",
                            "source_type": "radiology_docs",
                            "source_id": source_id,
                            "chunk_index": seq,
                            "text": c,
                        }
                    )
                    seq += 1
                if end >= n:
                    break
                start = max(0, end - overlap)

        _progress(f"DocsKB: 生成 chunks={len(chunks)}，开始向量化")

        # Embed
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
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
        model = None
        paths = get_model_paths()
        model_path = paths.get("embedding_model")
        if model_path and os.path.exists(model_path):
            try:
                model = SentenceTransformer(model_path, local_files_only=True, device=device)
            except Exception:
                tr = models.Transformer(model_path, local_files_only=True)
                pl = models.Pooling(tr.get_word_embedding_dimension())
                model = SentenceTransformer(modules=[tr, pl], device=device)
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True, device=device)

        texts = [c["text"] for c in chunks]
        vecs = model.encode(texts)
        rows = []
        for i, c in enumerate(chunks):
            row = dict(c)
            row["vector"] = vecs[i].tolist()
            rows.append(row)

        # Persist
        try:
            if table_name in (db.table_names() or []):
                db.drop_table(table_name)
        except Exception:
            pass
        db.create_table(table_name, data=rows)

        try:
            if meta_name in (db.table_names() or []):
                db.drop_table(meta_name)
            db.create_table(meta_name, data=hash_rows)
        except Exception:
            pass

        _progress(f"DocsKB: 索引完成 chunks={len(rows)}")
        return {"success": True, "chunks": len(rows), "sources": len(docs)}

    def query_global_docs(self, query: str, limit: int = 5) -> str:
        return self._query("global_docs_context", query, limit)

    def _query(self, table_name: str, query: str, limit: int, filters: dict = None) -> str:
        records = self._query_records(table_name, query, limit, filters=filters)
        context = []
        for r in records[: max(0, int(limit))]:
            if isinstance(r, dict) and "text" in r:
                context.append(r["text"])
            else:
                context.append(str(r))
        return "\n\n".join(context)

    def _query_records(self, table_name: str, query: str, limit: int, filters: dict = None) -> List[dict]:
        try:
            query_vector = None
            logger.info(f"[KB Debug] Querying table '{table_name}' with: '{query[:50]}...'")
            
            # 1. Try manual embedding (Preferred, to match ingestion)
            try:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                from sentence_transformers import SentenceTransformer, models
                try:
                    logger.info("[KB Debug] Loading SentenceTransformer locally...")
                    paths = get_model_paths()
                    model_path = paths.get("embedding_model")
                    if model_path and os.path.exists(model_path):
                        logger.info(f"[KB Debug] Using configured model path: {model_path}")
                        import torch
                        device = "cpu"
                        logger.info(f"[KB Debug] Target device: {device}")
                        try:
                            model = SentenceTransformer(model_path, local_files_only=True, device=device)
                        except Exception:
                            try:
                                tr = models.Transformer(model_path, local_files_only=True)
                                pl = models.Pooling(tr.get_word_embedding_dimension())
                                model = SentenceTransformer(modules=[tr, pl], device=device)
                            except Exception as e2:
                                raise e2
                    else:
                        import torch
                        device = "cpu"
                        logger.info(f"[KB Debug] Target device (default): {device}")
                        model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True, device=device)
                except Exception as e:
                    logger.warning(f"[KB Debug] Local embedding load failed: {e}. Falling back to text_to_embedding.")
                    model = None
                    
                if model is not None:
                    query_vector = model.encode(query).tolist()
                    logger.info(f"[KB Debug] Query vector generated. Length: {len(query_vector)}")
                else:
                    query_vector = None
            except ImportError:
                logger.error("[KB Debug] sentence_transformers not installed!")
                # 2. Fallback to CocoIndex flow function
                try:
                    query_vector = text_to_embedding.eval(query)
                except Exception as ex:
                    logger.error(f"[KB Debug] text_to_embedding fallback failed: {ex}")
                    query_vector = None
            except Exception as e:
                logger.error(f"[KB Debug] Embedding generation failed: {e}")
                query_vector = None
            
            lancedb = _get_lancedb()
            db = lancedb.connect(self.db_uri)
            try:
                try:
                    names = db.table_names()
                    if table_name not in names:
                        return []
                except Exception:
                    pass
                table = db.open_table(table_name)
            except Exception:
                return []
            
            if query_vector is not None:
                try:
                    # 逻辑隔离: 强化检索阶段的元数据硬过滤
                    if self.project_id and table_name == "project_context":
                        if filters is None:
                            filters = {}
                        filters["project_id"] = self.project_id
                        
                    q = table.search(query_vector)
                    if filters:
                        where_clauses = []
                        for k, v in filters.items():
                            vv = str(v).replace("'", "''")
                            where_clauses.append(f"{k} = '{vv}'")
                        where_expr = " AND ".join(where_clauses)
                        if where_expr:
                            q = q.where(where_expr)
                    base_limit = limit
                    if table_name == "project_context":
                        base_limit = max(limit * 4, 20)
                    results = q.limit(base_limit).to_list()
                    logger.info(f"[KB Debug] Search returned {len(results)} results.")
                except Exception as e:
                    logger.error(f"[KB Debug] Vector search failed: {e}")
                    query_vector = None
                    results = []
            else:
                logger.warning("[KB Debug] Query vector is None, falling back to keyword search.")
                try:
                    rows = table.to_arrow().to_pylist()
                except Exception:
                    rows = []
                q = (query or "").lower()
                scored = []
                for r in rows:
                    if filters:
                        ok = True
                        for k, v in filters.items():
                            if str(r.get(k, "")) != str(v):
                                ok = False
                                break
                        if not ok:
                            continue
                    text = str(r.get("text", "")).lower()
                    score = 0
                    for token in q.split():
                        if token and token in text:
                            score += 1
                    if score > 0:
                        scored.append((score, r))
                scored.sort(key=lambda x: x[0], reverse=True)
                results = [r for _, r in scored[:limit]]
            
            if query_vector is None and not results:
                try:
                    rows = table.to_arrow().to_pylist()
                except Exception:
                    rows = []
                q = (query or "").lower()
                scored = []
                for r in rows:
                    if filters:
                        ok = True
                        for k, v in filters.items():
                            if str(r.get(k, "")) != str(v):
                                ok = False
                                break
                        if not ok:
                            continue
                    text = str(r.get("text", "")).lower()
                    score = 0
                    for token in q.split():
                        if token and token in text:
                            score += 1
                    if score > 0:
                        scored.append((score, r))
                scored.sort(key=lambda x: x[0], reverse=True)
                results = [r for _, r in scored[:limit]]
            
            if table_name == "project_context":
                try:
                    results = self._expand_and_rerank_project_context(
                        table=table,
                        query=query,
                        query_vector=query_vector,
                        initial_results=results,
                        filters=filters or {},
                        final_k=limit,
                        neighbor_window=1,
                    )
                except Exception as e:
                    logger.warning(f"[KB Debug] expand/rerank failed, using raw results: {e}")
            return list(results or [])[: max(0, int(limit))]
        except Exception as e:
            # Downgrade to warning and return empty context to avoid noisy logs
            logger.warning(f"Query error for {table_name}: {e}")
            return []

    def _expand_and_rerank_project_context(
        self,
        table,
        query: str,
        query_vector,
        initial_results: List[dict],
        filters: Dict[str, str],
        final_k: int,
        neighbor_window: int = 1,
    ) -> List[dict]:
        project_id = (filters or {}).get("project_id", "")

        def _get_distance(r: dict) -> float:
            for k in ["_distance", "distance", "score", "_score"]:
                if k in r:
                    try:
                        return float(r.get(k))
                    except Exception:
                        return math.inf
            return math.inf

        def _normalize_text_val(v) -> str:
            try:
                return str(v or "")
            except Exception:
                return ""

        def _where_expr(extra: Dict[str, str]) -> str:
            where_clauses = []
            for k, v in extra.items():
                vv = str(v).replace("'", "''")
                where_clauses.append(f"{k} = '{vv}'")
            return " AND ".join(where_clauses)

        def _fetch_for_source(source_type: str, source_id: str, limit_n: int = 48) -> List[dict]:
            if query_vector is None:
                return []
            try:
                q = table.search(query_vector)
                extra = {}
                if project_id:
                    extra["project_id"] = project_id
                extra["source_type"] = source_type
                extra["source_id"] = source_id
                expr = _where_expr(extra)
                if expr:
                    q = q.where(expr)
                return q.limit(limit_n).to_list()
            except Exception:
                return []

        def _as_key(r: dict) -> Tuple[str, str, int]:
            st = _normalize_text_val(r.get("source_type"))
            sid = _normalize_text_val(r.get("source_id"))
            try:
                ci = int(r.get("chunk_index", -1))
            except Exception:
                ci = -1
            return st, sid, ci

        by_source: Dict[Tuple[str, str], Dict[int, dict]] = {}
        for r in initial_results or []:
            st, sid, ci = _as_key(r)
            if not st or not sid or ci < 0:
                continue
            by_source.setdefault((st, sid), {})[ci] = r

        top_hits = list(initial_results or [])[: min(10, len(initial_results or []))]
        expanded: Dict[str, dict] = {}
        for r in top_hits:
            rid = _normalize_text_val(r.get("id"))
            if rid:
                expanded[rid] = r
            st, sid, ci = _as_key(r)
            if not st or not sid or ci < 0:
                continue
            src_map = by_source.get((st, sid), {})
            missing = []
            for d in range(-neighbor_window, neighbor_window + 1):
                if d == 0:
                    continue
                if (ci + d) not in src_map:
                    missing.append(ci + d)
            if missing:
                fetched = _fetch_for_source(st, sid, limit_n=64)
                for fr in fetched:
                    fst, fsid, fci = _as_key(fr)
                    if fst and fsid and fci >= 0:
                        by_source.setdefault((fst, fsid), {})[fci] = fr
                src_map = by_source.get((st, sid), {})
            for d in range(-neighbor_window, neighbor_window + 1):
                if d == 0:
                    continue
                nr = src_map.get(ci + d)
                if not nr:
                    continue
                nid = _normalize_text_val(nr.get("id"))
                if nid:
                    expanded[nid] = nr

        for r in initial_results or []:
            rid = _normalize_text_val(r.get("id"))
            if rid and rid not in expanded:
                expanded[rid] = r

        candidates = list(expanded.values())

        q_lower = (query or "").lower()
        q_tokens = [t for t in re.split(r"\W+", q_lower) if t]
        q_token_set = set(q_tokens)

        method_hints = {
            "method",
            "methods",
            "protocol",
            "parameter",
            "scanner",
            "sequence",
            "reconstruction",
            "slice",
            "thickness",
            "spacing",
            "pixel",
            "kvp",
            "tesla",
            "tr",
            "te",
        }

        def _bonus(text: str) -> float:
            t = (text or "")
            tl = t.lower()
            s = 0.0
            if "[PROJECT_STATS]" in t:
                s += 3.0
            if "key_dicom_fields" in t or "key_dicom_fields:" in t:
                s += 2.0
            if "properties_key_fields" in t or "properties_key_fields:" in t:
                s += 2.0
            if "scope: method" in tl and (q_token_set & method_hints):
                s += 1.5
            if re.search(r"\d", t):
                s += 0.6
            if any(u in tl for u in [" mm", " cm", "kvp", "tesla", " ms", " hz", "bpm", "%"]):
                s += 0.4
            token_hits = 0
            for tok in q_tokens[:12]:
                if tok and tok in tl:
                    token_hits += 1
            s += min(2.0, token_hits * 0.2)
            return s

        scored = []
        for r in candidates:
            t = _normalize_text_val(r.get("text"))
            dist = _get_distance(r)
            base = 0.0 if dist == math.inf else -dist
            score = base + _bonus(t)
            scored.append((score, dist, r))
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        max_per_source = 2
        per_source: Dict[Tuple[str, str], int] = {}
        chosen: List[dict] = []
        for score, _, r in scored:
            st = _normalize_text_val(r.get("source_type"))
            sid = _normalize_text_val(r.get("source_id"))
            key = (st, sid)
            if st and sid:
                if per_source.get(key, 0) >= max_per_source:
                    continue
            chosen.append(r)
            if st and sid:
                per_source[key] = per_source.get(key, 0) + 1
            if len(chosen) >= max(0, int(final_k)):
                break

        return chosen
