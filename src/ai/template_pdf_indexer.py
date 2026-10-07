import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.ai.coco_flows import LANCEDB_URI
from src.ai.pdf_template_parser import parse_pdf_sections_from_text, summarize_pdf_template, build_style_profile
from src.ai.project_context_indexer import _chunk_text, _stable_hash, _load_embedding_model
from src.services.text_service import TextService
from src.ai.pdf_template_layout import analyze_pdf_template_layout_hybrid
from src.ai.section_progression import compute_section_progression

def _get_lancedb():
    try:
        import lancedb as _lancedb

        return _lancedb
    except Exception as e:
        raise RuntimeError(f"lancedb/pyarrow 不可用: {e}")


def _safe_basename(path: str) -> str:
    try:
        return os.path.splitext(os.path.basename(path))[0]
    except Exception:
        return "pdf_template"


def index_pdf_template_into_lancedb(
    project_id: str,
    pdf_path: str,
    template_name: str = "",
    lancedb_uri: str = "",
    max_pages: int = 80,
    progress_cb=None,
    pre_extracted_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pid = str(project_id or "").strip()
    if not pid:
        return {"success": False, "error": "empty project_id"}
    pid_sql = pid.replace("'", "''")
    if not pdf_path or not os.path.exists(pdf_path):
        return {"success": False, "error": f"pdf not found: {pdf_path}"}

    tname = (template_name or "").strip() or _safe_basename(pdf_path)
    db_uri = lancedb_uri or LANCEDB_URI
    lancedb = _get_lancedb()

    # --- Use pre-extracted data if available to avoid redundant OCR ---
    if pre_extracted_data and pre_extracted_data.get("content"):
        content = pre_extracted_data.get("content") or ""
        meta = pre_extracted_data.get("metadata") or {}
        sections = pre_extracted_data.get("sections") or {}
        constraints = pre_extracted_data.get("constraints") or {}
        layout = constraints.get("pdf_layout_a4")
        progression = constraints.get("section_progression")
    else:
        # Fallback to full OCR if no pre-extracted data provided
        ts = TextService()
        r = ts.read_pdf_document(
            pdf_path,
            max_pages=max_pages,
            use_ocr=False,
            deepseek_mode="free",
            ocr_engine="deepseek_ocr",
            progress_callback=(lambda p: progress_cb(p) if callable(progress_cb) else None),
        )
        if not r.get("success") or not (r.get("content") or "").strip():
            r2 = ts.read_pdf_document(pdf_path, max_pages=max_pages, use_ocr=True, deepseek_mode="free")
            if not r2.get("success"):
                return {"success": False, "error": r.get("error") or r2.get("error") or "pdf read failed"}
            content = r2.get("content") or ""
            meta = r2.get("metadata") or {}
        else:
            content = r.get("content") or ""
            meta = r.get("metadata") or {}

        sections = parse_pdf_sections_from_text(content)
        constraints = summarize_pdf_template(sections)
        try:
            pages_processed = int((meta or {}).get("processed_pages") or (meta or {}).get("page_count") or 20)
        except Exception:
            pages_processed = 20
        layout = analyze_pdf_template_layout_hybrid(pdf_path, deepseek_sections=sections, max_pages=min(20, pages_processed))
        progression = compute_section_progression(sections)
        
    # Standardize constraints structure for consistency
    constraints["pdf_extracted_sections"] = {k: (sections.get(k) or "")[:2000] for k in ["title", "introduction", "methods", "results", "discussion", "conclusion", "references"]}
    constraints["pdf_layout_a4"] = layout
    constraints["section_progression"] = progression
    constraints["draft_output_sections"] = ["Title", "Introduction", "Methods", "Results", "Discussion", "Conclusion", "References"]
    constraints["pdf_path"] = pdf_path
    constraints["pdf_metadata"] = meta

    embedder = _load_embedding_model()
    has_vector = embedder is not None

    db = lancedb.connect(db_uri)
    names = []
    try:
        names = db.table_names()
    except Exception:
        names = []

    now = datetime.utcnow().isoformat()

    init = {
        "id": "init",
        "project_id": "",
        "user_id": "",
        "source_type": "",
        "source_id": "",
        "chunk_index": 0,
        "content_hash": "",
        "updated_at": "",
        "text": "",
    }
    if has_vector:
        try:
            init["vector"] = embedder.encode(["init"], show_progress_bar=False)[0].tolist()
        except Exception:
            init["vector"] = [0.0] * 384
    def _open_or_create_project_context(require_vector: bool):
        init_row = init
        try:
            current_names = []
            try:
                current_names = db.table_names()
            except Exception:
                current_names = []

            if "project_context" not in (current_names or []):
                tbl = db.create_table("project_context", data=[init_row])
                try:
                    tbl.delete("id = 'init'")
                except Exception:
                    pass
                return tbl

            try:
                tbl = db.open_table("project_context")
            except Exception:
                try:
                    db.drop_table("project_context")
                except Exception:
                    pass
                tbl = db.create_table("project_context", data=[init_row])
                try:
                    tbl.delete("id = 'init'")
                except Exception:
                    pass
                return tbl

            try:
                if require_vector and "vector" not in tbl.schema.names:
                    try:
                        db.drop_table("project_context")
                    except Exception:
                        pass
                    tbl = db.create_table("project_context", data=[init_row])
                    try:
                        tbl.delete("id = 'init'")
                    except Exception:
                        pass
            except Exception:
                try:
                    db.drop_table("project_context")
                except Exception:
                    pass
                tbl = db.create_table("project_context", data=[init_row])
                try:
                    tbl.delete("id = 'init'")
                except Exception:
                    pass
            return tbl
        except Exception as e:
            raise e

    try:
        ctx_tbl = _open_or_create_project_context(require_vector=has_vector)
    except Exception as e:
        return {"success": False, "error": f"failed to open/create project_context: {e}"}

    table_has_vector = False
    try:
        table_has_vector = "vector" in ctx_tbl.schema.names
    except Exception:
        table_has_vector = has_vector

    vector_dim = 384
    if table_has_vector:
        try:
            sample_rows = ctx_tbl.to_arrow().to_pylist()
            for rr in sample_rows or []:
                vv = rr.get("vector")
                if isinstance(vv, (list, tuple)) and len(vv) > 0:
                    vector_dim = len(vv)
                    break
        except Exception:
            pass

    try:
        ctx_tbl.delete(f"project_id = '{pid_sql}' AND source_type = 'template_pdf'")
    except Exception:
        pass

    meta_row = {
        "pdf_path": pdf_path,
        "template_name": tname,
        "pdf_metadata": meta,
        "constraints": constraints,
    }
    meta_text = "[TEMPLATE_PDF_META]\n" + json.dumps(meta_row, ensure_ascii=False, indent=2)
    meta_hash = _stable_hash(meta_text)
    meta_chunks = _chunk_text(meta_text, chunk_size=1200, overlap=200) or [meta_text]
    meta_vectors = None
    if embedder is not None:
        try:
            meta_vectors = embedder.encode(meta_chunks, show_progress_bar=False)
        except Exception:
            meta_vectors = None

    records: List[Dict[str, Any]] = []
    for i, ch in enumerate(meta_chunks):
        vid = f"{pid}:template_pdf:meta:{meta_hash[:12]}:{i}"
        vec = None
        if meta_vectors is not None:
            vec = meta_vectors[i].tolist()
        elif table_has_vector:
            vec = [0.0] * max(1, int(vector_dim))
        records.append(
            {
                "id": vid,
                "project_id": pid,
                "user_id": "template",
                "source_type": "template_pdf",
                "source_id": f"{tname}:meta",
                "chunk_index": i,
                "content_hash": meta_hash,
                "updated_at": now,
                "text": ch,
            }
        )
        if vec is not None:
            records[-1]["vector"] = vec

    for sec in ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]:
        sec_text = (sections.get(sec) or "").strip()
        if not sec_text:
            continue
        profile = build_style_profile(sec_text)
        header = {
            "template_name": tname,
            "section": sec,
            "style_profile": profile,
            "constraints_hint": constraints,
        }
        packed = "[TEMPLATE_PDF_SECTION_STYLE]\n" + json.dumps(header, ensure_ascii=False, indent=2) + "\n\n[SECTION_TEXT]\n" + sec_text
        h = _stable_hash(packed)
        chunks = _chunk_text(packed, chunk_size=1200, overlap=200) or [packed]
        sec_vectors = None
        if embedder is not None:
            try:
                sec_vectors = embedder.encode(chunks, show_progress_bar=False)
            except Exception:
                sec_vectors = None
        for i, ch in enumerate(chunks):
            vid = f"{pid}:template_pdf:{sec}:{h[:12]}:{i}"
            vec = None
            if sec_vectors is not None:
                vec = sec_vectors[i].tolist()
            elif table_has_vector:
                vec = [0.0] * max(1, int(vector_dim))
            records.append(
                {
                    "id": vid,
                    "project_id": pid,
                    "user_id": "template",
                    "source_type": "template_pdf",
                    "source_id": f"{tname}:{sec}",
                    "chunk_index": i,
                    "content_hash": h,
                    "updated_at": now,
                    "text": ch,
                }
            )
            if vec is not None:
                records[-1]["vector"] = vec

    if records:
        try:
            ctx_tbl.add(records)
        except Exception:
            try:
                ctx_tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(records)
            except Exception:
                return {"success": False, "error": "failed to write records to lancedb"}

    return {
        "success": True,
        "project_id": pid,
        "template_name": tname,
        "db_uri": db_uri,
        "sections_indexed": len([k for k in sections.keys() if (sections.get(k) or "").strip()]),
        "records_written": len(records),
        "constraints": constraints,
    }
