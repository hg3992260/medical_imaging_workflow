import os
from typing import Dict, Any, List, Optional


def compute_section_progression(sections: Dict[str, str]) -> Dict[str, Any]:
    order = ["introduction", "methods", "results", "discussion", "conclusion", "references"]
    texts = {k: (sections.get(k) or "").strip() for k in order}
    keys = [k for k in order if texts.get(k)]

    out: Dict[str, Any] = {
        "order": keys,
        "edges": [],
        "matrix": {},
        "method": "",
        "available": False,
        "error": "",
    }
    if len(keys) < 2:
        out["available"] = True
        out["method"] = "insufficient_sections"
        return out

    os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
    os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")

    embedder = _load_embedder()
    if embedder is None:
        out["available"] = True
        out["method"] = "jaccard_fallback"
        return _jaccard_progression(out, keys, texts)

    try:
        vecs = embedder.encode([texts[k] for k in keys], normalize_embeddings=True, show_progress_bar=False)
        import numpy as np

        mat = np.matmul(vecs, np.transpose(vecs))
        out["available"] = True
        out["method"] = "sentence_transformers_cosine"
        out["matrix"] = {k: {} for k in keys}
        for i, a in enumerate(keys):
            for j, b in enumerate(keys):
                out["matrix"][a][b] = float(round(float(mat[i, j]), 4))
        for i in range(len(keys) - 1):
            a = keys[i]
            b = keys[i + 1]
            out["edges"].append({"from": a, "to": b, "similarity": out["matrix"][a][b]})
        return out
    except Exception as e:
        out["available"] = True
        out["method"] = "jaccard_fallback"
        out["error"] = str(e)
        return _jaccard_progression(out, keys, texts)


def _load_embedder():
    try:
        from sentence_transformers import SentenceTransformer, models
        import torch
        from src.utils.model_config_loader import get_model_paths

        device = "cpu"
        paths = get_model_paths()
        model_path = paths.get("embedding_model")
        if model_path and os.path.exists(model_path):
            try:
                return SentenceTransformer(model_path, local_files_only=True, device=device)
            except Exception:
                tr = models.Transformer(model_path, local_files_only=True)
                pl = models.Pooling(tr.get_word_embedding_dimension())
                return SentenceTransformer(modules=[tr, pl], device=device)
        try:
            return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True, device=device)
        except Exception:
            return None
    except Exception:
        return None


def _token_set(text: str) -> set:
    import re

    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return set([w for w in words if len(w) >= 3])


def _jaccard_progression(out: Dict[str, Any], keys: List[str], texts: Dict[str, str]) -> Dict[str, Any]:
    out["matrix"] = {k: {} for k in keys}
    tokens = {k: _token_set(texts[k]) for k in keys}
    for a in keys:
        for b in keys:
            ta = tokens[a]
            tb = tokens[b]
            denom = len(ta | tb)
            sim = (len(ta & tb) / denom) if denom else 0.0
            out["matrix"][a][b] = float(round(sim, 4))
    out["edges"] = []
    for i in range(len(keys) - 1):
        a = keys[i]
        b = keys[i + 1]
        out["edges"].append({"from": a, "to": b, "similarity": out["matrix"][a][b]})
    return out

