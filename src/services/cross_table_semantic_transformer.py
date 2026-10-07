import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.linear_model import LinearRegression
except Exception:
    mutual_info_regression = None
    LinearRegression = None

from src.services.tabular_branch_service import TabularBranchService


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("\x00", "").strip()


def _try_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = _safe_str(x)
    if not s:
        return None
    s = s.replace(",", "")
    try:
        v = float(s)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _flatten_json(obj: Any, prefix: str = "", out: Optional[Dict[str, Any]] = None, max_items: int = 200) -> Dict[str, Any]:
    if out is None:
        out = {}
    if len(out) >= max_items:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            kk = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            _flatten_json(v, kk, out, max_items=max_items)
            if len(out) >= max_items:
                break
        return out
    if isinstance(obj, list):
        for i, v in enumerate(obj[:20]):
            kk = f"{prefix}[{i}]"
            _flatten_json(v, kk, out, max_items=max_items)
            if len(out) >= max_items:
                break
        return out
    out[prefix] = obj
    return out


class CrossTableSemanticTransformer:
    def __init__(self, db_service):
        self.db = db_service
        self.tabular = TabularBranchService(db_service)

    def build_anchor_table(self, project_id: str, schema_cache: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Dict[str, Any]]:
        if pd is None:
            return None, {"error": "pandas_not_available"}
        mod_frames = []
        meta = {"modalities": {}}

        df_tab, m_tab = self._tabular_features(project_id, schema_cache)
        if df_tab is not None:
            mod_frames.append(df_tab)
        meta["modalities"]["tabular"] = m_tab

        df_roi, m_roi = self._roi_features(project_id)
        if df_roi is not None:
            mod_frames.append(df_roi)
        meta["modalities"]["roi"] = m_roi

        df_dicom, m_dicom = self._dicom_features(project_id)
        if df_dicom is not None:
            mod_frames.append(df_dicom)
        meta["modalities"]["dicom"] = m_dicom

        df_text, m_text = self._text_features(project_id)
        if df_text is not None:
            mod_frames.append(df_text)
        meta["modalities"]["text"] = m_text

        if not mod_frames:
            return None, {"error": "no_modalities"}

        out = mod_frames[0]
        for df in mod_frames[1:]:
            out = out.merge(df, on="anchor_id", how="outer")
        return out, meta

    def summarize_project(self, project_id: str, schema_cache: Optional[Dict[str, Any]] = None, max_chars: int = 5200) -> Tuple[str, Dict[str, Any]]:
        df, meta = self.build_anchor_table(project_id, schema_cache=schema_cache)
        if df is None or pd is None:
            return "", meta

        corr = self._cross_modal_correlations(df)
        mi = self._cross_modal_mutual_info(df)
        reg = self._simple_regression(df)

        lines = []
        lines.append(f"[CROSS_TABLE_SEMANTIC_TRANSFORMER] anchors={int(df.shape[0])} cols={int(df.shape[1])}")
        for k, v in (meta.get("modalities") or {}).items():
            if isinstance(v, dict):
                lines.append(f"- {k}: {v}")
        if corr:
            lines.append("[CROSS_MODAL_CORRELATION]")
            for x in corr[:18]:
                lines.append(f"- {x}")
        if mi:
            lines.append("[CROSS_MODAL_MUTUAL_INFO]")
            for x in mi[:18]:
                lines.append(f"- {x}")
        if reg:
            lines.append("[CROSS_MODAL_REGRESSION]")
            for x in reg[:18]:
                lines.append(f"- {x}")

        text = "\n".join(lines).strip()
        if len(text) > max_chars:
            text = text[: max(0, max_chars - 3)].rstrip() + "..."

        structured = {
            "cross_modal_correlations": corr[:25],
            "cross_modal_mutual_info": mi[:25],
            "cross_modal_regression": reg[:25],
            "modalities": meta.get("modalities") or {},
        }
        return text, structured

    def _tabular_features(self, project_id: str, schema_cache: Optional[Dict[str, Any]]) -> Tuple[Optional[Any], Dict[str, Any]]:
        if pd is None:
            return None, {"enabled": False}
        tree = self.tabular.build_tree(project_id, schema_cache=schema_cache)
        samples = tree.get("samples") or []
        rows = []
        leaf_count = 0
        for s in samples:
            uid = _safe_str(s.get("user_id"))
            for b in s.get("branches") or []:
                df = b.get("df")
                if df is None or getattr(df, "empty", False):
                    continue
                leaf_count += 1
                numeric_cols = []
                for c in list(df.columns):
                    cs = _safe_str(c)
                    if cs.startswith("__"):
                        continue
                    vals = []
                    for v in df[c].tolist()[:5000]:
                        fv = _try_float(v)
                        if fv is not None:
                            vals.append(fv)
                    if len(vals) >= 3:
                        numeric_cols.append((cs, float(sum(vals) / len(vals))))
                if not numeric_cols:
                    continue
                rec = {"anchor_id": uid}
                for name, meanv in numeric_cols[:80]:
                    rec[f"tab_{name}"] = meanv
                rows.append(rec)
        if not rows:
            return None, {"enabled": True, "leaf_tables": leaf_count, "anchors": 0}
        df_out = pd.DataFrame.from_records(rows)
        df_out = df_out.groupby("anchor_id", as_index=False).mean(numeric_only=True)
        return df_out, {"enabled": True, "leaf_tables": leaf_count, "anchors": int(df_out.shape[0]), "features": int(df_out.shape[1] - 1)}

    def _roi_features(self, project_id: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        if pd is None:
            return None, {"enabled": False}
        q = """
            SELECT rd.roi_id, rd.area, rd.perimeter, rd.properties, rd.session_id
            FROM roi_data rd
            JOIN dicom_sessions ds ON rd.session_id = ds.session_id
            WHERE ds.project_id = ?
            ORDER BY rd.created_at DESC
        """
        try:
            rows = self.db.execute_query(q, (project_id,)) or []
        except Exception:
            rows = []
        if not rows:
            return None, {"enabled": True, "anchors": 0}
        uid_map = self._dicom_session_user_map(project_id)
        recs = []
        for r in rows:
            sid = _safe_str(r.get("session_id"))
            uid = uid_map.get(sid) or ""
            if not uid:
                continue
            rec = {"anchor_id": uid}
            area = _try_float(r.get("area"))
            per = _try_float(r.get("perimeter"))
            if area is not None:
                rec["roi_area"] = area
            if per is not None:
                rec["roi_perimeter"] = per
            props = _safe_str(r.get("properties"))
            if props:
                try:
                    obj = json.loads(props)
                    flat = _flatten_json(obj, max_items=120)
                    for k, v in flat.items():
                        fv = _try_float(v)
                        if fv is not None:
                            rec[f"roi_{k}"] = fv
                except Exception:
                    pass
            if len(rec.keys()) > 1:
                recs.append(rec)
        if not recs:
            return None, {"enabled": True, "anchors": 0}
        df_out = pd.DataFrame.from_records(recs)
        df_out = df_out.groupby("anchor_id", as_index=False).mean(numeric_only=True)
        return df_out, {"enabled": True, "anchors": int(df_out.shape[0]), "features": int(df_out.shape[1] - 1)}

    def _dicom_session_user_map(self, project_id: str) -> Dict[str, str]:
        q = "SELECT session_id, user_id FROM dicom_sessions WHERE project_id = ?"
        try:
            rows = self.db.execute_query(q, (project_id,)) or []
        except Exception:
            rows = []
        out = {}
        for r in rows:
            out[_safe_str(r.get("session_id"))] = _safe_str(r.get("user_id"))
        return out

    def _dicom_features(self, project_id: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        if pd is None:
            return None, {"enabled": False}
        q = "SELECT session_id, user_id, dicom_info FROM dicom_sessions WHERE project_id = ? ORDER BY created_at DESC"
        try:
            rows = self.db.execute_query(q, (project_id,)) or []
        except Exception:
            rows = []
        if not rows:
            return None, {"enabled": True, "anchors": 0}
        recs = []
        for r in rows:
            uid = _safe_str(r.get("user_id"))
            if not uid:
                continue
            info = _safe_str(r.get("dicom_info"))
            if not info:
                continue
            rec = {"anchor_id": uid}
            try:
                obj = json.loads(info)
                flat = _flatten_json(obj, max_items=200)
            except Exception:
                flat = {}
            for k, v in flat.items():
                kk = _safe_str(k).lower()
                if not any(x in kk for x in ["slice", "thickness", "spacing", "pixelspacing", "rows", "columns", "kvp", "ma", "mas", "tr", "te", "flip", "echo", "dose"]):
                    continue
                fv = _try_float(v)
                if fv is None:
                    continue
                rec[f"dicom_{kk[:60]}"] = fv
            if len(rec.keys()) > 1:
                recs.append(rec)
        if not recs:
            return None, {"enabled": True, "anchors": 0}
        df_out = pd.DataFrame.from_records(recs)
        df_out = df_out.groupby("anchor_id", as_index=False).mean(numeric_only=True)
        return df_out, {"enabled": True, "anchors": int(df_out.shape[0]), "features": int(df_out.shape[1] - 1)}

    def _text_features(self, project_id: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        if pd is None:
            return None, {"enabled": False}
        recs = []
        q1 = """
            SELECT s.user_id, r.recognized_text
            FROM ocr_results r
            JOIN ocr_sessions s ON r.session_id = s.session_id
            WHERE s.project_id = ?
        """
        try:
            rows = self.db.execute_query(q1, (project_id,)) or []
        except Exception:
            rows = []
        for r in rows:
            uid = _safe_str(r.get("user_id"))
            txt = _safe_str(r.get("recognized_text"))
            if not uid or not txt:
                continue
            digits = sum(ch.isdigit() for ch in txt)
            recs.append({"anchor_id": uid, "text_ocr_len": float(len(txt)), "text_ocr_digit_ratio": float(digits) / float(max(1, len(txt)))})

        q2 = """
            SELECT ts.user_id, d.content
            FROM documents d
            JOIN text_sessions ts ON d.session_id = ts.session_id
            WHERE ts.project_id = ?
        """
        try:
            rows2 = self.db.execute_query(q2, (project_id,)) or []
        except Exception:
            rows2 = []
        for r in rows2:
            uid = _safe_str(r.get("user_id"))
            txt = _safe_str(r.get("content"))
            if not uid or not txt:
                continue
            digits = sum(ch.isdigit() for ch in txt)
            recs.append({"anchor_id": uid, "text_doc_len": float(len(txt)), "text_doc_digit_ratio": float(digits) / float(max(1, len(txt)))})

        if not recs:
            return None, {"enabled": True, "anchors": 0}
        df_out = pd.DataFrame.from_records(recs)
        df_out = df_out.groupby("anchor_id", as_index=False).mean(numeric_only=True)
        return df_out, {"enabled": True, "anchors": int(df_out.shape[0]), "features": int(df_out.shape[1] - 1)}

    def _cross_modal_correlations(self, df: Any) -> List[str]:
        if pd is None:
            return []
        cols = [c for c in df.columns if c != "anchor_id"]
        if len(cols) < 2:
            return []

        def mod(c: str) -> str:
            for p in ["tab_", "roi_", "dicom_", "text_"]:
                if c.startswith(p):
                    return p[:-1]
            return "other"

        scored = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = cols[i]
                b = cols[j]
                if mod(a) == mod(b):
                    continue
                s = df[[a, b]].dropna()
                if s.shape[0] < 3:
                    continue
                try:
                    r = float(s[a].corr(s[b]))
                except Exception:
                    continue
                if not math.isfinite(r) or abs(r) < 0.6:
                    continue
                scored.append((abs(r), r, int(s.shape[0]), a, b))
        scored.sort(key=lambda x: (x[0], x[2]), reverse=True)
        out = []
        for _, r, n, a, b in scored[:30]:
            out.append(f"{a} ~ {b}: pearson_r={r:.3f}  n={n}")
        return out

    def _cross_modal_mutual_info(self, df: Any) -> List[str]:
        if pd is None or mutual_info_regression is None:
            return []
        cols = [c for c in df.columns if c != "anchor_id"]
        if len(cols) < 2:
            return []

        def mod(c: str) -> str:
            for p in ["tab_", "roi_", "dicom_", "text_"]:
                if c.startswith(p):
                    return p[:-1]
            return "other"

        scored = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = cols[i]
                b = cols[j]
                if mod(a) == mod(b):
                    continue
                s = df[[a, b]].dropna()
                if s.shape[0] < 8:
                    continue
                x = s[[a]].to_numpy()
                y = s[b].to_numpy()
                try:
                    mi = float(mutual_info_regression(x, y, discrete_features=False, random_state=7)[0])
                except Exception:
                    continue
                if not math.isfinite(mi) or mi < 0.05:
                    continue
                scored.append((mi, int(s.shape[0]), a, b))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        out = []
        for mi, n, a, b in scored[:30]:
            out.append(f"{a} ~ {b}: mutual_info={mi:.3f}  n={n}")
        return out

    def _simple_regression(self, df: Any) -> List[str]:
        if pd is None or LinearRegression is None:
            return []
        target = None
        for c in df.columns:
            if c == "roi_area":
                target = c
                break
        if not target:
            return []
        feature_cols = [c for c in df.columns if c != "anchor_id" and c != target and any(c.startswith(p) for p in ["tab_", "dicom_", "text_"])]
        if len(feature_cols) < 2:
            return []
        s = df[[target] + feature_cols].dropna()
        if s.shape[0] < 8:
            return []
        x = s[feature_cols].to_numpy()
        y = s[target].to_numpy()
        try:
            model = LinearRegression()
            model.fit(x, y)
            r2 = float(model.score(x, y))
            coefs = list(model.coef_)
        except Exception:
            return []
        pairs = list(zip(feature_cols, coefs))
        pairs.sort(key=lambda t: abs(float(t[1])), reverse=True)
        out = [f"target={target}  n={int(s.shape[0])}  r2={r2:.3f}"]
        for name, w in pairs[:12]:
            out.append(f"{name}: coef={float(w):.3g}")
        return out

