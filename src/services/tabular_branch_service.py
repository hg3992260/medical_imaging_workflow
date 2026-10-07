import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None

from src.services.database_service import DatabaseService


def _norm_col(name: str) -> str:
    t = (name or "").strip().lower()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^a-z0-9_]+", "", t)
    return t


def _try_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


@dataclass
class BranchTable:
    user_id: str
    branch_path: str
    domain_tag: str
    priority_weight: float
    density: float
    n_rows: int
    n_cols: int
    df: Any


class TabularBranchService:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    def build_tree(self, project_id: str, schema_cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        project_name = self._get_project_name(project_id)
        q = """
            SELECT sf.file_id, sf.file_path, sf.metadata
            FROM storage_files sf
            JOIN project_storage ps ON ps.storage_id = sf.storage_id
            WHERE ps.project_id = ? AND sf.file_type = ?
            ORDER BY sf.created_at DESC
        """
        rows = self.db.execute_query(q, (project_id, "tabular"))
        branches = []
        for r in rows or []:
            fp = r.get("file_path")
            if not fp or not os.path.exists(fp):
                continue
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                continue

            meta = (obj or {}).get("metadata") or {}
            sheet = (obj or {}).get("sheet") or {}
            user_id = str(meta.get("user_id") or "").strip() or "unknown"
            branch_path = str(meta.get("branch_path") or "").strip() or fp
            domain_tag = str(meta.get("domain_tag") or "Tabular")
            priority_weight = float(meta.get("priority_weight") or 1.0)

            df = self._sheet_to_df(sheet)
            if df is None:
                continue
            df = self._apply_schema_cache(df, branch_path, schema_cache)
            df["__branch_path__"] = branch_path
            df["__sample_type__"] = domain_tag

            n_rows = int(meta.get("n_rows") or max(0, len(df)))
            n_cols = int(meta.get("n_cols") or max(0, len(df.columns)))
            density = self._data_density(df)
            branches.append(
                BranchTable(
                    user_id=user_id,
                    branch_path=branch_path,
                    domain_tag=domain_tag,
                    priority_weight=priority_weight,
                    density=density,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    df=df,
                )
            )

        return {
            "project": project_name or project_id,
            "project_id": project_id,
            "samples": self._group_by_sample(branches),
        }

    def summarize_project(self, project_id: str, max_chars: int = 5500, schema_cache: Optional[Dict[str, Any]] = None) -> str:
        schema_cache_obj = schema_cache
        tree = self.build_tree(project_id, schema_cache=schema_cache_obj)
        project_keyword = str(tree.get("project") or "").strip()
        branches = []
        for s in (tree.get("samples") or []):
            for b in (s.get("branches") or []):
                branches.append(b)

        schema_preview = self._build_schema_cache(branches)
        vertical = self._vertical_correlation(branches, schema_preview)
        inter = self._inter_branch_correlation(branches)
        pruning = self._dynamic_pruning(branches)

        lines = []
        lines.append(f"[TABULAR_TREE] Project={project_keyword}  Samples={len(tree.get('samples') or [])}  Branches={len(branches)}")
        canon_lines = self._canonical_cache_preview(schema_cache_obj)
        if canon_lines:
            lines.append("[SCHEMA_CANONICAL_CACHE]")
            lines.extend(canon_lines[:18])
        if pruning:
            lines.append("[DATA_DENSITY]")
            for p in pruning[:24]:
                lines.append(f"- {p}")
        if schema_preview:
            lines.append("[SCHEMA_CACHE_PREVIEW]")
            for c in schema_preview[:24]:
                lines.append(f"- {c}")
        if vertical:
            lines.append("[VERTICAL_CORRELATION]")
            for v in vertical[:28]:
                lines.append(f"- {v}")
        if inter:
            lines.append("[INTER_BRANCH_CORRELATION]")
            for v in inter[:22]:
                lines.append(f"- {v}")

        kw_align = self._keyword_alignment(branches, project_keyword)
        if kw_align:
            lines.append("[KEYWORD_ALIGNMENT]")
            for x in kw_align[:18]:
                lines.append(f"- {x}")

        out = "\n".join(lines).strip()
        if len(out) <= max_chars:
            return out
        return out[: max_chars - 3].rstrip() + "..."

    def _apply_schema_cache(self, df: Any, branch_path: str, schema_cache: Optional[Dict[str, Any]]) -> Any:
        if pd is None or df is None or df.empty:
            return df
        cache = schema_cache or {}
        cms = cache.get("canonical_mappings") if isinstance(cache, dict) else None
        if not isinstance(cms, list) or not cms:
            return df
        rename = {}
        for m in cms:
            if not isinstance(m, dict):
                continue
            canonical = str(m.get("canonical_name") or "").strip()
            if not canonical:
                continue
            aliases = m.get("aliases") or []
            if not isinstance(aliases, list):
                continue
            for a in aliases:
                if not isinstance(a, dict):
                    continue
                p = str(a.get("path") or "").strip()
                raw = str(a.get("raw_name") or "").strip()
                if p and raw and p == branch_path:
                    rename[raw] = canonical
        if rename:
            try:
                df = df.rename(columns=rename)
            except Exception:
                return df
        return df

    def _canonical_cache_preview(self, schema_cache: Optional[Dict[str, Any]]) -> List[str]:
        cache = schema_cache or {}
        cms = cache.get("canonical_mappings") if isinstance(cache, dict) else None
        if not isinstance(cms, list) or not cms:
            return []
        out = [f"canonical_mappings={len(cms)}"]
        for m in cms[:18]:
            if not isinstance(m, dict):
                continue
            cn = str(m.get("canonical_name") or "").strip()
            role = str(m.get("scientific_role") or "").strip()
            unit = str(m.get("unit_logic") or "").strip()
            aliases = m.get("aliases") or []
            out.append(f"- {cn} | role={role or 'NA'} | unit={unit or 'NA'} | aliases={len(aliases) if isinstance(aliases, list) else 0}")
        return out

    def _get_project_name(self, project_id: str) -> str:
        try:
            rows = self.db.execute_query("SELECT name FROM projects WHERE project_id = ?", (project_id,))
            row = (rows or [{}])[0]
            return str(row.get("name") or "").strip() if row else ""
        except Exception:
            return ""

    def _group_by_sample(self, branches: List[BranchTable]) -> List[Dict[str, Any]]:
        by: Dict[str, List[BranchTable]] = {}
        for b in branches:
            by.setdefault(b.user_id, []).append(b)
        out = []
        for uid, bs in sorted(by.items(), key=lambda x: x[0]):
            out.append(
                {
                    "user_id": uid,
                    "type": "Tabular",
                    "branches": [
                        {
                            "user_id": b.user_id,
                            "branch_path": b.branch_path,
                            "domain_tag": b.domain_tag,
                            "priority_weight": b.priority_weight,
                            "density": b.density,
                            "n_rows": b.n_rows,
                            "n_cols": b.n_cols,
                            "df": b.df,
                        }
                        for b in bs
                    ],
                }
            )
        return out

    def _sheet_to_df(self, sheet: Dict[str, Any]) -> Optional[Any]:
        if pd is None:
            return None
        if not isinstance(sheet, dict):
            return None
        records = sheet.get("records")
        if isinstance(records, list) and records:
            try:
                df = pd.DataFrame.from_records(records)
                return df
            except Exception:
                pass
        rows = sheet.get("rows")
        if isinstance(rows, list) and rows:
            headers = [str(x or "").strip() for x in (rows[0] or [])]
            body = rows[1:] if len(rows) > 1 else []
            try:
                df = pd.DataFrame(body, columns=headers[: len(body[0])] if body else headers)
                return df
            except Exception:
                return None
        kv = sheet.get("kv")
        if isinstance(kv, dict) and kv:
            try:
                df = pd.DataFrame([kv])
                return df
            except Exception:
                return None
        return None

    def _data_density(self, df: Any) -> float:
        try:
            if df is None or df.empty:
                return 0.0
            total = int(df.shape[0] * df.shape[1])
            if total <= 0:
                return 0.0
            non_empty = 0
            for c in df.columns:
                s = df[c]
                non_empty += int(s.astype(str).str.strip().ne("").sum())
            return float(non_empty) / float(total)
        except Exception:
            return 0.0

    def _dynamic_pruning(self, branches: List[Dict[str, Any]]) -> List[str]:
        out = []
        for b in branches:
            density = float(b.get("density") or 0.0)
            weight = 1.0 if density >= 0.30 else 0.10
            out.append(f"{b.get('branch_path')}  density={density:.2f}  weight={weight:.2f}")
        out.sort(key=lambda x: float(re.search(r"density=([0-9.]+)", x).group(1)) if re.search(r"density=([0-9.]+)", x) else 0.0, reverse=True)
        return out

    def _build_schema_cache(self, branches: List[Dict[str, Any]]) -> List[str]:
        mappings: Dict[str, List[Tuple[str, str]]] = {}
        for b in branches:
            df = b.get("df")
            if df is None:
                continue
            path = str(b.get("branch_path") or "")
            for col in df.columns:
                if str(col).startswith("__"):
                    continue
                canon = _norm_col(str(col))
                if not canon:
                    continue
                mappings.setdefault(canon, [])
                mappings[canon].append((path, str(col)))
        lines = []
        for canon, aliases in sorted(mappings.items(), key=lambda x: (-len(x[1]), x[0])):
            uniq = []
            seen = set()
            for p, raw in aliases:
                k = (p, raw)
                if k in seen:
                    continue
                seen.add(k)
                uniq.append({"path": p, "raw_name": raw})
            if len(uniq) < 2:
                continue
            lines.append(f"{canon}: {len(uniq)} aliases")
        return lines

    def _vertical_correlation(self, branches: List[Dict[str, Any]], schema_cache: List[str]) -> List[str]:
        col_stats: Dict[str, Dict[str, float]] = {}
        for b in branches:
            uid = str(b.get("user_id") or "")
            df = b.get("df")
            if df is None or df.empty:
                continue
            for c in df.columns:
                if str(c).startswith("__"):
                    continue
                canon = _norm_col(str(c))
                if not canon:
                    continue
                vals = []
                for v in df[c].tolist()[:5000]:
                    fv = _try_float(v)
                    if fv is not None and not math.isnan(fv) and math.isfinite(fv):
                        vals.append(fv)
                if len(vals) < 5:
                    continue
                m = float(sum(vals) / len(vals))
                col_stats.setdefault(canon, {})
                col_stats[canon][uid] = m
        out = []
        for canon, per_sample in col_stats.items():
            if len(per_sample) < 2:
                continue
            means = list(per_sample.values())
            out.append(f"{canon}: samples={len(per_sample)} mean_range=[{min(means):.3g},{max(means):.3g}]")
        out.sort(key=lambda x: int(re.search(r"samples=(\d+)", x).group(1)) if re.search(r"samples=(\d+)", x) else 0, reverse=True)
        return out

    def _inter_branch_correlation(self, branches: List[Dict[str, Any]]) -> List[str]:
        if pd is None:
            return []
        per_sample: Dict[str, Dict[str, float]] = {}
        for b in branches:
            uid = str(b.get("user_id") or "")
            df = b.get("df")
            if not uid or df is None or df.empty:
                continue
            for c in df.columns:
                if str(c).startswith("__"):
                    continue
                vals = []
                for v in df[c].tolist()[:5000]:
                    fv = _try_float(v)
                    if fv is not None and math.isfinite(fv):
                        vals.append(float(fv))
                if len(vals) < 5:
                    continue
                canon = _norm_col(str(c))
                if not canon:
                    continue
                per_sample.setdefault(uid, {})
                per_sample[uid][canon] = float(sum(vals) / len(vals))

        if len(per_sample) < 3:
            return []

        cols = {}
        for uid, m in per_sample.items():
            for k, v in m.items():
                cols.setdefault(k, [])
        col_list = sorted(cols.keys())

        def corr(xs: List[float], ys: List[float]) -> Optional[float]:
            if len(xs) != len(ys) or len(xs) < 3:
                return None
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            num = 0.0
            dx = 0.0
            dy = 0.0
            for x, y in zip(xs, ys):
                a = x - mx
                b = y - my
                num += a * b
                dx += a * a
                dy += b * b
            if dx <= 0 or dy <= 0:
                return None
            return num / math.sqrt(dx * dy)

        scored = []
        for i in range(len(col_list)):
            for j in range(i + 1, len(col_list)):
                c1 = col_list[i]
                c2 = col_list[j]
                xs = []
                ys = []
                for uid, m in per_sample.items():
                    if c1 in m and c2 in m:
                        xs.append(m[c1])
                        ys.append(m[c2])
                r = corr(xs, ys)
                if r is None:
                    continue
                if abs(r) < 0.55:
                    continue
                scored.append((abs(r), r, len(xs), c1, c2))

        scored.sort(key=lambda x: (x[0], x[2]), reverse=True)
        out = []
        for _, r, n, c1, c2 in scored[:28]:
            out.append(f"{c1} ~ {c2}: pearson_r={r:.3f}  n_samples={n}")
        return out

    def _keyword_alignment(self, branches: List[Dict[str, Any]], project_keyword: str) -> List[str]:
        kw = (project_keyword or "").strip().lower()
        tokens = [t for t in re.findall(r"[a-z0-9]+", kw) if len(t) >= 4]
        if not tokens:
            return []
        out = []
        for b in branches:
            df = b.get("df")
            if df is None:
                continue
            cols = [str(c).lower() for c in df.columns if not str(c).startswith("__")]
            hits = []
            for t in tokens:
                if any(t in c for c in cols):
                    hits.append(t)
            if hits:
                out.append(f"{b.get('branch_path')}: matched={','.join(sorted(set(hits)))}")
        return out
