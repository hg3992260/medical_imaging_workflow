import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from scipy.stats import chi2_contingency, f_oneway, mannwhitneyu, ttest_ind
except Exception:
    chi2_contingency = None
    f_oneway = None
    mannwhitneyu = None
    ttest_ind = None

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        recall_score,
        r2_score,
        silhouette_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except Exception:
    KMeans = None
    PCA = None
    RandomForestClassifier = None
    RandomForestRegressor = None
    LogisticRegression = None
    Ridge = None
    accuracy_score = None
    confusion_matrix = None
    f1_score = None
    mean_absolute_error = None
    mean_squared_error = None
    precision_score = None
    recall_score = None
    r2_score = None
    silhouette_score = None
    train_test_split = None
    StandardScaler = None


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


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = str(text or "")
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _generate_figure_notes_with_llm(llm_service, model: str, figure_ctx: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        if not llm_service or not figure_ctx:
            return {"captions": {}, "notes": ""}
        payload = json.dumps(figure_ctx, ensure_ascii=False)[:12000]
        sys_prompt = (
            "You are a medical data scientist. Generate concise figure captions and explanations.\n"
            "Return ONLY a JSON object with keys: captions, notes.\n"
            "captions: mapping from 'path' to a short caption (<= 24 Chinese characters, no punctuation at end).\n"
            "notes: bullet list in Chinese, one bullet per figure, 2-4 sentences each, explain what the figure shows and how to interpret it. Include key numbers if provided.\n"
            "Do NOT use conversational phrases. Do NOT speculate beyond provided context.\n"
        )
        prompt = f"{sys_prompt}\n\n[FIGURE_CONTEXT_JSON]\n{payload}\n"
        res = llm_service.generate(model, prompt, timeout=300)
        if not isinstance(res, dict) or not res.get("success"):
            return {"captions": {}, "notes": ""}
        obj = _extract_json_object(res.get("response", ""))
        if not isinstance(obj, dict):
            return {"captions": {}, "notes": ""}
        captions = obj.get("captions")
        notes = obj.get("notes")
        if not isinstance(captions, dict):
            captions = {}
        notes_str = str(notes or "").strip()
        clean_caps = {}
        for k, v in captions.items():
            kk = str(k).replace("\\", "/").strip()
            vv = str(v or "").strip()
            if kk and vv:
                clean_caps[kk] = vv[:80]
        return {"captions": clean_caps, "notes": notes_str}
    except Exception:
        return {"captions": {}, "notes": ""}


def _load_tabular_records_from_json(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return []
        # Check standard tabular metadata schema first
        sheet = obj.get("sheet") or {}
        recs = sheet.get("records") or []
        # Fallback to direct records list if available
        if not recs and obj.get("records"):
            recs = obj.get("records") or []
        if isinstance(recs, list):
            out = []
            for r in recs:
                if isinstance(r, dict):
                    out.append(r)
            return out
        return []
    except Exception:
        return []


def _load_tabular_package_from_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return {"records": [], "metadata": {}, "sheet": {}}
        metadata = obj.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        sheet = obj.get("sheet") or {}
        if not isinstance(sheet, dict):
            sheet = {}
        records = sheet.get("records") or obj.get("records") or []
        if not isinstance(records, list):
            records = []
        recs = [r for r in records if isinstance(r, dict)]
        return {"records": recs, "metadata": metadata, "sheet": sheet}
    except Exception:
        return {"records": [], "metadata": {}, "sheet": {}}


def _read_csv_excel_files(workspace: str, max_files: int = 20) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Load CSV/Excel files via duckdb for 10-100x faster reads."""
    import duckdb
    ws = str(workspace or "").strip()
    if not ws or not os.path.isdir(ws):
        return [], [], ["workspace_missing"]

    exts = {".csv", ".xlsx", ".xlsm", ".xls"}
    files = []
    for p in sorted(Path(ws).glob("**/*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
            if len(files) >= int(max_files):
                break

    if not files:
        return [], [], ["no_excel_csv_found"]

    warnings: List[str] = []
    records: List[Dict[str, Any]] = []
    sources: List[str] = []
    con = duckdb.connect()

    for p in files:
        try:
            ext = p.suffix.lower()
            if ext == ".csv":
                df = con.sql(f"SELECT * FROM read_csv_auto('{p}')").df()
            elif ext in (".xlsx", ".xlsm"):
                df = con.sql(f"SELECT * FROM st_read('{p}')").df()
            else:
                # .xls — fallback via xlrd (duckdb does not support .xls)
                try:
                    import xlrd
                    book = xlrd.open_workbook(str(p))
                    if book.nsheets < 1:
                        warnings.append(f"empty:{p.name}")
                        continue
                    sheet = book.sheet_by_index(0)
                    headers = [str(sheet.cell_value(0, c)) for c in range(sheet.ncols)]
                    rows = []
                    for r in range(1, min(sheet.nrows, 3000)):
                        row: Dict[str, Any] = {}
                        for c in range(sheet.ncols):
                            k = headers[c] if c < len(headers) else f"col_{c}"
                            row[str(k)] = sheet.cell_value(r, c)
                        rows.append(row)
                    df = pd.DataFrame(rows)
                except Exception:
                    warnings.append(f"xls_parse_failed:{p.name}")
                    continue

            if df is None or df.empty:
                warnings.append(f"empty:{p.name}")
                continue

            for c in df.columns:
                if str(c).startswith("Unnamed"):
                    df = df.drop(columns=[c])
            df = df.iloc[:2000]
            sources.append(os.path.relpath(str(p), ws).replace("\\", "/"))
            recs = df.to_dict(orient="records")
            for r in recs:
                if isinstance(r, dict):
                    records.append(r)
        except Exception as e:
            warnings.append(f"read_failed:{p.name}:{type(e).__name__}")
            continue

    con.close()
    return records, sources, warnings

    return records, sources, warnings


def load_project_tabular_records(
    project_storage_path: str,
    workspace: str,
    max_json_files: int = 200,
    max_records: int = 8000,
) -> Dict[str, Any]:
    base = str(project_storage_path or "").strip()
    sources: List[str] = []
    warnings: List[str] = []
    records: List[Dict[str, Any]] = []

    tabular_root = os.path.join(base, "tabular") if base else ""
    if not tabular_root or not os.path.isdir(tabular_root):
        if base:
            warnings.append("tabular_dir_missing")
    if tabular_root and os.path.isdir(tabular_root):
        json_files = []
        for p in sorted(Path(tabular_root).glob("**/*.json")):
            json_files.append(str(p))
            if len(json_files) >= int(max_json_files):
                break
        if not json_files:
            warnings.append("tabular_json_not_found")
        for p in json_files:
            pkg = _load_tabular_package_from_json(p)
            pkg_records = pkg.get("records") or []
            for r in pkg_records:
                records.append(r)
                if len(records) >= int(max_records):
                    break
            if len(records) >= int(max_records):
                break
        sources.extend([os.path.relpath(p, base).replace("\\", "/") for p in json_files])

    allow_workspace = str(os.environ.get("SKLEARN_ALLOW_WORKSPACE_CSV_EXCEL") or "1").strip().lower() in ("1", "true", "yes")
    if allow_workspace and (not records):
        recs, srcs, warns = _read_csv_excel_files(workspace, max_files=20)
        records = recs[:max_records]
        sources.extend([f"workspace:{s}" for s in (srcs or [])])
        warnings.extend(warns or [])
    if (not allow_workspace) and (not records):
        warnings.append("workspace_csv_excel_disabled")

    return {"records": records, "sources": sources, "warnings": warnings}


def _prepare_pca_inputs(df: "pd.DataFrame", max_rows: int = 5000) -> Dict[str, Any]:
    if pd is None or np is None or PCA is None or StandardScaler is None:
        return {"ok": False, "error": "missing_deps"}
    if df is None or df.empty:
        return {"ok": False, "error": "empty_df"}

    df0 = df.copy()
    df0 = df0.iloc[: int(max_rows)].copy()
    for c in list(df0.columns):
        if str(c).startswith("Unnamed"):
            df0 = df0.drop(columns=[c])

    numeric_cols: List[str] = []
    cat_cols: List[str] = []
    for c in df0.columns:
        s_num = pd.to_numeric(df0[c], errors="coerce")
        non_na = int(s_num.notna().sum())
        if non_na >= max(8, int(len(df0) * 0.6)):
            numeric_cols.append(str(c))
        else:
            cat_cols.append(str(c))

    X_num = pd.DataFrame(index=df0.index)
    if numeric_cols:
        X_num = df0[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X_cat = pd.DataFrame(index=df0.index)
    cat_meta: Dict[str, Any] = {}
    if cat_cols:
        limited = {}
        for c in cat_cols:
            s = df0[c].astype(str).fillna("")
            vc = s.value_counts(dropna=False)
            top = list(vc.index[:40])
            s2 = s.where(s.isin(top), "__OTHER__")
            limited[c] = s2
            cat_meta[c] = {"unique": int(vc.shape[0]), "kept": min(40, int(vc.shape[0]))}
        X_cat = pd.get_dummies(pd.DataFrame(limited), prefix_sep="=", dummy_na=False)

    X_all = pd.concat([X_num, X_cat], axis=1)
    if X_all.shape[1] < 2:
        return {"ok": False, "error": "not_enough_features", "n_features": int(X_all.shape[1])}

    Xv = X_all.astype(np.float32).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xv)

    if int(Xs.shape[0]) < 3:
        return {"ok": False, "error": "not_enough_samples", "n_samples": int(Xs.shape[0])}
    n_components = int(min(10, Xs.shape[1], int(Xs.shape[0]) - 1))
    if n_components < 2:
        return {"ok": False, "error": "not_enough_components", "n_components": int(n_components)}
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(Xs)
    evr = getattr(pca, "explained_variance_ratio_", None)
    if evr is None:
        return {"ok": False, "error": "pca_failed"}
    evr_list = [float(x) for x in evr.tolist()]

    cum = []
    s = 0.0
    for x in evr_list:
        s += float(x)
        cum.append(float(s))
    k_use = 0
    for i, v in enumerate(cum):
        if v >= 0.8:
            k_use = i + 1
            break
    if k_use <= 0:
        k_use = min(5, len(evr_list))

    feat_names = [str(x) for x in list(X_all.columns)]
    comps = getattr(pca, "components_", None)
    if comps is None:
        return {"ok": False, "error": "pca_components_missing"}
    comps = np.asarray(comps)[:k_use, :]
    weights = np.asarray(evr_list[:k_use], dtype=np.float32).reshape(-1, 1)
    scores = np.sum(np.abs(comps) * weights, axis=0)

    origin_scores: Dict[str, float] = {}
    for fname, sc in zip(feat_names, scores.tolist()):
        orig = fname.split("=", 1)[0]
        origin_scores[orig] = float(origin_scores.get(orig, 0.0) + float(sc))

    top = sorted(origin_scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
    top_cols = [k for k, _ in top[:12]]
    return {
        "ok": True,
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "cat_meta": cat_meta,
        "feature_names": feat_names,
        "top_original_columns": top[:20],
        "selected_columns": top_cols,
        "explained_variance_ratio": evr_list,
        "cumulative_explained_variance": cum,
        "Z": Z,
    }


def build_tabular_profile(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    cols: Dict[str, Dict[str, Any]] = {}
    for r in records[:5000]:
        if not isinstance(r, dict):
            continue
        for k, v in r.items():
            kk = str(k)
            c = cols.setdefault(kk, {"name": kk, "n": 0, "n_num": 0, "n_str": 0, "examples": []})
            c["n"] += 1
            fv = _try_float(v)
            if fv is not None:
                c["n_num"] += 1
            else:
                if _safe_str(v):
                    c["n_str"] += 1
            if len(c["examples"]) < 5 and _safe_str(v):
                c["examples"].append(_safe_str(v)[:60])

    columns = list(cols.values())
    columns.sort(key=lambda x: (-int(x.get("n", 0)), str(x.get("name", ""))))
    return {"n_records": len(records), "columns": columns[:60]}


def _discover_numeric_columns(df: "pd.DataFrame", max_cols: int = 20) -> List[str]:
    numeric_cols: List[str] = []
    if pd is None or df is None or df.empty:
        return numeric_cols
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        valid = int(s.notna().sum())
        if valid >= 8:
            numeric_cols.append(str(c))
    numeric_cols = sorted(
        numeric_cols,
        key=lambda c: (
            -int(pd.to_numeric(df[c], errors="coerce").notna().sum()),
            str(c),
        ),
    )
    return numeric_cols[: int(max_cols)]


def _discover_categorical_columns(df: "pd.DataFrame", max_cols: int = 10) -> List[str]:
    cat_cols: List[str] = []
    if pd is None or df is None or df.empty:
        return cat_cols
    for c in df.columns:
        s_num = pd.to_numeric(df[c], errors="coerce")
        valid_num = int(s_num.notna().sum())
        if valid_num >= max(8, int(len(df) * 0.6)):
            continue
        s = df[c].astype(str).fillna("")
        non_empty = int((s.str.strip() != "").sum())
        nunique = int(s.nunique(dropna=True))
        if non_empty >= 8 and 2 <= nunique <= 20:
            cat_cols.append(str(c))
    cat_cols = sorted(cat_cols, key=lambda c: (int(df[c].astype(str).nunique(dropna=True)), str(c)))
    return cat_cols[: int(max_cols)]


def _safe_json_value(v: Any) -> Any:
    try:
        if v is None:
            return None
        if isinstance(v, (str, int, bool)):
            return v
        if isinstance(v, float):
            if math.isfinite(v):
                return float(v)
            return None
        return str(v)
    except Exception:
        return None


def _cohens_d(x1: List[float], x2: List[float]) -> Optional[float]:
    try:
        if np is None:
            return None
        a = np.asarray(x1, dtype=np.float64)
        b = np.asarray(x2, dtype=np.float64)
        if a.size < 2 or b.size < 2:
            return None
        v1 = float(np.var(a, ddof=1))
        v2 = float(np.var(b, ddof=1))
        pooled_num = ((a.size - 1) * v1) + ((b.size - 1) * v2)
        pooled_den = float(a.size + b.size - 2)
        if pooled_den <= 0:
            return None
        pooled = math.sqrt(max(pooled_num / pooled_den, 0.0))
        if pooled <= 0:
            return None
        return float((float(np.mean(a)) - float(np.mean(b))) / pooled)
    except Exception:
        return None


def _rank_biserial_from_u(u_value: float, n1: int, n2: int) -> Optional[float]:
    try:
        if n1 <= 0 or n2 <= 0:
            return None
        return float((2.0 * float(u_value)) / float(n1 * n2) - 1.0)
    except Exception:
        return None


def _eta_squared(groups: List[List[float]]) -> Optional[float]:
    try:
        clean = []
        for g in groups:
            vals = [float(x) for x in g if x is not None and math.isfinite(float(x))]
            if vals:
                clean.append(vals)
        if len(clean) < 2:
            return None
        all_vals = [x for g in clean for x in g]
        if len(all_vals) < 3:
            return None
        grand_mean = float(sum(all_vals) / len(all_vals))
        ss_between = 0.0
        ss_total = 0.0
        for g in clean:
            m = float(sum(g) / len(g))
            ss_between += float(len(g)) * ((m - grand_mean) ** 2)
            for x in g:
                ss_total += (float(x) - grand_mean) ** 2
        if ss_total <= 0:
            return None
        return float(ss_between / ss_total)
    except Exception:
        return None


def _cramers_v_from_table(table: "pd.DataFrame", chi2_value: float) -> Optional[float]:
    try:
        if pd is None or table is None:
            return None
        n = float(table.to_numpy().sum())
        if n <= 0:
            return None
        r, k = table.shape
        denom = min(r - 1, k - 1)
        if denom <= 0:
            return None
        return float(math.sqrt(max(float(chi2_value) / (n * denom), 0.0)))
    except Exception:
        return None


def _sig_label(p_value: Optional[float]) -> str:
    try:
        if p_value is None or not math.isfinite(float(p_value)):
            return "NA"
        p = float(p_value)
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"
    except Exception:
        return "NA"


def _sig_interpretation(p_value: Optional[float]) -> str:
    try:
        if p_value is None or not math.isfinite(float(p_value)):
            return "significance unavailable"
        return "statistically significant" if float(p_value) < 0.05 else "not statistically significant"
    except Exception:
        return "significance unavailable"


def _format_inferential_test_summary(test: Dict[str, Any]) -> str:
    try:
        ttype = str(test.get("type") or "").strip()
        pval = test.get("p_value")
        stat = test.get("statistic")
        sig = str(test.get("significance") or "NA").strip() or "NA"
        eff = test.get("effect_size") or {}
        eff_name = str(eff.get("name") or "").strip()
        eff_val = eff.get("value")
        group_col = str(test.get("group_column") or "").strip()
        value_col = str(test.get("value_column") or "").strip()
        groups = [str(x).strip() for x in (test.get("groups") or []) if str(x).strip()]

        if ttype == "t_test":
            prefix = f"t-test on {value_col} across {group_col}"
        elif ttype == "mann_whitney_u":
            prefix = f"Mann-Whitney U on {value_col} across {group_col}"
        elif ttype == "anova":
            prefix = f"ANOVA on {value_col} across {group_col}"
        elif ttype == "chi_square":
            prefix = f"chi-square association between {group_col} and {value_col}"
        else:
            prefix = ttype or "inferential test"

        parts = [prefix]
        if groups:
            if len(groups) == 2:
                parts.append(f"groups={groups[0]} vs {groups[1]}")
            else:
                parts.append(f"groups={', '.join(groups[:6])}")
        if stat is not None and math.isfinite(float(stat)):
            parts.append(f"stat={float(stat):.3f}")
        if pval is not None and math.isfinite(float(pval)):
            parts.append(f"p={float(pval):.4g}")
        parts.append(f"result={_sig_interpretation(pval)}")
        parts.append(f"sig={sig}")
        if eff_name and eff_val is not None and math.isfinite(float(eff_val)):
            parts.append(f"{eff_name}={float(eff_val):.3f}")
        return ", ".join(parts)
    except Exception:
        return str(test or "")


def default_analysis_plan(profile: Dict[str, Any]) -> Dict[str, Any]:
    cols = profile.get("columns") or []
    numeric = [c["name"] for c in cols if float(c.get("n_num", 0)) >= 8]
    categorical = [c["name"] for c in cols if float(c.get("n_str", 0)) >= 8]

    tasks = []
    if len(numeric) >= 2:
        tasks.append({"type": "clustering", "method": "pca_kmeans", "k": 3, "metrics": ["silhouette"], "plot": "pca_scatter"})
        target = numeric[0]
        features = numeric[1: min(len(numeric), 7)]
        if features:
            tasks.append({"type": "regression", "target": target, "features": features, "models": ["ridge", "rf"], "metrics": ["r2", "mae"], "plot": "pred_vs_true"})
    if categorical and numeric:
        tasks.append({"type": "classification", "label": categorical[0], "features": numeric[: min(len(numeric), 8)], "models": ["logreg", "rf"], "metrics": ["accuracy", "f1_macro"], "plot": "confusion"})

    return {"tasks": tasks, "mode": "default"}


def plan_with_llm(llm_service, model: str, materials_text: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    sys_prompt = (
        "You are a data scientist building a SAFE analysis plan for tabular clinical/imaging data.\n"
        "Return ONLY a JSON object.\n"
        "Constraints:\n"
        "- Use ONLY columns that exist in the provided schema.\n"
        "- Propose at most 3 tasks total.\n"
        "- Task types allowed: regression, classification, clustering.\n"
        "- Metrics allowed: r2, mae, rmse, accuracy, f1_macro, precision_macro, recall_macro, silhouette.\n"
        "- Plots allowed: pca_scatter, pred_vs_true, confusion, feature_importance.\n"
        "- If no suitable task, return {\"tasks\":[]}.\n"
    )
    prompt = (
        f"{sys_prompt}\n\n"
        f"[MATERIALS_EXCERPT]\n{str(materials_text or '')[:1800]}\n\n"
        f"[TABULAR_SCHEMA_PROFILE_JSON]\n{json.dumps(profile, ensure_ascii=False)[:4000]}\n"
    )
    res = llm_service.generate(model, prompt, timeout=600)
    if not res.get("success"):
        return {"tasks": [], "mode": "llm_failed", "error": res.get("error")}
    obj = _extract_json_object(res.get("response", ""))
    if not obj:
        return {"tasks": [], "mode": "llm_parse_failed"}
    return obj


def _validate_plan(plan: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    cols = {c.get("name") for c in (profile.get("columns") or []) if isinstance(c, dict) and c.get("name")}
    tasks_in = plan.get("tasks") if isinstance(plan, dict) else None
    if not isinstance(tasks_in, list):
        return {"tasks": [], "mode": "invalid"}

    allowed_types = {"regression", "classification", "clustering"}
    allowed_metrics = {"r2", "mae", "rmse", "accuracy", "f1_macro", "precision_macro", "recall_macro", "silhouette"}
    allowed_plots = {"pca_scatter", "pred_vs_true", "confusion", "feature_importance"}

    tasks = []
    for t in tasks_in[:3]:
        if not isinstance(t, dict):
            continue
        tt = str(t.get("type") or "").strip().lower()
        if tt not in allowed_types:
            continue
        plot = str(t.get("plot") or "").strip().lower()
        if plot and plot not in allowed_plots:
            plot = ""
        metrics = t.get("metrics") or []
        if not isinstance(metrics, list):
            metrics = []
        metrics = [str(m).strip().lower() for m in metrics if str(m).strip().lower() in allowed_metrics]

        if tt == "regression":
            target = str(t.get("target") or "").strip()
            feats = t.get("features") or []
            if not target or target not in cols or not isinstance(feats, list):
                continue
            feats = [str(x).strip() for x in feats if str(x).strip() in cols][:12]
            if not feats:
                continue
            tasks.append({"type": "regression", "target": target, "features": feats, "models": ["ridge", "rf"], "metrics": metrics or ["r2", "mae"], "plot": plot or "pred_vs_true"})
        elif tt == "classification":
            label = str(t.get("label") or "").strip()
            feats = t.get("features") or []
            if not label or label not in cols or not isinstance(feats, list):
                continue
            feats = [str(x).strip() for x in feats if str(x).strip() in cols][:12]
            if not feats:
                continue
            tasks.append({"type": "classification", "label": label, "features": feats, "models": ["logreg", "rf"], "metrics": metrics or ["accuracy", "f1_macro"], "plot": plot or "confusion"})
        else:
            k = int(t.get("k") or 3)
            if k < 2 or k > 8:
                k = 3
            tasks.append({"type": "clustering", "method": "pca_kmeans", "k": k, "metrics": metrics or ["silhouette"], "plot": plot or "pca_scatter"})

    return {"tasks": tasks, "mode": str(plan.get("mode") or "llm").strip() or "llm"}


def _setup_plot_style():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        import matplotlib.font_manager as fm

        rcParams["axes.unicode_minus"] = False
        candidates = ["SimHei", "Microsoft YaHei", "PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC", "Segoe UI", "Tahoma"]
        
        # 不要因为字体找不到就直接 crash，至少保留基础绘图能力
        try:
            available = {f.name for f in fm.fontManager.ttflist}
            for name in candidates:
                if name in available:
                    rcParams["font.sans-serif"] = [name]
                    break
        except Exception:
            pass
            
        return plt
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Sklearn pipeline failed to init matplotlib: {e}")
        return None


def run_llm_guided_sklearn_pipeline(
    *,
    llm_service,
    model: str,
    materials_text: str,
    project_storage_path: str,
    workspace: str,
    out_dir: str,
) -> Dict[str, Any]:
    load_res = load_project_tabular_records(project_storage_path, workspace)
    records = load_res.get("records") or []
    sources = load_res.get("sources") or []
    warnings = load_res.get("warnings") or []
    if not records:
        return {"success": False, "error": "no_tabular_records", "warnings": warnings, "artifacts": [], "summary": ""}

    if pd is None or np is None:
        try:
            import numpy as _np
            import pandas as _pd
            globals()["np"] = _np
            globals()["pd"] = _pd
        except Exception:
            return {"success": False, "error": "missing_pandas_or_numpy", "warnings": warnings, "artifacts": [], "summary": ""}

    os.makedirs(out_dir, exist_ok=True)
    artifacts: List[str] = []

    df_full = pd.DataFrame(records)
    for c in list(df_full.columns):
        if str(c).startswith("Unnamed"):
            df_full = df_full.drop(columns=[c])

    plt = _setup_plot_style()

    def _rel(p: str) -> str:
        return os.path.relpath(p, project_storage_path).replace("\\", "/")

    numeric_cols_all = _discover_numeric_columns(df_full, max_cols=20)
    categorical_cols_all = _discover_categorical_columns(df_full, max_cols=10)

    pca_res = _prepare_pca_inputs(df_full)
    selected_cols: List[str] = []
    if isinstance(pca_res, dict) and pca_res.get("ok"):
        selected_cols = [str(x) for x in (pca_res.get("selected_columns") or []) if str(x).strip()]

    df = df_full
    if selected_cols:
        kept = [c for c in selected_cols if c in df_full.columns]
        if kept:
            df = df_full[kept].copy()
            records = df.to_dict(orient="records")

    profile = build_tabular_profile(records)
    llm_plan_raw = plan_with_llm(llm_service, model, materials_text, profile)
    validated = _validate_plan(llm_plan_raw, profile)
    if not validated.get("tasks"):
        validated = default_analysis_plan(profile)

    report: Dict[str, Any] = {
        "plan": validated,
        "sources": sources,
        "warnings": warnings,
        "profile": profile,
        "dataset_overview": {
            "n_rows": int(df_full.shape[0]),
            "n_columns": int(df_full.shape[1]),
            "numeric_columns": numeric_cols_all,
            "categorical_columns": categorical_cols_all,
        },
    }

    if numeric_cols_all:
        try:
            desc_rows = []
            for c in numeric_cols_all[:12]:
                s = pd.to_numeric(df_full[c], errors="coerce")
                if int(s.notna().sum()) < 8:
                    continue
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))
                desc_rows.append(
                    {
                        "column": str(c),
                        "n": int(s.notna().sum()),
                        "missing": int(s.isna().sum()),
                        "missing_ratio": float(s.isna().mean()),
                        "mean": float(s.mean()),
                        "std": float(s.std(ddof=0)) if int(s.notna().sum()) > 1 else 0.0,
                        "median": float(s.median()),
                        "q1": q1,
                        "q3": q3,
                        "iqr": float(q3 - q1),
                        "min": float(s.min()),
                        "max": float(s.max()),
                        "skew": float(s.skew()) if int(s.notna().sum()) > 2 else 0.0,
                        "kurtosis": float(s.kurt()) if int(s.notna().sum()) > 3 else 0.0,
                    }
                )
            if desc_rows:
                report["descriptive_stats"] = desc_rows
        except Exception:
            pass

    if len(numeric_cols_all) >= 2:
        try:
            corr_df = df_full[numeric_cols_all[:12]].apply(pd.to_numeric, errors="coerce")
            corr = corr_df.corr(method="pearson").replace([np.inf, -np.inf], np.nan)
            corr_round = corr.round(4).fillna(0.0)
            report["correlation_matrix"] = corr_round.to_dict()
            pairs = []
            cols_corr = list(corr.columns)
            for i in range(len(cols_corr)):
                for j in range(i + 1, len(cols_corr)):
                    v = corr.iloc[i, j]
                    if pd.isna(v):
                        continue
                    pairs.append(
                        {
                            "x": str(cols_corr[i]),
                            "y": str(cols_corr[j]),
                            "pearson_r": float(v),
                            "abs_r": float(abs(v)),
                        }
                    )
            pairs = sorted(pairs, key=lambda x: (-float(x.get("abs_r", 0.0)), str(x.get("x", "")), str(x.get("y", ""))))
            report["top_correlations"] = pairs[:20]

            if plt is not None:
                fig = plt.figure(figsize=(6.8, 5.4))
                ax = fig.add_subplot(111)
                im = ax.imshow(corr_round.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
                ax.set_xticks(range(len(corr_round.columns)))
                ax.set_yticks(range(len(corr_round.index)))
                ax.set_xticklabels(list(corr_round.columns), rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(list(corr_round.index), fontsize=8)
                ax.set_title("Pearson Correlation Heatmap")
                for i in range(len(corr_round.index)):
                    for j in range(len(corr_round.columns)):
                        ax.text(j, i, f"{float(corr_round.iloc[i, j]):.2f}", ha="center", va="center", fontsize=6, color="#111827")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                out_png = os.path.join(out_dir, "agent_sklearn_correlation_heatmap.png")
                fig.savefig(out_png, dpi=170)
                artifacts.append(_rel(out_png))

                strongest = pairs[0] if pairs else None
                if strongest:
                    x_col = str(strongest.get("x") or "")
                    y_col = str(strongest.get("y") or "")
                    scatter_df = df_full[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna().iloc[:3000]
                    if len(scatter_df) >= 8:
                        fig = plt.figure(figsize=(5.0, 4.0))
                        ax = fig.add_subplot(111)
                        ax.scatter(scatter_df[x_col], scatter_df[y_col], s=10, alpha=0.65, color="#2563eb")
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        ax.set_title(f"Strongest Pair Scatter (r={float(strongest.get('pearson_r', 0.0)):.3f})")
                        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
                        fig.tight_layout()
                        out_png = os.path.join(out_dir, "agent_sklearn_scatter_top_pair.png")
                        fig.savefig(out_png, dpi=170)
                        artifacts.append(_rel(out_png))
        except Exception:
            pass

    if numeric_cols_all and plt is not None:
        try:
            hist_cols = numeric_cols_all[:4]
            if hist_cols:
                fig, axes = plt.subplots(len(hist_cols), 1, figsize=(6.0, 2.6 * len(hist_cols)))
                if len(hist_cols) == 1:
                    axes = [axes]
                for ax, c in zip(axes, hist_cols):
                    s = pd.to_numeric(df_full[c], errors="coerce").dropna().iloc[:4000]
                    if len(s) == 0:
                        continue
                    ax.hist(s.values, bins=min(30, max(10, int(len(s) ** 0.5))), color="#60a5fa", alpha=0.85, edgecolor="white")
                    ax.set_title(f"Distribution: {c}")
                    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
                fig.tight_layout()
                out_png = os.path.join(out_dir, "agent_sklearn_distribution_panels.png")
                fig.savefig(out_png, dpi=170)
                artifacts.append(_rel(out_png))
        except Exception:
            pass

        try:
            box_cols = numeric_cols_all[:8]
            if box_cols:
                box_df = df_full[box_cols].apply(pd.to_numeric, errors="coerce")
                fig = plt.figure(figsize=(7.2, 4.6))
                ax = fig.add_subplot(111)
                box_df.boxplot(ax=ax, rot=35)
                ax.set_title("Boxplot of Numeric Variables")
                ax.set_ylabel("value")
                fig.tight_layout()
                out_png = os.path.join(out_dir, "agent_sklearn_boxplot_numeric.png")
                fig.savefig(out_png, dpi=170)
                artifacts.append(_rel(out_png))
        except Exception:
            pass

    if numeric_cols_all and categorical_cols_all:
        try:
            best_group = None
            for cat_col in categorical_cols_all[:6]:
                cats = df_full[cat_col].astype(str).fillna("")
                vc = cats.value_counts()
                keep = [str(x) for x in vc.index[:6] if str(x).strip()]
                if len(keep) < 2:
                    continue
                for num_col in numeric_cols_all[:8]:
                    tmp = pd.DataFrame({
                        "group": cats.where(cats.isin(keep), "__OTHER__"),
                        "value": pd.to_numeric(df_full[num_col], errors="coerce"),
                    }).dropna()
                    grp = tmp.groupby("group")["value"]
                    if int(tmp.shape[0]) < 12 or int(grp.ngroups) < 2:
                        continue
                    summary_rows = []
                    means = []
                    ns = []
                    for g, s in grp:
                        summary_rows.append(
                            {
                                "group": str(g),
                                "n": int(s.notna().sum()),
                                "mean": float(s.mean()),
                                "std": float(s.std(ddof=0)) if int(s.notna().sum()) > 1 else 0.0,
                                "median": float(s.median()),
                            }
                        )
                        means.append(float(s.mean()))
                        ns.append(int(s.notna().sum()))
                    if len(summary_rows) >= 2:
                        spread = max(means) - min(means) if means else 0.0
                        best_group = {
                            "group_column": str(cat_col),
                            "value_column": str(num_col),
                            "n_groups": len(summary_rows),
                            "groups": summary_rows,
                            "mean_spread": float(spread),
                        }
                        break
                if best_group:
                    break
            if best_group:
                report["group_comparison"] = best_group
                if plt is not None:
                    cat_col = str(best_group.get("group_column") or "")
                    num_col = str(best_group.get("value_column") or "")
                    cats = df_full[cat_col].astype(str).fillna("")
                    keep = [str(x.get("group") or "") for x in (best_group.get("groups") or [])][:6]
                    tmp = pd.DataFrame({
                        "group": cats.where(cats.isin(keep), "__OTHER__"),
                        "value": pd.to_numeric(df_full[num_col], errors="coerce"),
                    }).dropna()
                    if len(tmp) >= 8:
                        ordered = [g for g in keep if g in set(tmp["group"].astype(str).tolist())]
                        data = [tmp.loc[tmp["group"] == g, "value"].values for g in ordered]
                        fig = plt.figure(figsize=(7.0, 4.4))
                        ax = fig.add_subplot(111)
                        ax.boxplot(data, labels=ordered, patch_artist=True)
                        ax.set_title(f"Grouped Boxplot: {num_col} by {cat_col}")
                        ax.set_xlabel(cat_col)
                        ax.set_ylabel(num_col)
                        for tick in ax.get_xticklabels():
                            tick.set_rotation(25)
                        fig.tight_layout()
                        out_png = os.path.join(out_dir, "agent_sklearn_group_boxplot.png")
                        fig.savefig(out_png, dpi=170)
                        artifacts.append(_rel(out_png))
        except Exception:
            pass

    inferential_tests: List[Dict[str, Any]] = []
    if report.get("group_comparison"):
        try:
            g = report.get("group_comparison") or {}
            cat_col = str(g.get("group_column") or "")
            num_col = str(g.get("value_column") or "")
            if cat_col and num_col:
                cats = df_full[cat_col].astype(str).fillna("")
                keep = [str(x.get("group") or "") for x in (g.get("groups") or [])][:6]
                tmp = pd.DataFrame({
                    "group": cats.where(cats.isin(keep), "__OTHER__"),
                    "value": pd.to_numeric(df_full[num_col], errors="coerce"),
                }).dropna()
                ordered = [grp for grp in keep if grp in set(tmp["group"].astype(str).tolist())]
                group_arrays = [tmp.loc[tmp["group"] == grp, "value"].astype(float).tolist() for grp in ordered]
                valid_groups = [(grp, arr) for grp, arr in zip(ordered, group_arrays) if len(arr) >= 3]
                if len(valid_groups) == 2:
                    (g1, a1), (g2, a2) = valid_groups
                    if ttest_ind is not None:
                        t_res = ttest_ind(a1, a2, equal_var=False, nan_policy="omit")
                        inferential_tests.append(
                            {
                                "type": "t_test",
                                "group_column": cat_col,
                                "value_column": num_col,
                                "groups": [g1, g2],
                                "n1": len(a1),
                                "n2": len(a2),
                                "statistic": float(t_res.statistic),
                                "p_value": float(t_res.pvalue),
                                "significance": _sig_label(float(t_res.pvalue)),
                                "effect_size": {"name": "cohens_d", "value": _cohens_d(a1, a2)},
                            }
                        )
                    if mannwhitneyu is not None:
                        mw_res = mannwhitneyu(a1, a2, alternative="two-sided")
                        inferential_tests.append(
                            {
                                "type": "mann_whitney_u",
                                "group_column": cat_col,
                                "value_column": num_col,
                                "groups": [g1, g2],
                                "n1": len(a1),
                                "n2": len(a2),
                                "statistic": float(mw_res.statistic),
                                "p_value": float(mw_res.pvalue),
                                "significance": _sig_label(float(mw_res.pvalue)),
                                "effect_size": {
                                    "name": "rank_biserial",
                                    "value": _rank_biserial_from_u(float(mw_res.statistic), len(a1), len(a2)),
                                },
                            }
                        )
                elif len(valid_groups) >= 3 and f_oneway is not None:
                    labels = [x[0] for x in valid_groups[:6]]
                    arrays = [x[1] for x in valid_groups[:6]]
                    anova_res = f_oneway(*arrays)
                    inferential_tests.append(
                        {
                            "type": "anova",
                            "group_column": cat_col,
                            "value_column": num_col,
                            "groups": labels,
                            "group_sizes": {str(lb): len(arr) for lb, arr in zip(labels, arrays)},
                            "statistic": float(anova_res.statistic),
                            "p_value": float(anova_res.pvalue),
                            "significance": _sig_label(float(anova_res.pvalue)),
                            "effect_size": {"name": "eta_squared", "value": _eta_squared(arrays)},
                        }
                    )
        except Exception:
            pass

    if len(categorical_cols_all) >= 2:
        try:
            best_cat_pair = None
            for i in range(len(categorical_cols_all[:6])):
                for j in range(i + 1, len(categorical_cols_all[:6])):
                    c1 = str(categorical_cols_all[i])
                    c2 = str(categorical_cols_all[j])
                    s1 = df_full[c1].astype(str).fillna("")
                    s2 = df_full[c2].astype(str).fillna("")
                    top1 = list(s1.value_counts().index[:6])
                    top2 = list(s2.value_counts().index[:6])
                    tmp = pd.DataFrame({
                        c1: s1.where(s1.isin(top1), "__OTHER__"),
                        c2: s2.where(s2.isin(top2), "__OTHER__"),
                    })
                    table = pd.crosstab(tmp[c1], tmp[c2])
                    if int(table.shape[0]) < 2 or int(table.shape[1]) < 2:
                        continue
                    n_obs = int(table.to_numpy().sum())
                    if n_obs < 20:
                        continue
                    best_cat_pair = (c1, c2, table)
                    break
                if best_cat_pair:
                    break
            if best_cat_pair and chi2_contingency is not None:
                c1, c2, table = best_cat_pair
                chi2_stat, p_value, dof, expected = chi2_contingency(table)
                inferential_tests.append(
                    {
                        "type": "chi_square",
                        "group_column": c1,
                        "value_column": c2,
                        "table_shape": [int(table.shape[0]), int(table.shape[1])],
                        "n": int(table.to_numpy().sum()),
                        "statistic": float(chi2_stat),
                        "p_value": float(p_value),
                        "dof": int(dof),
                        "significance": _sig_label(float(p_value)),
                        "effect_size": {"name": "cramers_v", "value": _cramers_v_from_table(table, float(chi2_stat))},
                    }
                )
                report["categorical_association"] = {
                    "columns": [c1, c2],
                    "contingency_table": {str(idx): {str(k): int(v) for k, v in row.items()} for idx, row in table.to_dict(orient="index").items()},
                }
                if plt is not None:
                    fig = plt.figure(figsize=(6.6, 4.8))
                    ax = fig.add_subplot(111)
                    im = ax.imshow(table.values, cmap="YlOrRd")
                    ax.set_xticks(range(len(table.columns)))
                    ax.set_yticks(range(len(table.index)))
                    ax.set_xticklabels([str(x) for x in list(table.columns)], rotation=35, ha="right", fontsize=8)
                    ax.set_yticklabels([str(x) for x in list(table.index)], fontsize=8)
                    ax.set_xlabel(c2)
                    ax.set_ylabel(c1)
                    ax.set_title(f"Chi-square Heatmap ({_sig_label(float(p_value))}, p={float(p_value):.4g})")
                    for i in range(table.shape[0]):
                        for j in range(table.shape[1]):
                            ax.text(j, i, str(int(table.iloc[i, j])), ha="center", va="center", fontsize=7, color="#111827")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    out_png = os.path.join(out_dir, "agent_sklearn_chi_square_heatmap.png")
                    fig.savefig(out_png, dpi=170)
                    artifacts.append(_rel(out_png))
        except Exception:
            pass

    if inferential_tests and plt is not None:
        report["inferential_tests"] = inferential_tests
        report["inferential_summary"] = [_format_inferential_test_summary(x) for x in inferential_tests[:8]]
        try:
            n = min(len(inferential_tests), 8)
            fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
            for idx in range(n):
                ax = axes[0][idx]
                t = inferential_tests[idx]
                ttype = t.get("type", "")
                pval = t.get("p_value", 1.0)
                sig = t.get("significance", "ns")
                groups = t.get("groups") or []
                means = []
                stds = []
                if "group_sizes" in t:
                    group_sizes = t.get("group_sizes", {}) if isinstance(t.get("group_sizes"), dict) else {}
                    for g in groups:
                        sz = group_sizes.get(str(g), 0) or 0
                        means.append(float(sz))
                        stds.append(0.0)
                elif "n1" in t and "n2" in t:
                    means = [float(t.get("n1", 0)), float(t.get("n2", 0))]
                    stds = [0.0, 0.0]
                else:
                    means = [float(i + 1) for i in range(len(groups))]
                    stds = [0.0] * len(groups)
                bars = ax.bar(range(len(groups)), means, yerr=stds, capsize=4, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"][:len(groups)])
                ax.set_xticks(range(len(groups)))
                ax.set_xticklabels([str(g)[:12] for g in groups], rotation=25, fontsize=7)
                ax.set_ylabel(t.get("value_column", ""))
                ax.set_title(f"{ttype}\np={pval:.4g} {sig}", fontsize=9)
                for bar, m in zip(bars, means):
                    ax.text(bar.get_x() + bar.get_width() / 2, m, f"{m:.1f}", ha="center", va="bottom", fontsize=7)
            fig.tight_layout()
            fname = os.path.join(out_dir, "agent_sklearn_inferential_tests.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            artifacts.append(fname)
        except Exception:
            pass

    if isinstance(pca_res, dict) and pca_res.get("ok"):
        report["pca"] = {
            "n_rows": int(df_full.shape[0]),
            "n_features_onehot": int(len(pca_res.get("feature_names") or [])),
            "explained_variance_ratio": pca_res.get("explained_variance_ratio") or [],
            "cumulative_explained_variance": pca_res.get("cumulative_explained_variance") or [],
            "top_original_columns": pca_res.get("top_original_columns") or [],
            "selected_columns": selected_cols,
            "categorical_meta": pca_res.get("cat_meta") or {},
        }

        if plt is not None:
            try:
                evr = [float(x) for x in (pca_res.get("explained_variance_ratio") or [])]
                cum = [float(x) for x in (pca_res.get("cumulative_explained_variance") or [])]
                if evr and cum:
                    fig = plt.figure(figsize=(5.4, 3.6))
                    ax = fig.add_subplot(111)
                    xs = list(range(1, len(evr) + 1))
                    ax.bar(xs, evr, color="#93c5fd", alpha=0.85, label="explained variance")
                    ax.plot(xs, cum, marker="o", linewidth=1.6, color="#1d4ed8", label="cumulative")
                    ax.axhline(0.8, color="#64748b", linewidth=1.2, linestyle="--")
                    ax.set_ylim(0.0, 1.02)
                    ax.set_xlabel("n_components")
                    ax.set_ylabel("explained variance")
                    ax.set_title("PCA Scree")
                    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
                    ax.legend(loc="lower right", frameon=False, fontsize=8)
                    fig.tight_layout()
                    out_png = os.path.join(out_dir, "agent_sklearn_pca_scree.png")
                    fig.savefig(out_png, dpi=170)
                    artifacts.append(_rel(out_png))
            except Exception:
                pass

            try:
                top_items = pca_res.get("top_original_columns") or []
                items = [(str(k), float(v)) for k, v in top_items[:15] if str(k).strip()]
                if items:
                    labels = [k for k, _ in items][::-1]
                    raw_vals = [max(0.0, v) for _, v in items][::-1]
                    total = float(sum(raw_vals)) if raw_vals else 0.0
                    vals = [(v / total * 100.0) if total > 0 else 0.0 for v in raw_vals]
                    fig = plt.figure(figsize=(6.8, 4.2))
                    ax = fig.add_subplot(111)
                    bars = ax.barh(labels, vals, color="#3b82f6", alpha=0.85)
                    ax.set_xlabel("relative importance (%)")
                    ax.set_title("PCA Top Factors (relative)")
                    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.4)
                    xmax = max(vals) if vals else 0.0
                    ax.set_xlim(0.0, xmax * 1.18 + 1e-6)
                    for b, v in zip(bars, vals):
                        try:
                            ax.text(float(b.get_width()) + (xmax * 0.02 + 0.1), float(b.get_y()) + float(b.get_height()) / 2.0, f"{float(v):.1f}%", va="center", fontsize=8)
                        except Exception:
                            pass
                    fig.tight_layout()
                    out_png = os.path.join(out_dir, "agent_sklearn_pca_top_factors.png")
                    fig.savefig(out_png, dpi=170)
                    artifacts.append(_rel(out_png))
            except Exception:
                pass
    elif isinstance(pca_res, dict) and (pca_res.get("ok") is False):
        report["pca"] = {k: pca_res.get(k) for k in ["ok", "error", "n_samples", "n_features", "n_components"] if k in pca_res}

    for task in validated.get("tasks") or []:
        ttype = task.get("type")
        try:
            if ttype == "clustering" and PCA and StandardScaler and KMeans and silhouette_score and plt is not None:
                cols = [c for c in df.columns if str(c) in [x.get("name") for x in (profile.get("columns") or [])]]
                num_cols = []
                for c in cols:
                    s = pd.to_numeric(df[c], errors="coerce")
                    if int(s.notna().sum()) >= 8:
                        num_cols.append(str(c))
                if len(num_cols) < 2:
                    continue
                X = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).iloc[:5000]
                Xs = StandardScaler().fit_transform(X.values)
                Z = PCA(n_components=2).fit_transform(Xs)
                k = int(task.get("k") or 3)
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(Z)
                sil = float(silhouette_score(Z, labels)) if len(set(labels.tolist())) >= 2 else float("nan")
                unique_labels, counts = np.unique(labels, return_counts=True)
                report.setdefault("clustering", []).append(
                    {
                        "method": "pca_kmeans",
                        "k": k,
                        "silhouette": sil,
                        "inertia": float(getattr(km, "inertia_", 0.0)),
                        "cluster_sizes": {str(int(lb)): int(ct) for lb, ct in zip(unique_labels.tolist(), counts.tolist())},
                        "features": num_cols,
                    }
                )
                fig = plt.figure(figsize=(4.6, 3.5))
                ax = fig.add_subplot(111)
                ax.scatter(Z[:, 0], Z[:, 1], s=7, alpha=0.75, c=labels)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title(f"PCA+KMeans (k={k}, silhouette={sil:.3f})")
                fig.tight_layout()
                out_png = os.path.join(out_dir, "agent_sklearn_cluster.png")
                fig.savefig(out_png, dpi=170)
                artifacts.append(_rel(out_png))

            if ttype == "regression" and train_test_split and StandardScaler and Ridge and RandomForestRegressor and r2_score and mean_absolute_error and plt is not None:
                y_col = str(task.get("target") or "")
                x_cols = [str(x) for x in (task.get("features") or [])]
                if not y_col or not x_cols:
                    continue
                X = df[x_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                y = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0)
                if len(X) < 20:
                    continue
                X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                models = [("ridge", Ridge(alpha=1.0, random_state=42)), ("rf", RandomForestRegressor(n_estimators=200, random_state=42))]
                best = None
                for name, m in models:
                    try:
                        m.fit(X_train, y_train)
                        pred = m.predict(X_test)
                        r2 = float(r2_score(y_test, pred))
                        mae = float(mean_absolute_error(y_test, pred))
                        rmse = float(math.sqrt(mean_squared_error(y_test, pred))) if mean_squared_error is not None else float("nan")
                        if best is None or r2 > best["r2"]:
                            best = {"model": name, "r2": r2, "mae": mae, "rmse": rmse, "pred": pred, "estimator": m}
                    except Exception:
                        continue
                if not best:
                    continue
                report.setdefault("regression", []).append(
                    {
                        "target": y_col,
                        "features": x_cols,
                        "model": best["model"],
                        "r2": best["r2"],
                        "mae": best["mae"],
                        "rmse": best["rmse"],
                    }
                )
                fig = plt.figure(figsize=(4.6, 3.5))
                ax = fig.add_subplot(111)
                ax.scatter(y_test, best["pred"], s=8, alpha=0.65)
                ax.set_xlabel("y_true")
                ax.set_ylabel("y_pred")
                ax.set_title(f"Regression {best['model']} (R2={best['r2']:.3f}, RMSE={best['rmse']:.3f})")
                fig.tight_layout()
                out_png = os.path.join(out_dir, "agent_sklearn_regression.png")
                fig.savefig(out_png, dpi=170)
                artifacts.append(_rel(out_png))

                if best["model"] == "rf":
                    try:
                        importances = getattr(best.get("estimator"), "feature_importances_", None)
                        if importances is not None:
                            items = sorted(
                                [(str(f), float(v)) for f, v in zip(x_cols, importances.tolist())],
                                key=lambda kv: (-float(kv[1]), str(kv[0])),
                            )[:12]
                            if items:
                                labels = [k for k, _ in items][::-1]
                                vals = [float(v) for _, v in items][::-1]
                                fig = plt.figure(figsize=(6.2, 4.0))
                                ax = fig.add_subplot(111)
                                ax.barh(labels, vals, color="#2563eb", alpha=0.85)
                                ax.set_title("Regression Feature Importance")
                                ax.set_xlabel("importance")
                                fig.tight_layout()
                                out_png = os.path.join(out_dir, "agent_sklearn_regression_feature_importance.png")
                                fig.savefig(out_png, dpi=170)
                                artifacts.append(_rel(out_png))
                    except Exception:
                        pass

            if ttype == "classification" and train_test_split and StandardScaler and LogisticRegression and RandomForestClassifier and accuracy_score and f1_score and plt is not None:
                label = str(task.get("label") or "")
                x_cols = [str(x) for x in (task.get("features") or [])]
                if not label or not x_cols:
                    continue
                X = df[x_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                y_raw = df[label].astype(str).fillna("")
                y = y_raw.astype("category").cat.codes.values
                if len(X) < 30:
                    continue
                X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42, stratify=y if len(set(y.tolist())) >= 2 else None)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                models = [("logreg", LogisticRegression(max_iter=1000)), ("rf", RandomForestClassifier(n_estimators=200, random_state=42))]
                best = None
                for name, m in models:
                    try:
                        m.fit(X_train, y_train)
                        pred = m.predict(X_test)
                        acc = float(accuracy_score(y_test, pred))
                        f1 = float(f1_score(y_test, pred, average="macro")) if len(set(y_test.tolist())) >= 2 else float("nan")
                        prec = float(precision_score(y_test, pred, average="macro", zero_division=0)) if precision_score is not None and len(set(y_test.tolist())) >= 2 else float("nan")
                        rec = float(recall_score(y_test, pred, average="macro", zero_division=0)) if recall_score is not None and len(set(y_test.tolist())) >= 2 else float("nan")
                        if best is None or acc > best["acc"]:
                            best = {"model": name, "acc": acc, "f1_macro": f1, "precision_macro": prec, "recall_macro": rec, "estimator": m}
                    except Exception:
                        continue
                if not best:
                    continue
                class_balance = {}
                try:
                    uniq, cnts = np.unique(y, return_counts=True)
                    class_balance = {str(int(u)): int(c) for u, c in zip(uniq.tolist(), cnts.tolist())}
                except Exception:
                    class_balance = {}
                report.setdefault("classification", []).append(
                    {
                        "label": label,
                        "features": x_cols,
                        "model": best["model"],
                        "accuracy": best["acc"],
                        "f1_macro": best["f1_macro"],
                        "precision_macro": best["precision_macro"],
                        "recall_macro": best["recall_macro"],
                        "class_balance": class_balance,
                    }
                )
                fig = plt.figure(figsize=(4.8, 3.8))
                ax = fig.add_subplot(111)
                ax.set_title(f"Classification {best['model']} (acc={best['acc']:.3f}, f1={best['f1_macro']:.3f})")

                if confusion_matrix is not None:
                    try:
                        m = None
                        for name, model_obj in models:
                            if name == best["model"]:
                                model_obj.fit(X_train, y_train)
                                pred = model_obj.predict(X_test)
                                m = confusion_matrix(y_test, pred)
                                break
                        if m is not None:
                            im = ax.imshow(m, cmap="Blues")
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("True")
                            for (i, j), v in np.ndenumerate(m):
                                ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=8)
                    except Exception:
                        ax.text(0.02, 0.90, "Confusion matrix unavailable", transform=ax.transAxes)
                else:
                    ax.text(0.02, 0.90, "confusion_matrix unavailable", transform=ax.transAxes)
                    ax.axis("off")

                fig.tight_layout()
                out_png = os.path.join(out_dir, "agent_sklearn_classification.png")
                fig.savefig(out_png, dpi=170)
                artifacts.append(_rel(out_png))

                if best["model"] == "rf":
                    try:
                        importances = getattr(best.get("estimator"), "feature_importances_", None)
                        if importances is not None:
                            items = sorted(
                                [(str(f), float(v)) for f, v in zip(x_cols, importances.tolist())],
                                key=lambda kv: (-float(kv[1]), str(kv[0])),
                            )[:12]
                            if items:
                                labels = [k for k, _ in items][::-1]
                                vals = [float(v) for _, v in items][::-1]
                                fig = plt.figure(figsize=(6.2, 4.0))
                                ax = fig.add_subplot(111)
                                ax.barh(labels, vals, color="#7c3aed", alpha=0.85)
                                ax.set_title("Classification Feature Importance")
                                ax.set_xlabel("importance")
                                fig.tight_layout()
                                out_png = os.path.join(out_dir, "agent_sklearn_classification_feature_importance.png")
                                fig.savefig(out_png, dpi=170)
                                artifacts.append(_rel(out_png))
                    except Exception:
                        pass
        except Exception:
            continue

    figure_ctx: List[Dict[str, Any]] = []
    try:
        pngs = [a for a in artifacts if str(a).lower().endswith(".png")]
        if pngs:
            pca_block = report.get("pca") if isinstance(report, dict) else None
            clustering_block = (report.get("clustering") or []) if isinstance(report, dict) else []
            regression_block = (report.get("regression") or []) if isinstance(report, dict) else []
            classification_block = (report.get("classification") or []) if isinstance(report, dict) else []
            for p in pngs:
                kind = "figure"
                ctx = {}
                fn = os.path.basename(str(p))
                if "pca_scree" in fn:
                    kind = "pca_scree"
                    ctx = {
                        "explained_variance_ratio": (pca_block or {}).get("explained_variance_ratio") if isinstance(pca_block, dict) else [],
                        "cumulative_explained_variance": (pca_block or {}).get("cumulative_explained_variance") if isinstance(pca_block, dict) else [],
                        "n_rows": (pca_block or {}).get("n_rows") if isinstance(pca_block, dict) else None,
                        "n_features_onehot": (pca_block or {}).get("n_features_onehot") if isinstance(pca_block, dict) else None,
                    }
                elif "pca_top_factors" in fn:
                    kind = "pca_top_factors"
                    ctx = {
                        "top_original_columns": (pca_block or {}).get("top_original_columns") if isinstance(pca_block, dict) else [],
                        "selected_columns": (pca_block or {}).get("selected_columns") if isinstance(pca_block, dict) else [],
                    }
                elif "correlation_heatmap" in fn:
                    kind = "correlation_heatmap"
                    ctx = {"top_correlations": (report.get("top_correlations") or [])[:10]}
                elif "scatter_top_pair" in fn:
                    kind = "scatter_top_pair"
                    ctx = {"top_correlations": (report.get("top_correlations") or [])[:3]}
                elif "distribution_panels" in fn:
                    kind = "distribution_panels"
                    ctx = {"descriptive_stats": (report.get("descriptive_stats") or [])[:6]}
                elif "boxplot_numeric" in fn:
                    kind = "boxplot_numeric"
                    ctx = {"descriptive_stats": (report.get("descriptive_stats") or [])[:8]}
                elif "group_boxplot" in fn:
                    kind = "group_boxplot"
                    ctx = {
                        "group_comparison": report.get("group_comparison") or {},
                        "inferential_tests": [x for x in (report.get("inferential_tests") or []) if str(x.get("group_column") or "") == str((report.get("group_comparison") or {}).get("group_column") or "")]
                    }
                elif "chi_square_heatmap" in fn:
                    kind = "chi_square_heatmap"
                    ctx = {
                        "categorical_association": report.get("categorical_association") or {},
                        "inferential_tests": [x for x in (report.get("inferential_tests") or []) if str(x.get("type") or "") == "chi_square"],
                    }
                elif "cluster" in fn:
                    kind = "clustering"
                    ctx = (clustering_block[0] if clustering_block else {})
                elif "regression" in fn:
                    kind = "regression"
                    ctx = (regression_block[0] if regression_block else {})
                elif "classification" in fn:
                    kind = "classification"
                    ctx = (classification_block[0] if classification_block else {})
                figure_ctx.append({"path": str(p).replace("\\", "/"), "kind": kind, "context": ctx})
    except Exception:
        figure_ctx = []

    fig_llm = _generate_figure_notes_with_llm(llm_service, model, figure_ctx)
    figure_captions = fig_llm.get("captions") if isinstance(fig_llm, dict) else {}
    figure_notes = str((fig_llm.get("notes") if isinstance(fig_llm, dict) else "") or "").strip()
    if isinstance(report, dict) and (figure_captions or figure_notes):
        report["figure_explanations"] = {"captions": figure_captions or {}, "notes": figure_notes}

    summary_path = os.path.join(out_dir, "agent_sklearn_report.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    artifacts.append(_rel(summary_path))

    manifest_path = os.path.join(out_dir, "agent_ml_artifacts.json")
    try:
        arts = []
        for a in artifacts:
            ap = str(a).replace("\\", "/").strip()
            if not ap:
                continue
            item = {"path": ap}
            cap = ""
            if isinstance(figure_captions, dict):
                cap = str(figure_captions.get(ap) or "").strip()
            if cap:
                item["caption"] = cap
            arts.append(item)
        manifest = {
            "workspace": project_storage_path,
            "artifact_dir": os.path.relpath(out_dir, project_storage_path).replace("\\", "/"),
            "artifacts": arts,
            "summary_preview": json.dumps(
                {k: report.get(k) for k in ["descriptive_stats", "top_correlations", "group_comparison", "inferential_tests", "inferential_summary", "regression", "classification", "clustering"]},
                ensure_ascii=False,
            )[:2000],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        artifacts.append(_rel(manifest_path))
    except Exception:
        pass

    lines = []
    lines.append(f"Records loaded: {int(profile.get('n_records') or 0)}")
    overview = report.get("dataset_overview") or {}
    if overview:
        try:
            lines.append(
                f"Dataset: rows={int(overview.get('n_rows') or 0)} cols={int(overview.get('n_columns') or 0)} "
                f"numeric={len(overview.get('numeric_columns') or [])} categorical={len(overview.get('categorical_columns') or [])}"
            )
        except Exception:
            pass
    if report.get("warnings"):
        lines.append("Warnings: " + "; ".join([str(x) for x in (report.get("warnings") or [])[:6]]))
    if report.get("descriptive_stats"):
        try:
            top_desc = (report.get("descriptive_stats") or [])[:3]
            desc_parts = []
            for row in top_desc:
                desc_parts.append(
                    f"{row.get('column')}: mean={float(row.get('mean')):.3f}, median={float(row.get('median')):.3f}, IQR={float(row.get('iqr')):.3f}"
                )
            if desc_parts:
                lines.append("Descriptive stats: " + " | ".join(desc_parts))
        except Exception:
            pass
    if report.get("top_correlations"):
        try:
            tc = (report.get("top_correlations") or [])[0]
            lines.append(
                f"Top correlation: {tc.get('x')} vs {tc.get('y')} r={float(tc.get('pearson_r')):.3f}"
            )
        except Exception:
            pass
    if report.get("group_comparison"):
        try:
            g = report.get("group_comparison") or {}
            lines.append(
                f"Group comparison: {g.get('value_column')} by {g.get('group_column')} groups={int(g.get('n_groups') or 0)} mean_spread={float(g.get('mean_spread') or 0.0):.3f}"
            )
        except Exception:
            pass
    if report.get("inferential_tests"):
        try:
            test_lines = []
            for test in (report.get("inferential_tests") or [])[:4]:
                test_lines.append(_format_inferential_test_summary(test))
            if test_lines:
                lines.append("Inferential tests: " + " | ".join(test_lines))
        except Exception:
            pass
    if report.get("clustering"):
        c = (report.get("clustering") or [])[0]
        try:
            lines.append(
                f"Clustering: k={int(c.get('k'))} silhouette={float(c.get('silhouette')):.3f} inertia={float(c.get('inertia') or 0.0):.3f}"
            )
        except Exception:
            pass
    if report.get("regression"):
        r = (report.get("regression") or [])[0]
        try:
            lines.append(
                f"Regression: target={r.get('target')} model={r.get('model')} R2={float(r.get('r2')):.3f} MAE={float(r.get('mae')):.3f} RMSE={float(r.get('rmse')):.3f}"
            )
        except Exception:
            pass
    if report.get("classification"):
        c = (report.get("classification") or [])[0]
        try:
            lines.append(
                f"Classification: label={c.get('label')} model={c.get('model')} acc={float(c.get('accuracy')):.3f} "
                f"f1={float(c.get('f1_macro')):.3f} precision={float(c.get('precision_macro')):.3f} recall={float(c.get('recall_macro')):.3f}"
            )
        except Exception:
            pass
    pngs = [a for a in artifacts if str(a).lower().endswith(".png")]
    if pngs:
        lines.append("Figures: " + ", ".join(pngs[:12]))

    return {"success": True, "artifacts": artifacts, "summary": "\n".join(lines), "plan": validated, "profile": profile, "figure_notes": figure_notes, "figure_captions": figure_captions}
