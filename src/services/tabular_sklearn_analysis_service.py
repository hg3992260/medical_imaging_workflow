import json
import math
import os
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
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except Exception:
    KMeans = None
    PCA = None
    RandomForestClassifier = None
    RandomForestRegressor = None
    LogisticRegression = None
    Ridge = None
    mean_absolute_error = None
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


def _load_tabular_records_from_json(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return []
        sheet = obj.get("sheet") or {}
        recs = sheet.get("records") or []
        if isinstance(recs, list):
            out = []
            for r in recs:
                if isinstance(r, dict):
                    out.append(r)
            return out
        return []
    except Exception:
        return []


def run_tabular_sklearn_analysis(
    project_storage_path: str,
    out_dir: str,
    max_json_files: int = 120,
    max_records: int = 6000,
) -> Dict[str, Any]:
    base = str(project_storage_path or "").strip()
    if not base:
        return {"success": False, "error": "empty_project_storage_path", "artifacts": [], "summary": ""}
    tabular_root = os.path.join(base, "tabular")
    if not os.path.isdir(tabular_root):
        return {"success": False, "error": "tabular_dir_missing", "artifacts": [], "summary": ""}

    paths: List[str] = []
    for p in sorted(Path(tabular_root).glob("**/*.json")):
        paths.append(str(p))
        if len(paths) >= int(max_json_files):
            break

    if not paths:
        return {"success": False, "error": "no_tabular_json", "artifacts": [], "summary": ""}

    records: List[Dict[str, Any]] = []
    for p in paths:
        for r in _load_tabular_records_from_json(p):
            records.append(r)
            if len(records) >= int(max_records):
                break
        if len(records) >= int(max_records):
            break

    if not records:
        return {"success": False, "error": "no_records", "artifacts": [], "summary": ""}

    os.makedirs(out_dir, exist_ok=True)
    artifacts: List[str] = []

    numeric_cols: List[str] = []
    stats: Dict[str, Dict[str, float]] = {}

    if pd is None:
        num_vals: Dict[str, List[float]] = {}
        for r in records:
            for k, v in r.items():
                fv = _try_float(v)
                if fv is None:
                    continue
                num_vals.setdefault(str(k), []).append(fv)
        for k, xs in sorted(num_vals.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:12]:
            if not xs:
                continue
            xs2 = xs[:5000]
            mean = sum(xs2) / float(len(xs2))
            var = sum((x - mean) ** 2 for x in xs2) / float(len(xs2)) if len(xs2) > 1 else 0.0
            stats[k] = {"n": float(len(xs2)), "mean": float(mean), "stdev": float(math.sqrt(var)), "min": float(min(xs2)), "max": float(max(xs2))}
            numeric_cols.append(k)
    else:
        df = pd.DataFrame(records)
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            valid = int(s.notna().sum())
            if valid >= 8:
                numeric_cols.append(str(c))
                stats[str(c)] = {
                    "n": float(valid),
                    "mean": float(s.mean()),
                    "stdev": float(s.std(ddof=0)) if valid > 1 else 0.0,
                    "min": float(s.min()),
                    "max": float(s.max()),
                }
        numeric_cols = sorted(numeric_cols, key=lambda c: (-stats.get(c, {}).get("n", 0.0), c))[:20]

    summary_obj: Dict[str, Any] = {
        "sources": [os.path.relpath(p, base).replace("\\", "/") for p in paths],
        "records_loaded": len(records),
        "numeric_columns": numeric_cols,
        "stats": stats,
        "sklearn_available": bool(PCA and StandardScaler),
    }
    summary_path = os.path.join(out_dir, "agent_sklearn_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)
    artifacts.append(os.path.relpath(summary_path, base).replace("\\", "/"))

    def _setup_plot_style():
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import rcParams
            import matplotlib.font_manager as fm

            rcParams["axes.unicode_minus"] = False
            candidates = []
            for name in ["SimHei", "Microsoft YaHei", "PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC"]:
                try:
                    candidates.append(name)
                except Exception:
                    pass
            available = {f.name for f in fm.fontManager.ttflist}
            for name in candidates:
                if name in available:
                    rcParams["font.sans-serif"] = [name]
                    break
            return plt
        except Exception:
            return None

    fig_paths: List[str] = []
    ml_report: Dict[str, Any] = {}
    if pd is not None and np is not None and PCA and StandardScaler and len(numeric_cols) >= 2:
        try:
            X = pd.DataFrame(records)[numeric_cols]
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            X = X.iloc[: min(len(X), 5000)]
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X.values)
            pca = PCA(n_components=2)
            Z = pca.fit_transform(Xs)

            labels = None
            if KMeans and len(Z) >= 30:
                k = 3 if len(Z) >= 120 else 2
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(Z)

            plt = _setup_plot_style()
            if plt is None:
                raise RuntimeError("matplotlib_unavailable")

            fig = plt.figure(figsize=(4.4, 3.4))
            ax = fig.add_subplot(111)
            if labels is None:
                ax.scatter(Z[:, 0], Z[:, 1], s=7, alpha=0.65)
            else:
                ax.scatter(Z[:, 0], Z[:, 1], s=7, alpha=0.70, c=labels)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA of tabular numeric features")
            fig.tight_layout()
            out_png = os.path.join(out_dir, "agent_sklearn_pca.png")
            fig.savefig(out_png, dpi=170)
            fig_paths.append(out_png)

            if silhouette_score is not None and labels is not None and len(set(labels.tolist())) >= 2:
                try:
                    ml_report["clustering"] = {
                        "method": "kmeans",
                        "k": int(len(set(labels.tolist()))),
                        "silhouette_score": float(silhouette_score(Z, labels)),
                    }
                except Exception:
                    pass

            try:
                evr = [float(x) for x in (pca.explained_variance_ratio_ or [])]
                summary_obj["pca_explained_variance_ratio"] = evr
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary_obj, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        except Exception:
            fig_paths = []

    if pd is not None and np is not None and train_test_split and (Ridge or RandomForestRegressor or LogisticRegression or RandomForestClassifier):
        try:
            df = pd.DataFrame(records)
            numeric_df = pd.DataFrame()
            for c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce")
                if int(s.notna().sum()) >= 8:
                    numeric_df[str(c)] = s
            numeric_df = numeric_df.fillna(0.0)
            if numeric_df.shape[1] >= 2 and numeric_df.shape[0] >= 20:
                cols = list(numeric_df.columns)
                y_col = max(cols, key=lambda c: float(stats.get(c, {}).get("n", 0.0)))
                x_cols = [c for c in cols if c != y_col][: min(len(cols) - 1, 8)]
                if x_cols:
                    X = numeric_df[x_cols].values
                    y = numeric_df[y_col].values
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    scaler = StandardScaler() if StandardScaler else None
                    if scaler is not None:
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                    best = None
                    best_name = ""
                    best_r2 = -1e9
                    models = []
                    if Ridge is not None:
                        models.append(("ridge", Ridge(alpha=1.0, random_state=42)))
                    if RandomForestRegressor is not None:
                        models.append(("rf_reg", RandomForestRegressor(n_estimators=200, random_state=42)))

                    for name, m in models:
                        try:
                            m.fit(X_train, y_train)
                            pred = m.predict(X_test)
                            r2 = float(r2_score(y_test, pred)) if r2_score else float("nan")
                            mae = float(mean_absolute_error(y_test, pred)) if mean_absolute_error else float("nan")
                            if r2 > best_r2:
                                best_r2 = r2
                                best = (m, pred, mae)
                                best_name = name
                        except Exception:
                            continue

                    if best is not None:
                        _m, pred, mae = best
                        ml_report["regression"] = {
                            "target": y_col,
                            "features": x_cols,
                            "model": best_name,
                            "r2": float(best_r2),
                            "mae": float(mae),
                        }
                        plt = _setup_plot_style()
                        if plt is not None:
                            fig = plt.figure(figsize=(4.2, 3.4))
                            ax = fig.add_subplot(111)
                            ax.scatter(y_test, pred, s=8, alpha=0.65)
                            ax.set_xlabel("y_true")
                            ax.set_ylabel("y_pred")
                            ax.set_title(f"Regression ({best_name}) on {y_col}")
                            fig.tight_layout()
                            out_png = os.path.join(out_dir, "agent_sklearn_regression.png")
                            fig.savefig(out_png, dpi=170)
                            fig_paths.append(out_png)
        except Exception:
            pass

    for p in fig_paths:
        rp = os.path.relpath(p, base).replace("\\", "/")
        artifacts.append(rp)

    manifest_path = os.path.join(out_dir, "agent_ml_artifacts.json")
    try:
        manifest = {
            "workspace": base,
            "artifact_dir": os.path.relpath(out_dir, base).replace("\\", "/"),
            "artifacts": [{"path": a} for a in artifacts if str(a).strip()],
            "summary_preview": json.dumps({"stats": stats, "ml": ml_report}, ensure_ascii=False)[:2000],
            "ml_report": ml_report,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        artifacts.append(os.path.relpath(manifest_path, base).replace("\\", "/"))
    except Exception:
        pass

    summary_lines = []
    summary_lines.append(f"Tabular JSON files: {len(paths)}")
    summary_lines.append(f"Records loaded: {len(records)}")
    if numeric_cols:
        cols = ", ".join(numeric_cols[:10])
        summary_lines.append(f"Numeric columns (top): {cols}")
    if summary_obj.get("pca_explained_variance_ratio"):
        evr = summary_obj.get("pca_explained_variance_ratio") or []
        try:
            summary_lines.append(f"PCA explained variance ratio: {evr[0]:.3f}, {evr[1]:.3f}")
        except Exception:
            pass
    if any(a.lower().endswith(".png") for a in artifacts):
        imgs = [a for a in artifacts if a.lower().endswith(".png")]
        summary_lines.append("Figures: " + ", ".join(imgs[:6]))
    if ml_report.get("regression"):
        r = ml_report.get("regression") or {}
        try:
            summary_lines.append(f"Regression: target={r.get('target')} model={r.get('model')} R2={float(r.get('r2')):.3f} MAE={float(r.get('mae')):.3f}")
        except Exception:
            pass
    if ml_report.get("clustering"):
        c = ml_report.get("clustering") or {}
        try:
            summary_lines.append(f"Clustering: method={c.get('method')} k={int(c.get('k'))} silhouette={float(c.get('silhouette_score')):.3f}")
        except Exception:
            pass

    return {"success": True, "artifacts": artifacts, "summary": "\n".join(summary_lines), "stats": stats, "ml_report": ml_report}
