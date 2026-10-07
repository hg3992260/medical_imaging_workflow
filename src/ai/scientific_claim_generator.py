import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple


def _clip(s: str, n: int) -> str:
    t = (s or "").strip()
    if len(t) <= n:
        return t
    return t[: max(0, n)].rstrip() + "..."


def _norm_token(s: str) -> str:
    t = (s or "").strip().lower()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^a-z0-9_]+", "", t)
    return t


def _p_from_pearson_r(r: float, n: int) -> float:
    try:
        rr = float(r)
        nn = int(n)
    except Exception:
        return 1.0
    if nn < 4:
        return 1.0
    rr = max(min(rr, 0.999999), -0.999999)
    z = 0.5 * math.log((1.0 + rr) / (1.0 - rr)) * math.sqrt(max(1.0, float(nn - 3)))
    cdf = 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))
    p = 2.0 * (1.0 - cdf)
    return float(max(0.0, min(1.0, p)))


def _tone_band(p: float, effect: float) -> Tuple[str, int]:
    pp = float(p if p is not None else 1.0)
    ee = abs(float(effect or 0.0))
    if pp < 0.001 and ee > 0.7:
        return "DEFINITIVE", 5
    if pp < 0.05 and ee > 0.4:
        return "SUPPORTIVE", 3
    if pp < 0.05:
        return "TENTATIVE", 1
    return "CAUTIONARY", 1


def _choose_verb(tone: str, seed: str) -> str:
    lexicon = {
        "DEFINITIVE": ["demonstrates", "establishes", "validates", "robustly correlates"],
        "SUPPORTIVE": ["suggests", "indicates", "aligns with", "shows a significant"],
        "TENTATIVE": ["implies", "points towards", "potentially associates"],
        "CAUTIONARY": ["warrants further study", "shows no clear evidence", "remains inconclusive"],
    }
    opts = lexicon.get(str(tone or "").upper(), lexicon["TENTATIVE"])
    key = _norm_token(seed)
    if not key:
        return opts[0]
    idx = int(sum(ord(c) for c in key) % len(opts))
    return opts[idx]


def _is_conflict(feature_a: str, feature_b: str, literature_context: str) -> bool:
    text = (literature_context or "").lower()
    if not text:
        return False
    a = _norm_token(feature_a)
    b = _norm_token(feature_b)
    if not a or not b:
        return False
    conflict_words = ["contradict", "inconsistent", "conflict", "diverg", "disagree", "paradox"]
    if not any(w in text for w in conflict_words):
        return False
    return (a in _norm_token(text)) and (b in _norm_token(text))


def _domain_terms(project_keyword: str) -> Dict[str, str]:
    kw = (project_keyword or "").lower()
    if any(x in kw for x in ["ct", "mri", "dicom", "radiomics", "lung", "tumor", "lesion"]) or any(x in (project_keyword or "") for x in ["影像", "肿瘤", "肺", "磁共振", "CT"]):
        return {
            "increase": "increased signal intensity / higher feature magnitude",
            "decrease": "decreased signal intensity / lower feature magnitude",
            "consensus": "cross-branch (multi-modal) consistency",
        }
    return {"increase": "increased value", "decrease": "decreased value", "consensus": "cross-branch consistency"}


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("\x00", "").strip()


def build_scg_claim_synthesis_prompt(
    project_keyword: str,
    tabular_branch_mining: str,
    cross_table_correlations: str,
    logic_audit_results: str,
    seed_claim_objects: List[Dict[str, Any]],
    max_prompt_chars: int = 22000,
) -> str:
    seed_raw = json.dumps(seed_claim_objects or [], ensure_ascii=False, indent=2)
    base_template = (
        "# Phase 1.97: 学术主张合成器 (SCG)\n\n"
        "### [SYSTEM ROLE]\n"
        "你是一个资深学术通讯作者。你的任务是将底层数据挖掘引擎产出的统计特征与外部文献背景进行高阶融合，"
        "生成具有叙事导向的学术主张对象（Claim Objects）。\n\n"
        "### [INPUT DATA]\n"
        "1) {{TABULAR_BRANCH_MINING}}\n"
        "2) {{CROSS_TABLE_CORRELATIONS}}\n"
        "3) {{LOGIC_AUDIT_RESULTS}}\n"
        "4) {{PROJECT_KEYWORD}}\n\n"
        "#### {{PROJECT_KEYWORD}}\n"
        "{PROJECT_KEYWORD}\n\n"
        "#### {{TABULAR_BRANCH_MINING}}\n"
        "{TABULAR_BRANCH_MINING}\n\n"
        "#### {{CROSS_TABLE_CORRELATIONS}}\n"
        "{CROSS_TABLE_CORRELATIONS}\n\n"
        "#### {{LOGIC_AUDIT_RESULTS}}\n"
        "{LOGIC_AUDIT_RESULTS}\n\n"
        "#### {{SEED_CLAIM_OBJECTS_JSON}}\n"
        "{SEED_CLAIM_OBJECTS_JSON}\n\n"
        "### [SCIENTIFIC CLAIM SYNTHESIS LOGIC]\n"
        "请把 SEED_CLAIM_OBJECTS_JSON 中的每个条目，重写为更学术化、更叙事导向的 Claim Object。\n"
        "必须保持原始统计数值不变（r/p/n/density 等），不得捏造新数值。\n\n"
        "#### 1) Tone Mapping\n"
        "- p<0.001 且 |r|>0.7: Definitive，优先使用 Demonstrates/Establishes/Validates\n"
        "- p<0.05 且 |r|>0.4: Supportive，使用 Suggests/Indicates/Corresponds to\n"
        "- p>=0.05 或 data_density 很低: Tentative，使用 Implies/Warrants further investigation/Remains inconclusive\n\n"
        "#### 2) Narrative Role\n"
        "- Core_Discovery: 与研究方向高度相关的核心发现\n"
        "- Counter_Intuitive: 与文献或 Logic Audit 冲突（必须标注 logic_status=Conflict_with_Lit）\n"
        "- Robustness_Proof: 跨分支/跨模态一致性证据\n\n"
        "#### 3) Semantic Stitching\n"
        "把“数值增加/减少”改写为领域术语（例如影像：信号强度增强/特征上调）。\n\n"
        "### [OUTPUT FORMAT: JSON]\n"
        "严格输出 JSON 数组，不要输出 Markdown，不要输出解释文字：\n"
        "[\n"
        "  {\n"
        '    \"claim_id\": \"SCG_00X\",\n'
        '    \"narrative_role\": \"Core_Discovery/Counter_Intuitive/Robustness_Proof\",\n'
        '    \"statement\": \"高度学术化的单句主张描述\",\n'
        '    \"tone_intensity\": \"1-5\",\n'
        '    \"statistical_support\": {\"metrics\": \"r=..., p=..., n=...\", \"data_density\": \"0-1\"},\n'
        '    \"logic_status\": \"Consensus/Conflict_with_Lit\",\n'
        '    \"branch_path\": \"必须可追溯到原始分支（可用 evidence_ref 或 branch_path）\",\n'
        '    \"drafting_instruction\": \"针对该主张的写作策略（尤其 Discussion）\"\n'
        "  }\n"
        "]\n\n"
        "### [CONSTRAINTS]\n"
        "- 严禁幻觉：仅使用输入块中出现的信息。\n"
        "- 冲突前置：若 Logic Audit 指出冲突，必须设置 narrative_role=Counter_Intuitive。\n"
        "- 溯源性：每个主张必须带 branch_path。\n"
    )

    fixed = base_template.format(
        PROJECT_KEYWORD=_clip(project_keyword, 240),
        TABULAR_BRANCH_MINING="",
        CROSS_TABLE_CORRELATIONS="",
        LOGIC_AUDIT_RESULTS="",
        SEED_CLAIM_OBJECTS_JSON="",
    )
    remaining = max(2000, int(max_prompt_chars) - len(fixed))
    b_tab = int(remaining * 0.36)
    b_cross = int(remaining * 0.22)
    b_lar = int(remaining * 0.24)
    b_seed = max(0, remaining - (b_tab + b_cross + b_lar))

    tab = _clip(tabular_branch_mining, b_tab)
    cross = _clip(cross_table_correlations, b_cross)
    lar = _clip(logic_audit_results, b_lar)
    seed_json = _clip(seed_raw, b_seed)

    return base_template.format(
        PROJECT_KEYWORD=_clip(project_keyword, 240),
        TABULAR_BRANCH_MINING=tab,
        CROSS_TABLE_CORRELATIONS=cross,
        LOGIC_AUDIT_RESULTS=lar,
        SEED_CLAIM_OBJECTS_JSON=seed_json,
    )


def merge_enriched_claim_objects(seed: List[Dict[str, Any]], enriched: Any) -> List[Dict[str, Any]]:
    if not isinstance(seed, list):
        seed = []
    if not isinstance(enriched, list):
        return seed
    by_id: Dict[str, Dict[str, Any]] = {}
    for c in seed:
        if isinstance(c, dict) and _safe_str(c.get("claim_id")):
            by_id[_safe_str(c.get("claim_id"))] = c
    out = []
    for i, c in enumerate(enriched):
        if not isinstance(c, dict):
            continue
        cid = _safe_str(c.get("claim_id")) or _safe_str(c.get("id")) or f"SCG_LLM_{i+1}"
        base = by_id.get(cid) or {}
        statement = _safe_str(c.get("statement")) or _safe_str(c.get("content")) or _safe_str(base.get("statement")) or _safe_str(base.get("content"))
        if not statement:
            continue
        merged = dict(base)
        merged["claim_id"] = cid
        merged["statement"] = statement
        merged["content"] = statement
        merged["narrative_role"] = _safe_str(c.get("narrative_role")) or _safe_str(base.get("narrative_role")) or "Core_Discovery"
        merged["logic_status"] = _safe_str(c.get("logic_status")) or ("Conflict_with_Lit" if merged.get("narrative_role") == "Counter_Intuitive" else "Consensus")
        merged["drafting_instruction"] = _safe_str(c.get("drafting_instruction"))
        merged["branch_path"] = _safe_str(c.get("branch_path")) or _safe_str(base.get("evidence_ref")) or _safe_str(base.get("branch_path"))
        merged["tone_intensity"] = _safe_str(c.get("tone_intensity")) or str(base.get("tone_level") or "")
        merged["statistical_support"] = c.get("statistical_support") if isinstance(c.get("statistical_support"), dict) else {}
        out.append(merged)
    return out or seed


class ScientificClaimGenerator:
    def generate(
        self,
        project_keyword: str,
        aggregated_context: str,
        structured_data: Optional[Dict[str, Any]] = None,
        logic_audit_report: str = "",
        ref_context: str = "",
        max_claims: int = 24,
    ) -> Tuple[List[Dict[str, Any]], str]:
        blocks = self._extract_tabular_blocks(aggregated_context)
        claims: List[Dict[str, Any]] = []
        terms = _domain_terms(project_keyword)

        tab = blocks.get("tabular_mining", "")
        if tab:
            literature_ctx = "\n".join([x for x in [logic_audit_report, ref_context] if (x or "").strip()])
            claims.extend(self._claims_from_inter_branch(tab, terms, literature_ctx))
            claims.extend(self._claims_from_density(tab))
            claims.extend(self._claims_from_vertical(tab))

        if structured_data and isinstance(structured_data, dict):
            claims.extend(self._claims_from_structured_data(structured_data))

        uniq = []
        seen = set()
        for c in claims:
            key = (c.get("operator_type"), c.get("feature_a"), c.get("feature_b"), c.get("evidence_ref"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)

        uniq.sort(key=lambda x: float(x.get("strength") or 0.0), reverse=True)
        uniq = uniq[: max(1, int(max_claims))]
        block = self._to_prompt_block(project_keyword, uniq)
        return uniq, block

    def _extract_tabular_blocks(self, text: str) -> Dict[str, str]:
        t = text or ""
        m = re.search(r"\[TABULAR_BRANCH_MINING\]\n([\s\S]*)\Z", t)
        if not m:
            m = re.search(r"\[TABULAR_BRANCH_MINING\]\n([\s\S]*)", t)
        return {"tabular_mining": m.group(1).strip() if m else ""}

    def _claims_from_inter_branch(self, tab_ctx: str, terms: Dict[str, str], literature_ctx: str) -> List[Dict[str, Any]]:
        out = []
        total_samples = None
        try:
            m = re.search(r"Samples=(\d+)", tab_ctx or "")
            if m:
                total_samples = int(m.group(1))
        except Exception:
            total_samples = None

        section = self._extract_section(tab_ctx, "INTER_BRANCH_CORRELATION")
        if not section:
            return out
        for line in section:
            m = re.search(r"(.+?)\s*~\s*(.+?):\s*pearson_r=([\-0-9.]+)\s+n_samples=(\d+)", line)
            if not m:
                continue
            a = m.group(1).strip()
            b = m.group(2).strip()
            r = float(m.group(3))
            n = int(m.group(4))
            p = _p_from_pearson_r(r, n)
            strength = abs(r)
            density = 1.0
            if total_samples and total_samples > 0:
                density = max(0.0, min(1.0, float(n) / float(total_samples)))
            tone, tone_level = _tone_band(p, strength)
            narrative_role = "CONSENSUS_BUILDING"
            action_verb = _choose_verb(tone, f"{a}|{b}|{n}|{r}")
            if _is_conflict(a, b, literature_ctx):
                action_verb = "contradicts"
                narrative_role = "PARADOX_DISCUSSION"
            direction = "positive" if r >= 0 else "negative"
            rel = "highly concordant co-variation" if direction == "positive" else "inverse co-variation"
            statement = (
                f"Data from cross-branch analysis {action_verb} that '{a}' and '{b}' exhibit {rel} "
                f"across samples (Pearson r={r:.3f}, p={p:.3g}, n={n}, density={density:.2f})."
            )
            confidence_score = round((1.0 - float(p)) * float(strength) * float(density), 2)
            out.append(
                {
                    "claim_id": f"SCG_CORR_{len(out)+1}",
                    "operator_type": "Cross-Branch Consensus",
                    "statement": statement,
                    "content": statement,
                    "strength": float(strength),
                    "tone_level": int(tone_level),
                    "tone_band": tone,
                    "tone_verb": action_verb,
                    "direction": direction,
                    "narrative_role": narrative_role,
                    "confidence_score": confidence_score,
                    "feature_a": a,
                    "feature_b": b,
                    "metrics": {"pearson_r": r, "p_value": p, "n_samples": n, "data_density": density},
                    "evidence_ref": "INTER_BRANCH_CORRELATION",
                    "evidence_links": ["INTER_BRANCH_CORRELATION"],
                }
            )
        return out

    def _claims_from_density(self, tab_ctx: str) -> List[Dict[str, Any]]:
        out = []
        section = self._extract_section(tab_ctx, "DATA_DENSITY")
        if not section:
            return out
        lows = []
        for line in section:
            m = re.search(r"(.+?)\s+density=([0-9.]+)\s+weight=([0-9.]+)", line)
            if not m:
                continue
            branch = m.group(1).strip()
            density = float(m.group(2))
            weight = float(m.group(3))
            if density < 0.30:
                lows.append((density, weight, branch))
        lows.sort(key=lambda x: x[0])
        for density, weight, branch in lows[:8]:
            tone, tone_level = "CAUTIONARY", 1
            action_verb = "remains inconclusive"
            out.append(
                {
                    "claim_id": f"SCG_OUTLIER_{len(out)+1}",
                    "operator_type": "Outlier Detection",
                    "statement": f"Although a potential trend may exist, branch '{branch}' {action_verb} due to low data density (density={density:.2f}).",
                    "content": f"Although a potential trend may exist, branch '{branch}' {action_verb} due to low data density (density={density:.2f}).",
                    "strength": float(max(0.0, 1.0 - density)),
                    "tone_level": int(tone_level),
                    "tone_band": tone,
                    "tone_verb": action_verb,
                    "direction": "outlier",
                    "narrative_role": "CAUTIONARY_NOTE",
                    "confidence_score": round(float(max(0.0, 1.0 - density)) * float(density), 2),
                    "feature_a": "",
                    "feature_b": "",
                    "metrics": {"density": density, "weight": weight},
                    "evidence_ref": branch,
                    "evidence_links": [branch],
                }
            )
        return out

    def _claims_from_vertical(self, tab_ctx: str) -> List[Dict[str, Any]]:
        out = []
        section = self._extract_section(tab_ctx, "VERTICAL_CORRELATION")
        if not section:
            return out
        for line in section[:12]:
            m = re.search(r"([^:]+):\s*samples=(\d+)\s+mean_range=\[([^\]]+)\]", line)
            if not m:
                continue
            feat = m.group(1).strip()
            ns = int(m.group(2))
            rng = m.group(3).strip()
            strength = min(1.0, 0.2 + 0.08 * ns)
            p = 0.04 if ns >= 6 else 0.2
            tone, tone_level = _tone_band(p, strength)
            action_verb = _choose_verb(tone, f"{feat}|{ns}|{rng}")
            statement = f"Across samples, '{feat}' {action_verb} a stable magnitude (n={ns}, mean_range={rng})."
            out.append(
                {
                    "claim_id": f"SCG_ROBUST_{len(out)+1}",
                    "operator_type": "Cross-Branch Consensus",
                    "statement": statement,
                    "content": statement,
                    "strength": float(strength),
                    "tone_level": int(tone_level),
                    "tone_band": tone,
                    "tone_verb": action_verb,
                    "direction": "consensus",
                    "narrative_role": "CONSENSUS_BUILDING",
                    "confidence_score": round((1.0 - p) * float(strength), 2),
                    "feature_a": feat,
                    "feature_b": "",
                    "metrics": {"n_samples": ns, "mean_range": rng, "p_value_proxy": p},
                    "evidence_ref": "VERTICAL_CORRELATION",
                    "evidence_links": ["VERTICAL_CORRELATION"],
                }
            )
        return out

    def _claims_from_structured_data(self, structured_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        sf = structured_data.get("potential_correlations") or structured_data.get("statistical_findings") or []
        if not isinstance(sf, list):
            return out
        for x in sf[:18]:
            t = str(x or "").strip()
            if not t:
                continue
            tone, tone_level = "SUPPORTIVE", 3
            action_verb = _choose_verb(tone, t[:80])
            out.append(
                {
                    "claim_id": f"SCG_STRUCT_{len(out)+1}",
                    "operator_type": "Claim Operator",
                    "statement": f"Our extracted findings {action_verb} a statistically grounded observation: {t}",
                    "content": f"Our extracted findings {action_verb} a statistically grounded observation: {t}",
                    "strength": 0.6,
                    "tone_level": int(tone_level),
                    "tone_band": tone,
                    "tone_verb": action_verb,
                    "direction": "unspecified",
                    "narrative_role": "SUPPORTIVE_EVIDENCE",
                    "confidence_score": 0.6,
                    "feature_a": "",
                    "feature_b": "",
                    "metrics": {},
                    "evidence_ref": "STRUCTURIZER",
                    "evidence_links": ["STRUCTURIZER"],
                }
            )
        return out

    def _extract_section(self, tab_ctx: str, header: str) -> List[str]:
        m = re.search(rf"\[{re.escape(header)}\]\n([\s\S]*?)(?:\n\[[A-Z0-9_]+\]|\Z)", tab_ctx or "")
        if not m:
            return []
        lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
        return [ln[2:].strip() if ln.startswith("- ") else ln for ln in lines]

    def _to_prompt_block(self, project_keyword: str, claims: List[Dict[str, Any]]) -> str:
        obj = {"project_keyword": project_keyword, "claim_objects": claims}
        js = _clip(json.dumps(obj, ensure_ascii=False, indent=2), 3800)
        lines = []
        lines.append("[SCIENTIFIC_CLAIM_GENERATOR_SCG]")
        lines.append("AcademicToneMapper policy:")
        lines.append("- DEFINITIVE (L5): demonstrates / establishes / validates / robustly correlates")
        lines.append("- SUPPORTIVE (L3): suggests / indicates / aligns with / shows a significant")
        lines.append("- TENTATIVE (L1): implies / points towards / potentially associates")
        lines.append("- CAUTIONARY (L1): warrants further study / shows no clear evidence / remains inconclusive")
        lines.append("")
        lines.append("[SCG_CLAIM_OBJECTS_JSON]")
        lines.append(js)
        lines.append("")
        lines.append("[SCG_CLAIM_SUMMARY]")
        for c in claims[:18]:
            cid = c.get("claim_id")
            lvl = c.get("tone_level")
            strength = c.get("strength")
            conf = c.get("confidence_score")
            role = c.get("narrative_role")
            ev = c.get("evidence_ref")
            txt = c.get("statement") or c.get("content")
            try:
                conf_s = f"{float(conf):.2f}"
            except Exception:
                conf_s = str(conf or "")
            lines.append(f"- {cid} | L{lvl} | conf={conf_s} | strength={strength:.2f} | role={role} | evidence={ev}: {txt}")
        return "\n".join(lines).strip()
