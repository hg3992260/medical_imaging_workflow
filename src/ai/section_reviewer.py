import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.services.llm_judgement_client import LLMJudgementClient


def _bool_env(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() not in ["0", "false", "no", "off", ""]


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _extract_json_block(text: str) -> Optional[dict]:
    s = str(text or "").strip()
    if not s:
        return None
    # Strip Ollama thinking-model tags
    s = re.sub(r"<\s*/\s*think\s*>.*?<\s*response\s*>", "", s, flags=re.DOTALL)
    s = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", s, flags=re.DOTALL)
    s = re.sub(r"\u0001?THINK\s.*?END\s?THINK", "", s, flags=re.DOTALL)
    s = re.sub(r"^\u0001?>\s*", "", s)
    s = re.sub(r"\n\u0001?>\s*", "\n", s)
    # Strip markdown code fences
    md_m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", s)
    if md_m:
        s = md_m.group(1).strip()
    # Brace-counting JSON extraction: find first '{', track nesting until balanced
    idx = s.find("{")
    if idx >= 0:
        depth = 0
        in_string = False
        escape = False
        for i in range(idx, len(s)):
            ch = s[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[idx:i+1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                    break
    # Fallback: try non-JSON key:value patterns for common judge fields
    if re.search(r"\bscore\b", s, re.IGNORECASE) or re.search(r"\bpassed\b", s, re.IGNORECASE):
        out: Dict[str, Any] = {"score": 5.0, "passed": True, "issues": [], "missing_data": [], "hallucination_flags": [], "suggested_edits": []}
        m = re.search(r"\bscore\s*[:=]\s*(\d+(?:\.\d+)?)", s, re.IGNORECASE)
        if m:
            out["score"] = float(m.group(1))
        m = re.search(r"\bpassed\s*[:=]\s*(true|false|yes|no)", s, re.IGNORECASE)
        if m:
            out["passed"] = m.group(1).lower() in ("true", "yes")
        for field in ("issues", "missing_data", "hallucination_flags", "suggested_edits"):
            items = re.findall(rf"\b{field}\s*[:=]\s*\[(.*?)\]", s, re.IGNORECASE | re.DOTALL)
            if items:
                field_text = items[0]
                vals = re.findall(r'"([^"]*)"', field_text)
                out[field] = vals[:10]
        if out["score"] != 5.0 or not out["passed"] or out["issues"]:
            return out
    # Last resort: try the entire text as JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Absolute last resort: salvage from any text content
    if len(s) > 10:
        return {
            "score": 5.0,
            "passed": False,
            "issues": ["llm_raw_text: could not extract structured review"],
            "missing_data": [],
            "hallucination_flags": [],
            "suggested_edits": ["AUTO: Proceed with refinement to improve logical flow, evidence grounding, and readability."],
        }
    return None


@dataclass
class SectionReviewerConfig:
    enabled: bool
    base_url: str
    api_key: str
    model: str
    min_score: float
    temperature: float
    max_tokens: int
    timeout_s: float

    @classmethod
    def from_env(cls):
        enabled = _bool_env("RSNA_JUDGE_ENABLED", True)
        base_url = os.environ.get("RSNA_JUDGE_BASE_URL", "http://localhost:11434/v1").strip()
        api_key = os.environ.get("RSNA_JUDGE_API_KEY", "").strip()
        model = os.environ.get("RSNA_JUDGE_MODEL", "qwen3:8b").strip()
        min_score = _float_env("RSNA_JUDGE_MIN_SCORE", 8.0)
        temperature = _float_env("RSNA_JUDGE_TEMPERATURE", 0.0)
        max_tokens = _int_env("RSNA_JUDGE_MAX_TOKENS", 1024)
        timeout_s = _float_env("RSNA_JUDGE_TIMEOUT_S", 120.0)
        return cls(
            enabled=enabled,
            base_url=base_url,
            api_key=api_key,
            model=model,
            min_score=float(min_score),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            timeout_s=float(timeout_s),
        )

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]):
        env = cls.from_env()
        if not cfg:
            return env
        d = dict(cfg)
        return cls(
            enabled=bool(d.get("enabled", env.enabled)),
            base_url=str(d.get("base_url", env.base_url) or "").strip(),
            api_key=str(d.get("api_key", env.api_key) or "").strip(),
            model=str(d.get("model", env.model) or "").strip(),
            min_score=float(d.get("min_score", env.min_score)),
            temperature=float(d.get("temperature", env.temperature)),
            max_tokens=int(d.get("max_tokens", env.max_tokens)),
            timeout_s=float(d.get("timeout_s", env.timeout_s)),
        )


class SectionReviewer:
    # Pool of local Ollama models eligible for dual-review (random 2 per review)
    PANEL_MODELS = ["qwen3:8b", "llama3:8b", "qwen2.5:7b", "gemma4:e4b"]

    def __init__(self, config: SectionReviewerConfig, llm_service=None, model: str = ""):
        self.config = config
        self.llm_service = llm_service
        self.model = str(model or "").strip()
        self.client = LLMJudgementClient(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout_s=config.timeout_s,
        )

    @property
    def enabled(self) -> bool:
        if not bool(self.config.enabled):
            return False
        if self.llm_service is not None and bool(self.model or self.config.model):
            return True
        return bool(self.config.base_url) and bool(self.config.model)

    def review(
        self,
        *,
        unit_name: str,
        outline_title: str,
        core_claims: List[str],
        ref_context: str,
        structured_data: Optional[Dict[str, Any]],
        draft_text: str,
        raw_context_excerpt: str,
    ) -> Dict[str, Any]:
        core_claims_str = "; ".join([str(x) for x in (core_claims or []) if str(x).strip()][:10])
        evidence_findings = ""
        evidence_lines = ""
        if structured_data:
            evidence_findings = ", ".join([str(x) for x in (structured_data.get("statistical_findings") or [])][:40])
            evidence_lines = "\n".join([str(x) for x in (structured_data.get("numeric_evidence_lines") or [])][:40])

        prompt = (
            "You are a strict peer reviewer for a medical imaging / radiology paper.\n"
            "Your job: evaluate ONE section draft for factuality, evidence-grounding, structure, and scientific writing.\n"
            "Hard rules:\n"
            "- Do NOT reward fabricated statistics. If numbers/metrics appear without evidence, flag as hallucination.\n"
            "- Citations like [1] are allowed, but do not invent new references.\n"
            "- If required parameters/metrics are missing, identify them as missing instead of guessing.\n"
            "Return ONLY JSON with keys:\n"
            "- score (number 0-10, higher is better)\n"
            "- passed (bool)\n"
            "- issues (list[string], max 10)\n"
            "- missing_data (list[string], max 10)\n"
            "- hallucination_flags (list[string], max 10)\n"
            "- suggested_edits (list[string], max 10)\n\n"
            f"[PAPER_TITLE]\n{outline_title}\n\n"
            f"[SECTION]\n{unit_name}\n\n"
            f"[CORE_CLAIMS]\n{core_claims_str}\n\n"
            f"[EVIDENCE_FINDINGS]\n{evidence_findings}\n\n"
            f"[EVIDENCE_LINES]\n{evidence_lines}\n\n"
            f"[REFERENCES_LIST]\n{str(ref_context or '')[:2000]}\n\n"
            f"[RAW_CONTEXT_EXCERPT]\n{str(raw_context_excerpt or '')[:2000]}\n\n"
            f"[DRAFT]\n{str(draft_text or '')[:5000]}\n"
        )
        try:
            from src.ai.prompt_registry import PromptRegistry
            tpl = PromptRegistry.instance().get("section_review")
            if tpl.strip():
                tpl = (tpl
                    .replace("{outline_title}", str(outline_title or ""))
                    .replace("{unit_name}", str(unit_name or ""))
                    .replace("{core_claims}", core_claims_str)
                    .replace("{evidence_findings}", evidence_findings)
                    .replace("{evidence_lines}", evidence_lines)
                    .replace("{references}", str(ref_context or '')[:2000])
                    .replace("{raw_context}", str(raw_context_excerpt or '')[:2000])
                    .replace("{draft_text}", str(draft_text or '')[:5000])
                )
                prompt = tpl
        except Exception:
            pass

        messages = [{"role": "user", "content": prompt}]
        panel = list(self.PANEL_MODELS)
        random.shuffle(panel)
        panel = panel[:2]
        if not panel:
            panel = [self.model] if self.model else ["qwen3:8b"]

        # Detect custom API: if llm_service is a remote service (not local Ollama),
        # the panel model names are local-only and won't work. Use main model instead.
        svc_type = type(self.llm_service).__name__ if self.llm_service else ""
        if "Custom" in svc_type or "Remote" in svc_type:
            if self.model:
                panel = [self.model, self.model]
            else:
                panel = ["deepseek-v4-flash", "deepseek-v4-flash"]
            self._custom_api_judge = True
        else:
            self._custom_api_judge = False

        reviews = []
        for mdl in panel:
            rev = self._review_with_model(mdl, prompt)
            reviews.append(rev)

        return self._merge_reviews(reviews)

    def _review_with_model(self, model, prompt):
        score_unparseable = 7.0
        try:
            resp = self.llm_service.generate(
                model,
                prompt,
                timeout=int(self.config.timeout_s),
                options={
                    "temperature": float(self.config.temperature),
                    "num_predict": int(self.config.max_tokens),
                },
                stream_callback=None,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            return {"score": score_unparseable, "passed": False, "issues": [f"judge_exception:{type(e).__name__}"], "missing_data": [], "hallucination_flags": [], "suggested_edits": [], "_model": model}
        if not resp.get("success"):
            return {"score": score_unparseable, "passed": False, "issues": [f"judge_failed:{resp.get('error', 'unknown')}"], "missing_data": [], "hallucination_flags": [], "suggested_edits": [], "_model": model}
        content = resp.get("response", "")
        obj = _extract_json_block(content)
        if not obj:
            import logging
            logging.getLogger("SectionReviewer").warning(
                "Judge model=%s raw content (len=%d first=%r last=%r)",
                model, len(content or ""),
                (content or "")[:120],
                (content or "")[-120:],
            )
            # Retry once with stronger prompt when first attempt fails to produce parseable JSON
            retry_prompt = (
                "EMERGENCY RETRY: Your previous response was NOT valid JSON.\n"
                "CRITICAL INSTRUCTION: Output ONLY a single Python dictionary. Start with { and end with }.\n"
                "No markdown, no explanation, no code fences. Just the dict.\n\n"
                + prompt[:1500] + "\n\n...[truncated]...\n\n" + prompt[-2000:]
            )
            try:
                resp2 = self.llm_service.generate(
                    model, retry_prompt,
                    timeout=int(self.config.timeout_s),
                    options={"temperature": 0.0, "num_predict": min(int(self.config.max_tokens), 512)},
                    stream_callback=None,
                    response_format={"type": "json_object"},
                )
                if resp2.get("success"):
                    obj_retry = _extract_json_block(resp2.get("response", ""))
                    if obj_retry:
                        obj = obj_retry
                        if not (obj_retry.get("issues") or obj_retry.get("suggested_edits")):
                            obj["issues"] = ["judge returned empty review — auto-assigning generic issues"]
                            obj["suggested_edits"] = ["Re-evaluate clinical relevance, statistical rigor, and evidence grounding."]
            except Exception:
                pass
            if obj is None:
                obj = {"score": score_unparseable, "passed": False, "issues": ["could not parse judge output"], "missing_data": [], "hallucination_flags": [], "suggested_edits": [str(content or "")[:300] if content else "review the draft manually"]}
        score = float(obj.get("score", 5.0))
        score = max(0.0, min(10.0, score))
        passed = bool(obj.get("passed", score >= 8.0))

        def _list(k: str):
            v = obj.get(k, [])
            if isinstance(v, str): return [v]
            if isinstance(v, list): return [str(x) for x in v if str(x).strip()][:10]
            return []
        return {
            "score": score,
            "passed": passed,
            "issues": _list("issues"),
            "missing_data": _list("missing_data"),
            "hallucination_flags": _list("hallucination_flags"),
            "suggested_edits": _list("suggested_edits"),
            "_model": model,
        }

    @staticmethod
    def _merge_reviews(reviews):
        if not reviews:
            return {"score": 7.0, "passed": False, "issues": ["no review available"], "missing_data": [], "hallucination_flags": [], "suggested_edits": []}
        scores = [r.get("score", 0) for r in reviews]
        avg_score = sum(scores) / len(scores) if scores else 5.0
        passed = sum(1 for r in reviews if r.get("passed")) >= len(reviews) / 2
        seen = set()

        def _uniq(key):
            items = []
            for r in reviews:
                for item in r.get(key, []):
                    if item not in seen:
                        seen.add(item)
                        items.append(item)
            return items[:15]

        return {
            "score": round(avg_score, 1),
            "passed": passed,
            "issues": _uniq("issues"),
            "missing_data": _uniq("missing_data"),
            "hallucination_flags": _uniq("hallucination_flags"),
            "suggested_edits": _uniq("suggested_edits"),
            "_panel": [r.get("_model", "?") for r in reviews],
        }

    def should_refine(self, review: Dict[str, Any]) -> bool:
        if not review:
            return False
        issues = review.get("issues") or []
        try:
            issue_list = [str(x) for x in issues]
        except Exception:
            issue_list = []
        if any(x.startswith("judge_failed:") for x in issue_list):
            return False
        score = review.get("score", 0)
        try:
            score = float(score)
        except Exception:
            score = 0.0
        return score < float(self.config.min_score) and not review.get("passed")

    def build_refine_prompt(self, draft_text: str, review: Dict[str, Any]) -> str:
        issues = ", ".join([str(x) for x in (review.get("issues") or [])][:10])
        missing = ", ".join([str(x) for x in (review.get("missing_data") or [])][:10])
        halluc = ", ".join([str(x) for x in (review.get("hallucination_flags") or [])][:10])
        edits = "\n".join([f"- {x}" for x in (review.get("suggested_edits") or [])][:10])
        try:
            from src.ai.prompt_registry import PromptRegistry
            tpl = PromptRegistry.instance().get("refine_prompt")
            if tpl.strip():
                return (tpl
                    .replace("{issues}", issues).replace("{missing_data}", missing)
                    .replace("{hallucination_flags}", halluc).replace("{suggested_edits}", edits)
                    + f"\n\n[ORIGINAL_DRAFT]\n{draft_text}\n")
        except Exception:
            pass
        return (
            "Rewrite the draft section to address the peer review issues.\n"
            "Hard constraints:\n"
            "- Do not invent any statistics, p-values, means, SD, R², PSNR, SSIM, etc.\n"
            "- If data is missing, use placeholders like [MISSING_DATA: MetricName] or [PARAM_MISSING].\n"
            "- Keep in-text citations format [1], [2] if present; do not add new references.\n\n"
            f"[REVIEW_ISSUES]\n{issues}\n\n"
            f"[MISSING_DATA]\n{missing}\n\n"
            f"[HALLUCINATION_FLAGS]\n{halluc}\n\n"
            f"[SUGGESTED_EDITS]\n{edits}\n\n"
            f"[ORIGINAL_DRAFT]\n{draft_text}\n"
        )
