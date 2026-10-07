import json
import logging
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


@dataclass
class CausalEdge:
    src: str
    dst: str
    mechanism: str = ""
    evidence: str = ""
    strength: float = 0.5  # heuristic, not a learned score


@dataclass
class CausalModel:
    nodes: List[str] = field(default_factory=list)          # X, Y, Z candidates
    edges: List[CausalEdge] = field(default_factory=list)   # DAG edges
    confounders: List[str] = field(default_factory=list)    # Z
    treatments: List[str] = field(default_factory=list)     # X
    outcomes: List[str] = field(default_factory=list)       # Y
    causal_claims: List[str] = field(default_factory=list)  # English-only, hedged if needed
    limitations: List[str] = field(default_factory=list)    # what prevents causal identification
    counterfactual_questions: List[str] = field(default_factory=list)
    intervention_notes: List[str] = field(default_factory=list)
    raw_json: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["edges"] = [asdict(e) for e in self.edges]
        return d


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Prefer fenced JSON.
    m = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # Fallback to first {...} block.
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


class CausalEnhancer:
    """
    Causal enhancement for the agent loop (Pearl ladder inspired):
    - Association: summarize variables and correlations (already handled elsewhere)
    - Intervention: propose DAG (SCM) and backdoor adjustment candidates
    - Counterfactual: generate counterfactual audit questions grounded in confounders
    """

    def __init__(self, llm_call, stream_cb=None):
        self._call_llm = llm_call
        self._stream = stream_cb or (lambda *_: None)

    def extract(
        self,
        aggregated_context: str,
        structured_data: Optional[Dict[str, Any]] = None,
        tabular_summary: str = "",
        project_title: str = "",
    ) -> Tuple[Optional[CausalModel], str]:
        """
        Returns (CausalModel|None, status_text). Never raises.
        """
        try:
            ctx = str(aggregated_context or "")
            sd = structured_data or {}
            cognition = str(sd.get("data_cognition_summary") or "")
            stats = sd.get("statistical_findings") or []
            pvals = sd.get("p_values") or []

            self._stream("> 🕸️ [CausalGraph]: Building causal structure model (SCM/DAG)...\n")

            prompt = (
                "[LANGUAGE_LOCK]\n"
                "Output MUST be English-only. Do NOT use Chinese.\n\n"
                "You are helping draft a medical imaging paper. Build a causal DAG (SCM) grounded in the provided evidence.\n"
                "Do NOT invent patient demographics or statistical values. If missing, list them under limitations.\n\n"
                "Return ONLY valid JSON with keys:\n"
                "- treatments: [string]\n"
                "- outcomes: [string]\n"
                "- confounders: [string]\n"
                "- nodes: [string]\n"
                "- edges: [{\"src\": \"\", \"dst\": \"\", \"mechanism\": \"\", \"evidence\": \"\"}]\n"
                "- causal_claims: [string]  (hedged unless identification is explicit)\n"
                "- limitations: [string]\n"
                "- intervention_notes: [string] (what would happen under do(X=...); qualitative)\n"
                "- counterfactual_questions: [string] (at least 3, grounded in confounders)\n\n"
                f"Project title: {str(project_title or '')[:200]}\n\n"
                "[Association summary]\n"
                f"{cognition[:1200]}\n\n"
                "[Structured findings]\n"
                f"statistical_findings: {', '.join([str(x) for x in stats[:12]])}\n"
                f"p_values: {', '.join([str(x) for x in pvals[:12]])}\n\n"
                "[Tabular mining summary]\n"
                f"{str(tabular_summary or '')[:1800]}\n\n"
                "[Raw context excerpt]\n"
                f"{ctx[:4200]}\n"
            )

            res = self._call_llm(prompt)
            obj = _extract_json_object(res)
            if not isinstance(obj, dict):
                return None, "FAIL: causal extractor did not return valid JSON."

            cm = CausalModel()
            cm.raw_json = obj
            cm.treatments = [str(x).strip() for x in _ensure_list(obj.get("treatments")) if str(x).strip()]
            cm.outcomes = [str(x).strip() for x in _ensure_list(obj.get("outcomes")) if str(x).strip()]
            cm.confounders = [str(x).strip() for x in _ensure_list(obj.get("confounders")) if str(x).strip()]
            cm.nodes = [str(x).strip() for x in _ensure_list(obj.get("nodes")) if str(x).strip()]
            cm.causal_claims = [str(x).strip() for x in _ensure_list(obj.get("causal_claims")) if str(x).strip()]
            cm.limitations = [str(x).strip() for x in _ensure_list(obj.get("limitations")) if str(x).strip()]
            cm.counterfactual_questions = [
                str(x).strip() for x in _ensure_list(obj.get("counterfactual_questions")) if str(x).strip()
            ]
            cm.intervention_notes = [
                str(x).strip() for x in _ensure_list(obj.get("intervention_notes")) if str(x).strip()
            ]

            edges = []
            for e in _ensure_list(obj.get("edges")):
                if not isinstance(e, dict):
                    continue
                src = str(e.get("src") or "").strip()
                dst = str(e.get("dst") or "").strip()
                if not src or not dst:
                    continue
                edges.append(
                    CausalEdge(
                        src=src,
                        dst=dst,
                        mechanism=str(e.get("mechanism") or "").strip(),
                        evidence=str(e.get("evidence") or "").strip(),
                    )
                )
            cm.edges = edges

            # Minimal sanity checks.
            if not cm.nodes:
                cm.nodes = list(dict.fromkeys(cm.treatments + cm.outcomes + cm.confounders))
            if not cm.counterfactual_questions:
                cm.counterfactual_questions = [
                    "If the treatment were not applied, would the observed outcome still be expected given the same acquisition protocol?",
                    "If patient size (as a confounder) were different, would the outcome remain directionally consistent?",
                    "If scanner hardware or reconstruction kernel changed, would the effect attribution still hold?",
                ]
            return cm, "PASS"
        except Exception as e:
            logger.exception("Causal extraction failed: %s", e)
            return None, f"FAIL: {type(e).__name__}: {e}"

