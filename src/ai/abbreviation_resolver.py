"""
Abbreviation Resolution Engine

System-wide abbreviation detector and resolver that prevents hallucination
caused by ambiguous medical abbreviations (e.g., CAT → Computerized Axial Tomography
vs. cat → feline).

Three-tier resolution:
  1. Embedded KB (resources/abbreviations.json): 150+ radiology abbreviations
  2. Contextual scanning: find full expansion in source document text
  3. Risk classification: flag high-risk abbreviations for RAG sanitization

Usage:
    from src.ai.abbreviation_resolver import AbbreviationResolver
    resolver = AbbreviationResolver()
    resolved = resolver.resolve(text, source_document_text)
    sanitized = resolver.sanitize_rag_query(query_text)
    risks = resolver.check_domain_risk(draft_text)
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("AbbreviationResolver")

_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "abbreviations.json")


class AbbreviationResolver:
    """Centralized abbreviation resolution for hallucination prevention."""

    _instance: Optional["AbbreviationResolver"] = None

    @classmethod
    def instance(cls) -> "AbbreviationResolver":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.kb: Dict[str, Any] = {"entries": {}, "global_vet_exclusions": [], "human_clinical_anchors": []}
        self._load_kb()
        self._sort_entries_by_length_desc()

    def _load_kb(self) -> None:
        try:
            if os.path.isfile(_KB_PATH):
                with open(_KB_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.kb["entries"] = data.get("entries", {})
                self.kb["global_vet_exclusions"] = data.get("global_vet_exclusions", [])
                self.kb["human_clinical_anchors"] = data.get("human_clinical_anchors", [])
                logger.info(f"AbbreviationResolver: loaded %d entries from %s", len(self.kb["entries"]), _KB_PATH)
            else:
                logger.warning("AbbreviationResolver: KB not found at %s, using empty registry", _KB_PATH)
        except Exception as e:
            logger.warning("AbbreviationResolver: failed to load KB: %s", e)

    def _sort_entries_by_length_desc(self) -> None:
        entries = self.kb.get("entries", {})
        sorted_keys = sorted(entries.keys(), key=len, reverse=True)
        self.kb["entries"] = {k: entries[k] for k in sorted_keys}

    def resolve(self, text: str, source_context: str = "") -> str:
        """Resolve known abbreviations in text using KB + contextual scanning.
        
        Priority:
          1. Source context scan (find "Computed Tomography (CT)" pattern)
          2. KB expansion
          3. Leave unchanged if unresolvable
        """
        if not text or not self.kb["entries"]:
            return str(text or "")

        t = str(text)
        entries = self.kb["entries"]

        for abbr, entry in entries.items():
            # Match exact abbreviation case-sensitively first, then case-insensitively
            if not re.search(rf'\b{re.escape(abbr)}\b', t):
                continue

            expansion = entry.get("expansion", abbr)
            vet_risk = entry.get("vet_risk", "none")

            context_expansion = self._find_in_source_context(abbr, source_context)
            if context_expansion:
                expansion = context_expansion

            if vet_risk == "high":
                replacement = f"{expansion} ({abbr})" if abbr != expansion else expansion
                t = re.sub(rf'\b{re.escape(abbr)}\b', replacement, t)
            else:
                if abbr != expansion and expansion not in t:
                    first_occurrence = True
                    def _replace_first(match):
                        nonlocal first_occurrence
                        if first_occurrence:
                            first_occurrence = False
                            return f"{expansion} ({match.group(0)})"
                        return match.group(0)
                    t = re.sub(rf'\b{re.escape(abbr)}\b', _replace_first, t)

        return t

    def _find_in_source_context(self, abbr: str, source_text: str) -> Optional[str]:
        """Scan source text for the full expansion of an abbreviation.
        
        Patterns matched:
          - "Computed Tomography (CT)"
          - "CT — Computed Tomography"
          - "CT: Computed Tomography"
        """
        if not source_text or not abbr:
            return None
        entry = self.kb["entries"].get(abbr, {})
        full = entry.get("expansion", "")
        if not full:
            return None
        patterns = [
            rf'{re.escape(full)}\s*\(\s*{re.escape(abbr)}\s*\)',
            rf'{re.escape(abbr)}\s*[-–—:]\s*{re.escape(full)}',
            rf'{re.escape(full)}\s*\(\s*{re.escape(abbr.upper())}\s*\)',
        ]
        for pat in patterns:
            if re.search(pat, source_text, re.IGNORECASE):
                return full
        return None

    def sanitize_rag_query(self, query: str) -> str:
        """Purify a RAG/retrieval query:
        1. Resolve abbreviations
        2. Append exclusion terms for high-risk abbreviations
        3. Append global veterinary exclusions
        """
        if not query:
            return str(query or "")

        q = self.resolve(query)

        exclusions: Set[str] = set()
        entries = self.kb["entries"]
        for abbr, entry in entries.items():
            if re.search(rf'\b{re.escape(abbr)}\b', query, re.IGNORECASE):
                rag_exclude = entry.get("rag_exclude", [])
                exclusions.update(rag_exclude)

        if exclusions:
            existing_exclude = "NOT (" in q.upper()
            if not existing_exclude:
                q += " NOT (" + " OR ".join(sorted(exclusions)[:30]) + ")"

        return q

    def check_domain_risk(self, text: str) -> List[str]:
        """Check text for domain risks:
        - Veterinary/animal terms without human clinical context
        - Ambiguous abbreviations used in risky contexts
        """
        if not text:
            return []

        risks: List[str] = []
        t = str(text or "").lower()

        vet_terms = self.kb.get("global_vet_exclusions", [])
        human_anchors = self.kb.get("human_clinical_anchors", [])

        found_vet = [w for w in vet_terms if w.lower() in t]
        found_human = [w for w in human_anchors if w.lower() in t]

        if found_vet and not found_human:
            risks.append(f"Domain risk: veterinary/animal terms ({', '.join(found_vet[:5])}) without human clinical anchors")

        entries = self.kb["entries"]
        for abbr, entry in entries.items():
            if entry.get("vet_risk") != "high":
                continue
            if re.search(rf'\b{re.escape(abbr)}\b', t):
                if not any(w in t for w in human_anchors[:10]):
                    risks.append(f"High-risk abbreviation '{abbr}' used without clinical context anchors")

        return risks

    def get_expansion(self, abbr: str) -> Optional[str]:
        """Get the full expansion of an abbreviation from KB."""
        entries = self.kb.get("entries", {})
        entry = entries.get(abbr) or entries.get(abbr.upper()) or entries.get(abbr.lower())
        return entry.get("expansion") if entry else None

    def is_ambiguous(self, abbr: str) -> bool:
        """Check if an abbreviation is flagged as ambiguous."""
        entries = self.kb.get("entries", {})
        entry = entries.get(abbr) or entries.get(abbr.upper()) or entries.get(abbr.lower())
        return entry.get("ambiguous", False) if entry else False
