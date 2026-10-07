import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.agent_core_loop import _build_logic_audit_prompt, _normalize_lar, _try_parse_json


def main():
    p = _build_logic_audit_prompt("INTERNAL", "WEB")
    assert "SEED/KB_FACTS" in p
    assert "WEB_LATEST_INSIGHTS" in p
    assert "consensus_table" in p

    sample = {
        "consensus_table": [{"topic": "t", "fact": "f", "confidence": "high"}],
        "conflict_analysis": [{"point": "x", "internal_version": "i", "external_version": "e", "reasoning": "r", "paper_slot": "Discussion"}],
        "technical_gap": ["g"],
    }
    norm = _normalize_lar(sample)
    assert isinstance(norm, dict)
    assert isinstance(norm.get("consensus_table"), list)

    parsed = _try_parse_json('{"consensus_table":[],"conflict_analysis":[],"technical_gap":[]}')
    assert isinstance(parsed, dict)


if __name__ == "__main__":
    main()

