import os
import sys
import inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.agent_core_loop import AgentCoreLoopExecutor
from src.ai.coco_flows import ResearchOutline, UnitSummary


def main():
    ex = AgentCoreLoopExecutor(kb=None, db_service=None, progress_cb=None, stream_cb=None)
    captured = {"prompt": ""}

    def fake_call_llm(prompt: str, options=None) -> str:
        captured["prompt"] = prompt or ""
        return "## 4. Discussion\n\n5.1 Rationalization\n\n5.2 Conflict & Divergence Analysis\n\n5.3 Implications\n\n5.4 Limitations & Outlook\n"

    ex._call_llm = fake_call_llm

    u1 = UnitSummary(unit_id="Unit 1", summary_text="intro summary", significance="High")
    u3 = UnitSummary(unit_id="Unit 3", summary_text="results summary", significance="High")
    outline = ResearchOutline(
        title="Test Title",
        core_claims=["c1", "c2"],
        primary_hypothesis="H1",
        null_hypothesis="H0",
        methodology_highlights="m",
        key_results=["r"],
        target_audience="a",
    )
    sig = inspect.signature(ex._draft_discussion)
    if "project_id" in sig.parameters:
        ex._draft_discussion(
            u1,
            u3,
            outline,
            project_id="p",
            ref_context="- A (2025). Paper. J.\n",
            novelty_report="",
            fact_matrix="",
            logic_audit_report='{"consensus_table":[],"conflict_analysis":[],"technical_gap":[]}',
            structured_data={},
        )
    else:
        ex._draft_discussion(
            u1,
            u3,
            outline,
            ref_context="- A (2025). Paper. J.\n",
            novelty_report="",
        )

    p = captured["prompt"]
    assert "Relevant Literature" in p
    assert "Introduction Summary" in p
    assert "Results Summary" in p
    if "LOGIC_AUDIT_REPORT_LAR" in p:
        assert "MANDATORY CITATION CONSTRAINT" in p or "Cite" in p


if __name__ == "__main__":
    main()
