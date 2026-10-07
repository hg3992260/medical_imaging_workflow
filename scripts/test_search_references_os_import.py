import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.agent_core_loop import AgentCoreLoopExecutor
from src.ai.coco_flows import ResearchOutline


class DummyPubMed:
    def search(self, query: str, max_results: int = 5):
        return []

    def search_guidelines(self, query: str, max_results: int = 5):
        return []


class DummyEPMC:
    def search(self, query: str, max_results: int = 10):
        return []

    def search_guidelines(self, query: str, max_results: int = 10):
        return []


def main():
    ex = AgentCoreLoopExecutor(kb=None, db_service=None, progress_cb=None, stream_cb=None)
    ex.pubmed_service = DummyPubMed()
    ex.europe_pmc_service = DummyEPMC()
    outline = ResearchOutline(
        title="Test Title",
        core_claims=["claim a", "claim b"],
        primary_hypothesis="H1",
        null_hypothesis="H0",
        methodology_highlights="m",
        key_results=["r"],
        target_audience="a",
    )
    refs = ex._search_references(outline)
    assert isinstance(refs, list)


if __name__ == "__main__":
    main()

