import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai.agent_core_loop import _merge_and_rank_references


def main():
    q = "CT reconstruction deep learning"
    pub = [
        {"source": "pubmed", "pmid": "1", "doi": "", "title": "Deep learning for CT reconstruction", "abstract": "We found AUC 0.91", "year": "2023", "citation": "A (2023). T. J."},
        {"source": "pubmed", "pmid": "2", "doi": "", "title": "Background only study", "abstract": "Background: ...", "year": "2010", "citation": "B (2010). T. J."},
    ]
    epmc = [
        {"source": "europepmc", "pmid": "1", "doi": "10.1/x", "title": "Deep learning for CT reconstruction", "abstract": "AUC 0.91 p<0.001", "year": "2023", "citation": "A (2023). T. J."},
        {"source": "europepmc", "pmid": "", "doi": "10.2/y", "title": "CT reconstruction with diffusion", "abstract": "Results: 120 patients", "year": "2022", "citation": "C (2022). T. J."},
    ]
    guidelines = [
        {"source": "pubmed", "category": "guideline", "pub_types": "Practice Guideline", "pmid": "9", "doi": "", "title": "ACR Appropriateness Criteria for CT reconstruction", "abstract": "Recommendation: use iterative reconstruction when appropriate.", "year": "2016", "citation": "ACR (2016). Criteria. Radiology."},
    ]
    merged = _merge_and_rank_references([guidelines, pub, epmc], q, 10)
    assert len(merged) >= 2
    keys = [(r.get("pmid") or "", r.get("doi") or "", r.get("title") or "") for r in merged]
    assert keys.count(("1", "10.1/x", "Deep learning for CT reconstruction")) == 0
    assert sum(1 for r in merged if r.get("pmid") == "1") == 1
    assert merged[0].get("pmid") == "9"


if __name__ == "__main__":
    main()
