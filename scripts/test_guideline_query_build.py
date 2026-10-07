import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.pubmed_service import PubMedService
from src.services.europe_pmc_service import EuropePMCService


def main():
    p = PubMedService()
    term = p._build_guideline_term("CT pulmonary angiography")
    assert "[Publication Type]" in term
    assert "Practice Guideline" in term
    assert "[Title/Abstract]" in term

    e = EuropePMCService()
    q = e._build_guideline_query("CT pulmonary angiography")
    assert "guideline" in q.lower()
    assert "consensus" in q.lower()


if __name__ == "__main__":
    main()

