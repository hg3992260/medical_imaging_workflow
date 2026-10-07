import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.europe_pmc_service import EuropePMCService


def main():
    sample = {
        "hitCount": 2,
        "resultList": {
            "result": [
                {
                    "title": "Deep learning for CT reconstruction",
                    "journalTitle": "Radiology",
                    "pubYear": "2023",
                    "authorString": "Zhang X, Li Y, Wang Z",
                    "doi": "10.1000/xyz",
                    "pmid": "11111111",
                    "abstractText": "RESULTS: AUC improved from 0.82 to 0.91 (p<0.001). CONCLUSION: Better performance.",
                },
                {
                    "title": "Another paper",
                    "journalTitle": "Med Imaging",
                    "pubYear": "2020",
                    "authorString": "Chen A",
                    "doi": "",
                    "pmid": "",
                    "abstractText": "",
                },
            ]
        },
    }
    svc = EuropePMCService()
    out = svc._parse_json(sample)
    assert len(out) == 2
    assert out[0].get("source") == "europepmc"
    assert out[0].get("pmid") == "11111111"
    assert out[0].get("doi") == "10.1000/xyz"
    assert out[0].get("url", "").endswith("/11111111/")


if __name__ == "__main__":
    main()

