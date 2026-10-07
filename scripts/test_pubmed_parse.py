import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.pubmed_service import PubMedService


def main():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal>
          <Title>Radiology</Title>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Deep learning for MRI reconstruction</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">We evaluate TR and TE.</AbstractText>
          <AbstractText>SliceThickness was 0.6 mm.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1000/test.doi</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""
    svc = PubMedService()
    out = svc._parse_xml(xml)
    assert len(out) == 1
    r = out[0]
    assert r.get("pmid") == "12345678"
    assert "MRI reconstruction" in r.get("title", "")
    assert r.get("doi") == "10.1000/test.doi"
    assert "SliceThickness" in r.get("abstract", "") or "SliceThickness" not in r.get("abstract", "")
    assert "SliceThickness was 0.6 mm" in r.get("abstract", "")
    assert r.get("year") == "2024"
    assert r.get("url", "").endswith("/12345678/")


if __name__ == "__main__":
    main()

