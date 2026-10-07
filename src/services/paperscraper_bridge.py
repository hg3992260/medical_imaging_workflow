"""
paperscraper bridge — extends PubMed search with arXiv + bioRxiv + citation metrics.

Usage:
    from src.services.paperscraper_bridge import search_paperscraper
    results = search_paperscraper("CT deep learning dose reduction", max_results=5)
    # Returns normalized list of dicts with keys: pmid, doi, title, abstract, journal, year, authors, url
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("paperscraper_bridge")

_SEARCH_CACHE: Dict[str, Any] = {}
_MAX_CACHE = 200


def search_paperscraper(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search arXiv + bioRxiv via paperscraper, return normalized results."""
    results: List[Dict[str, Any]] = []
    cache_key = f"paperscraper:{query}:{max_results}"
    if cache_key in _SEARCH_CACHE:
        return _SEARCH_CACHE[cache_key]

    # arXiv
    try:
        arxiv_results = _search_arxiv(query, max_results)
        results.extend(arxiv_results)
        logger.info(f"paperscraper: arXiv returned {len(arxiv_results)} results for '{query[:60]}'")
    except Exception as e:
        logger.warning(f"paperscraper: arXiv search failed: {e}")

    # bioRxiv — needs a server check first
    try:
        biorxiv_results = _search_biorxiv(query, max_results)
        if biorxiv_results:
            results.extend(biorxiv_results)
            logger.info(f"paperscraper: bioRxiv returned {len(biorxiv_results)} results")
    except Exception as e:
        logger.debug(f"paperscraper: bioRxiv skipped: {e}")

    # Google Scholar (fallback, slow — limit to 2)
    try:
        if len(results) < max_results:
            scholar_results = _search_scholar(query, max_results=2)
            results.extend(scholar_results)
    except Exception as e:
        logger.debug(f"paperscraper: Google Scholar failed: {e}")

    _SEARCH_CACHE[cache_key] = results[:max_results]
    if len(_SEARCH_CACHE) > _MAX_CACHE:
        _SEARCH_CACHE.clear()
    return results[:max_results]


def _search_arxiv(query: str, max_results: int) -> List[Dict[str, Any]]:
    from arxiv import Search, SortCriterion
    search = Search(query=query, max_results=max_results, sort_by=SortCriterion.Relevance)
    out = []
    for r in search.results():
        out.append({
            "pmid": "",
            "doi": str(r.doi or ""),
            "title": str(r.title or "").strip(),
            "abstract": str(r.summary or "").strip()[:500],
            "journal": "arXiv",
            "year": "",
            "authors": [str(a) for a in r.authors],
            "url": str(r.entry_id or ""),
            "source": "arxiv",
            "citation_count": 0,
        })
    return out


def _search_biorxiv(query: str, max_results: int) -> List[Dict[str, Any]]:
    try:
        from paperscraper.get_xml import get_xml_from_biorxiv
        import urllib.request
        import urllib.parse
        import json
        import xml.etree.ElementTree as ET
        encoded = urllib.parse.quote_plus(query)
        url = f"https://api.biorxiv.org/search?query={encoded}&max={max_results}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
        root = ET.fromstring(body)
        ns = {"a": "http://www.w3.org/2005/Atom", "b": "http://biorxiv.org/ns"}
        out = []
        for entry in root.findall("a:entry", ns):
            title = entry.findtext("a:title", "", ns).strip()
            summary = entry.findtext("a:summary", "", ns).strip()
            doi = (entry.findtext("a:id", "", ns) or "").replace("doi:", "")
            out.append({
                "pmid": "", "doi": doi, "title": title, "abstract": summary[:500],
                "journal": "bioRxiv", "year": "", "authors": [], "url": f"https://doi.org/{doi}",
                "source": "biorxiv", "citation_count": 0,
            })
        return out
    except Exception:
        return []


def _search_scholar(query: str, max_results: int) -> List[Dict[str, Any]]:
    try:
        from scholarly import scholarly
        scholarly.set_timeout(10)
        search_query = scholarly.search_pubs(query)
        out = []
        for i in range(max_results):
            try:
                pub = next(search_query)
                bib = pub.get("bib", {})
                out.append({
                    "pmid": "", "doi": pub.get("pub_url", ""),
                    "title": str(bib.get("title", "") or "").strip(),
                    "abstract": str(bib.get("abstract", "") or "").strip()[:500],
                    "journal": str(bib.get("journal", "") or ""),
                    "year": str(bib.get("pub_year", "") or ""),
                    "authors": bib.get("author", []),
                    "url": str(pub.get("pub_url", "") or ""),
                    "source": "google_scholar",
                    "citation_count": int(pub.get("cites", "0") or "0"),
                })
            except StopIteration:
                break
        return out
    except Exception:
        return []


def merge_results(pubmed_results: List[Dict], paperscraper_results: List[Dict],
                  max_total: int = 8) -> List[Dict]:
    """Merge PubMed + paperscraper results, dedup by DOI, rank by citation count."""
    seen_dois: set = set()
    merged: List[Dict] = []
    for r in pubmed_results + paperscraper_results:
        doi = str(r.get("doi", "") or "").strip().lower()
        if doi and doi in seen_dois:
            continue
        if doi:
            seen_dois.add(doi)
        merged.append(r)
    ranked = sorted(merged, key=lambda x: -int(x.get("citation_count", 0) or 0))
    return ranked[:max_total]
