import os
import time
import logging
import re
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class EuropePMCService:
    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"

    def __init__(self):
        self.timeout = float(os.environ.get("EUROPE_PMC_TIMEOUT", "15"))
        self.verify_ssl = os.environ.get("EUROPE_PMC_VERIFY_SSL", "1").strip().lower() not in ["0", "false", "no", "off", ""]
        self.cache_ttl_s = int(os.environ.get("EUROPE_PMC_CACHE_TTL_S", "600"))
        self.max_results_cap = int(os.environ.get("EUROPE_PMC_MAX_RESULTS_CAP", "50"))
        self._cache: Dict[Tuple[str, int], Tuple[float, List[Dict[str, str]]]] = {}

        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        q = (query or "").strip()
        if not q:
            return []
        max_results = max(1, min(int(max_results), self.max_results_cap))

        key = (q, max_results)
        now = time.time()
        cached = self._cache.get(key)
        if cached and (now - cached[0] <= self.cache_ttl_s):
            return cached[1]

        params = {
            "query": q,
            "format": "json",
            "pageSize": str(max_results),
            "sort": "relevance",
            "resultType": "core",
        }
        url = f"{self.BASE_URL}/search"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout, verify=self.verify_ssl)
            resp.raise_for_status()
            data = resp.json() if resp.text else {}
            parsed = self._parse_json(data)
            self._cache[key] = (now, parsed)
            return parsed
        except Exception as e:
            logger.error(f"Europe PMC execution error: {e}")
            return []

    def search_guidelines(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        q = (query or "").strip()
        if not q:
            return []
        max_results = max(1, min(int(max_results), self.max_results_cap))
        term = self._build_guideline_query(q)
        results = self.search(term, max_results=max_results) or []
        for r in results or []:
            if isinstance(r, dict):
                r["category"] = "guideline"
        return results

    def _build_guideline_query(self, query: str) -> str:
        q = (query or "").strip()
        if not q:
            return ""
        intent = "(guideline OR consensus OR statement OR recommendation OR appropriateness)"
        org = "(ACR OR RSNA OR ESR OR ECR)"
        return f"({q}) AND {intent} AND {org}"

    def _parse_json(self, data: dict) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        results = (data or {}).get("resultList", {}).get("result", []) or []
        for it in results:
            if not isinstance(it, dict):
                continue
            title = (it.get("title") or "").strip()
            journal = (it.get("journalTitle") or "").strip()
            year = (it.get("pubYear") or "").strip() or "n.d."
            author_string = (it.get("authorString") or "").strip()
            doi = (it.get("doi") or "").strip()
            pmid = (it.get("pmid") or "").strip()
            abstract = (it.get("abstractText") or "").strip()
            pub_type = (it.get("pubType") or "").strip()
            pub_types = ""
            if it.get("pubTypeList") and isinstance(it.get("pubTypeList"), dict):
                pts = []
                for pt in it.get("pubTypeList", {}).get("pubType", []) or []:
                    if isinstance(pt, str) and pt.strip():
                        if pt.strip() not in pts:
                            pts.append(pt.strip())
                pub_types = "; ".join(pts)

            author_text = author_string
            if not author_text and it.get("authorList") and isinstance(it.get("authorList"), dict):
                authors = []
                for a in it.get("authorList", {}).get("author", []) or []:
                    if not isinstance(a, dict):
                        continue
                    last = (a.get("lastName") or "").strip()
                    initials = (a.get("initials") or "").strip()
                    if last:
                        authors.append(f"{last} {initials}".strip() if initials else last)
                author_text = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")

            citation = f"{author_text} ({year}). {title}. {journal}.".strip()
            url = ""
            if pmid:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            elif doi:
                url = f"https://doi.org/{doi}"

            if not title and not abstract:
                continue

            out.append(
                {
                    "source": "europepmc",
                    "pmid": pmid,
                    "doi": doi,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "pub_types": pub_types or pub_type,
                    "year": year,
                    "citation": citation,
                    "url": url,
                }
            )
        return out


def normalize_title_for_dedup(title: str) -> str:
    t = (title or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", "", t)
    return t
