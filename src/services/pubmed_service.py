import os
import time
import requests
import xml.etree.ElementTree as ET
import re
import logging
from typing import List, Dict, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class PubMedService:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
        "should", "would", "may", "might", "must", "study", "analysis", "using", "based",
        "result", "results", "conclusion", "background", "aim", "method", "methods",
        "patient", "patients", "group", "groups", "case", "cases", "report", "reports",
        "review", "meta-analysis", "systematic", "cohort", "retrospective", "prospective",
        "clinical", "significance", "significant", "correlated", "correlates", "detection",
        "identification", "evaluation", "assessment", "found", "showed", "demonstrated",
        "between", "among", "during", "within", "without", "through", "under", "above"
    }

    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self.timeout = float(os.environ.get("PUBMED_TIMEOUT", "15"))
        self.verify_ssl = os.environ.get("PUBMED_VERIFY_SSL", "1").strip().lower() not in ["0", "false", "no", "off", ""]
        self.tool = os.environ.get("PUBMED_TOOL", "medical_imaging_workflow")
        self.email = os.environ.get("PUBMED_EMAIL", "")
        self.api_key = os.environ.get("NCBI_API_KEY", os.environ.get("PUBMED_API_KEY", ""))
        self.max_results_cap = int(os.environ.get("PUBMED_MAX_RESULTS_CAP", "20"))
        self.cache_ttl_s = int(os.environ.get("PUBMED_CACHE_TTL_S", "600"))
        self._cache: Dict[Tuple[str, int], Tuple[float, List[Dict[str, str]]]] = {}

        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search PubMed with optimized query strategy.
        1. Try raw query (if short)
        2. Try extracted keywords (AND)
        3. Try relaxed keywords (OR/subset)
        """
        results = []
        q = (query or "").strip()
        if not q:
            return []

        max_results = max(1, min(int(max_results), self.max_results_cap))
        
        # Strategy 1: Raw query if it's short enough to be a specific title or term
        if len(q.split()) < 10:
            title_q = self._as_title_query(q)
            logger.info(f"PubMed: Attempting title query: '{title_q}'")
            results = self._execute_search(title_q, max_results)
            if results:
                return results
            tiab_q = self._as_tiab_query(q)
            logger.info(f"PubMed: Attempting tiab query: '{tiab_q}'")
            results = self._execute_search(tiab_q, max_results)
            if results: return results

        # Strategy 2: Keyword Extraction
        keywords = self._extract_keywords(q)
        if not keywords:
            logger.warning("PubMed: No keywords extracted.")
            return []
            
        # Try full keyword combination
        optimized_query = self._keywords_to_tiab_query(keywords[:4], op="AND")
        logger.info(f"PubMed: Attempting optimized query (AND): '{optimized_query}'")
        results = self._execute_search(optimized_query, max_results)
        if results: return results
        
        # Strategy 3: Relaxed Search (Top 3 keywords or OR)
        if len(keywords) > 3:
            relaxed_query = self._keywords_to_tiab_query(keywords[:3], op="AND")
            logger.info(f"PubMed: Attempting relaxed query (Top 3): '{relaxed_query}'")
            results = self._execute_search(relaxed_query, max_results)
            if results: return results
            
        # Strategy 4: Fallback to OR for broad coverage
        broad_query = self._keywords_to_tiab_query(keywords[:6], op="OR")
        logger.info(f"PubMed: Attempting broad query (OR): '{broad_query}'")
        results = self._execute_search(broad_query, max_results)
        
        return results

    def search_guidelines(self, query: str, max_results: int = 6) -> List[Dict[str, str]]:
        q = (query or "").strip()
        if not q:
            return []
        max_results = max(1, min(int(max_results), self.max_results_cap))
        term = self._build_guideline_term(q)
        if not term:
            return []
        results = self._execute_search(term, max_results)
        for r in results:
            if isinstance(r, dict):
                r["category"] = "guideline"
        return results

    def _build_guideline_term(self, query: str) -> str:
        q = (query or "").strip()
        if not q:
            return ""
        q_tiab = self._as_tiab_query(q)
        pub_types = [
            "\"Practice Guideline\"[Publication Type]",
            "\"Guideline\"[Publication Type]",
            "\"Consensus Development Conference\"[Publication Type]",
            "\"Consensus Development Conference, NIH\"[Publication Type]",
        ]
        intent = [
            "(guideline[Title/Abstract])",
            "(consensus[Title/Abstract])",
            "(statement[Title/Abstract])",
            "(recommendation[Title/Abstract])",
            "(appropriateness[Title/Abstract])",
        ]
        org = [
            "(ACR[Title/Abstract])",
            "(RSNA[Title/Abstract])",
            "(ESR[Title/Abstract])",
            "(ECR[Title/Abstract])",
        ]
        pt_block = " OR ".join(pub_types)
        intent_block = " OR ".join(intent)
        org_block = " OR ".join(org)
        return f"({q_tiab}) AND (({pt_block}) OR ({intent_block})) AND ({org_block} OR ({intent_block}))"

    def _execute_search(self, term: str, max_results: int) -> List[Dict[str, str]]:
        try:
            key = (term, int(max_results))
            now = time.time()
            cached = self._cache.get(key)
            if cached and (now - cached[0] <= self.cache_ttl_s):
                return cached[1]

            common = {"tool": self.tool}
            if self.email:
                common["email"] = self.email
            if self.api_key:
                common["api_key"] = self.api_key

            # 1. Search for IDs
            search_url = f"{self.BASE_URL}/esearch.fcgi"
            params = {
                **common,
                "db": "pubmed",
                "term": term,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance"
            }
            resp = self.session.get(search_url, params=params, timeout=self.timeout, verify=self.verify_ssl)
            resp.raise_for_status()
            data = resp.json() if resp.text else {}
            id_list = data.get("esearchresult", {}).get("idlist", [])
            
            if not id_list:
                return []

            # 2. Fetch details
            fetch_url = f"{self.BASE_URL}/efetch.fcgi"
            fetch_params = {
                **common,
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml"
            }
            fetch_resp = self.session.get(fetch_url, params=fetch_params, timeout=self.timeout, verify=self.verify_ssl)
            fetch_resp.raise_for_status()

            parsed = self._parse_xml(fetch_resp.text)
            self._cache[key] = (now, parsed)
            return parsed
        except Exception as e:
            logger.error(f"PubMed execution error: {e}")
            return []

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        raw = (text or "").strip()
        if not raw:
            return []

        priority = []
        for w in re.findall(r"\b[A-Z]{2,}\b", raw):
            wl = w.lower()
            if wl not in self.STOP_WORDS and wl not in priority:
                priority.append(w)

        clean_text = re.sub(r"[^\w\s-]", " ", raw.lower())
        words = []
        for w in clean_text.split():
            if w in self.STOP_WORDS:
                continue
            if len(w) <= 2 or w.isdigit():
                continue
            if w not in words:
                words.append(w)

        final = priority + [w for w in words if w.upper() not in priority]
        return final[:8]

    def _quote_term(self, term: str) -> str:
        t = re.sub(r'\s+', " ", (term or "").strip())
        t = t.replace('"', "")
        if " " in t or "-" in t:
            return f"\"{t}\""
        return t

    def _as_title_query(self, query: str) -> str:
        q = self._quote_term(query)
        return f"{q}[Title]"

    def _as_tiab_query(self, query: str) -> str:
        q = self._quote_term(query)
        return f"{q}[Title/Abstract]"

    def _keywords_to_tiab_query(self, keywords: List[str], op: str) -> str:
        parts = []
        for k in keywords:
            kt = self._quote_term(k)
            if not kt:
                continue
            parts.append(f"({kt}[Title/Abstract])")
        if not parts:
            return ""
        joiner = f" {op} "
        return joiner.join(parts)

    def _parse_xml(self, xml_content: str) -> List[Dict[str, str]]:
        results = []
        try:
            root = ET.fromstring(xml_content)
            for article in root.findall(".//PubmedArticle"):
                pmid = (article.findtext(".//PMID") or "").strip()

                title_node = article.find(".//ArticleTitle")
                title_text = "".join(title_node.itertext()).strip() if title_node is not None else "No Title"

                journal_text = (article.findtext(".//Journal/Title") or "").strip()

                year = (article.findtext(".//PubDate/Year") or "").strip()
                if not year:
                    medline_date = (article.findtext(".//PubDate/MedlineDate") or "").strip()
                    m = re.search(r"\b(19|20)\d{2}\b", medline_date)
                    year = m.group(0) if m else "n.d."

                authors = []
                for author in article.findall(".//Author"):
                    last = (author.findtext("LastName") or "").strip()
                    initials = (author.findtext("Initials") or "").strip()
                    if last:
                        authors.append(f"{last} {initials}".strip() if initials else last)

                author_text = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_text += " et al."

                abstract_parts = []
                for abs_node in article.findall(".//Abstract/AbstractText"):
                    label = abs_node.attrib.get("Label")
                    txt = "".join(abs_node.itertext()).strip()
                    if not txt:
                        continue
                    abstract_parts.append(f"{label}: {txt}" if label else txt)
                abstract_text = "\n".join(abstract_parts).strip()

                pub_types = []
                for pt in article.findall(".//PublicationTypeList/PublicationType"):
                    t = "".join(pt.itertext()).strip()
                    if t and t not in pub_types:
                        pub_types.append(t)

                doi = ""
                for aid in article.findall(".//ArticleIdList/ArticleId"):
                    if aid.attrib.get("IdType") == "doi":
                        doi = (aid.text or "").strip()
                        break

                citation = f"{author_text} ({year}). {title_text}. {journal_text}."
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                results.append({
                    "source": "pubmed",
                    "pmid": pmid,
                    "doi": doi,
                    "title": title_text,
                    "abstract": abstract_text,
                    "journal": journal_text,
                    "pub_types": "; ".join(pub_types),
                    "citation": citation,
                    "year": year,
                    "url": url
                })
        except Exception as e:
            logger.error(f"XML parse error: {e}")
        return results
