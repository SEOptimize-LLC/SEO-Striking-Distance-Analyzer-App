"""
SERP Analyzer Module

Fetches top organic SERP results for target keywords via DataForSEO,
scrapes competitor pages, and extracts structured SEO element patterns
(titles, H1s, H2s) for competitive intelligence.

Only runs on pre-filtered, top-priority queries — never on raw GSC data.
"""

import requests
import time
import base64
from typing import List, Dict, Optional, Any
from bs4 import BeautifulSoup
import streamlit as st


class SerpAnalyzer:
    """Fetch SERP results and extract competitive patterns from top-ranking pages."""

    BASE_URL = "https://api.dataforseo.com/v3"
    SERP_ENDPOINT = "/serp/google/organic/live/regular"
    MIN_REQUEST_INTERVAL = 5.0  # seconds — matches DataForSEO rate limit

    SCRAPER_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }

    def __init__(self, login: Optional[str] = None, password: Optional[str] = None):
        """Initialize SERP analyzer with DataForSEO credentials.

        Args:
            login: DataForSEO login email (optional, uses st.secrets if not provided)
            password: DataForSEO password (optional, uses st.secrets if not provided)
        """
        if login and password:
            self.login = login
            self.password = password
        else:
            try:
                self.login = st.secrets["DATAFORSEO_LOGIN"]
                self.password = st.secrets["DATAFORSEO_PASSWORD"]
            except (KeyError, FileNotFoundError) as e:
                raise ValueError(
                    "DataForSEO credentials not found. Add DATAFORSEO_LOGIN and "
                    "DATAFORSEO_PASSWORD to Streamlit secrets."
                ) from e

        credentials = f"{self.login}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        self.api_headers = {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json"
        }

        self.last_request_time = 0
        self._serp_cache: Dict[str, List[Dict]] = {}

    def _rate_limit(self):
        """Enforce DataForSEO rate limit between API requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()

    def fetch_serp_results(
        self,
        keyword: str,
        location_code: int = 2840,
        language_code: str = "en",
        depth: int = 10
    ) -> List[Dict[str, Any]]:
        """Fetch top organic SERP results for a keyword via DataForSEO.

        Args:
            keyword: Target keyword to look up
            location_code: DataForSEO location code (2840 = United States)
            language_code: Language code
            depth: Number of results to retrieve (max 10 for cost control)

        Returns:
            List of SERP result dicts: {position, url, title, description, domain}
        """
        cache_key = f"{keyword}:{location_code}:{depth}"
        if cache_key in self._serp_cache:
            return self._serp_cache[cache_key]

        self._rate_limit()

        payload = [{
            "keyword": keyword,
            "location_code": location_code,
            "language_code": language_code,
            "depth": depth,
            "se_domain": "google.com"
        }]

        try:
            response = requests.post(
                f"{self.BASE_URL}{self.SERP_ENDPOINT}",
                headers=self.api_headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status_code") != 20000:
                raise Exception(
                    f"DataForSEO SERP error: {data.get('status_message', 'Unknown error')}"
                )

            results = []
            for task in data.get("tasks", []):
                if task.get("status_code") != 20000:
                    continue
                for task_result in task.get("result", []):
                    for item in task_result.get("items", []):
                        if item.get("type") == "organic":
                            results.append({
                                "position": item.get("rank_absolute", 0),
                                "url": item.get("url", ""),
                                "title": item.get("title", ""),
                                "description": item.get("description", ""),
                                "domain": item.get("domain", "")
                            })

            results = sorted(results, key=lambda x: x["position"])[:depth]
            self._serp_cache[cache_key] = results
            return results

        except requests.exceptions.HTTPError as e:
            print(f"⚠️ DataForSEO SERP HTTP error for '{keyword}': {e.response.status_code}")
            return []
        except Exception as e:
            print(f"⚠️ DataForSEO SERP error for '{keyword}': {str(e)}")
            return []

    def scrape_competitor_page(self, url: str, timeout: int = 15) -> Dict[str, Any]:
        """Scrape a competitor page and extract structured SEO elements.

        Extracts title, H1, H2s, meta description, and main content text.

        Args:
            url: Competitor page URL
            timeout: Request timeout in seconds

        Returns:
            Dict with: url, title, h1, h2s, meta_description, content, success
        """
        try:
            response = requests.get(
                url,
                headers=self.SCRAPER_HEADERS,
                timeout=timeout
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract title tag
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else ""

            # Extract first H1
            h1_tag = soup.find('h1')
            h1 = h1_tag.get_text(strip=True) if h1_tag else ""

            # Extract H2s (up to 10)
            h2s = [tag.get_text(strip=True) for tag in soup.find_all('h2')][:10]

            # Extract meta description
            meta_tag = soup.find('meta', attrs={'name': 'description'})
            if not meta_tag:
                meta_tag = soup.find('meta', attrs={'property': 'og:description'})
            meta_description = meta_tag.get('content', '') if meta_tag else ""

            # Extract main content — strip nav/footer noise first
            for unwanted in soup(['script', 'style', 'header', 'nav', 'footer', 'aside']):
                unwanted.decompose()

            main = None
            for selector in ['main', 'article', '.content', '.main-content',
                              '.post-content', '.entry-content', '#content', '#main']:
                main = soup.select_one(selector)
                if main:
                    break

            if not main:
                main = soup.body or soup

            # Limit content to 5000 chars — enough for entity analysis without waste
            content = main.get_text(separator=' ', strip=True)[:5000]

            return {
                "url": url,
                "title": title,
                "h1": h1,
                "h2s": h2s,
                "meta_description": meta_description,
                "content": content,
                "success": bool(content and len(content) > 100)
            }

        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout scraping competitor: {url}")
        except requests.exceptions.HTTPError as e:
            print(f"⚠️ HTTP {e.response.status_code} scraping competitor: {url}")
        except Exception as e:
            print(f"⚠️ Error scraping competitor {url}: {str(e)[:100]}")

        return {
            "url": url, "title": "", "h1": "", "h2s": [],
            "meta_description": "", "content": "", "success": False
        }

    def scrape_top_competitors(
        self,
        serp_results: List[Dict],
        max_pages: int = 5,
        exclude_domains: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Scrape the top N competitor pages from SERP results.

        Args:
            serp_results: List of SERP result dicts from fetch_serp_results()
            max_pages: Maximum number of pages to scrape
            exclude_domains: Domains to skip (e.g. the user's own domain)

        Returns:
            List of successfully scraped page dicts with SEO elements
        """
        exclude_domains = exclude_domains or []
        scraped = []

        for result in serp_results:
            if len(scraped) >= max_pages:
                break

            url = result.get("url", "")
            domain = result.get("domain", "")

            if not url:
                continue

            # Skip excluded domains (e.g. user's own site)
            if any(ex in domain or ex in url for ex in exclude_domains):
                continue

            page_data = self.scrape_competitor_page(url)

            if page_data["success"]:
                page_data["serp_position"] = result.get("position", 0)
                scraped.append(page_data)

            # Polite delay between competitor scrapes
            time.sleep(1.5)

        return scraped

    def extract_title_patterns(self, competitor_pages: List[Dict]) -> Dict[str, Any]:
        """Summarize title tag patterns across competitor pages.

        Args:
            competitor_pages: List of scraped page dicts

        Returns:
            Dict with: examples, avg_length, common_separator, count
        """
        titles = [p["title"] for p in competitor_pages if p.get("title")]
        if not titles:
            return {}

        avg_length = sum(len(t) for t in titles) / len(titles)

        has_pipe = sum(1 for t in titles if '|' in t)
        has_dash = sum(1 for t in titles if ' - ' in t)
        has_colon = sum(1 for t in titles if ':' in t)

        if has_pipe >= has_dash and has_pipe >= has_colon:
            common_separator = "|"
        elif has_dash >= has_colon:
            common_separator = "-"
        elif has_colon > 0:
            common_separator = ":"
        else:
            common_separator = None

        return {
            "examples": titles,
            "avg_length": round(avg_length),
            "common_separator": common_separator,
            "count": len(titles)
        }

    def extract_h1_patterns(self, competitor_pages: List[Dict]) -> Dict[str, Any]:
        """Summarize H1 patterns across competitor pages.

        Args:
            competitor_pages: List of scraped page dicts

        Returns:
            Dict with: examples, avg_length, count
        """
        h1s = [p["h1"] for p in competitor_pages if p.get("h1")]
        if not h1s:
            return {}

        avg_length = sum(len(h) for h in h1s) / len(h1s)

        return {
            "examples": h1s,
            "avg_length": round(avg_length),
            "count": len(h1s)
        }

    def build_competitive_brief(
        self,
        keyword: str,
        serp_results: List[Dict],
        scraped_pages: List[Dict]
    ) -> Dict[str, Any]:
        """Build a competitive intelligence brief for a keyword.

        Combines SERP data and scraped competitor page analysis into a
        single dict that gets passed to the optimization generator.

        Args:
            keyword: The target keyword
            serp_results: Raw SERP results from DataForSEO
            scraped_pages: Scraped and parsed competitor pages

        Returns:
            Competitive brief dict with patterns and raw competitor data
        """
        return {
            "keyword": keyword,
            "serp_results_count": len(serp_results),
            "pages_analyzed": len(scraped_pages),
            "title_patterns": self.extract_title_patterns(scraped_pages),
            "h1_patterns": self.extract_h1_patterns(scraped_pages),
            "competitor_h2s": [
                {"url": p["url"], "position": p.get("serp_position", 0), "h2s": p["h2s"][:6]}
                for p in scraped_pages if p.get("h2s")
            ],
            "competitor_pages": scraped_pages
        }
