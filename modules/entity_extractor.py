"""
Entity Extractor Module

Extracts named entities from competitor pages using a two-layer approach:
1. Google Natural Language API — entity detection + salience scoring
2. AI (Claude Sonnet via OpenRouter) — strategic interpretation of entity gaps

Only runs on pre-filtered, top-priority queries — never on raw GSC data.
"""

import requests
import json
from typing import List, Dict, Optional, Any
from collections import defaultdict
import streamlit as st
from modules.ai_analysis import AIAnalyzer


class EntityExtractor:
    """Extract and analyze topical entities using Google NLP + AI interpretation.

    Supports two authentication methods (in priority order):
    1. Simple API key  — GOOGLE_NLP_API_KEY in Streamlit secrets
    2. OAuth2 bearer token — reuses the Google OAuth token from the GSC
       authentication flow (requires the user to have re-authenticated after
       the cloud-language scope was added to the OAuth scopes list)
    """

    GOOGLE_NLP_URL = "https://language.googleapis.com/v1/documents:analyzeEntities"

    def __init__(self, api_key: Optional[str] = None, oauth_token: Optional[str] = None):
        """Initialize entity extractor.

        Auth resolution order:
        1. api_key parameter
        2. GOOGLE_NLP_API_KEY in st.secrets
        3. oauth_token parameter
        4. OAuth token from gsc_credentials in st.session_state

        Args:
            api_key: Simple Google API key (optional)
            oauth_token: OAuth2 bearer token string (optional)
        """
        self.api_key = None
        self.oauth_token = None

        # Priority 1 & 2: Simple API key
        if api_key:
            self.api_key = api_key
        elif "GOOGLE_NLP_API_KEY" in st.secrets and st.secrets.get("GOOGLE_NLP_API_KEY", "").strip():
            self.api_key = st.secrets["GOOGLE_NLP_API_KEY"].strip()

        # Priority 3 & 4: OAuth2 bearer token
        if not self.api_key:
            if oauth_token:
                self.oauth_token = oauth_token
            else:
                # Reuse the Google OAuth token already authenticated for GSC
                creds_dict = st.session_state.get("gsc_credentials", {})
                token = creds_dict.get("token", "").strip() if creds_dict else ""
                if token:
                    self.oauth_token = token

        if not self.api_key and not self.oauth_token:
            raise ValueError(
                "No Google credentials available for Natural Language API. "
                "Either add GOOGLE_NLP_API_KEY to Streamlit secrets, or "
                "authenticate with Google via the 'Connect to GSC' tab first "
                "(you may need to disconnect and re-authenticate to include the "
                "cloud-language scope)."
            )

        # AI analyzer for strategic interpretation — optional, gracefully degrades
        try:
            self.ai = AIAnalyzer()
        except Exception:
            self.ai = None

    def _build_request_url_and_headers(self) -> tuple:
        """Build the NLP API URL and headers based on available auth method.

        Returns:
            Tuple of (url, headers)
        """
        if self.api_key:
            return f"{self.GOOGLE_NLP_URL}?key={self.api_key}", {}
        else:
            return self.GOOGLE_NLP_URL, {"Authorization": f"Bearer {self.oauth_token}"}

    def extract_entities_google_nlp(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using Google Natural Language API.

        Args:
            text: Text content to analyze. Truncated to 10,000 chars for cost control.

        Returns:
            List of entity dicts sorted by salience (highest first):
            [{name, type, salience, mention_count, wikipedia_url}]
        """
        if not text or len(text.strip()) < 50:
            return []

        # Truncate to control cost (1 unit = 1,000 chars)
        text = text[:10000]

        payload = {
            "document": {
                "type": "PLAIN_TEXT",
                "content": text
            },
            "encodingType": "UTF8"
        }

        try:
            url, headers = self._build_request_url_and_headers()
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            entities = []
            for entity in data.get("entities", []):
                entity_type = entity.get("type", "OTHER")
                salience = entity.get("salience", 0)

                # Skip very low salience noise (< 0.01)
                if salience < 0.01:
                    continue

                entities.append({
                    "name": entity.get("name", ""),
                    "type": entity_type,
                    "salience": round(salience, 4),
                    "mention_count": len(entity.get("mentions", [])),
                    "wikipedia_url": entity.get("metadata", {}).get("wikipedia_url", "")
                })

            entities.sort(key=lambda x: x["salience"], reverse=True)
            return entities

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            detail = e.response.text[:300] if e.response.text else ""
            print(f"⚠️ Google NLP API HTTP {status}: {detail}")
            return []
        except Exception as e:
            print(f"⚠️ Entity extraction error: {str(e)}")
            return []

    def analyze_competitor_entities(
        self,
        competitor_pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run entity extraction on all competitor pages and aggregate results.

        Aggregates entity salience and page frequency across all analyzed pages.
        Importance score = avg_salience × page_frequency × 100

        Args:
            competitor_pages: List of scraped competitor page dicts (must have 'content' key)

        Returns:
            Aggregated entity list sorted by importance_score (highest first):
            [{name, type, avg_salience, page_count, page_frequency,
              total_mentions, importance_score, wikipedia_url, pages}]
        """
        entity_aggregator: Dict[str, Dict] = defaultdict(lambda: {
            "name": "",
            "type": "OTHER",
            "total_salience": 0.0,
            "page_count": 0,
            "total_mentions": 0,
            "pages": [],
            "wikipedia_url": ""
        })

        total_pages = len(competitor_pages)

        for page in competitor_pages:
            content = page.get("content", "")
            url = page.get("url", "")

            if not content or len(content) < 100:
                continue

            entities = self.extract_entities_google_nlp(content)

            for entity in entities:
                name_key = entity["name"].lower().strip()
                if not name_key or len(name_key) < 2:
                    continue

                agg = entity_aggregator[name_key]
                agg["name"] = entity["name"]  # Preserve original casing
                agg["type"] = entity["type"]
                agg["total_salience"] += entity["salience"]
                agg["page_count"] += 1
                agg["total_mentions"] += entity["mention_count"]
                agg["pages"].append(url)
                if entity.get("wikipedia_url") and not agg["wikipedia_url"]:
                    agg["wikipedia_url"] = entity["wikipedia_url"]

        results = []
        for name_key, agg in entity_aggregator.items():
            page_count = agg["page_count"]
            if page_count == 0:
                continue

            avg_salience = agg["total_salience"] / page_count
            page_frequency = page_count / total_pages if total_pages > 0 else 0
            importance_score = avg_salience * page_frequency * 100

            results.append({
                "name": agg["name"],
                "type": agg["type"],
                "avg_salience": round(avg_salience, 4),
                "page_count": page_count,
                "total_pages": total_pages,
                "page_frequency": round(page_frequency, 2),
                "total_mentions": agg["total_mentions"],
                "importance_score": round(importance_score, 4),
                "wikipedia_url": agg["wikipedia_url"],
                "pages": list(set(agg["pages"]))
            })

        results.sort(key=lambda x: x["importance_score"], reverse=True)
        return results

    def extract_your_page_entities(self, content: str) -> set:
        """Extract entity names (lowercase) from the user's own page.

        Args:
            content: Page content text

        Returns:
            Set of lowercase entity names found on the page
        """
        if not content:
            return set()
        entities = self.extract_entities_google_nlp(content)
        return {e["name"].lower().strip() for e in entities if e.get("name")}

    def find_entity_gaps(
        self,
        your_page_content: str,
        competitor_entities: List[Dict[str, Any]],
        min_page_frequency: float = 0.4,
        top_n: int = 20
    ) -> List[Dict[str, Any]]:
        """Find entities present in competitor pages but absent from your page.

        Filters to entities appearing in at least min_page_frequency of competitor
        pages (default: 40%) to surface only the most consensual gaps.

        Args:
            your_page_content: Text content of the user's page
            competitor_entities: Aggregated list from analyze_competitor_entities()
            min_page_frequency: Minimum proportion of competitor pages (0.0–1.0)
            top_n: Maximum number of gap entities to return

        Returns:
            List of gap entity dicts, sorted by importance_score
        """
        your_entities = self.extract_your_page_entities(your_page_content)
        gaps = []

        for entity in competitor_entities:
            if entity["page_frequency"] < min_page_frequency:
                continue

            entity_name_lower = entity["name"].lower().strip()
            if entity_name_lower not in your_entities:
                gaps.append({
                    **entity,
                    "missing_from_your_page": True
                })

        return gaps[:top_n]

    def get_ai_interpretation(
        self,
        entity_gaps: List[Dict[str, Any]],
        page_topic: str,
        keyword: str
    ) -> str:
        """Use Claude to provide strategic interpretation of entity gaps.

        Args:
            entity_gaps: List of gap entity dicts
            page_topic: Main topic/intent of the user's page
            keyword: Target keyword being analyzed

        Returns:
            AI-generated strategic analysis as a string (3–5 bullet points)
        """
        if not self.ai or not entity_gaps:
            return ""

        gap_list = "\n".join([
            f"- {e['name']} ({e['type'].lower()}) — "
            f"in {int(e['page_frequency'] * 100)}% of top-ranking pages, "
            f"avg salience: {e['avg_salience']:.3f}"
            for e in entity_gaps[:15]
        ])

        prompt = f"""You are an expert SEO content strategist. A webpage is trying to rank for a target keyword but is missing key entities that top-ranking competitors consistently reference.

Page Topic: "{page_topic}"
Target Keyword: "{keyword}"

Entities found in top-ranking competitor pages but MISSING from this page:
{gap_list}

Provide a concise strategic analysis in 4–5 bullet points covering:
- Which entity gaps are most critical to close and why (focus on topical authority)
- How these missing entities signal content depth gaps to Google
- Natural, specific ways to incorporate the top 3–5 missing entities
- What these gaps reveal about the semantic gap between this page and competitors

Be specific, actionable, and SEO-focused. Avoid generic advice."""

        try:
            return self.ai._call_openrouter(
                model="claude-sonnet-4",
                prompt=prompt,
                max_tokens=600,
                temperature=0.4
            )
        except Exception as e:
            print(f"⚠️ AI entity interpretation error: {str(e)}")
            return ""

    def run_full_analysis(
        self,
        competitor_pages: List[Dict[str, Any]],
        your_page_content: str,
        page_topic: str,
        keyword: str
    ) -> Dict[str, Any]:
        """Run the complete entity analysis pipeline for a keyword.

        Steps:
        1. Google NLP extracts + aggregates entities from all competitor pages
        2. Identifies entities missing from the user's page (min 40% page frequency)
        3. Claude provides strategic interpretation of the gaps

        Args:
            competitor_pages: Scraped competitor pages (from SerpAnalyzer)
            your_page_content: Text content of the user's own page
            page_topic: Main topic/intent of the user's page
            keyword: Target keyword being analyzed

        Returns:
            Dict with: keyword, competitor_entities, your_entities,
                       entity_gaps, ai_interpretation, pages_analyzed
        """
        # Step 1: Aggregate competitor entities
        competitor_entities = self.analyze_competitor_entities(competitor_pages)

        # Step 2: Find gaps vs. user's page
        entity_gaps = self.find_entity_gaps(
            your_page_content=your_page_content,
            competitor_entities=competitor_entities
        )

        # Step 3: Extract user's own entities for display
        your_entities = self.extract_entities_google_nlp(your_page_content)

        # Step 4: AI strategic interpretation
        ai_interpretation = self.get_ai_interpretation(
            entity_gaps=entity_gaps,
            page_topic=page_topic,
            keyword=keyword
        )

        return {
            "keyword": keyword,
            "pages_analyzed": len(competitor_pages),
            "competitor_entities": competitor_entities[:30],
            "your_entities": [e["name"] for e in your_entities[:20]],
            "entity_gaps": entity_gaps,
            "ai_interpretation": ai_interpretation
        }
