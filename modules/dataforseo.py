"""
Data For SEO API Integration Module

Fetches search volume and keyword difficulty data from DataForSEO API.
Optimized for cost efficiency with 1,000-keyword batching.
"""

import requests
import time
import streamlit as st
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import base64


class DataForSEOClient:
    """Client for DataForSEO API with optimized batching and rate limiting."""

    # API Configuration
    BASE_URL = "https://api.dataforseo.com/v3"
    ENDPOINT = "/keywords_data/google_ads/search_volume/live"

    # Rate limiting: 12 requests per minute = 1 request every 5 seconds
    MIN_REQUEST_INTERVAL = 5.0  # seconds

    # Batching configuration
    MAX_KEYWORDS_PER_BATCH = 1000  # Maximum keywords per API request

    # US Location code for DataForSEO
    US_LOCATION_CODE = 2840

    def __init__(self, login: Optional[str] = None, password: Optional[str] = None):
        """Initialize DataForSEO client with credentials from Streamlit secrets.

        Args:
            login: DataForSEO login email (optional, uses st.secrets if not provided)
            password: DataForSEO password (optional, uses st.secrets if not provided)
        """
        # Try to get credentials from parameters or Streamlit secrets
        if login and password:
            self.login = login
            self.password = password
        else:
            try:
                self.login = st.secrets["DATAFORSEO_LOGIN"]
                self.password = st.secrets["DATAFORSEO_PASSWORD"]
            except (KeyError, FileNotFoundError) as e:
                raise ValueError(
                    "DataForSEO credentials not found. Please add DATAFORSEO_LOGIN and "
                    "DATAFORSEO_PASSWORD to your Streamlit secrets."
                ) from e

        # Create base64 encoded auth header
        credentials = f"{self.login}:{self.password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        self.headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/json"
        }

        # Track last request time for rate limiting
        self.last_request_time = 0

        # Cache for API responses to avoid duplicate calls
        self.cache: Dict[str, Dict] = {}

    def _rate_limit(self):
        """Enforce rate limiting: wait if needed before making next request."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.MIN_REQUEST_INTERVAL:
            sleep_time = self.MIN_REQUEST_INTERVAL - time_since_last_request
            print(f"⏳ Rate limiting: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _create_batch_payload(self, keywords: List[str]) -> List[Dict]:
        """Create API payload for a batch of keywords.

        Args:
            keywords: List of keywords (max 1000)

        Returns:
            List with single task payload
        """
        if len(keywords) > self.MAX_KEYWORDS_PER_BATCH:
            raise ValueError(
                f"Batch size {len(keywords)} exceeds maximum of {self.MAX_KEYWORDS_PER_BATCH}"
            )

        # Calculate date range for last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        payload = [{
            "location_code": self.US_LOCATION_CODE,  # United States
            "language_code": "en",  # English
            "keywords": keywords,
            "date_from": start_date.strftime("%Y-%m-%d"),
            "date_to": end_date.strftime("%Y-%m-%d"),
            # Only get search volume data (no historic trends)
            "search_partners": False,
            "include_adult_keywords": False
        }]

        return payload

    def _make_request(self, payload: List[Dict]) -> Dict:
        """Make API request with error handling.

        Args:
            payload: Request payload

        Returns:
            API response data
        """
        url = f"{self.BASE_URL}{self.ENDPOINT}"

        # Apply rate limiting
        self._rate_limit()

        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if data.get("status_code") != 20000:
                error_msg = data.get("status_message", "Unknown error")
                raise Exception(f"DataForSEO API error: {error_msg}")

            return data

        except requests.exceptions.Timeout:
            raise Exception("DataForSEO API request timed out")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"DataForSEO API HTTP error: {e.response.status_code}")
        except Exception as e:
            raise Exception(f"DataForSEO API error: {str(e)}")

    def _parse_response(self, response_data: Dict) -> Dict[str, Dict]:
        """Parse API response and extract keyword metrics.

        Args:
            response_data: Raw API response

        Returns:
            Dictionary mapping keywords to their metrics
        """
        results = {}

        try:
            tasks = response_data.get("tasks", [])

            for task in tasks:
                if task.get("status_code") != 20000:
                    continue

                task_result = task.get("result", [])

                for result in task_result:
                    keyword_data = result.get("keyword_data", {})
                    keyword = result.get("keyword", "")

                    if keyword:
                        results[keyword.lower()] = {
                            "keyword": keyword,
                            "search_volume": keyword_data.get("keyword_info", {}).get("search_volume", 0),
                            "competition": keyword_data.get("keyword_info", {}).get("competition", 0),
                            "cpc": keyword_data.get("keyword_info", {}).get("cpc", 0),
                            # Map competition (0-1) to keyword difficulty (0-100)
                            "keyword_difficulty": int(keyword_data.get("keyword_info", {}).get("competition", 0) * 100),
                            "monthly_searches": keyword_data.get("keyword_info", {}).get("monthly_searches", [])
                        }

        except Exception as e:
            print(f"⚠️ Error parsing DataForSEO response: {str(e)}")

        return results

    def get_keyword_metrics(
        self,
        keywords: List[str],
        show_progress: bool = True
    ) -> Dict[str, Dict]:
        """Get search volume and keyword difficulty for a list of keywords.

        Automatically batches keywords into groups of 1,000 for cost efficiency.

        Args:
            keywords: List of keywords to analyze
            show_progress: Whether to print progress messages

        Returns:
            Dictionary mapping keywords to their metrics:
            {
                "keyword": {
                    "search_volume": int,
                    "keyword_difficulty": int (0-100),
                    "competition": float (0-1),
                    "cpc": float
                }
            }
        """
        # Remove duplicates and normalize
        unique_keywords = list(set([kw.strip().lower() for kw in keywords if kw.strip()]))

        if not unique_keywords:
            return {}

        # Check cache first
        uncached_keywords = [kw for kw in unique_keywords if kw not in self.cache]

        if show_progress:
            cached_count = len(unique_keywords) - len(uncached_keywords)
            if cached_count > 0:
                print(f"✓ Found {cached_count} keywords in cache")
            print(f"📡 Fetching metrics for {len(uncached_keywords)} keywords...")

        # Split into batches of 1,000 keywords
        batches = [
            uncached_keywords[i:i + self.MAX_KEYWORDS_PER_BATCH]
            for i in range(0, len(uncached_keywords), self.MAX_KEYWORDS_PER_BATCH)
        ]

        if show_progress and batches:
            print(f"📦 Processing {len(batches)} batch(es) of up to {self.MAX_KEYWORDS_PER_BATCH} keywords...")

        # Process each batch
        all_results = {}

        for i, batch in enumerate(batches, 1):
            if show_progress:
                print(f"🔄 Batch {i}/{len(batches)}: {len(batch)} keywords...")

            try:
                # Create payload
                payload = self._create_batch_payload(batch)

                # Make request
                response = self._make_request(payload)

                # Parse response
                batch_results = self._parse_response(response)

                # Update cache and results
                self.cache.update(batch_results)
                all_results.update(batch_results)

                if show_progress:
                    print(f"✓ Batch {i}/{len(batches)} complete: {len(batch_results)} keywords processed")

            except Exception as e:
                if show_progress:
                    print(f"❌ Error processing batch {i}: {str(e)}")
                # Continue with next batch even if one fails
                continue

        # Combine cached and newly fetched results
        final_results = {**{kw: self.cache[kw] for kw in unique_keywords if kw in self.cache}}

        if show_progress:
            print(f"✅ Total keywords enriched: {len(final_results)}/{len(unique_keywords)}")

        return final_results

    def enrich_dataframe(self, df, keyword_column: str = 'query'):
        """Enrich a dataframe with search volume and keyword difficulty.

        Args:
            df: Pandas DataFrame with keywords
            keyword_column: Name of the column containing keywords

        Returns:
            DataFrame with added columns: search_volume, keyword_difficulty, cpc
        """
        import pandas as pd

        # Get unique keywords from dataframe
        keywords = df[keyword_column].dropna().unique().tolist()

        # Fetch metrics
        metrics = self.get_keyword_metrics(keywords)

        # Map metrics back to dataframe
        df['search_volume'] = df[keyword_column].apply(
            lambda x: metrics.get(str(x).lower(), {}).get('search_volume', 0) if pd.notna(x) else 0
        )
        df['keyword_difficulty'] = df[keyword_column].apply(
            lambda x: metrics.get(str(x).lower(), {}).get('keyword_difficulty', 0) if pd.notna(x) else 0
        )
        df['cpc'] = df[keyword_column].apply(
            lambda x: metrics.get(str(x).lower(), {}).get('cpc', 0) if pd.notna(x) else 0
        )

        return df


# Convenience function for quick access
def get_keyword_data(keywords: List[str], show_progress: bool = True) -> Dict[str, Dict]:
    """Quick function to get keyword data using Streamlit secrets.

    Args:
        keywords: List of keywords to analyze
        show_progress: Whether to show progress messages

    Returns:
        Dictionary of keyword metrics
    """
    client = DataForSEOClient()
    return client.get_keyword_metrics(keywords, show_progress)
