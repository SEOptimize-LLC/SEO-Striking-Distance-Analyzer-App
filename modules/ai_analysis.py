"""
AI-Powered Semantic Analysis Module

Provides tiered AI analysis for keyword relevancy scoring:
- Gemini 2.0 Flash: Bulk filtering
- GPT-4o: Semantic scoring
- Claude Sonnet 4.5: Deep analysis

Optimized for cost efficiency and accuracy via OpenRouter.
"""

import streamlit as st
import requests
import json
import re
import time
from typing import List, Dict, Optional


class AIAnalyzer:
    """Tiered AI analysis for keyword relevancy scoring via OpenRouter."""

    # OpenRouter API configuration
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    # Model mapping to OpenRouter identifiers
    MODELS = {
        "gemini-flash": "google/gemini-2.0-flash-exp:free",
        "gpt-4o": "openai/gpt-4o",
        "claude-sonnet-4": "anthropic/claude-sonnet-4.5"
    }

    def __init__(self, openrouter_key: Optional[str] = None):
        """Initialize AI analyzer with OpenRouter API key.

        Args:
            openrouter_key: OpenRouter API key (optional, uses st.secrets if not provided)
        """
        # Get API key from parameter or Streamlit secrets
        if openrouter_key:
            self.api_key = openrouter_key
        else:
            try:
                self.api_key = st.secrets["OPENROUTER_API_KEY"]
            except (KeyError, FileNotFoundError):
                raise ValueError(
                    "OpenRouter API key not found. Please add OPENROUTER_API_KEY to Streamlit secrets."
                )

        # Configure headers for OpenRouter API
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": st.secrets.get("app_url", "http://localhost:8501"),
            "X-Title": "SEO Striking Distance Analyzer",
            "Content-Type": "application/json"
        }

        # Cache for topic extractions
        self.topic_cache: Dict[str, str] = {}

    def _call_openrouter(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.3
    ) -> str:
        """Make unified API call to OpenRouter.

        Args:
            model: Model identifier ('gemini-flash', 'gpt-4o', 'claude-sonnet-4')
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation (0.0-1.0)

        Returns:
            Model response text

        Raises:
            Exception: If API call fails
        """
        # Get OpenRouter model ID
        model_id = self.MODELS.get(model, model)

        try:
            response = requests.post(
                f"{self.OPENROUTER_BASE_URL}/chat/completions",
                headers=self.headers,
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=60
            )
            response.raise_for_status()

            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except requests.exceptions.Timeout:
            raise Exception(f"OpenRouter API timeout for model {model}")
        except requests.exceptions.HTTPError as e:
            error_msg = f"OpenRouter API HTTP error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_msg += f" - {error_data['error'].get('message', '')}"
            except:
                pass
            raise Exception(error_msg)
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}")

    def extract_url_topic(
        self,
        url: str,
        title: str,
        h1: str,
        content_snippet: str,
        model: str = "gpt-4o"
    ) -> str:
        """Extract the main topic/intent of a URL using AI.

        Args:
            url: The URL to analyze
            title: Page title
            h1: H1 heading
            content_snippet: First ~500 chars of content
            model: AI model to use ('gpt-4o' or 'claude-sonnet-4')

        Returns:
            Concise topic description (2-3 sentences)
        """
        # Check cache first
        cache_key = f"{url}:{title}:{h1}"
        if cache_key in self.topic_cache:
            return self.topic_cache[cache_key]

        prompt = f"""Analyze this webpage and extract its main topic/intent in 2-3 concise sentences.

URL: {url}
Title: {title}
H1: {h1}
Content snippet: {content_snippet[:500]}

Respond with ONLY the topic description, no preamble or explanation."""

        try:
            topic = self._call_openrouter(
                model=model,
                prompt=prompt,
                max_tokens=150,
                temperature=0.3
            )

            # Cache the result
            self.topic_cache[cache_key] = topic
            return topic

        except Exception as e:
            print(f"⚠️ Error extracting topic for {url}: {str(e)}")
            # Fallback: create topic from title + H1
            return f"{title}. {h1}" if h1 and h1 != title else title

    def score_keyword_relevancy(
        self,
        keyword: str,
        url_topic: str,
        model: str = "gpt-4o"
    ) -> int:
        """Score how relevant a keyword is to a URL's topic.

        Args:
            keyword: The keyword to score
            url_topic: Topic description of the URL
            model: AI model to use ('gemini-flash', 'gpt-4o', or 'claude-sonnet-4')

        Returns:
            Relevancy score (0-100)
        """
        prompt = f"""Rate the semantic relevancy between this keyword and URL topic on a scale of 0-100.

Keyword: "{keyword}"
URL Topic: "{url_topic}"

Guidelines:
- 90-100: Exact match or primary topic
- 70-89: Highly relevant, strong semantic connection
- 50-69: Moderately relevant, related concepts
- 30-49: Loosely related, tangential connection
- 0-29: Not relevant or different topic

Respond with ONLY the number (0-100), no explanation."""

        try:
            score_text = self._call_openrouter(
                model=model,
                prompt=prompt,
                max_tokens=10,
                temperature=0.1
            )

            # Extract number from response
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                score = int(numbers[0])
                return min(100, max(0, score))  # Clamp to 0-100
            else:
                return 50  # Default if can't parse

        except Exception as e:
            print(f"⚠️ Error scoring keyword '{keyword}': {str(e)}")
            # Fallback: simple keyword matching
            keyword_lower = keyword.lower()
            topic_lower = url_topic.lower()

            if keyword_lower in topic_lower:
                return 85
            elif any(word in topic_lower for word in keyword_lower.split()):
                return 60
            else:
                return 30

    def batch_score_keywords(
        self,
        keywords: List[str],
        url_topic: str,
        model: str = "gpt-4o",
        show_progress: bool = True
    ) -> Dict[str, int]:
        """Score multiple keywords for relevancy to a URL topic.

        Args:
            keywords: List of keywords to score
            url_topic: Topic description of the URL
            model: AI model to use
            show_progress: Whether to show progress

        Returns:
            Dictionary mapping keywords to relevancy scores (0-100)
        """
        scores = {}

        for i, keyword in enumerate(keywords, 1):
            if show_progress and i % 10 == 0:
                print(f"Scoring keyword {i}/{len(keywords)}...")

            scores[keyword] = self.score_keyword_relevancy(keyword, url_topic, model)

            # Small delay to avoid rate limits
            if i < len(keywords):
                time.sleep(0.1)

        return scores

    def tiered_analysis(
        self,
        keywords: List[str],
        url_topic: str,
        show_progress: bool = True
    ) -> Dict[str, Dict]:
        """Run tiered AI analysis for cost optimization.

        Stage 1: Gemini Flash - bulk filtering (keep score >= 40)
        Stage 2: GPT-4o - detailed scoring on filtered keywords
        Stage 3: Claude Sonnet 4 - deep analysis on top 20%

        Args:
            keywords: List of keywords to analyze
            url_topic: Topic description of the URL
            show_progress: Whether to show progress

        Returns:
            Dictionary mapping keywords to analysis results:
            {
                "keyword": {
                    "relevancy_score": int (0-100),
                    "tier": str ("flash", "gpt4o", "claude"),
                    "confidence": str ("low", "medium", "high")
                }
            }
        """
        results = {}

        # Stage 1: Gemini Flash - Quick filtering
        if show_progress:
            print(f"🔍 Stage 1: Quick filtering with Gemini Flash ({len(keywords)} keywords)...")

        flash_scores = self.batch_score_keywords(keywords, url_topic, model="gemini-flash", show_progress=False)

        # Filter: Keep keywords with score >= 40
        filtered_keywords = [kw for kw, score in flash_scores.items() if score >= 40]

        if show_progress:
            print(f"✓ Stage 1 complete: {len(filtered_keywords)}/{len(keywords)} keywords passed filter")

        # Stage 2: GPT-4o - Detailed scoring
        if filtered_keywords and show_progress:
            print(f"🔬 Stage 2: Detailed scoring with GPT-4o ({len(filtered_keywords)} keywords)...")

        gpt4o_scores = self.batch_score_keywords(filtered_keywords, url_topic, model="gpt-4o", show_progress=False) if filtered_keywords else {}

        if show_progress and filtered_keywords:
            print(f"✓ Stage 2 complete")

        # Stage 3: Claude Sonnet 4 - Top 20% deep analysis
        top_20_percent = int(len(filtered_keywords) * 0.2)
        top_keywords = sorted(gpt4o_scores.items(), key=lambda x: x[1], reverse=True)[:top_20_percent]

        if top_keywords and show_progress:
            print(f"🎯 Stage 3: Deep analysis with Claude Sonnet 4 ({len(top_keywords)} keywords)...")

        claude_scores = {}
        for kw, _ in top_keywords:
            claude_scores[kw] = self.score_keyword_relevancy(kw, url_topic, model="claude-sonnet-4")

        if show_progress and top_keywords:
            print(f"✓ Stage 3 complete")

        # Compile results
        for keyword in keywords:
            if keyword in claude_scores:
                # Claude analysis (highest confidence)
                results[keyword] = {
                    "relevancy_score": claude_scores[keyword],
                    "tier": "claude-sonnet-4",
                    "confidence": "high"
                }
            elif keyword in gpt4o_scores:
                # GPT-4o analysis (medium confidence)
                results[keyword] = {
                    "relevancy_score": gpt4o_scores[keyword],
                    "tier": "gpt-4o",
                    "confidence": "medium"
                }
            else:
                # Gemini Flash only (low confidence)
                results[keyword] = {
                    "relevancy_score": flash_scores.get(keyword, 0),
                    "tier": "gemini-flash",
                    "confidence": "low"
                }

        return results


# Convenience function
def analyze_keywords_with_ai(
    keywords: List[str],
    url_topic: str,
    use_tiered: bool = True
) -> Dict[str, int]:
    """Quick function to analyze keywords with AI.

    Args:
        keywords: List of keywords to analyze
        url_topic: Topic of the URL
        use_tiered: Whether to use tiered analysis (recommended)

    Returns:
        Dictionary mapping keywords to relevancy scores
    """
    analyzer = AIAnalyzer()

    if use_tiered:
        results = analyzer.tiered_analysis(keywords, url_topic)
        return {kw: data["relevancy_score"] for kw, data in results.items()}
    else:
        return analyzer.batch_score_keywords(keywords, url_topic, model="gpt-4o")
