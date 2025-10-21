"""
AI-Powered Semantic Analysis Module

Provides tiered AI analysis for keyword relevancy scoring:
- Gemini 2.0 Flash: Bulk filtering
- GPT-4o: Semantic scoring
- Claude Sonnet 4.5: Deep analysis

Optimized for cost efficiency and accuracy.
"""

import streamlit as st
from typing import List, Dict, Optional, Tuple
import time
from anthropic import Anthropic
import openai
import google.generativeai as genai


class AIAnalyzer:
    """Tiered AI analysis for keyword relevancy scoring."""

    def __init__(
        self,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        google_key: Optional[str] = None
    ):
        """Initialize AI clients with API keys from Streamlit secrets.

        Args:
            openai_key: OpenAI API key (optional, uses st.secrets if not provided)
            anthropic_key: Anthropic API key (optional, uses st.secrets if not provided)
            google_key: Google AI API key (optional, uses st.secrets if not provided)
        """
        # Initialize OpenAI
        if openai_key:
            self.openai_client = openai.OpenAI(api_key=openai_key)
        else:
            try:
                self.openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            except (KeyError, FileNotFoundError):
                self.openai_client = None

        # Initialize Anthropic
        if anthropic_key:
            self.anthropic_client = Anthropic(api_key=anthropic_key)
        else:
            try:
                self.anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
            except (KeyError, FileNotFoundError):
                self.anthropic_client = None

        # Initialize Google AI
        if google_key:
            genai.configure(api_key=google_key)
            self.google_configured = True
        else:
            try:
                genai.configure(api_key=st.secrets["GOOGLE_AI_API_KEY"])
                self.google_configured = True
            except (KeyError, FileNotFoundError):
                self.google_configured = False

        # Cache for topic extractions
        self.topic_cache: Dict[str, str] = {}

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
            if model == "claude-sonnet-4" and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=150,
                    temperature=0.3,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                topic = response.content[0].text.strip()

            elif model == "gpt-4o" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=150,
                    temperature=0.3,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                topic = response.choices[0].message.content.strip()

            else:
                # Fallback: create topic from title + H1
                topic = f"{title}. {h1}" if h1 and h1 != title else title

            # Cache the result
            self.topic_cache[cache_key] = topic
            return topic

        except Exception as e:
            print(f"⚠️ Error extracting topic for {url}: {str(e)}")
            # Fallback
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
            if model == "gemini-flash" and self.google_configured:
                model_obj = genai.GenerativeModel("gemini-2.0-flash-exp")
                response = model_obj.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=10,
                        temperature=0.1
                    )
                )
                score_text = response.text.strip()

            elif model == "gpt-4o" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=10,
                    temperature=0.1,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                score_text = response.choices[0].message.content.strip()

            elif model == "claude-sonnet-4" and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=10,
                    temperature=0.1,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                score_text = response.content[0].text.strip()

            else:
                # Fallback: simple keyword matching
                keyword_lower = keyword.lower()
                topic_lower = url_topic.lower()

                if keyword_lower in topic_lower:
                    return 85
                elif any(word in topic_lower for word in keyword_lower.split()):
                    return 60
                else:
                    return 30

            # Extract number from response
            import re
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                score = int(numbers[0])
                return min(100, max(0, score))  # Clamp to 0-100
            else:
                return 50  # Default if can't parse

        except Exception as e:
            print(f"⚠️ Error scoring keyword '{keyword}': {str(e)}")
            return 50  # Default fallback score

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
