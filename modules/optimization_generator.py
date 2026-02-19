"""
AI-Powered Optimization Recommendations Generator

Generates actionable SEO optimization suggestions for striking distance opportunities.
Includes smart keyword filtering with intent matching and relevancy validation.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from modules.ai_analysis import AIAnalyzer


class OptimizationGenerator:
    """Generate actionable SEO optimization recommendations with smart filtering."""

    def __init__(self, ai_analyzer: Optional[AIAnalyzer] = None):
        """Initialize optimization generator.

        Args:
            ai_analyzer: AIAnalyzer instance for semantic analysis
        """
        self.ai = ai_analyzer or AIAnalyzer()

    def filter_keywords_by_relevancy(
        self,
        keywords: List[Dict[str, Any]],
        url_topic: str,
        min_relevancy: int = 60
    ) -> Tuple[List[Dict], List[Dict]]:
        """Filter keywords by semantic relevancy to URL topic.

        Args:
            keywords: List of keyword dicts with 'query', 'position', 'clicks', etc.
            url_topic: Main topic/intent of the URL
            min_relevancy: Minimum relevancy score (0-100)

        Returns:
            Tuple of (relevant_keywords, filtered_out_keywords)
        """
        relevant = []
        filtered = []

        for kw in keywords:
            query = kw.get('query', '')

            # Score relevancy
            relevancy_score = self.ai.score_keyword_relevancy(
                keyword=query,
                url_topic=url_topic,
                model="gpt-4o"
            )

            kw_with_score = {**kw, 'relevancy_score': relevancy_score}

            if relevancy_score >= min_relevancy:
                relevant.append(kw_with_score)
            else:
                filtered.append(kw_with_score)

        return relevant, filtered

    def identify_query_intent(self, query: str) -> str:
        """Identify the primary user intent for a query.

        Args:
            query: Search query to analyze

        Returns:
            Intent type: 'informational', 'navigational', 'transactional', 'commercial'
        """
        prompt = f"""Identify the primary search intent for this query:

Query: "{query}"

Respond with ONLY ONE of these intent types:
- informational: User wants to learn or find information
- navigational: User wants to find a specific website or page
- transactional: User wants to buy, download, sign up, or take action
- commercial: User is researching products/services before purchase

Respond with ONLY the intent type, no explanation."""

        try:
            intent = self.ai._call_openrouter(
                model="gpt-4o",
                prompt=prompt,
                max_tokens=10,
                temperature=0.1
            ).strip().lower()

            # Validate response
            valid_intents = ['informational', 'navigational', 'transactional', 'commercial']
            if intent in valid_intents:
                return intent
            else:
                # Fallback: parse from response
                for valid_intent in valid_intents:
                    if valid_intent in intent:
                        return valid_intent
                return 'informational'  # Default

        except Exception as e:
            print(f"⚠️ Error identifying intent for '{query}': {str(e)}")
            return 'informational'  # Default fallback

    def validate_natural_fit(
        self,
        keyword: str,
        page_topic: str,
        current_title: str
    ) -> Dict[str, Any]:
        """Check if keyword can be naturally incorporated into content.

        Args:
            keyword: Keyword to validate
            page_topic: Main topic of the page
            current_title: Current page title

        Returns:
            Dict with 'can_incorporate' (bool), 'confidence' (str), 'reasoning' (str)
        """
        prompt = f"""Evaluate if this keyword can be naturally incorporated into the page content.

Page Topic: "{page_topic}"
Current Title: "{current_title}"
Keyword to Add: "{keyword}"

Can this keyword be naturally added to the page without:
1. Forcing unrelated terms
2. Keyword stuffing
3. Hurting user experience
4. Creating awkward phrasing

Respond in JSON format:
{{
  "can_incorporate": true/false,
  "confidence": "high/medium/low",
  "reasoning": "Brief explanation"
}}"""

        try:
            response = self.ai._call_openrouter(
                model="gpt-4o",
                prompt=prompt,
                max_tokens=150,
                temperature=0.2
            )

            # Parse JSON response
            result = json.loads(response)
            return result

        except Exception as e:
            print(f"⚠️ Error validating natural fit for '{keyword}': {str(e)}")
            # Conservative fallback
            return {
                "can_incorporate": False,
                "confidence": "low",
                "reasoning": "Error during validation"
            }

    def filter_and_rank_keywords(
        self,
        missing_keywords: List[Dict[str, Any]],
        page_context: Dict[str, str],
        min_relevancy: int = 60,
        enforce_intent_match: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """Smart keyword filtering with intent matching and natural language validation.

        Args:
            missing_keywords: Keywords missing from on-page optimization
            page_context: Dict with 'topic', 'current_title', 'primary_intent'
            min_relevancy: Minimum semantic relevancy score
            enforce_intent_match: Whether to filter by intent mismatch

        Returns:
            Tuple of (recommended_keywords, filtered_keywords)
        """
        recommended = []
        filtered = []

        url_topic = page_context.get('topic', '')
        current_title = page_context.get('current_title', '')
        page_intent = page_context.get('primary_intent', 'informational')

        for kw in missing_keywords:
            query = kw.get('query', '')
            reasons_filtered = []

            # Stage 1: Semantic Relevancy
            relevancy_score = self.ai.score_keyword_relevancy(
                keyword=query,
                url_topic=url_topic,
                model="gpt-4o"
            )

            kw_enhanced = {**kw, 'relevancy_score': relevancy_score}

            if relevancy_score < min_relevancy:
                reasons_filtered.append(f"Low relevancy: {relevancy_score}/100")

            # Stage 2: Intent Matching
            if enforce_intent_match:
                query_intent = self.identify_query_intent(query)
                kw_enhanced['intent'] = query_intent

                if query_intent != page_intent:
                    reasons_filtered.append(f"Intent mismatch: query={query_intent}, page={page_intent}")

            # Stage 3: Natural Fit Validation
            natural_fit = self.validate_natural_fit(query, url_topic, current_title)
            kw_enhanced['natural_fit'] = natural_fit

            if not natural_fit.get('can_incorporate', False):
                reasons_filtered.append(f"Cannot naturally incorporate: {natural_fit.get('reasoning', '')}")

            # Decision
            if reasons_filtered:
                kw_enhanced['filtered_reasons'] = reasons_filtered
                filtered.append(kw_enhanced)
            else:
                # Calculate priority score
                priority_score = (
                    relevancy_score * 0.4 +
                    kw.get('clicks', 0) * 2 +
                    kw.get('impressions', 0) * 0.01 +
                    (100 - kw.get('position', 100)) * 0.5
                )
                kw_enhanced['priority_score'] = priority_score
                recommended.append(kw_enhanced)

        # Sort recommended by priority
        recommended.sort(key=lambda x: x.get('priority_score', 0), reverse=True)

        return recommended, filtered

    def generate_title_variations(
        self,
        current_title: str,
        missing_keywords: List[str],
        url: str,
        n_variations: int = 3,
        max_length: int = 60,
        competitive_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate optimized title variations incorporating missing keywords.

        Args:
            current_title: Current page title
            missing_keywords: List of high-priority keywords to incorporate
            url: Page URL for context
            n_variations: Number of variations to generate
            max_length: Maximum title length in characters
            competitive_context: Optional competitive intelligence dict with
                                 competitor title patterns and entity gaps

        Returns:
            List of title variation dicts with 'title', 'keywords_added', 'length', 'reasoning'
        """
        # Limit keywords to top 5 most important
        keywords_str = ', '.join(missing_keywords[:5])

        # Build competitive context section if available
        competitor_section = ""
        if competitive_context:
            brief = competitive_context.get("competitive_brief", {})
            title_patterns = brief.get("title_patterns", {})
            entity_analysis = competitive_context.get("entity_analysis", {})

            if title_patterns.get("examples"):
                examples = "\n".join([f"  - {t}" for t in title_patterns["examples"][:5]])
                separator = title_patterns.get("common_separator")
                avg_len = title_patterns.get("avg_length", "unknown")
                competitor_section += f"""
Competitor Title Intelligence (top-ranking pages for "{competitive_context.get('primary_query', '')}"):
- Competitor titles (positions 1–{len(title_patterns['examples'])}):
{examples}
- Average competitor title length: {avg_len} chars
- Most common separator: {separator or 'none'}
"""

            if entity_analysis and entity_analysis.get("entity_gaps"):
                top_gaps = entity_analysis["entity_gaps"][:5]
                gap_names = ", ".join([e["name"] for e in top_gaps])
                competitor_section += f"\nKey entities missing from this page (vs competitors): {gap_names}\n"

        prompt = f"""You are an SEO expert optimizing page titles with full competitive context.

Current Title: {current_title}
URL: {url}

High-value keywords missing from this title:
{keywords_str}
{competitor_section}
Task: Generate {n_variations} optimized title variations that:
1. Incorporate missing keywords naturally
2. Learn from competitor title patterns above (format, length, style)
3. Stay under {max_length} characters
4. Preserve brand name if present in original
5. Are compelling and click-worthy
6. DON'T copy competitor titles — differentiate while matching the proven format

For each variation, provide:
- The new title
- Which keywords were incorporated
- Brief reasoning (reference competitor patterns where relevant)

Format your response as JSON array:
[
  {{
    "title": "...",
    "keywords_added": ["kw1", "kw2"],
    "reasoning": "..."
  }}
]"""

        try:
            response = self.ai._call_openrouter(
                model="claude-sonnet-4",
                prompt=prompt,
                max_tokens=500,
                temperature=0.7
            )

            # Parse JSON response
            variations = json.loads(response)

            # Add length to each variation
            for var in variations:
                var['length'] = len(var.get('title', ''))

            return variations[:n_variations]

        except Exception as e:
            print(f"⚠️ Error generating title variations: {str(e)}")
            # Fallback: simple title with first keyword
            if missing_keywords:
                fallback_title = f"{missing_keywords[0]} | {current_title}"[:max_length]
                return [{
                    "title": fallback_title,
                    "keywords_added": [missing_keywords[0]],
                    "length": len(fallback_title),
                    "reasoning": "Fallback: prepended top keyword to current title"
                }]
            return []

    def generate_h1_suggestion(
        self,
        current_h1: str,
        missing_keywords: List[str],
        primary_intent: str,
        competitive_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate optimized H1 that incorporates missing keywords.

        Args:
            current_h1: Current H1 heading
            missing_keywords: High-priority keywords to incorporate
            primary_intent: Primary user intent (informational, transactional, etc.)
            competitive_context: Optional competitive intelligence dict

        Returns:
            Dict with 'h1', 'keywords_added', 'reasoning'
        """
        keywords_str = ', '.join(missing_keywords[:3])

        # Build competitor H1 context if available
        competitor_h1_section = ""
        if competitive_context:
            brief = competitive_context.get("competitive_brief", {})
            h1_patterns = brief.get("h1_patterns", {})
            entity_analysis = competitive_context.get("entity_analysis", {})

            if h1_patterns.get("examples"):
                examples = "\n".join(
                    [f"  - {h}" for h in h1_patterns["examples"][:5]]
                )
                competitor_h1_section += (
                    f"\nCompetitor H1s (top-ranking pages):\n{examples}\n"
                    f"Average competitor H1 length: "
                    f"{h1_patterns.get('avg_length', '?')} chars\n"
                )

            if entity_analysis and entity_analysis.get("entity_gaps"):
                top_gaps = [
                    e["name"] for e in entity_analysis["entity_gaps"][:3]
                ]
                competitor_h1_section += (
                    f"Top missing entities: {', '.join(top_gaps)}\n"
                )

        prompt = f"""Generate an optimized H1 heading for SEO with competitive context.

Current H1: {current_h1}
Primary Intent: {primary_intent}
Missing Keywords: {keywords_str}
{competitor_h1_section}
Create ONE optimized H1 that:
1. Incorporates 1-2 of the most important missing keywords naturally
2. Matches the {primary_intent} intent
3. Is clear, concise, and compelling
4. Stays under 70 characters
5. Learns from competitor H1 patterns above without copying them

Respond in JSON format:
{{
  "h1": "...",
  "keywords_added": ["kw1"],
  "reasoning": "Brief explanation referencing competitor patterns"
}}"""

        try:
            response = self.ai._call_openrouter(
                model="gpt-4o",
                prompt=prompt,
                max_tokens=200,
                temperature=0.6
            )

            return json.loads(response)

        except Exception as e:
            print(f"⚠️ Error generating H1: {str(e)}")
            return {
                "h1": f"{missing_keywords[0] if missing_keywords else ''} | {current_h1}",
                "keywords_added": [missing_keywords[0]] if missing_keywords else [],
                "reasoning": "Fallback: prepended top keyword"
            }

    def generate_h2_structure(
        self,
        current_h2s: List[str],
        missing_keyword_clusters: List[Dict],
        content_topic: str
    ) -> List[Dict[str, Any]]:
        """Suggest H2 structure based on keyword clusters.

        Args:
            current_h2s: Current H2 headings on page
            missing_keyword_clusters: Clustered missing keywords
            content_topic: Main topic of content

        Returns:
            List of H2 suggestions with 'h2', 'keywords_targeted', 'content_brief'
        """
        # Limit to top 5 clusters
        clusters_str = json.dumps(missing_keyword_clusters[:5], indent=2)
        current_h2s_str = '\n'.join([f"- {h2}" for h2 in current_h2s[:10]])

        prompt = f"""Suggest an optimized H2 structure for this content.

Content Topic: {content_topic}

Current H2s:
{current_h2s_str}

Missing Keyword Clusters:
{clusters_str}

Generate 3-5 new or improved H2 headings that:
1. Target the missing keyword clusters
2. Complement existing H2 structure
3. Create logical content flow
4. Are clear and compelling
5. Incorporate keywords naturally

For each H2, provide:
- The H2 text
- Keywords targeted
- Brief content outline for that section

Respond in JSON format:
[
  {{
    "h2": "...",
    "keywords_targeted": ["kw1", "kw2"],
    "content_brief": "What to cover in this section..."
  }}
]"""

        try:
            response = self.ai._call_openrouter(
                model="claude-sonnet-4",
                prompt=prompt,
                max_tokens=600,
                temperature=0.6
            )

            return json.loads(response)

        except Exception as e:
            print(f"⚠️ Error generating H2 structure: {str(e)}")
            return []

    def generate_meta_description(
        self,
        current_meta: str,
        missing_keywords: List[str],
        top_queries: List[str],
        max_length: int = 160,
        competitive_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate meta description variations.

        Args:
            current_meta: Current meta description
            missing_keywords: Keywords to incorporate
            top_queries: Top-performing queries for this page
            max_length: Maximum length in characters
            competitive_context: Optional competitive intelligence dict

        Returns:
            List of meta description variations
        """
        keywords_str = ', '.join(missing_keywords[:5])
        queries_str = ', '.join(top_queries[:5])

        # Build competitor meta context if available
        competitor_meta_section = ""
        if competitive_context:
            brief = competitive_context.get("competitive_brief", {})
            pages = brief.get("competitor_pages", [])
            entity_analysis = competitive_context.get("entity_analysis", {})

            competitor_metas = [
                p["meta_description"] for p in pages
                if p.get("meta_description")
            ][:4]
            if competitor_metas:
                examples = "\n".join(
                    [f"  - {m}" for m in competitor_metas]
                )
                competitor_meta_section += (
                    f"\nCompetitor meta descriptions (top-ranking pages):\n"
                    f"{examples}\n"
                )

            if entity_analysis and entity_analysis.get("entity_gaps"):
                top_gaps = [
                    e["name"] for e in entity_analysis["entity_gaps"][:4]
                ]
                competitor_meta_section += (
                    f"Key entities to consider including: "
                    f"{', '.join(top_gaps)}\n"
                )

        prompt = f"""Generate 2 optimized meta description variations with competitive context.

Current Meta Description: {current_meta}
Missing Keywords: {keywords_str}
Top Queries: {queries_str}
{competitor_meta_section}
Create 2 compelling meta descriptions that:
1. Incorporate missing keywords naturally
2. Include a clear call-to-action
3. Stay under {max_length} characters
4. Are compelling and click-worthy
5. Learn from competitor meta approaches above without copying them

Respond in JSON format:
[
  {{
    "meta_description": "...",
    "keywords_added": ["kw1", "kw2"],
    "length": 155,
    "reasoning": "..."
  }}
]"""

        try:
            response = self.ai._call_openrouter(
                model="gpt-4o",
                prompt=prompt,
                max_tokens=400,
                temperature=0.7
            )

            variations = json.loads(response)

            # Add length if not present
            for var in variations:
                if 'length' not in var:
                    var['length'] = len(var.get('meta_description', ''))

            return variations

        except Exception as e:
            print(f"⚠️ Error generating meta descriptions: {str(e)}")
            return []

    def generate_url_optimization_report(
        self,
        url: str,
        current_elements: Dict[str, str],
        missing_keywords: List[Dict],
        ranking_data: Dict,
        page_intent: str = 'informational',
        competitive_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate complete optimization report for a single URL.

        Args:
            url: Page URL
            current_elements: Dict with 'title', 'h1', 'h2s', 'meta_description'
            missing_keywords: List of missing keyword dicts
            ranking_data: Dict with 'position', 'clicks', 'impressions'
            page_intent: Primary intent of the page
            competitive_context: Optional competitive intelligence dict containing
                'primary_query', 'competitive_brief', and 'entity_analysis'

        Returns:
            Comprehensive optimization report
        """
        # Extract URL topic from current elements
        url_topic = (
            f"{current_elements.get('title', '')}. "
            f"{current_elements.get('h1', '')}"
        )

        # Smart keyword filtering
        page_context = {
            'topic': url_topic,
            'current_title': current_elements.get('title', ''),
            'primary_intent': page_intent
        }

        recommended_kws, filtered_kws = self.filter_and_rank_keywords(
            missing_keywords=missing_keywords,
            page_context=page_context,
            min_relevancy=60,
            enforce_intent_match=True
        )

        # Extract top keyword queries
        recommended_queries = [kw['query'] for kw in recommended_kws[:10]]

        report = {
            'url': url,
            'current_performance': {
                'avg_position': ranking_data.get('position', 0),
                'total_clicks': ranking_data.get('clicks', 0),
                'total_impressions': ranking_data.get('impressions', 0)
            },
            'current': current_elements,
            'keyword_analysis': {
                'total_missing': len(missing_keywords),
                'recommended_count': len(recommended_kws),
                'filtered_count': len(filtered_kws),
                'top_recommended': recommended_queries[:5]
            }
        }

        # Attach competitive intelligence to report if available
        if competitive_context:
            report['competitive_context'] = competitive_context

        # Generate title variations if we have recommended keywords
        if recommended_queries:
            report['title_variations'] = self.generate_title_variations(
                current_title=current_elements.get('title', ''),
                missing_keywords=recommended_queries,
                url=url,
                n_variations=3,
                max_length=60,
                competitive_context=competitive_context
            )

            # Generate H1 suggestion
            report['h1_suggestion'] = self.generate_h1_suggestion(
                current_h1=current_elements.get('h1', ''),
                missing_keywords=recommended_queries,
                primary_intent=page_intent,
                competitive_context=competitive_context
            )

            # Generate meta descriptions
            top_queries = [kw['query'] for kw in missing_keywords[:10]]
            report['meta_descriptions'] = self.generate_meta_description(
                current_meta=current_elements.get('meta_description', ''),
                missing_keywords=recommended_queries,
                top_queries=top_queries,
                max_length=160,
                competitive_context=competitive_context
            )
        else:
            report['recommendation'] = (
                "No high-priority keyword opportunities found that fit "
                "naturally with page content."
            )

        return report
