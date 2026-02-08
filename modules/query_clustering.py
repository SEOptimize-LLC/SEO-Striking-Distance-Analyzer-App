"""
Query Clustering Module

Groups semantically related queries to identify content themes and keyword clusters.
Helps understand which keyword groups should be targeted together.
"""

import json
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict
from modules.ai_analysis import AIAnalyzer


def cluster_queries_by_url(
    results_df: pd.DataFrame,
    ai_analyzer: AIAnalyzer
) -> Dict[str, List[Dict]]:
    """Cluster queries for each URL by semantic similarity.

    Args:
        results_df: Striking distance analysis results DataFrame
        ai_analyzer: AI analyzer for semantic grouping

    Returns:
        Dictionary mapping URLs to their query clusters:
        {
            "url1": [
                {
                    "cluster_name": "CRM Software Comparison",
                    "queries": [
                        {"query": "best crm", "volume": 5000, "position": 8},
                        ...
                    ],
                    "total_volume": 7500,
                    "avg_position": 10.0,
                    "optimized_count": 1,
                    "missing_count": 1
                },
                ...
            ]
        }
    """
    clustered_data = {}

    # Group by URL
    for url in results_df['url'].unique():
        url_data = results_df[results_df['url'] == url]

        # Get all queries for this URL
        queries = url_data['query'].tolist()

        if len(queries) == 0:
            continue

        # Cluster queries semantically
        clusters = semantic_cluster_queries(queries, ai_analyzer)

        # Enrich clusters with metrics from dataframe
        enriched_clusters = []
        for cluster in clusters:
            cluster_queries = cluster.get('queries', [])

            # Get metrics for each query in cluster
            enriched_queries = []
            optimized_count = 0
            missing_count = 0

            for query_text in cluster_queries:
                query_row = url_data[url_data['query'] == query_text]

                if not query_row.empty:
                    query_row = query_row.iloc[0]

                    query_dict = {
                        'query': query_text,
                        'position': query_row.get('position', 0),
                        'clicks': query_row.get('clicks', 0),
                        'impressions': query_row.get('impressions', 0),
                        'search_volume': query_row.get('search_volume', 0) if 'search_volume' in query_row else None
                    }

                    enriched_queries.append(query_dict)

                    # Count optimization status
                    if query_row.get('overall_optimized', False):
                        optimized_count += 1
                    else:
                        missing_count += 1

            # Calculate cluster metrics
            total_volume = sum(q.get('search_volume', 0) or 0 for q in enriched_queries)
            avg_position = sum(q.get('position', 0) for q in enriched_queries) / len(enriched_queries) if enriched_queries else 0

            enriched_cluster = {
                'cluster_name': cluster.get('cluster_name', 'Unnamed Cluster'),
                'queries': enriched_queries,
                'total_volume': total_volume,
                'avg_position': avg_position,
                'optimized_count': optimized_count,
                'missing_count': missing_count,
                'total_queries': len(enriched_queries)
            }

            enriched_clusters.append(enriched_cluster)

        # Sort clusters by total volume (or missing count if volume not available)
        if any(c['total_volume'] > 0 for c in enriched_clusters):
            enriched_clusters.sort(key=lambda x: x['total_volume'], reverse=True)
        else:
            enriched_clusters.sort(key=lambda x: x['missing_count'], reverse=True)

        clustered_data[url] = enriched_clusters

    return clustered_data


def semantic_cluster_queries(
    query_list: List[str],
    ai_analyzer: AIAnalyzer,
    max_clusters: int = 5
) -> List[Dict[str, Any]]:
    """Group queries into semantic clusters using AI.

    Args:
        query_list: List of queries to cluster
        ai_analyzer: AI analyzer instance
        max_clusters: Maximum number of clusters to create

    Returns:
        List of cluster dicts with 'cluster_name' and 'queries'
    """
    if len(query_list) <= 3:
        # Too few queries to cluster meaningfully
        return [{
            'cluster_name': 'All Queries',
            'queries': query_list
        }]

    # Prepare queries string
    queries_str = '\n'.join([f"{i+1}. {q}" for i, q in enumerate(query_list)])

    prompt = f"""Analyze these search queries and group them into {min(max_clusters, len(query_list)//2)} semantic clusters.

Queries:
{queries_str}

Group these queries by:
1. Topic similarity
2. User intent
3. Semantic relationship

For each cluster:
- Give it a descriptive name
- List the query numbers that belong to it

Respond in JSON format:
[
  {{
    "cluster_name": "Descriptive name",
    "query_indices": [1, 3, 5]
  }}
]"""

    try:
        response = ai_analyzer._call_openrouter(
            model="gpt-4o",
            prompt=prompt,
            max_tokens=500,
            temperature=0.3
        )

        clusters_data = json.loads(response)

        # Convert query indices back to actual queries
        clusters = []
        for cluster in clusters_data:
            cluster_queries = [
                query_list[idx - 1]
                for idx in cluster.get('query_indices', [])
                if 0 < idx <= len(query_list)
            ]

            if cluster_queries:
                clusters.append({
                    'cluster_name': cluster.get('cluster_name', 'Unnamed'),
                    'queries': cluster_queries
                })

        return clusters

    except Exception as e:
        print(f"⚠️ Error clustering queries: {str(e)}")
        # Fallback: single cluster with all queries
        return [{
            'cluster_name': 'All Queries',
            'queries': query_list
        }]


def extract_semantic_entities(
    query_list: List[str],
    ai_analyzer: AIAnalyzer
) -> List[str]:
    """Extract common entities/themes from query list.

    Args:
        query_list: List of queries to analyze
        ai_analyzer: AI analyzer instance

    Returns:
        List of semantic entities/themes found in queries

    Example:
        Input: ["best crm software", "crm pricing", "crm vs erp"]
        Output: ["pricing", "comparison", "vs competitors"]
    """
    if not query_list:
        return []

    queries_str = '\n'.join([f"- {q}" for q in query_list[:20]])

    prompt = f"""Extract common themes, modifiers, and entities from these queries.

Queries:
{queries_str}

Identify recurring:
1. Modifiers (best, top, cheap, free, etc.)
2. Intent signals (how to, what is, vs, pricing, etc.)
3. Common entities (product names, categories, etc.)

Return a JSON array of unique themes found:
["theme1", "theme2", ...]

Limit to 10 most significant themes."""

    try:
        response = ai_analyzer._call_openrouter(
            model="gpt-4o",
            prompt=prompt,
            max_tokens=200,
            temperature=0.2
        )

        entities = json.loads(response)
        return entities[:10]

    except Exception as e:
        print(f"⚠️ Error extracting entities: {str(e)}")
        return []


def identify_primary_intent(
    query_cluster: List[str],
    ai_analyzer: AIAnalyzer
) -> str:
    """Identify the primary user intent for a cluster of queries.

    Args:
        query_cluster: List of related queries
        ai_analyzer: AI analyzer instance

    Returns:
        Intent type: 'informational', 'navigational', 'transactional', 'commercial'
    """
    if not query_cluster:
        return 'informational'

    # Sample up to 10 queries
    sample_queries = query_cluster[:10]
    queries_str = '\n'.join([f"- {q}" for q in sample_queries])

    prompt = f"""Identify the PRIMARY user intent for this group of related queries.

Queries:
{queries_str}

What is the dominant search intent?
- informational: User wants to learn or find information
- navigational: User wants to find a specific website or page
- transactional: User wants to buy, download, sign up, or take action
- commercial: User is researching products/services before purchase

Respond with ONLY the intent type (one word), no explanation."""

    try:
        intent = ai_analyzer._call_openrouter(
            model="gpt-4o",
            prompt=prompt,
            max_tokens=10,
            temperature=0.1
        ).strip().lower()

        # Validate
        valid_intents = ['informational', 'navigational', 'transactional', 'commercial']
        if intent in valid_intents:
            return intent

        # Fallback: parse from response
        for valid_intent in valid_intents:
            if valid_intent in intent:
                return valid_intent

        return 'informational'

    except Exception as e:
        print(f"⚠️ Error identifying intent: {str(e)}")
        return 'informational'


def group_by_intent(
    results_df: pd.DataFrame,
    ai_analyzer: AIAnalyzer
) -> Dict[str, pd.DataFrame]:
    """Group striking distance results by search intent.

    Args:
        results_df: Striking distance results
        ai_analyzer: AI analyzer instance

    Returns:
        Dictionary mapping intent types to filtered DataFrames
    """
    intent_groups = defaultdict(list)

    # Identify intent for each query
    for idx, row in results_df.iterrows():
        query = row.get('query', '')

        if not query:
            continue

        # Get intent
        intent = identify_primary_intent([query], ai_analyzer)

        intent_groups[intent].append(idx)

    # Create DataFrames for each intent group
    intent_dfs = {}
    for intent, indices in intent_groups.items():
        intent_dfs[intent] = results_df.loc[indices]

    return intent_dfs


def analyze_content_gaps(
    url_clusters: List[Dict],
    current_h2s: List[str]
) -> List[Dict[str, Any]]:
    """Identify content gaps based on keyword clusters vs current H2 structure.

    Args:
        url_clusters: Keyword clusters for a URL
        current_h2s: Current H2 headings on the page

    Returns:
        List of content gap recommendations
    """
    gaps = []

    for cluster in url_clusters:
        cluster_name = cluster.get('cluster_name', '')
        missing_count = cluster.get('missing_count', 0)

        if missing_count == 0:
            continue  # No gaps for this cluster

        # Check if cluster theme is covered in current H2s
        is_covered = any(
            cluster_name.lower() in h2.lower()
            for h2 in current_h2s
        )

        if not is_covered:
            gap = {
                'cluster_theme': cluster_name,
                'missing_queries': missing_count,
                'total_queries': cluster.get('total_queries', 0),
                'avg_position': cluster.get('avg_position', 0),
                'recommendation': f"Add H2 section targeting '{cluster_name}' theme",
                'priority': 'high' if missing_count >= 5 else 'medium' if missing_count >= 2 else 'low'
            }
            gaps.append(gap)

    # Sort by missing count
    gaps.sort(key=lambda x: x['missing_queries'], reverse=True)

    return gaps
