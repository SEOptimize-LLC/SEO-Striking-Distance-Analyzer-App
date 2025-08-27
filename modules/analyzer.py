import pandas as pd
import re
from typing import Dict, List
from collections import defaultdict

def analyze_striking_distance(
    meta_df: pd.DataFrame,
    organic_df: pd.DataFrame,
    scraped_data: Dict[str, str],
    meta_columns: Dict[str, str],
    organic_columns: Dict[str, str],
    min_clicks: int,
    top_queries: int,
    use_impressions_weighted: bool
) -> pd.DataFrame:
    """Main analysis function to determine striking distance optimization."""

    results = []

    # Group organic data by URL
    organic_by_url = defaultdict(list)
    for _, row in organic_df.iterrows():
        url = row[organic_columns['url']]
        query = row[organic_columns['query']]
        clicks = row[organic_columns['clicks']]
        impressions = row[organic_columns['impressions']]
        position = row[organic_columns['position']]

        if pd.isna(clicks) or clicks < min_clicks:
            continue

        # Calculate score
        if use_impressions_weighted and impressions > 0:
            score = clicks / impressions
        else:
            score = clicks

        organic_by_url[url].append({
            'query': query,
            'clicks': clicks,
            'impressions': impressions,
            'position': position,
            'score': score
        })

    # Sort queries by score for each URL
    for url in organic_by_url:
        organic_by_url[url].sort(key=lambda x: x['score'], reverse=True)
        organic_by_url[url] = organic_by_url[url][:top_queries]

    # Process each URL from meta data
    for _, meta_row in meta_df.iterrows():
        url = meta_row[meta_columns['url']]

        if url not in organic_by_url:
            continue

        # Get page data
        title = str(meta_row.get(meta_columns.get('title', ''), ''))
        h1 = str(meta_row.get(meta_columns.get('h1', ''), ''))
        h2 = str(meta_row.get(meta_columns.get('h2', ''), ''))
        meta_desc = str(meta_row.get(meta_columns.get('meta_description', ''), ''))
        content = scraped_data.get(url, '')

        # Combine all H2s if multiple columns
        h2_texts = []
        for col in meta_df.columns:
            if col.lower().startswith('h2') and col in meta_row:
                h2_text = str(meta_row[col])
                if h2_text:
                    h2_texts.append(h2_text)

        # Create page text for checking
        page_elements = {
            'title': title,
            'h1': h1,
            'h2': ' '.join(h2_texts),
            'meta_description': meta_desc,
            'content': content
        }

        # Check each top query
        for query_data in organic_by_url[url]:
            query = query_data['query']

            # Check presence in each element
            checks = {}
            for element, text in page_elements.items():
                checks[f'{element}_optimized'] = check_query_in_text(query, text)

            # Overall optimization
            checks['overall_optimized'] = any(checks.values())

            # Add to results
            result = {
                'url': url,
                'query': query,
                'clicks': query_data['clicks'],
                'impressions': query_data['impressions'],
                'position': query_data['position'],
                'score': query_data['score'],
                **checks
            }
            results.append(result)

    return pd.DataFrame(results)

def check_query_in_text(query: str, text: str) -> bool:
    """Check if query terms are present in text."""
    if not text or not query:
        return False

    # Normalize text
    text = text.lower()
    query = query.lower()

    # Split query into words
    query_words = re.findall(r'\b\w+\b', query)

    # Check if all query words are present
    for word in query_words:
        if word not in text:
            return False

    return True