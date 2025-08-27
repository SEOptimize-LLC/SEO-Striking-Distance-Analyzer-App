import pandas as pd
import re
from typing import Dict, List

def load_excel_files(file) -> pd.DataFrame:
    """Load Excel file and return DataFrame."""
    try:
        df = pd.read_excel(file, engine='openpyxl')
        return df
    except Exception as e:
        raise ValueError(f"Error loading Excel file: {str(e)}")

def detect_columns(df: pd.DataFrame, report_type: str) -> Dict[str, str]:
    """Auto-detect column names based on common patterns."""
    columns = df.columns.str.lower().str.strip()

    detected = {}

    if report_type == 'meta':
        # URL patterns
        url_patterns = ['url', 'page', 'link', 'address', 'website']
        detected['url'] = find_best_match(columns, url_patterns)

        # Title patterns
        title_patterns = ['title', 'page title', 'meta title']
        detected['title'] = find_best_match(columns, title_patterns)

        # H1 patterns
        h1_patterns = ['h1', 'heading1', 'main heading']
        detected['h1'] = find_best_match(columns, h1_patterns)

        # H2 patterns (may have multiple)
        h2_patterns = ['h2', 'heading2', 'subheading']
        detected['h2'] = find_best_match(columns, h2_patterns)

        # Meta description patterns
        desc_patterns = ['description', 'meta description', 'meta desc']
        detected['meta_description'] = find_best_match(columns, desc_patterns)

    elif report_type == 'organic':
        # URL patterns
        url_patterns = ['url', 'page', 'landing page', 'final url']
        detected['url'] = find_best_match(columns, url_patterns)

        # Query patterns
        query_patterns = ['query', 'keyword', 'search query', 'search term']
        detected['query'] = find_best_match(columns, query_patterns)

        # Clicks patterns
        clicks_patterns = ['clicks', 'click']
        detected['clicks'] = find_best_match(columns, clicks_patterns)

        # Impressions patterns
        impressions_patterns = ['impressions', 'impression', 'impr']
        detected['impressions'] = find_best_match(columns, impressions_patterns)

        # Position patterns
        position_patterns = ['position', 'avg position', 'ranking', 'rank']
        detected['position'] = find_best_match(columns, position_patterns)

    # Check for missing required columns
    required_cols = {
        'meta': ['url', 'title', 'h1', 'meta_description'],
        'organic': ['url', 'query', 'clicks', 'impressions', 'position']
    }

    missing_cols = []
    for col in required_cols[report_type]:
        if not detected.get(col):
            missing_cols.append(col)

    if missing_cols:
        raise ValueError(f"Could not detect required columns: {', '.join(missing_cols)}")

    return detected

def find_best_match(columns: pd.Index, patterns: List[str]) -> str:
    """Find the best matching column name from a list of patterns."""
    for pattern in patterns:
        # Exact match
        if pattern in columns.values:
            return columns[columns == pattern].iloc[0]

        # Fuzzy match using regex
        regex_pattern = re.compile(r'\b' + re.escape(pattern) + r'\b', re.IGNORECASE)
        matches = columns[columns.str.contains(regex_pattern, na=False)]
        if not matches.empty:
            return matches.iloc[0]

    return None