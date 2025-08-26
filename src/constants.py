"""
Application constants and configuration for SEO Striking Distance Analyzer
"""
from typing import Dict, List, Tuple
from enum import Enum

class HTMLTags(Enum):
    """HTML tags to analyze for keyword presence"""
    TITLE = "title"
    H1 = "h1"
    H2 = "h2"
    META_DESCRIPTION = "meta_description"
    CONTENT = "content"

class GSCColumns(Enum):
    """Standard column mappings for GSC data"""
    URL = "url"
    QUERY = "query"
    CLICKS = "clicks"
    IMPRESSIONS = "impressions"
    CTR = "ctr"
    POSITION = "position"

# Configuration constants
MAX_FILE_SIZE_MB = 200
STRIKING_DISTANCE_RANGE = (4, 20)  # Position range for striking distance
TOP_QUERIES_THRESHOLD = 5  # Number of top queries to analyze per URL
MIN_CLICKS_THRESHOLD = 1  # Minimum clicks to consider a query
MIN_IMPRESSIONS_THRESHOLD = 10  # Minimum impressions to consider

# Column mapping patterns for different export tools
COLUMN_MAPPINGS: Dict[str, Dict[str, List[str]]] = {
    "standard": {
        "url": ["url", "page", "landing page", "top pages", "landing_page", "page_url"],
        "query": ["query", "queries", "search query", "keyword", "search_query", "keywords"],
        "clicks": ["clicks", "click", "total clicks", "total_clicks"],
        "impressions": ["impressions", "impression", "total impressions", "total_impressions"],
        "ctr": ["ctr", "click through rate", "clickthrough rate", "click_through_rate", "avg_ctr"],
        "position": ["position", "avg position", "average position", "avg. position", "average_position", "avg_position"]
    }
}

# HTML column mapping patterns
HTML_COLUMN_MAPPINGS: Dict[str, List[str]] = {
    "url": ["url", "page", "address", "page_url", "link"],
    "title": ["title", "page title", "page_title", "meta title", "meta_title", "title tag", "title_tag"],
    "h1": ["h1", "h1 tag", "h1_tag", "heading 1", "heading_1"],
    "meta_description": ["meta description", "meta_description", "description", "meta desc", "meta_desc"],
    "content": ["content", "body", "text", "page content", "page_content", "body_text"]
}

# Scoring weights for determining "top performing"
PERFORMANCE_WEIGHTS = {
    "clicks": 0.5,
    "impressions": 0.2,
    "ctr": 0.2,
    "position_inverse": 0.1  # Lower position = better
}

# Priority thresholds
PRIORITY_THRESHOLDS = {
    "high_clicks": 10,
    "high_impressions": 500,
    "high_position": 10,
    "medium_impressions": 50,
    "medium_position": 20
}

# Color schemes for visualizations
COLOR_SCHEME = {
    "high": "#FF6B6B",
    "medium": "#FFD93D", 
    "low": "#6BCB77",
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "success": "#4CAF50",
    "warning": "#FF9800",
    "error": "#F44336"
}
