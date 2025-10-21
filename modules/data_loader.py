import pandas as pd
import re
from typing import Dict, List

def load_data_file(file) -> pd.DataFrame:
    """Load Excel or CSV file and return DataFrame."""
    try:
        # Get file extension
        file_name = file.name.lower()

        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            df = pd.read_excel(file, engine='openpyxl')
        elif file_name.endswith('.csv'):
            # Try reading CSV with more robust settings
            try:
                # First attempt: standard CSV with quoting
                df = pd.read_csv(
                    file,
                    encoding='utf-8',
                    quotechar='"',
                    skipinitialspace=True,
                    on_bad_lines='skip',  # Skip malformed lines
                    engine='python'  # More flexible than C engine
                )
            except Exception as e1:
                # Second attempt: Try different encodings
                file.seek(0)  # Reset file pointer
                try:
                    df = pd.read_csv(
                        file,
                        encoding='latin-1',
                        quotechar='"',
                        skipinitialspace=True,
                        on_bad_lines='skip',
                        engine='python'
                    )
                except Exception as e2:
                    # Third attempt: Try tab-separated
                    file.seek(0)
                    try:
                        df = pd.read_csv(
                            file,
                            sep='\t',
                            encoding='utf-8',
                            on_bad_lines='skip',
                            engine='python'
                        )
                    except Exception as e3:
                        # Give up and raise original error
                        raise ValueError(f"Could not parse CSV file. Tried multiple formats. Original error: {str(e1)}")
        else:
            raise ValueError("Unsupported file format. Please upload .xlsx, .xls, or .csv files.")

        return df
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")

def detect_columns(df: pd.DataFrame, report_type: str) -> Dict[str, str]:
    """Auto-detect column names based on common patterns."""
    columns = df.columns.str.lower().str.strip()

    detected = {}

    if report_type == 'meta':
        # URL patterns (prioritize 'address' for meta reports)
        url_patterns = ['address', 'url', 'page', 'link', 'website']
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
        # URL patterns (prioritize 'landing page' for organic reports)
        url_patterns = ['landing page', 'url', 'page', 'final url', 'address']
        detected['url'] = find_best_match(columns, url_patterns)

        # Query patterns
        query_patterns = ['query', 'keyword', 'search query', 'search term', 'queries', 'keywords']
        detected['query'] = find_best_match(columns, query_patterns)

        # Clicks patterns
        clicks_patterns = ['clicks', 'click']
        detected['clicks'] = find_best_match(columns, clicks_patterns)

        # Impressions patterns
        impressions_patterns = ['impressions', 'impression', 'impr']
        detected['impressions'] = find_best_match(columns, impressions_patterns)

        # Position patterns
        position_patterns = ['position', 'avg position', 'ranking', 'rank', 'avg. pos', 'avg pos', 'average position', 'avg position']
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
    # Columns to exclude from matching (irrelevant columns)
    exclude_patterns = ['redirect url', 'redirect', 'redirect_url']

    # Filter out excluded columns
    valid_columns = []
    valid_indices = []
    for i, col in enumerate(columns):
        col_lower = col.lower().strip()
        should_exclude = False
        for exclude_pattern in exclude_patterns:
            if exclude_pattern in col_lower:
                should_exclude = True
                break
        if not should_exclude:
            valid_columns.append(col)
            valid_indices.append(i)

    if not valid_columns:
        return None

    # Create index of valid columns
    valid_columns_idx = pd.Index(valid_columns)
    normalized_columns = valid_columns_idx.str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()

    for pattern in patterns:
        # Normalize pattern for comparison
        normalized_pattern = pattern.lower().replace('.', '').replace(' ', '').strip()

        # Check exact matches first (case-insensitive)
        for i, col in enumerate(valid_columns_idx):
            if col.lower().strip() == pattern.lower().strip():
                return valid_columns_idx[i]

        # Check normalized matches
        for i, norm_col in enumerate(normalized_columns):
            if normalized_pattern in norm_col or norm_col in normalized_pattern:
                return valid_columns_idx[i]

        # Fuzzy match using regex (more flexible)
        regex_pattern = re.compile(r'\b' + re.escape(pattern) + r'\b', re.IGNORECASE)
        matches = valid_columns_idx[valid_columns_idx.str.contains(regex_pattern, na=False)]
        if not matches.empty:
            return matches[0]

    return None