"""
Multi-Source Data Parsers

Supports parsing keyword data from:
- Ahrefs (v1 and v2 exports)
- Semrush
- Google Search Console
- Screaming Frog crawls

Provides a unified data model for all sources.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
import re


class DataParser:
    """Unified parser for multiple SEO data sources."""

    # Unified column names for standardization
    STANDARD_COLUMNS = {
        'url': 'url',
        'query': 'query',
        'keyword': 'query',  # Alias
        'position': 'position',
        'clicks': 'clicks',
        'impressions': 'impressions',
        'search_volume': 'search_volume',
        'keyword_difficulty': 'keyword_difficulty',
        'cpc': 'cpc',
        'traffic': 'traffic',
        'title': 'title',
        'h1': 'h1',
        'h2': 'h2',
        'meta_description': 'meta_description'
    }

    @staticmethod
    def detect_source(df: pd.DataFrame) -> str:
        """Detect which tool the data export came from.

        Args:
            df: DataFrame to analyze

        Returns:
            Source identifier: 'ahrefs', 'semrush', 'gsc', 'screaming_frog', 'unknown'
        """
        columns = [col.lower().strip() for col in df.columns]

        # Screaming Frog signatures (check first, as it's most specific for meta data)
        # Primary: Address + Title columns
        has_address = any('address' in col for col in columns)
        has_title_1 = any('title 1' in col for col in columns)
        has_h1_pattern = any('h1-1' in col or 'h1-2' in col for col in columns)
        has_meta_desc_1 = any('meta description 1' in col for col in columns)

        # If has Address AND (Title 1 OR H1-1), it's likely Screaming Frog
        if has_address and (has_title_1 or has_h1_pattern or has_meta_desc_1):
            return 'screaming_frog'

        # Google Search Console signatures
        gsc_signatures = ['landing page', 'query', 'clicks', 'impressions']
        if all(any(sig in col for col in columns) for sig in gsc_signatures):
            return 'gsc'

        # Ahrefs signatures
        ahrefs_signatures = ['volume', 'keyword difficulty', 'cpc', 'parent topic']
        if any('ahrefs' in col for col in columns) or \
           sum(1 for sig in ahrefs_signatures if any(sig in col for col in columns)) >= 2:
            return 'ahrefs'

        # Semrush signatures
        semrush_signatures = ['kd %', 'search intent', 'serp features', 'number of results']
        if any('semrush' in col for col in columns) or \
           sum(1 for sig in semrush_signatures if any(sig in col for col in columns)) >= 2:
            return 'semrush'

        return 'unknown'

    @staticmethod
    def parse_ahrefs_keywords(df: pd.DataFrame) -> pd.DataFrame:
        """Parse Ahrefs keyword export (v1 or v2).

        Expected columns (flexible):
        - Keyword / Query
        - URL / Current URL
        - Volume / Search Volume
        - Position / Ranking
        - Keyword Difficulty / KD
        - CPC
        - Traffic

        Returns:
            Standardized DataFrame
        """
        # Create column mapping
        column_map = {}

        columns_lower = {col: col.lower().strip() for col in df.columns}

        # Map URL column
        for original_col, lower_col in columns_lower.items():
            if 'current url' in lower_col or lower_col == 'url':
                column_map[original_col] = 'url'
                break

        # Map keyword column
        for original_col, lower_col in columns_lower.items():
            if lower_col in ['keyword', 'query', 'keywords']:
                column_map[original_col] = 'query'
                break

        # Map volume column
        for original_col, lower_col in columns_lower.items():
            if 'volume' in lower_col or 'search volume' in lower_col:
                column_map[original_col] = 'search_volume'
                break

        # Map position column
        for original_col, lower_col in columns_lower.items():
            if 'position' in lower_col or 'ranking' in lower_col or lower_col == 'pos':
                column_map[original_col] = 'position'
                break

        # Map keyword difficulty
        for original_col, lower_col in columns_lower.items():
            if 'keyword difficulty' in lower_col or lower_col == 'kd':
                column_map[original_col] = 'keyword_difficulty'
                break

        # Map CPC
        for original_col, lower_col in columns_lower.items():
            if lower_col == 'cpc':
                column_map[original_col] = 'cpc'
                break

        # Map traffic
        for original_col, lower_col in columns_lower.items():
            if lower_col == 'traffic':
                column_map[original_col] = 'traffic'
                break

        # Rename columns
        standardized_df = df.rename(columns=column_map)

        # Clean volume column if it has "0-10" string values (Ahrefs v1)
        if 'search_volume' in standardized_df.columns:
            if standardized_df['search_volume'].dtype == 'object':
                standardized_df['search_volume'] = standardized_df['search_volume'].astype(str).str.replace('0-10', '5')
                standardized_df['search_volume'] = pd.to_numeric(standardized_df['search_volume'], errors='coerce').fillna(0).astype(int)

        return standardized_df

    @staticmethod
    def parse_semrush_keywords(df: pd.DataFrame) -> pd.DataFrame:
        """Parse Semrush keyword export.

        Expected columns (flexible):
        - Keyword
        - URL
        - Position
        - Search Volume
        - KD %
        - CPC
        - Traffic

        Returns:
            Standardized DataFrame
        """
        column_map = {}
        columns_lower = {col: col.lower().strip() for col in df.columns}

        # Map URL
        for original_col, lower_col in columns_lower.items():
            if lower_col in ['url', 'landing page']:
                column_map[original_col] = 'url'
                break

        # Map keyword
        for original_col, lower_col in columns_lower.items():
            if lower_col in ['keyword', 'query', 'keywords']:
                column_map[original_col] = 'query'
                break

        # Map position
        for original_col, lower_col in columns_lower.items():
            if 'position' in lower_col or 'ranking' in lower_col:
                column_map[original_col] = 'position'
                break

        # Map search volume
        for original_col, lower_col in columns_lower.items():
            if 'search volume' in lower_col or lower_col == 'volume':
                column_map[original_col] = 'search_volume'
                break

        # Map keyword difficulty (KD %)
        for original_col, lower_col in columns_lower.items():
            if 'kd %' in lower_col or 'keyword difficulty' in lower_col:
                column_map[original_col] = 'keyword_difficulty'
                break

        # Map CPC
        for original_col, lower_col in columns_lower.items():
            if lower_col == 'cpc' or 'cpc (usd)' in lower_col:
                column_map[original_col] = 'cpc'
                break

        # Map traffic
        for original_col, lower_col in columns_lower.items():
            if lower_col == 'traffic':
                column_map[original_col] = 'traffic'
                break

        standardized_df = df.rename(columns=column_map)

        # Convert KD from percentage to 0-100 scale if needed
        if 'keyword_difficulty' in standardized_df.columns:
            # If values are like "15%", remove % and convert
            if standardized_df['keyword_difficulty'].dtype == 'object':
                standardized_df['keyword_difficulty'] = standardized_df['keyword_difficulty'].astype(str).str.replace('%', '')
            standardized_df['keyword_difficulty'] = pd.to_numeric(standardized_df['keyword_difficulty'], errors='coerce').fillna(0).astype(int)

        return standardized_df

    @staticmethod
    def parse_screaming_frog_crawl(df: pd.DataFrame) -> pd.DataFrame:
        """Parse Screaming Frog crawl export.

        Expected columns:
        - Address (URL)
        - Title 1
        - H1-1
        - H2-1, H2-2, etc.
        - Meta Description 1
        - Indexability

        Returns:
            Standardized DataFrame with meta tags
        """
        column_map = {}
        columns_lower = {col: col.lower().strip() for col in df.columns}

        # Map URL (Address in Screaming Frog)
        for original_col, lower_col in columns_lower.items():
            if lower_col == 'address':
                column_map[original_col] = 'url'
                break

        # Map Title
        for original_col, lower_col in columns_lower.items():
            if 'title 1' in lower_col or lower_col == 'title':
                column_map[original_col] = 'title'
                break

        # Map H1
        for original_col, lower_col in columns_lower.items():
            if 'h1-1' in lower_col or lower_col == 'h1':
                column_map[original_col] = 'h1'
                break

        # Map Meta Description
        for original_col, lower_col in columns_lower.items():
            if 'meta description 1' in lower_col or lower_col == 'meta description':
                column_map[original_col] = 'meta_description'
                break

        standardized_df = df.rename(columns=column_map)

        # Filter to only indexable pages if Indexability column exists
        if 'Indexability' in df.columns:
            standardized_df = standardized_df[standardized_df['Indexability'] == 'Indexable'].copy()

        # Collect all H2 columns
        h2_columns = [col for col in df.columns if col.lower().startswith('h2-')]
        if h2_columns:
            # Combine all H2s into a single column
            standardized_df['h2'] = df[h2_columns].apply(
                lambda row: ' '.join(row.dropna().astype(str)), axis=1
            )

        return standardized_df

    @staticmethod
    def parse_gsc_export(df: pd.DataFrame) -> pd.DataFrame:
        """Parse Google Search Console export.

        Expected columns (case-insensitive):
        - Landing Page / Page / URL / Address
        - Query / Keyword / Top queries
        - Clicks
        - Impressions
        - Avg. Pos / Average Position / Position / Avg Position

        Returns:
            Standardized DataFrame
        """
        column_map = {}

        # Create mapping of lowercase column to original column for easier matching
        col_lower_to_original = {col.lower().strip(): col for col in df.columns}

        # Map URL - try multiple variations
        url_variations = ['landing page', 'page', 'url', 'address', 'top pages']
        for variant in url_variations:
            if variant in col_lower_to_original:
                column_map[col_lower_to_original[variant]] = 'url'
                break

        # Map query - try multiple variations
        query_variations = ['query', 'queries', 'keyword', 'keywords', 'top queries', 'search query']
        for variant in query_variations:
            if variant in col_lower_to_original:
                column_map[col_lower_to_original[variant]] = 'query'
                break

        # Map clicks - exact match
        if 'clicks' in col_lower_to_original:
            column_map[col_lower_to_original['clicks']] = 'clicks'

        # Map impressions
        impressions_variations = ['impressions', 'impression', 'impr']
        for variant in impressions_variations:
            if variant in col_lower_to_original:
                column_map[col_lower_to_original[variant]] = 'impressions'
                break

        # Map position - try multiple variations including with periods
        position_variations = ['avg. pos', 'avg.pos', 'avg pos', 'average position', 'position', 'avg position']
        for variant in position_variations:
            if variant in col_lower_to_original:
                column_map[col_lower_to_original[variant]] = 'position'
                break

        # Rename columns
        standardized_df = df.rename(columns=column_map)

        # Keep only standardized columns (drop unmapped columns)
        standard_cols = ['url', 'query', 'clicks', 'impressions', 'position']
        available_cols = [col for col in standard_cols if col in standardized_df.columns]

        return standardized_df[available_cols]

    @staticmethod
    def parse_auto(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Automatically detect source and parse accordingly.

        Args:
            df: Raw DataFrame from any supported source

        Returns:
            Tuple of (standardized DataFrame, source name)
        """
        source = DataParser.detect_source(df)

        if source == 'ahrefs':
            return DataParser.parse_ahrefs_keywords(df), 'ahrefs'
        elif source == 'semrush':
            return DataParser.parse_semrush_keywords(df), 'semrush'
        elif source == 'gsc':
            return DataParser.parse_gsc_export(df), 'gsc'
        elif source == 'screaming_frog':
            return DataParser.parse_screaming_frog_crawl(df), 'screaming_frog'
        else:
            # Return as-is with unknown source
            return df, 'unknown'

    @staticmethod
    def merge_keyword_sources(
        gsc_df: Optional[pd.DataFrame] = None,
        ahrefs_df: Optional[pd.DataFrame] = None,
        semrush_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Merge keyword data from multiple sources.

        Args:
            gsc_df: Google Search Console data (optional)
            ahrefs_df: Ahrefs keyword data (optional)
            semrush_df: Semrush keyword data (optional)

        Returns:
            Merged DataFrame with best data from each source
        """
        all_dfs = []

        # Add each source to the list if provided
        if gsc_df is not None and len(gsc_df) > 0:
            gsc_df['source'] = 'gsc'
            all_dfs.append(gsc_df)

        if ahrefs_df is not None and len(ahrefs_df) > 0:
            ahrefs_df['source'] = 'ahrefs'
            all_dfs.append(ahrefs_df)

        if semrush_df is not None and len(semrush_df) > 0:
            semrush_df['source'] = 'semrush'
            all_dfs.append(semrush_df)

        if not all_dfs:
            return pd.DataFrame()

        # Concatenate all sources
        merged_df = pd.concat(all_dfs, ignore_index=True, sort=False)

        # Remove duplicates, keeping the one with most data
        # Priority: url + query combination
        merged_df['data_completeness'] = merged_df.notna().sum(axis=1)

        # Sort by completeness (descending) and drop duplicates
        merged_df = merged_df.sort_values('data_completeness', ascending=False)
        merged_df = merged_df.drop_duplicates(subset=['url', 'query'], keep='first')
        merged_df = merged_df.drop('data_completeness', axis=1)

        return merged_df


# Convenience functions
def load_and_parse(file_path: str) -> Tuple[pd.DataFrame, str]:
    """Load a file and automatically parse it.

    Args:
        file_path: Path to CSV or Excel file

    Returns:
        Tuple of (standardized DataFrame, source name)
    """
    # Load file
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Auto-parse
    return DataParser.parse_auto(df)
