
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import re
from datetime import datetime
import io
import time

# Lazy imports - only load when needed
def lazy_import_requests():
    import requests
    import json
    return requests, json

def lazy_import_beautifulsoup():
    from bs4 import BeautifulSoup
    return BeautifulSoup

# Page configuration
st.set_page_config(
    page_title="SEO Striking Distance Analyzer",
    page_icon="🎯",
    layout="wide"
)

# Full CSS with performance optimization
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stButton>button { 
        width: 100%; 
        background-color: #2E86AB; 
        color: white; 
        border-radius: 8px;
    }
    .stButton>button:hover { background-color: #1E5A7A; }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #e8f5e8;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #fff8e1;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .branded-terms-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DataforSEOClient:
    """Client for DataforSEO API integration using the correct clickstream endpoint"""

    def __init__(self, username: str = None, password: str = None):
        self.username = username
        self.password = password
        self.base_url = "https://api.dataforseo.com/v3/keywords_data/clickstream_data/dataforseo_search_volume/live"

    def get_search_volumes(self, keywords: List[str], location_code: int = 2840, language_code: str = "en") -> Dict[str, int]:
        """
        Get search volumes for keywords using DataforSEO API
        Batches up to 1000 keywords per request for cost efficiency
        """
        if not self.username or not self.password:
            st.warning("DataforSEO credentials not provided")
            return {}

        # Lazy load requests and json
        requests, json = lazy_import_requests()

        volumes = {}

        # Process in batches of 1000 (API limit for cost efficiency)
        for i in range(0, len(keywords), 1000):
            batch = keywords[i:i+1000]

            try:
                # Prepare request data according to API documentation
                post_data = [{
                    "keywords": batch,
                    "location_code": location_code,
                    "language_code": language_code,
                    "use_clickstream": True  # Use clickstream data for better accuracy
                }]

                # Make API request
                response = requests.post(
                    self.base_url,
                    auth=(self.username, self.password),
                    headers={
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    data=json.dumps(post_data),
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()

                    # Check if request was successful
                    if data.get('status_code') == 20000 and data.get('tasks'):
                        task = data['tasks'][0]

                        if task.get('status_code') == 20000 and task.get('result'):
                            result = task['result'][0]

                            # Extract search volumes from items
                            for item in result.get('items', []):
                                keyword = item.get('keyword', '')
                                search_volume = item.get('search_volume', 0)
                                volumes[keyword] = search_volume
                        else:
                            st.warning(f"API Task failed: {task.get('status_message', 'Unknown error')}")
                    else:
                        st.warning(f"API request failed: {data.get('status_message', 'Unknown error')}")
                else:
                    st.warning(f"HTTP error {response.status_code}: {response.text}")

                # Rate limiting - wait between batches
                if i + 1000 < len(keywords):
                    time.sleep(1)

            except Exception as e:
                st.warning(f"Error processing batch {i//1000 + 1}: {str(e)}")
                continue

        return volumes

class ContentExtractor:
    """Extract main content from URLs excluding navigation, footer, etc."""

    def __init__(self):
        # Lazy initialization
        self.session = None

    def _get_session(self):
        """Initialize session only when needed"""
        if self.session is None:
            requests, _ = lazy_import_requests()
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        return self.session

    def extract_main_content(self, url: str) -> str:
        """
        Extract main content from URL, excluding navigation, footer, sidebar
        """
        try:
            session = self._get_session()
            BeautifulSoup = lazy_import_beautifulsoup()

            response = session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted elements
            unwanted_tags = [
                'nav', 'header', 'footer', 'aside', 'sidebar',
                '[class*="nav"]', '[class*="menu"]', '[class*="header"]',
                '[class*="footer"]', '[class*="sidebar"]', '[id*="nav"]',
                '[id*="menu"]', '[id*="header"]', '[id*="footer"]',
                'script', 'style', 'meta', 'link'
            ]

            for tag in unwanted_tags:
                for element in soup.select(tag):
                    element.decompose()

            # Try to find main content area
            main_content = None

            # Look for common main content selectors
            main_selectors = [
                'main', '[role="main"]', '.main-content', '#main-content',
                '.content', '#content', '.post-content', '.entry-content',
                'article', '.article'
            ]

            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break

            # If no main content found, use body but remove known non-content areas
            if not main_content:
                main_content = soup.find('body')

            if main_content:
                # Extract text and clean it
                text = main_content.get_text(separator=' ', strip=True)
                # Clean up whitespace
                text = ' '.join(text.split())
                return text.lower()

            return ""

        except Exception as e:
            st.warning(f"Could not extract content from {url}: {str(e)}")
            return ""

class DataProcessor:
    """Handles data processing and validation"""

    def __init__(self):
        self.gsc_data = None
        self.html_data = None
        self.content_data = {}
        self.column_mappings = {}
        self.content_extractor = None

    def _get_content_extractor(self):
        """Lazy load content extractor"""
        if self.content_extractor is None:
            self.content_extractor = ContentExtractor()
        return self.content_extractor

    def detect_columns(self, df: pd.DataFrame, expected_columns: Dict[str, List[str]]) -> Dict[str, str]:
        """Auto-detect column mappings based on common variations"""
        mappings = {}
        df_columns = [col.lower().strip() for col in df.columns]

        for standard_name, variations in expected_columns.items():
            for col in df_columns:
                for variation in variations:
                    if variation.lower() in col or col in variation.lower():
                        original_col = df.columns[df_columns.index(col)]
                        mappings[original_col] = standard_name
                        break
                if standard_name in mappings.values():
                    break

        return mappings

    def filter_branded_terms(self, df: pd.DataFrame, branded_terms: str) -> pd.DataFrame:
        """Filter out branded terms from the dataset"""
        if not branded_terms.strip():
            return df

        # Parse branded terms (comma or newline separated)
        terms = [term.strip().lower() for term in re.split('[,\n]', branded_terms) if term.strip()]

        if not terms:
            return df

        # Filter out queries containing any branded term
        mask = ~df['query'].str.lower().str.contains('|'.join(terms), regex=True, na=False)
        filtered_df = df[mask].copy()

        dropped_count = len(df) - len(filtered_df)
        if dropped_count > 0:
            st.info(f"🚫 Filtered out {dropped_count} queries containing branded terms")

        return filtered_df

    def process_gsc_data(self, file, branded_terms: str = "") -> pd.DataFrame:
        """Process Google Search Console data"""
        try:
            # Read file with optimized settings
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, low_memory=False)
            else:
                df = pd.read_excel(file)

            # Expected GSC columns and their variations
            gsc_columns = {
                'url': ['url', 'page', 'landing page', 'top pages'],
                'query': ['query', 'queries', 'search query', 'keyword'],
                'clicks': ['clicks', 'click', 'total clicks'],
                'impressions': ['impressions', 'impression', 'total impressions'],
                'position': ['position', 'avg position', 'average position', 'avg. position']
            }

            # Auto-detect columns
            mappings = self.detect_columns(df, gsc_columns)
            self.column_mappings.update(mappings)

            # Rename columns
            df = df.rename(columns=mappings)

            # Validate required columns
            required = ['url', 'query', 'clicks', 'position']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Clean and convert data types
            df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce').fillna(0).astype(int)
            df['impressions'] = pd.to_numeric(df.get('impressions', 0), errors='coerce').fillna(0).astype(int)
            df['position'] = pd.to_numeric(df['position'], errors='coerce').fillna(100).round(1)

            # Clean URLs
            df['url_clean'] = df['url'].str.lower().str.strip().str.rstrip('/')

            # Filter branded terms
            df = self.filter_branded_terms(df, branded_terms)

            self.gsc_data = df
            return df

        except Exception as e:
            raise ValueError(f"Error processing GSC data: {str(e)}")

    def process_html_data(self, file) -> pd.DataFrame:
        """Process HTML metadata report"""
        try:
            # Read file with optimized settings
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, low_memory=False)
            else:
                df = pd.read_excel(file)

            # Expected HTML columns and their variations
            html_columns = {
                'url': ['url', 'address', 'page'],
                'title': ['title', 'page title', 'title tag'],
                'h1': ['h1', 'h1 tag', 'heading 1'],
                'h2': ['h2', 'h2 tag', 'heading 2', 'h2_1'],
                'meta_description': ['meta description', 'description', 'meta desc'],
                'content': ['content', 'body', 'text', 'copy', 'main content']
            }

            # Auto-detect columns
            mappings = self.detect_columns(df, html_columns)
            self.column_mappings.update(mappings)

            # Rename columns
            df = df.rename(columns=mappings)

            # Ensure URL column exists
            if 'url' not in df.columns:
                raise ValueError("URL column not found in HTML data")

            # Clean URLs to match GSC format
            df['url_clean'] = df['url'].str.lower().str.strip().str.rstrip('/')

            # Fill missing values and clean text
            text_columns = ['title', 'h1', 'h2', 'meta_description', 'content']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str)
                else:
                    df[col] = ''

            # Combine multiple H2 columns if they exist
            h2_cols = [col for col in df.columns if col.startswith('h2') and col != 'h2']
            if h2_cols:
                df['h2_combined'] = df[h2_cols].fillna('').apply(lambda x: ' '.join(x), axis=1)
                df['h2'] = df['h2'] + ' ' + df['h2_combined']

            self.html_data = df
            return df

        except Exception as e:
            raise ValueError(f"Error processing HTML data: {str(e)}")

    def extract_content_for_urls(self, urls: List[str], max_urls: int = 50) -> None:
        """Extract content for URLs if not provided in HTML data"""
        if len(urls) > max_urls:
            st.warning(f"⚠️ Content extraction limited to first {max_urls} URLs for performance")
            urls = urls[:max_urls]

        progress_bar = st.progress(0)
        status_text = st.empty()

        extractor = self._get_content_extractor()

        for i, url in enumerate(urls):
            if url not in self.content_data:
                status_text.text(f"Extracting content from URL {i+1} of {len(urls)}")
                content = extractor.extract_main_content(url)
                self.content_data[url] = content
                progress_bar.progress((i + 1) / len(urls))
                time.sleep(0.5)  # Rate limiting

        progress_bar.empty()
        status_text.empty()

class AnalysisEngine:
    """Core analysis engine for keyword optimization opportunities"""

    def __init__(self, gsc_data: pd.DataFrame, html_data: pd.DataFrame, content_data: Dict[str, str] = None):
        self.gsc_data = gsc_data
        self.html_data = html_data
        self.content_data = content_data or {}

    def analyze_keywords(self, top_n_queries: int = 5, min_clicks: int = 1, 
                        striking_distance_range: Tuple[int, int] = (4, 20),
                        search_volumes: Dict[str, int] = None) -> pd.DataFrame:
        """Main analysis function"""
        results = []

        # Get unique URLs that exist in both datasets
        gsc_urls = set(self.gsc_data['url_clean'].unique())
        html_urls = set(self.html_data['url_clean'].unique())
        common_urls = gsc_urls.intersection(html_urls)

        if not common_urls:
            raise ValueError("No matching URLs found between GSC and HTML data")

        # Performance limit with user notification
        if len(common_urls) > 200:
            st.warning(f"⚠️ Analyzing first 200 URLs for performance (found {len(common_urls)} total). "
                      f"For larger datasets, consider filtering your data before upload.")
            common_urls = list(common_urls)[:200]

        for url in common_urls:
            url_results = self._analyze_url(
                url, top_n_queries, min_clicks, striking_distance_range, search_volumes
            )
            results.extend(url_results)

        if results:
            df_results = pd.DataFrame(results)
            # Sort by priority (striking distance first, then by clicks)
            df_results['priority_score'] = (
                df_results['is_striking_distance'].astype(int) * 1000 +
                df_results['clicks']
            )
            df_results = df_results.sort_values('priority_score', ascending=False)
            return df_results
        else:
            return pd.DataFrame()

    def _analyze_url(self, url: str, top_n: int, min_clicks: int, 
                    striking_range: Tuple[int, int], search_volumes: Dict[str, int] = None) -> List[Dict]:
        """Analyze a single URL for keyword opportunities"""

        # Get HTML data for this URL
        html_row = self.html_data[self.html_data['url_clean'] == url]
        if html_row.empty:
            return []
        html_row = html_row.iloc[0]

        # Get GSC queries for this URL
        url_queries = self.gsc_data[self.gsc_data['url_clean'] == url].copy()

        if url_queries.empty:
            return []

        # Get top performing queries - improved logic
        top_queries = self._get_top_queries(url_queries, top_n, min_clicks)

        results = []
        for _, query_row in top_queries.iterrows():
            analysis = self._analyze_query_presence(query_row, html_row, url, striking_range, search_volumes)
            results.append(analysis)

        return results

    def _get_top_queries(self, df: pd.DataFrame, top_n: int, min_clicks: int) -> pd.DataFrame:
        """
        Get top N queries with improved logic:
        1. First get queries with clicks >= min_clicks, sorted by clicks
        2. If we need more to reach top_n, fill with highest impression queries
        """
        # Get queries with minimum clicks
        with_clicks = df[df['clicks'] >= min_clicks].nlargest(top_n, 'clicks')

        # If we have enough, return them
        if len(with_clicks) >= top_n:
            return with_clicks.head(top_n)

        # If we need more, fill with highest impression queries (excluding those already selected)
        remaining_needed = top_n - len(with_clicks)
        remaining_df = df[~df.index.isin(with_clicks.index)]

        if not remaining_df.empty:
            by_impressions = remaining_df.nlargest(remaining_needed, 'impressions')
            result = pd.concat([with_clicks, by_impressions])
        else:
            result = with_clicks

        return result.head(top_n)

    def _analyze_query_presence(self, query_row: pd.Series, html_row: pd.Series, 
                               url: str, striking_range: Tuple[int, int],
                               search_volumes: Dict[str, int] = None) -> Dict:
        """Check if query is present in HTML elements and content"""

        query = query_row['query'].lower()
        position = round(query_row['position'], 1)

        # Check presence in HTML elements
        in_title = self._check_presence(query, html_row.get('title', ''))
        in_h1 = self._check_presence(query, html_row.get('h1', ''))
        in_h2 = self._check_presence(query, html_row.get('h2', ''))
        in_meta = self._check_presence(query, html_row.get('meta_description', ''))

        # Check presence in content
        content_text = html_row.get('content', '') or self.content_data.get(url, '')
        in_content = self._check_presence(query, content_text)

        # Striking distance check
        is_striking = striking_range[0] <= position <= striking_range[1]

        # Determine if optimization is needed
        optimization_needed = not any([in_title, in_h1, in_h2, in_meta, in_content])

        # Priority determination
        if is_striking and optimization_needed:
            priority = "High"
        elif optimization_needed:
            priority = "Medium"
        else:
            priority = "Low"  # Already optimized

        # Get search volume if available
        search_volume = search_volumes.get(query_row['query'], 0) if search_volumes else 0

        return {
            'url': url,
            'query': query_row['query'],
            'clicks': query_row['clicks'],
            'impressions': query_row.get('impressions', 0),
            'position': position,
            'search_volume': search_volume,
            'is_striking_distance': is_striking,
            'in_title': in_title,
            'in_h1': in_h1,
            'in_h2': in_h2,
            'in_meta_description': in_meta,
            'in_content': in_content,
            'optimization_needed': optimization_needed,
            'priority': priority
        }

    def _check_presence(self, query: str, text: str) -> bool:
        """Advanced text matching for query presence"""
        if not text:
            return False

        text_lower = text.lower()
        query_lower = query.lower()

        # Strategy 1: Exact phrase match
        if query_lower in text_lower:
            return True

        # Strategy 2: All significant words present
        query_words = set(query_lower.split())
        # Remove common stop words that don't matter for matching
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        significant_words = query_words - stop_words

        if len(significant_words) > 0:
            text_words = set(text_lower.split())
            # If 70% or more of significant words are present, consider it a match
            matches = significant_words.intersection(text_words)
            if len(matches) >= len(significant_words) * 0.7:
                return True

        return False

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = DataProcessor()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def main():
    # Header
    st.title("🎯 SEO Striking Distance Analyzer")
    st.markdown("""
    **Identify high-impact keyword optimization opportunities by analyzing if your top-performing queries are present in key HTML elements and content.**

    Upload your Google Search Console data and HTML metadata report to discover which pages need optimization.
    """)

    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Analysis Settings")

        # Branded terms filter
        st.subheader("🚫 Branded Terms Filter")
        branded_terms = st.text_area(
            "Exclude branded terms (comma or line separated)",
            placeholder="your brand name, competitor names, etc.",
            help="These terms will be excluded from the analysis",
            height=80
        )

        st.divider()

        top_n_queries = st.slider(
            "Top queries per URL", 
            min_value=3, 
            max_value=15, 
            value=5,
            help="Number of top-performing queries to analyze for each URL"
        )

        min_clicks = st.number_input(
            "Minimum clicks threshold", 
            min_value=0, 
            max_value=10, 
            value=1,
            help="Minimum clicks required for a query to be analyzed"
        )

        striking_distance_range = st.slider(
            "Striking distance positions",
            min_value=2,
            max_value=30,
            value=(4, 20),
            help="Position range considered 'striking distance'"
        )

        st.divider()

        # DataforSEO API integration
        st.subheader("📊 Search Volume Data")
        enable_search_volume = st.checkbox("Enable search volume data (DataforSEO API)")

        if enable_search_volume:
            st.info("💰 Cost-optimized: Sends up to 1,000 keywords per API request")
            api_username = st.text_input("DataforSEO Username", type="password")
            api_password = st.text_input("DataforSEO Password", type="password")

            # Location settings
            location_code = st.number_input(
                "Location Code", 
                value=2840, 
                help="Location code (2840 = USA, 2826 = UK, etc.)"
            )
            language_code = st.selectbox(
                "Language Code",
                options=["en", "es", "fr", "de", "it"],
                index=0,
                help="Language code for search volume data"
            )
        else:
            api_username = api_password = None
            location_code = 2840
            language_code = "en"

        st.divider()

        # Content extraction
        st.subheader("📄 Content Extraction")
        extract_content = st.checkbox(
            "Extract content from URLs",
            help="If HTML report doesn't include content, extract it from live URLs"
        )

        st.divider()
        st.info("""
        **💡 How it works:**
        1. Upload your GSC data and HTML report
        2. App identifies top queries by clicks for each URL
        3. Checks if queries are present in Title, H1, H2, Meta Description, and Content
        4. Flags optimization opportunities
        """)

    # File upload section
    st.subheader("📤 Upload Your Data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**1. Google Search Console Report**")
        st.caption("CSV or Excel file with URL, Query, Clicks, Position data")

        gsc_file = st.file_uploader(
            "Upload GSC data",
            type=['csv', 'xlsx', 'xls'],
            key='gsc_file',
            help="Export from GSC with queries and page data"
        )

        if gsc_file:
            try:
                with st.spinner("Processing GSC data..."):
                    gsc_data = st.session_state.processor.process_gsc_data(gsc_file, branded_terms)
                st.success(f"✅ Loaded {len(gsc_data):,} query records")

                with st.expander("Preview GSC Data"):
                    st.dataframe(gsc_data.head())

                # Show column mapping
                if st.session_state.processor.column_mappings:
                    st.caption("Auto-detected column mappings:")
                    for orig, mapped in st.session_state.processor.column_mappings.items():
                        if orig != mapped:
                            st.text(f"  {orig} → {mapped}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    with col2:
        st.markdown("**2. HTML Metadata Report**")
        st.caption("CSV or Excel file with URL, Title, H1, H2, Meta Description, Content")

        html_file = st.file_uploader(
            "Upload HTML report",
            type=['csv', 'xlsx', 'xls'],
            key='html_file',
            help="Crawl export with HTML metadata and content"
        )

        if html_file:
            try:
                with st.spinner("Processing HTML data..."):
                    html_data = st.session_state.processor.process_html_data(html_file)
                st.success(f"✅ Loaded {len(html_data):,} URLs")

                with st.expander("Preview HTML Data"):
                    display_cols = ['url', 'title', 'h1', 'meta_description']
                    if 'content' in html_data.columns and html_data['content'].str.len().sum() > 0:
                        display_cols.append('content')
                    available_cols = [col for col in display_cols if col in html_data.columns]
                    st.dataframe(html_data[available_cols].head())

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Analysis section
    if gsc_file and html_file and st.session_state.processor.gsc_data is not None and st.session_state.processor.html_data is not None:
        st.divider()

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Analyze Keyword Opportunities", type="primary"):
                try:
                    with st.spinner("Analyzing keyword opportunities..."):
                        # Content extraction if needed
                        if extract_content:
                            common_urls = list(set(st.session_state.processor.gsc_data['url_clean'].unique()) & 
                                             set(st.session_state.processor.html_data['url_clean'].unique()))
                            if common_urls:
                                st.session_state.processor.extract_content_for_urls(common_urls)

                        # Search volume data if enabled
                        search_volumes = {}
                        if enable_search_volume and api_username and api_password:
                            with st.spinner("Fetching search volume data..."):
                                unique_queries = st.session_state.processor.gsc_data['query'].unique().tolist()
                                st.info(f"📊 Fetching search volumes for {len(unique_queries)} keywords (batches of 1,000)...")
                                dataforseo_client = DataforSEOClient(api_username, api_password)
                                search_volumes = dataforseo_client.get_search_volumes(
                                    unique_queries, 
                                    location_code=location_code,
                                    language_code=language_code
                                )
                                if search_volumes:
                                    st.success(f"✅ Retrieved search volumes for {len(search_volumes)} queries")

                        # Run analysis
                        analyzer = AnalysisEngine(
                            st.session_state.processor.gsc_data,
                            st.session_state.processor.html_data,
                            st.session_state.processor.content_data
                        )

                        results = analyzer.analyze_keywords(
                            top_n_queries=top_n_queries,
                            min_clicks=min_clicks,
                            striking_distance_range=striking_distance_range,
                            search_volumes=search_volumes
                        )

                        st.session_state.analysis_results = results

                    if not results.empty:
                        st.success(f"✅ Analysis complete! Found {len(results)} keyword opportunities.")
                    else:
                        st.warning("⚠️ No opportunities found. Try adjusting your settings.")

                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")

    # Results section
    if st.session_state.analysis_results is not None and not st.session_state.analysis_results.empty:
        results = st.session_state.analysis_results

        st.divider()
        st.subheader("📊 Analysis Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_opportunities = len(results)
            optimization_needed = results['optimization_needed'].sum()
            st.metric("Total Analyzed", total_opportunities, f"{optimization_needed} need optimization")

        with col2:
            high_priority = (results['priority'] == 'High').sum()
            st.metric("High Priority", high_priority, "Striking distance + not optimized")

        with col3:
            striking_distance = results['is_striking_distance'].sum()
            st.metric("Striking Distance", striking_distance, f"Positions {striking_distance_range[0]}-{striking_distance_range[1]}")

        with col4:
            potential_clicks = results[results['optimization_needed']]['clicks'].sum()
            if 'search_volume' in results.columns:
                potential_volume = results[results['optimization_needed']]['search_volume'].sum()
                st.metric("Opportunity Volume", f"{potential_volume:,}", f"{potential_clicks:,} current clicks")
            else:
                st.metric("Opportunity Clicks", f"{potential_clicks:,}", "From unoptimized queries")

        # Filter controls
        st.subheader("🔍 Filter Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            show_only_opportunities = st.checkbox(
                "Show only optimization opportunities", 
                value=True,
                help="Hide queries already found in HTML tags/content"
            )

        with col2:
            priority_filter = st.multiselect(
                "Priority Level",
                options=['High', 'Medium', 'Low'],
                default=['High', 'Medium']
            )

        with col3:
            min_clicks_filter = st.number_input(
                "Minimum clicks to display",
                min_value=0,
                value=0
            )

        # Apply filters
        filtered_results = results.copy()

        if show_only_opportunities:
            filtered_results = filtered_results[filtered_results['optimization_needed'] == True]

        if priority_filter:
            filtered_results = filtered_results[filtered_results['priority'].isin(priority_filter)]

        if min_clicks_filter > 0:
            filtered_results = filtered_results[filtered_results['clicks'] >= min_clicks_filter]

        # Display results
        st.subheader(f"Results ({len(filtered_results)} of {len(results)})")

        if not filtered_results.empty:
            # Prepare display dataframe
            display_columns = [
                'url', 'query', 'clicks', 'impressions', 'position',
                'in_title', 'in_h1', 'in_h2', 'in_meta_description', 'in_content', 'priority'
            ]

            # Add search volume if available
            if 'search_volume' in filtered_results.columns:
                display_columns.insert(5, 'search_volume')

            display_df = filtered_results[display_columns].copy()

            # Format columns
            display_df['position'] = display_df['position'].round(1)
            if 'search_volume' in display_df.columns:
                display_df['search_volume'] = display_df['search_volume'].fillna(0).astype(int)

            # Rename columns for better display
            column_renames = {
                'search_volume': 'Search Volume',
                'in_title': 'In Title',
                'in_h1': 'In H1',
                'in_h2': 'In H2',
                'in_meta_description': 'In Meta Desc',
                'in_content': 'In Content'
            }

            display_df = display_df.rename(columns=column_renames)

            # Color coding for the dataframe
            def highlight_priority(row):
                if row['priority'] == 'High':
                    return ['background-color: #ffebee'] * len(row)
                elif row['priority'] == 'Medium':
                    return ['background-color: #fff3e0'] * len(row)
                else:
                    return [''] * len(row)

            styled_df = display_df.style.apply(highlight_priority, axis=1)

            st.dataframe(
                styled_df,
                use_container_width=True,
                height=500
            )

            # Export functionality
            st.subheader("📥 Export Results")
            col1, col2 = st.columns(2)

            with col1:
                csv_data = filtered_results.to_csv(index=False)
                st.download_button(
                    label="📄 Download Filtered Results as CSV",
                    data=csv_data,
                    file_name=f"striking_distance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col2:
                # Create summary report
                summary_metrics = {
                    'Metric': [
                        'Total URLs Analyzed',
                        'Total Queries Analyzed', 
                        'Queries Needing Optimization',
                        'High Priority Opportunities',
                        'Striking Distance Keywords',
                        'Potential Traffic Impact (clicks)'
                    ],
                    'Value': [
                        results['url'].nunique(),
                        len(results),
                        results['optimization_needed'].sum(),
                        (results['priority'] == 'High').sum(),
                        results['is_striking_distance'].sum(),
                        results[results['optimization_needed']]['clicks'].sum()
                    ]
                }

                if 'search_volume' in results.columns:
                    summary_metrics['Metric'].append('Total Search Volume Opportunity')
                    summary_metrics['Value'].append(results[results['optimization_needed']]['search_volume'].sum())

                summary_df = pd.DataFrame(summary_metrics)
                summary_csv = summary_df.to_csv(index=False)

                st.download_button(
                    label="📈 Download Summary Report",
                    data=summary_csv,
                    file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No results match your current filters. Try adjusting the filter settings.")

if __name__ == "__main__":
    main()
