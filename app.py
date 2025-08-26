import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="SEO Striking Distance Analyzer",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for better UI
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
</style>
""", unsafe_allow_html=True)

class DataProcessor:
    """Handles data processing and validation"""
    
    def __init__(self):
        self.gsc_data = None
        self.html_data = None
        self.column_mappings = {}
    
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
    
    def process_gsc_data(self, file) -> pd.DataFrame:
        """Process Google Search Console data"""
        try:
            # Read file
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
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
            df['position'] = pd.to_numeric(df['position'], errors='coerce').fillna(100)
            
            # Clean URLs
            df['url_clean'] = df['url'].str.lower().str.strip().str.rstrip('/')
            
            self.gsc_data = df
            return df
            
        except Exception as e:
            raise ValueError(f"Error processing GSC data: {str(e)}")
    
    def process_html_data(self, file) -> pd.DataFrame:
        """Process HTML metadata report"""
        try:
            # Read file
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            # Expected HTML columns and their variations
            html_columns = {
                'url': ['url', 'address', 'page'],
                'title': ['title', 'page title', 'title tag'],
                'h1': ['h1', 'h1 tag', 'heading 1'],
                'h2': ['h2', 'h2 tag', 'heading 2', 'h2_1'],
                'meta_description': ['meta description', 'description', 'meta desc']
            }
            
            # Auto-detect columns
            mappings = self.detect_columns(df, html_columns)
            
            # Rename columns
            df = df.rename(columns=mappings)
            
            # Ensure URL column exists
            if 'url' not in df.columns:
                raise ValueError("URL column not found in HTML data")
            
            # Clean URLs to match GSC format
            df['url_clean'] = df['url'].str.lower().str.strip().str.rstrip('/')
            
            # Fill missing values and clean text
            text_columns = ['title', 'h1', 'h2', 'meta_description']
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

class AnalysisEngine:
    """Core analysis engine for keyword optimization opportunities"""
    
    def __init__(self, gsc_data: pd.DataFrame, html_data: pd.DataFrame):
        self.gsc_data = gsc_data
        self.html_data = html_data
    
    def analyze_keywords(self, top_n_queries: int = 5, min_clicks: int = 1, 
                        striking_distance_range: Tuple[int, int] = (4, 20)) -> pd.DataFrame:
        """Main analysis function"""
        results = []
        
        # Get unique URLs that exist in both datasets
        gsc_urls = set(self.gsc_data['url_clean'].unique())
        html_urls = set(self.html_data['url_clean'].unique())
        common_urls = gsc_urls.intersection(html_urls)
        
        if not common_urls:
            raise ValueError("No matching URLs found between GSC and HTML data")
        
        for url in common_urls:
            url_results = self._analyze_url(
                url, top_n_queries, min_clicks, striking_distance_range
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
                    striking_range: Tuple[int, int]) -> List[Dict]:
        """Analyze a single URL for keyword opportunities"""
        
        # Get HTML data for this URL
        html_row = self.html_data[self.html_data['url_clean'] == url]
        if html_row.empty:
            return []
        html_row = html_row.iloc[0]
        
        # Get GSC queries for this URL
        url_queries = self.gsc_data[
            (self.gsc_data['url_clean'] == url) & 
            (self.gsc_data['clicks'] >= min_clicks)
        ].copy()
        
        if url_queries.empty:
            return []
        
        # Get top performing queries by clicks
        top_queries = url_queries.nlargest(top_n, 'clicks')
        
        results = []
        for _, query_row in top_queries.iterrows():
            analysis = self._analyze_query_presence(query_row, html_row, url, striking_range)
            results.append(analysis)
        
        return results
    
    def _analyze_query_presence(self, query_row: pd.Series, html_row: pd.Series, 
                               url: str, striking_range: Tuple[int, int]) -> Dict:
        """Check if query is present in HTML elements"""
        
        query = query_row['query'].lower()
        position = query_row['position']
        
        # Check presence in HTML elements
        in_title = self._check_presence(query, html_row.get('title', ''))
        in_h1 = self._check_presence(query, html_row.get('h1', ''))
        in_h2 = self._check_presence(query, html_row.get('h2', ''))
        in_meta = self._check_presence(query, html_row.get('meta_description', ''))
        
        # Overall presence flag
        query_found_in_tags = any([in_title, in_h1, in_h2, in_meta])
        
        # Striking distance check
        is_striking = striking_range[0] <= position <= striking_range[1]
        
        # Priority determination
        if is_striking and not query_found_in_tags:
            priority = "High"
        elif query_found_in_tags:
            priority = "Low"  # Already optimized
        else:
            priority = "Medium"
        
        return {
            'url': url,
            'query': query_row['query'],
            'clicks': query_row['clicks'],
            'impressions': query_row.get('impressions', 0),
            'position': position,
            'is_striking_distance': is_striking,
            'in_title': in_title,
            'in_h1': in_h1,
            'in_h2': in_h2,
            'in_meta_description': in_meta,
            'query_found_in_tags': query_found_in_tags,
            'optimization_needed': not query_found_in_tags,
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
    **Identify high-impact keyword optimization opportunities by analyzing if your top-performing queries are present in key HTML elements.**
    
    Upload your Google Search Console data and HTML metadata report to discover which pages need optimization.
    """)
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        
        top_n_queries = st.slider(
            "Top queries per URL", 
            min_value=3, 
            max_value=10, 
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
        st.info("""
        **💡 How it works:**
        1. Upload your GSC data and HTML report
        2. App identifies top queries by clicks for each URL
        3. Checks if queries are present in Title, H1, H2, Meta Description
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
                    gsc_data = st.session_state.processor.process_gsc_data(gsc_file)
                st.success(f"✅ Loaded {len(gsc_data):,} query records")
                
                with st.expander("Preview GSC Data"):
                    st.dataframe(gsc_data.head())
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.markdown("**2. HTML Metadata Report**")
        st.caption("CSV or Excel file with URL, Title, H1, H2, Meta Description")
        
        html_file = st.file_uploader(
            "Upload HTML report",
            type=['csv', 'xlsx', 'xls'],
            key='html_file',
            help="Crawl export with HTML metadata"
        )
        
        if html_file:
            try:
                with st.spinner("Processing HTML data..."):
                    html_data = st.session_state.processor.process_html_data(html_file)
                st.success(f"✅ Loaded {len(html_data):,} URLs")
                
                with st.expander("Preview HTML Data"):
                    display_cols = ['url', 'title', 'h1', 'meta_description']
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
                        analyzer = AnalysisEngine(
                            st.session_state.processor.gsc_data,
                            st.session_state.processor.html_data
                        )
                        
                        results = analyzer.analyze_keywords(
                            top_n_queries=top_n_queries,
                            min_clicks=min_clicks,
                            striking_distance_range=striking_distance_range
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
            st.metric("Opportunity Clicks", f"{potential_clicks:,}", "From unoptimized queries")
        
        # Filter controls
        st.subheader("🔍 Filter Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_only_opportunities = st.checkbox(
                "Show only optimization opportunities", 
                value=True,
                help="Hide queries already found in HTML tags"
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
            display_df = filtered_results[[
                'url', 'query', 'clicks', 'position', 'is_striking_distance',
                'in_title', 'in_h1', 'in_h2', 'in_meta_description', 
                'query_found_in_tags', 'priority'
            ]].copy()
            
            # Format for better readability
            display_df['position'] = display_df['position'].round(1)
            display_df = display_df.rename(columns={
                'is_striking_distance': 'Striking Distance',
                'in_title': 'In Title',
                'in_h1': 'In H1',
                'in_h2': 'In H2',
                'in_meta_description': 'In Meta Desc',
                'query_found_in_tags': 'Found in Tags'
            })
            
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
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name=f"striking_distance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create summary report
                summary_data = {
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
                summary_df = pd.DataFrame(summary_data)
                summary_csv = summary_df.to_csv(index=False)
                
                st.download_button(
                    label="📈 Download Summary",
                    data=summary_csv,
                    file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No results match your current filters. Try adjusting the filter settings.")

if __name__ == "__main__":
    main()
