import streamlit as st
import pandas as pd
from modules.data_loader import load_data_file
from modules.scraper import scrape_urls
from modules.analyzer import analyze_striking_distance
from modules.dataforseo import DataForSEOClient
from modules.data_parsers import DataParser
from modules.ai_analysis import AIAnalyzer
from modules.prioritization import PrioritizationEngine
from modules.gsc_connector import (
    get_auth_url,
    get_credentials_from_code,
    save_credentials,
    load_credentials,
    get_search_console_service,
    get_verified_sites,
    fetch_striking_distance_data
)
from datetime import datetime, timedelta

st.set_page_config(page_title="SEO Striking Distance Analyzer", page_icon="🔍", layout="wide")

st.title("🔍 SEO Striking Distance Analyzer")
st.markdown("Analyze if your top organic queries are properly optimized in your HTML tags and content.")

# Sidebar configuration
st.sidebar.header("Configuration")

st.sidebar.subheader("📊 Analysis Settings")
min_clicks = st.sidebar.slider("Minimum Clicks Threshold", 1, 5000, 10, 1)
top_queries = st.sidebar.slider("Top Queries per URL", 1, 20, 5, 1)
use_impressions_weighted = st.sidebar.checkbox("Use Impressions-Weighted Clicks", value=True)

# Branded terms exclusion
st.sidebar.markdown("**🏷️ Branded Terms Exclusion**")
branded_terms_input = st.sidebar.text_input(
    "Branded terms (comma-separated)",
    value="",
    placeholder="e.g., your brand, company name, product",
    help="Exclude queries containing these terms. Case-insensitive, comma-separated."
)

# Parse branded terms
if branded_terms_input.strip():
    branded_terms = [term.strip().lower() for term in branded_terms_input.split(',') if term.strip()]
    st.sidebar.caption(f"🚫 Excluding {len(branded_terms)} branded term(s)")
else:
    branded_terms = []

st.sidebar.subheader("🚀 Data Enrichment")
use_dataforseo = st.sidebar.checkbox(
    "Enrich with DataForSEO",
    value=False,
    help="Add search volume and keyword difficulty scores from DataForSEO API"
)

if use_dataforseo:
    st.sidebar.info("💡 Ensure DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD are set in Streamlit secrets")
    st.sidebar.caption("⚡ Keywords are batched in groups of 1,000 for cost efficiency")

st.sidebar.subheader("🤖 AI-Powered Analysis")
use_ai_analysis = st.sidebar.checkbox(
    "Enable AI Semantic Analysis",
    value=False,
    help="Use AI models to score keyword relevancy and extract topics"
)

if use_ai_analysis:
    ai_model_option = st.sidebar.selectbox(
        "AI Model Strategy",
        options=[
            "Tiered (Gemini → GPT-4o → Claude)",
            "Gemini 2.0 Flash Only",
            "GPT-4o Only",
            "Claude Sonnet 4.5 Only"
        ],
        help="Choose your AI analysis strategy. Tiered is most cost-effective."
    )

    # Check OpenRouter API key
    has_openrouter = "OPENROUTER_API_KEY" in st.secrets and st.secrets.get("OPENROUTER_API_KEY", "").strip()

    # Show API key status
    st.sidebar.markdown("**🔑 API Key Status:**")

    if has_openrouter:
        st.sidebar.success("✅ OpenRouter connected")
        st.sidebar.caption("🌐 Access to all models via single API key")
    else:
        st.sidebar.error("⚠️ Add OPENROUTER_API_KEY to Streamlit secrets")
        st.sidebar.caption("Get your key at: https://openrouter.ai/keys")

    # Model strategy info
    if "Tiered" in ai_model_option:
        st.sidebar.caption("⚡ Most cost-effective: Gemini filters → GPT-4o scores → Claude deep analysis on top 20%")
    elif "Gemini" in ai_model_option:
        st.sidebar.caption("⚡ Fastest & cheapest option (~$0.075/M tokens)")
    elif "GPT-4o" in ai_model_option:
        st.sidebar.caption("⚡ Balanced cost & quality ($2.50/M input)")
    elif "Claude" in ai_model_option:
        st.sidebar.caption("⚡ Highest quality analysis ($3/M input)")
else:
    ai_model_option = None

# Handle OAuth callback from Google
auth_code = st.query_params.get("code")
if auth_code and "gsc_credentials" not in st.session_state:
    credentials = get_credentials_from_code(auth_code)
    if credentials:
        save_credentials(credentials)
        st.query_params.clear()
        st.success("✅ Successfully authenticated with Google!")
        st.rerun()

# File uploads
st.subheader("📁 Data Sources")

# Tabs for different data input methods
tab1, tab2, tab3 = st.tabs(["🔄 Standard Upload", "🚀 Multi-Source Upload", "🔗 Connect to GSC"])

with tab1:
    st.markdown("### 📤 Standard Upload Mode")
    st.success("💡 **Recommended**: Upload Screaming Frog + Google Search Console (both required)")
    st.caption("✨ Tip: Enable DataForSEO enrichment in sidebar for search volume & keyword difficulty")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📄 1. Screaming Frog Export (Required)**")
        st.caption("Contains: URL, Title, H1, H2s, Meta Description")
        st.caption("Provides: On-page SEO data")
        meta_file = st.file_uploader(
            "Upload Screaming Frog crawl",
            type=["xlsx", "csv"],
            key="meta_standard",
            help="Export from Screaming Frog with meta tags and on-page elements"
        )

    with col2:
        st.markdown("**📊 2. Google Search Console (Required)**")
        st.caption("Contains: URL, Query, Position, Clicks, Impressions")
        st.caption("Provides: Actual ranking performance data")
        organic_file = st.file_uploader(
            "Upload GSC Performance Report",
            type=["xlsx", "csv"],
            key="organic_standard",
            help="Export from GSC with keyword performance data (clicks, impressions, position)"
        )

with tab2:
    st.markdown("### 🚀 Multi-Source Upload Mode")
    st.info("💡 **Use this when**: You don't have GSC access, want to merge multiple sources, or only have Ahrefs/Semrush data")
    st.warning("⚠️ Note: Ahrefs/Semrush don't provide clicks/impressions. For best results, use Standard Upload with GSC + DataForSEO enrichment")

    st.markdown("**📊 Keyword Sources** (Upload at least one)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Google Search Console**")
        st.caption("Optional")
        gsc_file = st.file_uploader(
            "GSC export",
            type=["xlsx", "csv"],
            key="gsc_multi",
            help="GSC performance report with URL, Query, Position, Clicks, Impressions"
        )

    with col2:
        st.markdown("**Ahrefs**")
        st.caption("Optional")
        ahrefs_file = st.file_uploader(
            "Ahrefs keywords",
            type=["xlsx", "csv"],
            key="ahrefs_multi",
            help="Ahrefs organic keywords export"
        )

    with col3:
        st.markdown("**Semrush**")
        st.caption("Optional")
        semrush_file = st.file_uploader(
            "Semrush keywords",
            type=["xlsx", "csv"],
            key="semrush_multi",
            help="Semrush organic positions export"
        )

    st.markdown("---")
    st.markdown("**📄 Meta Tags / Crawl Data** (Optional but recommended)")
    st.caption("For better analysis, upload a Screaming Frog crawl with URL, Title, H1, Meta Description")
    meta_file_multi = st.file_uploader(
        "Screaming Frog or meta tags file",
        type=["xlsx", "csv"],
        key="meta_multi",
        help="Provides on-page SEO data for more accurate keyword relevancy analysis"
    )

with tab3:
    st.markdown("### 🔗 Direct Google Search Console Connection")
    st.success("💡 **Recommended**: Connect directly to GSC - no manual exports needed!")
    st.caption("✨ Automatically fetch striking distance data with position filtering")

    # Load existing credentials
    credentials = load_credentials()

    if not credentials:
        # Show authentication UI
        st.info("Please authenticate with Google to access your Search Console data.")
        st.markdown("**Why connect?**")
        st.markdown("- ✅ No manual CSV exports")
        st.markdown("- ✅ Always up-to-date data")
        st.markdown("- ✅ Position filtering at API level")
        st.markdown("- ✅ Access all your verified properties")

        if st.button("🔑 Sign in with Google", type="primary"):
            auth_result = get_auth_url()
            if auth_result:
                auth_url, _ = auth_result
                st.markdown(f"**Click here to authorize:** [🔗 Authorize with Google]({auth_url})")
                st.info("After authorizing, you'll be redirected back to this app.")
    else:
        # Show connected state
        service = get_search_console_service(credentials)

        if service:
            sites = get_verified_sites(service)

            if sites:
                st.success(f"✅ Connected! Found {len(sites)} verified site(s)")

                # Site selection with session state persistence
                if 'selected_gsc_site' not in st.session_state:
                    st.session_state['selected_gsc_site'] = sites[0]

                selected_site = st.selectbox(
                    "🌐 Select Website",
                    sites,
                    index=sites.index(st.session_state['selected_gsc_site']) if st.session_state['selected_gsc_site'] in sites else 0,
                    key="gsc_site_selector",
                    help="Choose which GSC property to analyze"
                )

                # Update session state when selection changes
                st.session_state['selected_gsc_site'] = selected_site

                # Configuration
                col1, col2 = st.columns(2)

                with col1:
                    days_back = st.slider(
                        "📅 Days of data",
                        min_value=7,
                        max_value=90,
                        value=28,
                        help="How many days back to fetch data"
                    )

                with col2:
                    position_range = st.slider(
                        "📊 Position range (striking distance)",
                        min_value=1,
                        max_value=100,
                        value=(8, 20),
                        help="Filter by average position range"
                    )

                min_clicks_gsc = st.slider(
                    "Minimum Clicks Threshold",
                    min_value=1,
                    max_value=100,
                    value=min_clicks,
                    help="Only include queries with at least this many clicks"
                )

                # Also need Screaming Frog data
                st.markdown("---")
                st.markdown("**📄 Screaming Frog Export (Required)**")
                st.caption("Upload your Screaming Frog crawl to complete the analysis")
                meta_file_gsc = st.file_uploader(
                    "Upload Screaming Frog crawl",
                    type=["xlsx", "csv"],
                    key="meta_gsc",
                    help="Export from Screaming Frog with meta tags and on-page elements"
                )

                if st.button("📊 Fetch GSC Data & Analyze", type="primary") and meta_file_gsc:
                    with st.spinner("Fetching data from Google Search Console..."):
                        # Calculate date range
                        end_date = datetime.now() - timedelta(days=1)
                        start_date = end_date - timedelta(days=days_back)

                        # Fetch data using connector
                        organic_df = fetch_striking_distance_data(
                            service=service,
                            site_url=selected_site,
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d'),
                            min_position=position_range[0],
                            max_position=position_range[1],
                            min_clicks=min_clicks_gsc
                        )

                        if not organic_df.empty:
                            st.success(f"✅ Fetched {len(organic_df)} URL-query combinations from GSC!")

                            # Store in session state to continue with analysis
                            st.session_state['gsc_data'] = organic_df
                            st.session_state['gsc_meta_file'] = meta_file_gsc
                            st.session_state['using_gsc'] = True

                            st.info("📊 Data loaded! Scroll down to see the analysis results...")
                        else:
                            st.warning("No data found for the selected criteria. Try adjusting your filters.")

                elif not meta_file_gsc:
                    st.warning("⚠️ Please upload Screaming Frog data to continue")

                # Disconnect option
                st.markdown("---")
                if st.button("🔓 Disconnect Google Account"):
                    if "gsc_credentials" in st.session_state:
                        del st.session_state["gsc_credentials"]
                    st.rerun()
            else:
                st.error("No verified sites found in your Google Search Console account.")
        else:
            st.error("Failed to connect to Google Search Console. Please try reconnecting.")

# Determine which mode we're in
using_gsc = 'gsc_data' in st.session_state and 'gsc_meta_file' in st.session_state
using_multi_source = any([gsc_file, ahrefs_file, semrush_file, meta_file_multi])
using_standard = meta_file and organic_file

if using_standard or using_multi_source or using_gsc:
    try:
        with st.spinner("Loading and parsing data..."):

            # GSC MODE: Direct API fetch + Screaming Frog upload
            if using_gsc and not using_standard and not using_multi_source:
                # Load GSC data from session state
                organic_df = st.session_state['gsc_data']
                organic_source = "Google Search Console (Direct API)"
                st.info(f"✓ GSC data loaded from API: {len(organic_df)} rows")

                # Load meta file from session state
                try:
                    meta_df_raw = load_data_file(st.session_state['gsc_meta_file'])
                    st.info(f"✓ Raw file loaded: {len(meta_df_raw)} rows, {len(meta_df_raw.columns)} columns")
                    st.info(f"📋 Raw columns: {', '.join(meta_df_raw.columns.tolist())}")

                    meta_df, meta_source = DataParser.parse_auto(meta_df_raw)
                    st.info(f"✓ Meta tags file loaded ({meta_source}): {len(meta_df)} rows")
                except Exception as e:
                    st.error(f"❌ Error loading meta tags file: {str(e)}")
                    st.stop()

            # STANDARD MODE: Single meta file + single organic file
            elif using_standard and not using_multi_source:
                # Load meta file
                try:
                    meta_df_raw = load_data_file(meta_file)
                    st.info(f"✓ Raw file loaded: {len(meta_df_raw)} rows, {len(meta_df_raw.columns)} columns")
                    st.info(f"📋 Raw columns: {', '.join(meta_df_raw.columns.tolist())}")

                    meta_df, meta_source = DataParser.parse_auto(meta_df_raw)
                    st.info(f"✓ Meta tags file loaded ({meta_source}): {len(meta_df)} rows")
                except Exception as e:
                    st.error(f"❌ Error loading meta tags file: {str(e)}")
                    st.stop()

                # Load organic file with auto-detection
                try:
                    organic_df_raw = load_data_file(organic_file)
                    st.info(f"✓ Raw file loaded: {len(organic_df_raw)} rows, {len(organic_df_raw.columns)} columns")
                    st.info(f"📋 Raw columns: {', '.join(organic_df_raw.columns.tolist())}")

                    organic_df, organic_source = DataParser.parse_auto(organic_df_raw)
                    st.info(f"✓ Keyword data loaded ({organic_source}): {len(organic_df)} rows")
                except Exception as e:
                    st.error(f"❌ Error loading keyword data file: {str(e)}")
                    st.stop()

            # MULTI-SOURCE MODE: Merge multiple keyword sources
            elif using_multi_source:
                # Load and parse each keyword source
                parsed_sources = []
                source_names = []

                if gsc_file:
                    try:
                        gsc_df_raw = load_data_file(gsc_file)
                        gsc_df, _ = DataParser.parse_auto(gsc_df_raw)
                        parsed_sources.append(gsc_df)
                        source_names.append('GSC')
                        st.info(f"✓ GSC data loaded: {len(gsc_df)} rows")
                    except Exception as e:
                        st.warning(f"⚠️ Error loading GSC file: {str(e)}")

                if ahrefs_file:
                    try:
                        ahrefs_df_raw = load_data_file(ahrefs_file)
                        ahrefs_df, _ = DataParser.parse_auto(ahrefs_df_raw)
                        parsed_sources.append(ahrefs_df)
                        source_names.append('Ahrefs')
                        st.info(f"✓ Ahrefs data loaded: {len(ahrefs_df)} rows")
                    except Exception as e:
                        st.warning(f"⚠️ Error loading Ahrefs file: {str(e)}")

                if semrush_file:
                    try:
                        semrush_df_raw = load_data_file(semrush_file)
                        semrush_df, _ = DataParser.parse_auto(semrush_df_raw)
                        parsed_sources.append(semrush_df)
                        source_names.append('Semrush')
                        st.info(f"✓ Semrush data loaded: {len(semrush_df)} rows")
                    except Exception as e:
                        st.warning(f"⚠️ Error loading Semrush file: {str(e)}")

                if not parsed_sources:
                    st.error("❌ No keyword sources loaded successfully")
                    st.stop()

                # Merge all keyword sources
                organic_df = DataParser.merge_keyword_sources(
                    gsc_df=parsed_sources[0] if 'GSC' in source_names else None,
                    ahrefs_df=parsed_sources[source_names.index('Ahrefs')] if 'Ahrefs' in source_names else None,
                    semrush_df=parsed_sources[source_names.index('Semrush')] if 'Semrush' in source_names else None
                )
                st.success(f"✅ Merged keyword data from {', '.join(source_names)}: {len(organic_df)} unique URL-keyword combinations")

                # Load meta file
                if not meta_file_multi:
                    st.error("❌ Please upload a meta tags/crawl file in multi-source mode")
                    st.stop()

                try:
                    meta_df_raw = load_data_file(meta_file_multi)
                    meta_df, meta_source = DataParser.parse_auto(meta_df_raw)
                    st.info(f"✓ Meta tags file loaded ({meta_source}): {len(meta_df)} rows")
                except Exception as e:
                    st.error(f"❌ Error loading meta tags file: {str(e)}")
                    st.stop()

            # DataParser already standardized column names, so use them directly
            # Standardized column names from DataParser:
            # Meta: url, title, h1, h2, meta_description
            # Organic: url, query, position, clicks, impressions

            meta_columns = {
                'url': 'url',
                'title': 'title',
                'h1': 'h1',
                'h2': 'h2',
                'meta_description': 'meta_description'
            }

            organic_columns = {
                'url': 'url',
                'query': 'query',
                'position': 'position',
                'clicks': 'clicks',
                'impressions': 'impressions'
            }

            # Verify required columns exist
            missing_meta = [col for col in ['url', 'title'] if col not in meta_df.columns]
            missing_organic = [col for col in ['url', 'query', 'position'] if col not in organic_df.columns]

            if missing_meta:
                st.error(f"❌ Missing required columns in meta file: {', '.join(missing_meta)}")
                st.error(f"📋 Columns found in file: {', '.join(meta_df.columns.tolist())}")
                st.error(f"🔍 Detected source: {meta_source}")
                st.info("💡 For Screaming Frog: Export should include 'Address', 'Title 1', 'H1-1', 'Meta Description 1'")
                st.stop()

            if missing_organic:
                st.error(f"❌ Missing required columns in keyword file: {', '.join(missing_organic)}")
                st.error(f"📋 Columns found in file: {', '.join(organic_df.columns.tolist())}")
                st.error(f"🔍 Detected source: {organic_source}")
                st.info("💡 For GSC: Export should include 'Landing Page', 'Query', 'Clicks', 'Impressions', 'Average Position'")
                st.stop()

            st.success("✅ Data loaded and standardized successfully!")

            # Filter out branded terms if specified
            if branded_terms:
                initial_count = len(organic_df)
                query_col = organic_columns['query']

                # Filter out queries containing any branded term (case-insensitive)
                def contains_branded_term(query):
                    query_lower = str(query).lower()
                    return any(branded_term in query_lower for branded_term in branded_terms)

                organic_df = organic_df[~organic_df[query_col].apply(contains_branded_term)]
                excluded_count = initial_count - len(organic_df)

                if excluded_count > 0:
                    st.info(f"🚫 Excluded {excluded_count:,} branded queries from analysis ({len(organic_df):,} remaining)")

            # Show standardized columns in expander
            with st.expander("🔍 View Standardized Columns"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Meta Tags File:**")
                    st.write(f"✓ Available columns: {', '.join([f'`{col}`' for col in meta_df.columns if col in meta_columns.values()])}")

                with col2:
                    st.write("**Keyword Data File:**")
                    st.write(f"✓ Available columns: {', '.join([f'`{col}`' for col in organic_df.columns if col in organic_columns.values()])}")

        if st.button("🚀 Analyze Striking Distance", type="primary"):
            try:
                with st.spinner("Scraping URLs and analyzing... This may take a while."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Scrape URLs for content
                    status_text.text("📡 Scraping URLs for content...")
                    progress_bar.progress(10)

                    urls_to_scrape = meta_df[meta_columns['url']].tolist()
                    st.info(f"🔍 Scraping {len(urls_to_scrape)} URLs...")

                    try:
                        scraped_data = scrape_urls(urls_to_scrape)
                        successful_scrapes = sum(1 for v in scraped_data.values() if v)
                        st.info(f"✓ Successfully scraped {successful_scrapes}/{len(urls_to_scrape)} URLs")
                    except Exception as e:
                        st.error(f"❌ Scraping error: {str(e)}")
                        st.warning("⚠️ Continuing with partial data...")
                        scraped_data = {}

                    # Analyze striking distance
                    status_text.text("🔬 Analyzing keyword opportunities...")
                    progress_bar.progress(50)

                    try:
                        results = analyze_striking_distance(
                            meta_df, organic_df, scraped_data,
                            meta_columns, organic_columns,
                            min_clicks, top_queries, use_impressions_weighted
                        )
                    except Exception as e:
                        st.error(f"❌ Analysis error: {str(e)}")
                        st.stop()

                    # Enrich with DataForSEO if enabled
                    if use_dataforseo and len(results) > 0:
                        status_text.text("🔍 Enriching with search volume & keyword difficulty...")
                        progress_bar.progress(75)

                        try:
                            st.info("📡 Connecting to DataForSEO API...")
                            client = DataForSEOClient()

                            # Get unique keywords before enrichment
                            unique_keywords = results['query'].nunique()
                            st.info(f"🔍 Fetching metrics for {unique_keywords} unique keywords...")

                            # Enrich the dataframe
                            results = client.enrich_dataframe(results, keyword_column='query')

                            # Count how many were successfully enriched
                            enriched_count = results[results['search_volume'] > 0].shape[0]
                            st.success(f"✓ Successfully enriched {enriched_count}/{len(results)} keywords with DataForSEO")

                            # Show sample if successful
                            if enriched_count > 0:
                                sample = results[results['search_volume'] > 0][['query', 'search_volume', 'keyword_difficulty']].head(3)
                                st.info("Sample enriched keywords:")
                                st.dataframe(sample, use_container_width=True)
                            else:
                                st.warning("⚠️ No keywords were enriched. Check DataForSEO API response.")

                        except ValueError as e:
                            st.warning(f"⚠️ DataForSEO credentials not found: {str(e)}")
                            st.info("Add DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD to Streamlit secrets to enable enrichment")
                        except Exception as e:
                            st.error(f"❌ DataForSEO enrichment error: {str(e)}")
                            import traceback
                            st.error(f"Error details: {traceback.format_exc()}")
                            st.warning("⚠️ Continuing without enrichment...")

                    # AI Semantic Analysis if enabled
                    if use_ai_analysis and len(results) > 0:
                        status_text.text("🤖 Running AI semantic analysis...")
                        progress_bar.progress(85)

                        try:
                            analyzer = AIAnalyzer()

                            # Determine which model(s) to use based on user selection
                            if "Tiered" in ai_model_option:
                                use_tiered = True
                                topic_model = "gpt-4o"
                                analysis_mode = "tiered"
                            elif "Gemini" in ai_model_option:
                                use_tiered = False
                                topic_model = "gemini-flash"
                                analysis_mode = "gemini-flash"
                            elif "GPT-4o" in ai_model_option:
                                use_tiered = False
                                topic_model = "gpt-4o"
                                analysis_mode = "gpt-4o"
                            elif "Claude" in ai_model_option:
                                use_tiered = False
                                topic_model = "claude-sonnet-4"
                                analysis_mode = "claude-sonnet-4"

                            # Extract topics and score keywords per URL
                            ai_scores = []
                            urls_processed = results['url'].unique()

                            st.info(f"🔬 Analyzing {len(urls_processed)} URLs with {ai_model_option}...")

                            for url in urls_processed:
                                # Get URL data
                                url_data = meta_df[meta_df[meta_columns['url']] == url].iloc[0]
                                url_keywords = results[results['url'] == url]['query'].tolist()

                                # Extract URL topic from scraped content
                                scraped_content = scraped_data.get(url, {})
                                content_snippet = scraped_content.get('content', '')[:500] if scraped_content else ''

                                url_topic = analyzer.extract_url_topic(
                                    url=url,
                                    title=url_data[meta_columns['title']],
                                    h1=url_data.get(meta_columns.get('h1'), ''),
                                    content_snippet=content_snippet,
                                    model=topic_model
                                )

                                # Run analysis based on selected mode
                                if use_tiered:
                                    # Tiered analysis
                                    keyword_analysis = analyzer.tiered_analysis(
                                        keywords=url_keywords,
                                        url_topic=url_topic,
                                        show_progress=False
                                    )
                                else:
                                    # Single model analysis
                                    keyword_analysis = analyzer.batch_score_keywords(
                                        keywords=url_keywords,
                                        url_topic=url_topic,
                                        model=analysis_mode,
                                        show_progress=False
                                    )
                                    # Convert to expected format
                                    keyword_analysis = {
                                        kw: {
                                            'relevancy_score': score,
                                            'tier': analysis_mode,
                                            'confidence': 'high' if score >= 70 else 'medium' if score >= 40 else 'low'
                                        }
                                        for kw, score in keyword_analysis.items()
                                    }

                                # Add scores to results
                                for keyword, analysis_data in keyword_analysis.items():
                                    ai_scores.append({
                                        'url': url,
                                        'query': keyword,
                                        'ai_relevancy_score': analysis_data['relevancy_score'],
                                        'ai_tier': analysis_data['tier'],
                                        'ai_confidence': analysis_data['confidence']
                                    })

                            # Merge AI scores into results
                            ai_df = pd.DataFrame(ai_scores)
                            results = results.merge(ai_df, on=['url', 'query'], how='left')

                            st.info(f"✓ AI analysis complete: {len(ai_scores)} keywords scored using {ai_model_option}")

                        except Exception as e:
                            st.warning(f"⚠️ AI analysis error: {str(e)}")
                            st.info("Ensure OPENAI_API_KEY, ANTHROPIC_API_KEY, and GOOGLE_AI_API_KEY are set in secrets")
                            st.warning("⚠️ Continuing without AI analysis...")

                    # Calculate SEO Value Scores (Smart Prioritization)
                    if len(results) > 0:
                        status_text.text("🎯 Calculating SEO Value Scores...")
                        progress_bar.progress(95)

                        try:
                            # Initialize prioritization engine with custom weights
                            # Relevancy: 0.40, Traffic: 0.35, Ranking: 0.25
                            engine = PrioritizationEngine(
                                relevancy_weight=0.40,
                                traffic_weight=0.35,
                                ranking_weight=0.25
                            )

                            # Add prioritization scores
                            results = engine.prioritize_dataframe(results)

                            st.info(f"✓ SEO Value Scores calculated for {len(results)} opportunities")

                        except Exception as e:
                            st.warning(f"⚠️ Prioritization error: {str(e)}")
                            st.warning("⚠️ Continuing without prioritization scores...")

                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")

                st.success(f"🎉 Analysis complete! Found {len(results)} keyword opportunities.")

            except Exception as e:
                st.error(f"❌ Unexpected error during analysis: {str(e)}")
                import traceback
                with st.expander("🔍 View Error Details"):
                    st.code(traceback.format_exc())
                st.stop()

            # Display results
            st.subheader("📋 Results")

            # Show top opportunities if SEO Value Scores are available
            if 'seo_value_score' in results.columns:
                st.markdown("### 🎯 Top 10 Priority Opportunities")
                st.caption("Ranked by SEO Value Score (Relevancy: 40%, Traffic: 35%, Ranking: 25%)")

                # Display top 10
                top_10 = results.head(10)[['url', 'query', 'position', 'seo_value_score',
                                           'relevancy_score', 'traffic_potential_score',
                                           'ranking_opportunity_score']].copy()

                # Format scores
                if 'search_volume' in results.columns:
                    top_10 = results.head(10)[['url', 'query', 'position', 'search_volume',
                                               'keyword_difficulty', 'seo_value_score',
                                               'relevancy_score', 'traffic_potential_score',
                                               'ranking_opportunity_score']].copy()

                st.dataframe(top_10, use_container_width=True)

                st.markdown("---")

            st.markdown("### 📊 All Results")
            st.dataframe(results, use_container_width=True)

            # Summary statistics
            total_checks = len(results)
            optimized_count = results['overall_optimized'].sum() if 'overall_optimized' in results.columns else 0
            optimization_rate = (optimized_count / total_checks * 100) if total_checks > 0 else 0

            # Create metrics columns based on available data
            if use_dataforseo and 'search_volume' in results.columns:
                col1, col2, col3, col4 = st.columns(4)
            else:
                col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total URL-Query Combinations", f"{total_checks:,}")
            with col2:
                st.metric("Optimized Combinations", f"{optimized_count:,}")
            with col3:
                st.metric("Optimization Rate", f"{optimization_rate:.1f}%")

            if use_dataforseo and 'search_volume' in results.columns:
                with col4:
                    total_volume = results['search_volume'].sum()
                    st.metric("Total Search Volume", f"{int(total_volume):,}")

            # Prioritization insights
            if 'seo_value_score' in results.columns:
                st.subheader("🎯 Smart Prioritization Insights")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    avg_seo_score = results['seo_value_score'].mean()
                    st.metric("Avg. SEO Value Score", f"{avg_seo_score:.1f}/100")
                with col2:
                    high_priority = len(results[results['seo_value_score'] >= 75])
                    st.metric("High Priority (≥75)", f"{high_priority:,}")
                with col3:
                    medium_priority = len(results[(results['seo_value_score'] >= 50) & (results['seo_value_score'] < 75)])
                    st.metric("Medium Priority (50-74)", f"{medium_priority:,}")
                with col4:
                    low_priority = len(results[results['seo_value_score'] < 50])
                    st.metric("Low Priority (<50)", f"{low_priority:,}")

                # Component score breakdown
                st.markdown("#### 📊 Score Component Breakdown")
                col1, col2, col3 = st.columns(3)

                with col1:
                    avg_relevancy = results['relevancy_score'].mean()
                    st.metric("Avg. Relevancy Score", f"{avg_relevancy:.1f}/100", help="Weight: 40%")
                with col2:
                    avg_traffic = results['traffic_potential_score'].mean()
                    st.metric("Avg. Traffic Potential", f"{avg_traffic:.1f}/100", help="Weight: 35%")
                with col3:
                    avg_ranking = results['ranking_opportunity_score'].mean()
                    st.metric("Avg. Ranking Opportunity", f"{avg_ranking:.1f}/100", help="Weight: 25%")

            # Additional metrics for DataForSEO enriched data
            if use_dataforseo and 'search_volume' in results.columns:
                st.subheader("📈 Traffic Potential Insights")
                col1, col2, col3 = st.columns(3)

                with col1:
                    avg_volume = results['search_volume'].mean()
                    st.metric("Avg. Search Volume", f"{int(avg_volume):,}")
                with col2:
                    avg_kd = results['keyword_difficulty'].mean()
                    st.metric("Avg. Keyword Difficulty", f"{int(avg_kd)}/100")
                with col3:
                    # High-value opportunities: high volume + low difficulty + not optimized
                    high_value = results[
                        (results['search_volume'] > avg_volume) &
                        (results['keyword_difficulty'] < 50) &
                        (results['overall_optimized'] == False)
                    ]
                    st.metric("High-Value Opportunities", f"{len(high_value):,}")

            # Download results (filtered columns)
            # Remove "fluff" score columns from export
            columns_to_remove = [
                'score', 'ai_relevancy_score', 'ai_tier', 'ai_confidence',
                'relevancy_score', 'traffic_potential_score',
                'ranking_opportunity_score', 'seo_value_score'
            ]
            export_columns = [col for col in results.columns if col not in columns_to_remove]
            csv = results[export_columns].to_csv(index=False)

            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="striking_distance_results.csv",
                mime="text/csv",
                key="download_csv_button"
            )

            # Optimization Recommendations Section
            st.markdown("---")
            st.header("✍️ AI-Powered Optimization Recommendations")
            st.markdown("Generate actionable recommendations to optimize your striking distance opportunities.")

            # Import optimization modules
            from modules.optimization_generator import OptimizationGenerator
            from modules.query_clustering import cluster_queries_by_url

            # Select URLs for optimization
            st.markdown("### 🎯 Select URLs to Optimize")

            # Get top opportunities (up to 20 URLs)
            if 'seo_value_score' in results.columns:
                top_urls = results.nlargest(20, 'seo_value_score')['url'].unique()
            else:
                # Fallback: sort by clicks
                url_clicks = results.groupby('url')['clicks'].sum().sort_values(ascending=False)
                top_urls = url_clicks.head(20).index.tolist()

            selected_urls = st.multiselect(
                "Choose URLs to generate optimization plans (up to 10)",
                options=top_urls,
                default=list(top_urls[:3]),
                max_selections=10,
                help="Select URLs where you want AI-powered optimization recommendations"
            )

            if selected_urls and st.button("🚀 Generate Optimization Plans", type="primary", key="generate_optimization_button"):
                with st.spinner("Generating AI-powered recommendations... This may take a few minutes."):
                    try:
                        # Initialize generator
                        opt_generator = OptimizationGenerator()

                        optimization_reports = {}

                        for idx, url in enumerate(selected_urls, 1):
                            st.info(f"Processing {idx}/{len(selected_urls)}: {url}")

                            # Get URL data
                            url_results = results[results['url'] == url]

                            # Get meta data
                            url_meta_data = meta_df[meta_df['url'] == url]

                            if url_meta_data.empty:
                                st.warning(f"⚠️ Skipping {url} - no meta data found")
                                continue

                            url_meta = url_meta_data.iloc[0]

                            # Extract missing keywords (not optimized)
                            missing_kws = url_results[
                                url_results.get('overall_optimized', True) is False
                            ].to_dict('records')

                            if not missing_kws:
                                st.info(f"✓ {url} - all keywords already optimized!")
                                continue

                            # Prepare current elements
                            current_elements = {
                                'title': url_meta.get('title', ''),
                                'h1': url_meta.get('h1', ''),
                                'h2s': str(url_meta.get('h2', '')).split(',') if url_meta.get('h2') else [],
                                'meta_description': url_meta.get('meta_description', '')
                            }

                            # Calculate ranking data
                            ranking_data = {
                                'position': url_results['position'].mean(),
                                'clicks': url_results['clicks'].sum(),
                                'impressions': url_results['impressions'].sum()
                            }

                            # Identify primary intent
                            primary_intent = opt_generator.identify_query_intent(
                                missing_kws[0].get('query', '')
                            )

                            # Generate optimization report
                            report = opt_generator.generate_url_optimization_report(
                                url=url,
                                current_elements=current_elements,
                                missing_keywords=missing_kws,
                                ranking_data=ranking_data,
                                page_intent=primary_intent
                            )

                            optimization_reports[url] = report

                        # Store in session state for export
                        st.session_state['optimization_reports'] = optimization_reports

                        st.success(f"✅ Generated {len(optimization_reports)} optimization reports!")

                    except Exception as e:
                        st.error(f"❌ Error generating recommendations: {str(e)}")
                        import traceback
                        with st.expander("🔍 View Error Details"):
                            st.code(traceback.format_exc())

            # Display reports if they exist
            if 'optimization_reports' in st.session_state and st.session_state['optimization_reports']:
                st.markdown("---")
                st.markdown("### 📋 Optimization Reports")

                for url, report in st.session_state['optimization_reports'].items():
                    with st.expander(f"📄 {url}", expanded=False):
                        # Current Performance
                        st.markdown("#### 📊 Current Performance")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            avg_pos = report['current_performance']['avg_position']
                            st.metric("Avg Position", f"{avg_pos:.1f}")

                        with col2:
                            total_clicks = report['current_performance']['total_clicks']
                            st.metric("Total Clicks", f"{int(total_clicks):,}")

                        with col3:
                            total_impr = report['current_performance']['total_impressions']
                            st.metric("Total Impressions", f"{int(total_impr):,}")

                        # Keyword Analysis
                        st.markdown("#### 🔍 Keyword Analysis")
                        kw_analysis = report.get('keyword_analysis', {})

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Missing", kw_analysis.get('total_missing', 0))
                        with col2:
                            st.metric("Recommended", kw_analysis.get('recommended_count', 0),
                                     help="Keywords that fit naturally and match intent")
                        with col3:
                            st.metric("Filtered Out", kw_analysis.get('filtered_count', 0),
                                     help="Keywords filtered for low relevancy or intent mismatch")

                        if kw_analysis.get('top_recommended'):
                            st.markdown("**Top Recommended Keywords:**")
                            st.markdown(", ".join([f"`{kw}`" for kw in kw_analysis['top_recommended']]))

                        st.markdown("---")

                        # Title Optimization
                        if 'title_variations' in report:
                            st.markdown("#### 📝 Title Optimization")
                            st.markdown(f"**Current Title:** {report['current']['title']}")
                            st.caption(f"Length: {len(report['current']['title'])} characters")

                            st.markdown("**Recommended Variations:**")

                            for i, variation in enumerate(report['title_variations'], 1):
                                st.markdown(f"**Option {i}:**")
                                st.success(variation['title'])
                                st.caption(f"✅ Keywords added: {', '.join(variation.get('keywords_added', []))}")
                                st.caption(f"📏 Length: {variation['length']} chars")
                                st.caption(f"💡 {variation.get('reasoning', '')}")
                                st.markdown("")

                        st.markdown("---")

                        # H1 Optimization
                        if 'h1_suggestion' in report:
                            st.markdown("#### 🎯 H1 Optimization")
                            st.markdown(f"**Current H1:** {report['current']['h1']}")
                            st.markdown(f"**Recommended H1:**")
                            st.success(report['h1_suggestion']['h1'])
                            st.caption(f"✅ Keywords incorporated: {', '.join(report['h1_suggestion'].get('keywords_added', []))}")
                            st.caption(f"💡 {report['h1_suggestion'].get('reasoning', '')}")

                        st.markdown("---")

                        # Meta Description
                        if 'meta_descriptions' in report and report['meta_descriptions']:
                            st.markdown("#### 📄 Meta Description Optimization")
                            st.markdown(f"**Current Meta:** {report['current']['meta_description']}")
                            st.caption(f"Length: {len(report['current']['meta_description'])} characters")

                            st.markdown("**Recommended Variations:**")

                            for i, meta_var in enumerate(report['meta_descriptions'], 1):
                                st.markdown(f"**Option {i}:**")
                                st.success(meta_var['meta_description'])
                                st.caption(f"✅ Keywords added: {', '.join(meta_var.get('keywords_added', []))}")
                                st.caption(f"📏 Length: {meta_var['length']} chars")
                                if 'reasoning' in meta_var:
                                    st.caption(f"💡 {meta_var['reasoning']}")
                                st.markdown("")

                        # No recommendations message
                        if 'recommendation' in report:
                            st.info(report['recommendation'])

                # Export option
                st.markdown("---")
                st.markdown("### 📥 Export Optimization Reports")

                # JSON export
                import json
                json_export = json.dumps(st.session_state['optimization_reports'], indent=2)

                st.download_button(
                    label="📋 Download Reports (JSON)",
                    data=json_export,
                    file_name=f"optimization_reports_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    help="Download all optimization reports in JSON format",
                    key="download_json_button"
                )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your file formats and try again.")

else:
    st.info("Please upload files or connect to Google Search Console to begin the analysis.")
    st.caption("💡 Tip: Use the 'Connect to GSC' tab for the fastest workflow!")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | SEO Striking Distance Analysis Tool")
