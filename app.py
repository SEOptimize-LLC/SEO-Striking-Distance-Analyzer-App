import streamlit as st
import pandas as pd
from modules.data_loader import load_data_file
from modules.scraper import scrape_urls
from modules.analyzer import analyze_striking_distance
from modules.dataforseo import DataForSEOClient
from modules.data_parsers import DataParser
from modules.ai_analysis import AIAnalyzer
from modules.prioritization import PrioritizationEngine
import time

st.set_page_config(page_title="SEO Striking Distance Analyzer", page_icon="🔍", layout="wide")

st.title("🔍 SEO Striking Distance Analyzer")
st.markdown("Analyze if your top organic queries are properly optimized in your HTML tags and content.")

# Sidebar configuration
st.sidebar.header("Configuration")

st.sidebar.subheader("📊 Analysis Settings")
min_clicks = st.sidebar.slider("Minimum Clicks Threshold", 1, 5000, 10, 10)
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

    # Check which API keys are available (key exists AND has a value)
    has_openai = "OPENAI_API_KEY" in st.secrets and st.secrets.get("OPENAI_API_KEY", "").strip()
    has_anthropic = "ANTHROPIC_API_KEY" in st.secrets and st.secrets.get("ANTHROPIC_API_KEY", "").strip()
    has_google = "GOOGLE_AI_API_KEY" in st.secrets and st.secrets.get("GOOGLE_AI_API_KEY", "").strip()

    # Show API key status based on selected model
    st.sidebar.markdown("**🔑 API Key Status:**")

    if "Tiered" in ai_model_option:
        # Tiered needs all three
        st.sidebar.markdown(f"{'✅' if has_google else '❌'} Google AI (Gemini)")
        st.sidebar.markdown(f"{'✅' if has_openai else '❌'} OpenAI (GPT-4o)")
        st.sidebar.markdown(f"{'✅' if has_anthropic else '❌'} Anthropic (Claude)")

        if not (has_openai and has_anthropic and has_google):
            st.sidebar.error("⚠️ Missing API keys - add them in Streamlit secrets")

        st.sidebar.caption("⚡ Most cost-effective: Gemini filters → GPT-4o scores → Claude deep analysis on top 20%")

    elif "Gemini" in ai_model_option:
        st.sidebar.markdown(f"{'✅' if has_google else '❌'} Google AI (Gemini)")
        if not has_google:
            st.sidebar.error("⚠️ Add GOOGLE_AI_API_KEY to Streamlit secrets")
        st.sidebar.caption("⚡ Fastest & cheapest option (~$0.075/M tokens)")

    elif "GPT-4o" in ai_model_option:
        st.sidebar.markdown(f"{'✅' if has_openai else '❌'} OpenAI (GPT-4o)")
        if not has_openai:
            st.sidebar.error("⚠️ Add OPENAI_API_KEY to Streamlit secrets")
        st.sidebar.caption("⚡ Balanced cost & quality ($2.50/M input)")

    elif "Claude" in ai_model_option:
        st.sidebar.markdown(f"{'✅' if has_anthropic else '❌'} Anthropic (Claude)")
        if not has_anthropic:
            st.sidebar.error("⚠️ Add ANTHROPIC_API_KEY to Streamlit secrets")
        st.sidebar.caption("⚡ Highest quality analysis ($3/M input)")
else:
    ai_model_option = None

# File uploads
st.subheader("📁 Data Sources")

# Tabs for different data input methods
tab1, tab2 = st.tabs(["🔄 Standard Upload", "🚀 Multi-Source Upload"])

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

# Determine which mode we're in
using_multi_source = any([gsc_file, ahrefs_file, semrush_file, meta_file_multi])
using_standard = meta_file and organic_file

if using_standard or using_multi_source:
    try:
        with st.spinner("Loading and parsing data..."):

            # STANDARD MODE: Single meta file + single organic file
            if using_standard and not using_multi_source:
                # Load meta file
                try:
                    meta_df_raw = load_data_file(meta_file)
                    meta_df, meta_source = DataParser.parse_auto(meta_df_raw)
                    st.info(f"✓ Meta tags file loaded ({meta_source}): {len(meta_df)} rows")
                except Exception as e:
                    st.error(f"❌ Error loading meta tags file: {str(e)}")
                    st.stop()

                # Load organic file with auto-detection
                try:
                    organic_df_raw = load_data_file(organic_file)
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
                st.info("💡 Please ensure your file is a valid Screaming Frog export or contains: URL, Title, H1, Meta Description")
                st.stop()

            if missing_organic:
                st.error(f"❌ Missing required columns in keyword file: {', '.join(missing_organic)}")
                st.info("💡 Please ensure your file is a valid GSC export or contains: URL, Query, Position, Clicks, Impressions")
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
                            client = DataForSEOClient()
                            results = client.enrich_dataframe(results, keyword_column='query')
                            st.info(f"✓ Enriched {len(results)} keywords with search volume and KD scores")
                        except ValueError as e:
                            st.warning(f"⚠️ DataForSEO credentials not found: {str(e)}")
                            st.info("Add DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD to Streamlit secrets to enable enrichment")
                        except Exception as e:
                            st.error(f"❌ DataForSEO enrichment error: {str(e)}")
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

            # Download results
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="striking_distance_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your file formats and try again.")

else:
    st.info("Please upload both Excel files to begin the analysis.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | SEO Striking Distance Analysis Tool")