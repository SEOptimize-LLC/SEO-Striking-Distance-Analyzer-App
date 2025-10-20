import streamlit as st
import pandas as pd
from modules.data_loader import load_data_file, detect_columns
from modules.scraper import scrape_urls
from modules.analyzer import analyze_striking_distance
from modules.dataforseo import DataForSEOClient
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

st.sidebar.subheader("🚀 Data Enrichment")
use_dataforseo = st.sidebar.checkbox(
    "Enrich with DataForSEO",
    value=False,
    help="Add search volume and keyword difficulty scores from DataForSEO API"
)

if use_dataforseo:
    st.sidebar.info("💡 Ensure DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD are set in Streamlit secrets")
    st.sidebar.caption("⚡ Keywords are batched in groups of 1,000 for cost efficiency")

# File uploads
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Meta Tags Report")
    meta_file = st.file_uploader("Upload Excel or CSV file with meta tags (Title, H1, H2s, Meta Description)", type=["xlsx", "csv"])

with col2:
    st.subheader("📊 Organic Performance Report")
    organic_file = st.file_uploader("Upload Google Search Console export (URLs, Queries, Clicks, Impressions, Position)", type=["xlsx", "csv"])

if meta_file and organic_file:
    try:
        with st.spinner("Loading data..."):
            # Load data
            try:
                meta_df = load_data_file(meta_file)
                st.info(f"✓ Meta tags file loaded: {len(meta_df)} rows")
            except Exception as e:
                st.error(f"❌ Error loading meta tags file: {str(e)}")
                st.stop()

            try:
                organic_df = load_data_file(organic_file)
                st.info(f"✓ Organic performance file loaded: {len(organic_df)} rows")
            except Exception as e:
                st.error(f"❌ Error loading organic performance file: {str(e)}")
                st.stop()

            # Detect columns
            try:
                meta_columns = detect_columns(meta_df, 'meta')
            except ValueError as e:
                st.error(f"❌ Meta tags file column detection failed: {str(e)}")
                st.info("💡 Please ensure your file contains columns for: URL/Address, Title, H1, and Meta Description")
                st.stop()

            try:
                organic_columns = detect_columns(organic_df, 'organic')
            except ValueError as e:
                st.error(f"❌ Organic performance file column detection failed: {str(e)}")
                st.info("💡 Please ensure your file contains columns for: URL/Landing Page, Query, Clicks, Impressions, and Position")
                st.stop()

            st.success("✅ Data loaded successfully!")
            st.subheader("Detected Columns")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Meta Tags Report:**")
                for key, value in meta_columns.items():
                    st.write(f"✓ {key}: `{value}`")

            with col2:
                st.write("**Organic Report:**")
                for key, value in organic_columns.items():
                    st.write(f"✓ {key}: `{value}`")

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