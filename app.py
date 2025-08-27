import streamlit as st
import pandas as pd
from modules.data_loader import load_excel_files, detect_columns
from modules.scraper import scrape_urls
from modules.analyzer import analyze_striking_distance
import time

st.set_page_config(page_title="SEO Striking Distance Analyzer", page_icon="🔍", layout="wide")

st.title("🔍 SEO Striking Distance Analyzer")
st.markdown("Analyze if your top organic queries are properly optimized in your HTML tags and content.")

# Sidebar configuration
st.sidebar.header("Configuration")
min_clicks = st.sidebar.slider("Minimum Clicks Threshold", 0, 1, 10, 100,1000)
top_queries = st.sidebar.slider("Top Queries per URL", 1, 20, 5, 1)
use_impressions_weighted = st.sidebar.checkbox("Use Impressions-Weighted Clicks", value=True)

# File uploads
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Meta Tags Report")
    meta_file = st.file_uploader("Upload Excel file with meta tags (Title, H1, H2s, Meta Description)", type=["xlsx"])

with col2:
    st.subheader("📊 Organic Performance Report")
    organic_file = st.file_uploader("Upload Google Search Console export (URLs, Queries, Clicks, Impressions, Position)", type=["xlsx"])

if meta_file and organic_file:
    try:
        with st.spinner("Loading data..."):
            # Load data
            meta_df = load_excel_files(meta_file)
            organic_df = load_excel_files(organic_file)

            # Detect columns
            meta_columns = detect_columns(meta_df, 'meta')
            organic_columns = detect_columns(organic_df, 'organic')

            st.success("Data loaded successfully!")
            st.subheader("Detected Columns")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Meta Tags Report:**")
                for key, value in meta_columns.items():
                    st.write(f"{key}: {value}")

            with col2:
                st.write("**Organic Report:**")
                for key, value in organic_columns.items():
                    st.write(f"{key}: {value}")

        if st.button("🚀 Analyze Striking Distance", type="primary"):
            with st.spinner("Scraping URLs and analyzing... This may take a while."):
                progress_bar = st.progress(0)

                # Scrape URLs for content
                progress_bar.progress(10, "Scraping URLs...")
                scraped_data = scrape_urls(meta_df[meta_columns['url']].tolist())

                # Analyze striking distance
                progress_bar.progress(50, "Analyzing data...")
                results = analyze_striking_distance(
                    meta_df, organic_df, scraped_data,
                    meta_columns, organic_columns,
                    min_clicks, top_queries, use_impressions_weighted
                )

                progress_bar.progress(100, "Complete!")

            st.success("Analysis complete!")

            # Display results
            st.subheader("📋 Results")
            st.dataframe(results, use_container_width=True)

            # Summary statistics
            total_checks = len(results)
            optimized_count = results['Overall_Optimized'].sum()
            optimization_rate = (optimized_count / total_checks * 100) if total_checks > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total URL-Query Combinations", total_checks)
            with col2:
                st.metric("Optimized Combinations", optimized_count)
            with col3:
                st.metric("Optimization Rate", ".1f")

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
