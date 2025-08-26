# SEO Striking Distance Analyzer

A Streamlit application that identifies keyword optimization opportunities by analyzing if top-performing organic queries are present in key HTML elements.

## Features

- **Intelligent Column Detection**: Automatically maps columns from different tools (GSC, Screaming Frog, etc.)
- **Advanced Text Matching**: Uses multiple strategies to detect query presence in HTML elements
- **Striking Distance Analysis**: Focuses on keywords in positions 4-20 for quick wins
- **Priority Scoring**: Identifies high-impact opportunities first
- **Export Functionality**: Download results and summary reports

## How It Works

1. **Upload Data**: Provide your Google Search Console export and HTML metadata report
2. **Auto-Detection**: App automatically detects and maps relevant columns
3. **Analysis**: Identifies top queries per URL and checks presence in Title, H1, H2, Meta Description
4. **Results**: Shows optimization opportunities with priority scoring

## File Requirements

### Google Search Console Data
- Must contain: URL, Query, Clicks, Position
- Optional: Impressions
- Formats: CSV or Excel

### HTML Metadata Report  
- Must contain: URL, Title
- Optional: H1, H2, Meta Description
- Formats: CSV or Excel

## Quick Start

1. Export your GSC data (Queries report with pages)
2. Export HTML metadata from your crawler (Screaming Frog, etc.)
3. Upload both files to the app
4. Review the analysis and download results

## Deployment

This app can be deployed on Streamlit Cloud:

1. Fork this repository
2. Connect to Streamlit Cloud
3. Deploy directly from GitHub

## Configuration

- **Top Queries**: Number of top-performing queries to analyze per URL (default: 5)
- **Minimum Clicks**: Threshold for including queries in analysis (default: 1)
- **Striking Distance**: Position range to prioritize (default: 4-20)
