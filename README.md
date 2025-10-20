# SEO Striking Distance Analyzer

An AI-powered Streamlit application that intelligently analyzes and prioritizes SEO keyword opportunities based on semantic relevance, traffic potential, and ranking opportunity.

## Features

- **Multi-format File Upload**: Upload Excel (.xlsx) or CSV files - one with meta tags data and another with Google Search Console organic performance data
- **Auto Column Detection**: Automatically detects column names based on common patterns
- **Web Scraping**: Extracts main content from URLs excluding navigation, footer, and sidebars
- **DataForSEO Integration**: Enrich keywords with search volume and keyword difficulty scores (optional)
- **Intelligent Batching**: Automatically batches 1,000 keywords per API request for cost efficiency
- **Flexible Criteria**: Configurable minimum clicks threshold and number of top queries to analyze
- **Impressions-Weighted Analysis**: Option to use clicks/impressions ratio for better query evaluation
- **Traffic Potential Analysis**: Identifies high-value opportunities (high volume + low difficulty)
- **Detailed Results**: TRUE/FALSE analysis for each URL-query combination across multiple HTML elements
- **Export Functionality**: Download results as CSV for further analysis

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your data files:**
   - **Meta Tags Report**: Excel (.xlsx) or CSV file containing URLs and their corresponding meta tags (Title, H1, H2s, Meta Description)
   - **Organic Performance Report**: Google Search Console export as Excel (.xlsx) or CSV file with URLs, queries, clicks, impressions, and average position

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Upload your files** and configure the analysis parameters in the sidebar

4. **Click "🚀 Analyze Striking Distance"** to start the analysis

5. **Review results** and download the CSV export

## Configuration Options

### Analysis Settings
- **Minimum Clicks Threshold**: Filter out queries with fewer than this number of clicks
- **Top Queries per URL**: Number of highest-performing queries to analyze per URL
- **Use Impressions-Weighted Clicks**: Calculate query performance using clicks/impressions ratio

### Data Enrichment (Optional)
- **Enrich with DataForSEO**: Add search volume and keyword difficulty scores to your analysis

#### Setting up DataForSEO (Optional but Recommended)

1. **Sign up for DataForSEO**: Get your API credentials at [https://app.dataforseo.com/api-dashboard](https://app.dataforseo.com/api-dashboard)

2. **For Local Development**:
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Add your credentials:
     ```toml
     DATAFORSEO_LOGIN = "your-email@example.com"
     DATAFORSEO_PASSWORD = "your-api-password"
     ```

3. **For Streamlit Cloud**:
   - Go to your app settings
   - Navigate to the "Secrets" tab
   - Add your credentials:
     ```toml
     DATAFORSEO_LOGIN = "your-email@example.com"
     DATAFORSEO_PASSWORD = "your-api-password"
     ```

4. **Cost Optimization**:
   - Keywords are automatically batched in groups of 1,000
   - Rate limited to 12 requests/minute (API requirement)
   - Example: 10,000 keywords = 10 API requests = ~$1
   - Caching prevents duplicate API calls for the same keywords

## Analysis Elements

The app checks if query terms appear in:
- Page Title
- H1 heading
- H2 headings (up to 5)
- Meta description
- Main content (excluding navigation, footer, sidebars)

## Deployment on Streamlit Cloud

1. Create a new app on [Streamlit Cloud](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set the main file path to `app.py`
4. Deploy!

## Requirements

- Python 3.9+
- Streamlit
- Pandas
- OpenPyXL (for Excel files)
- Requests
- BeautifulSoup4

**Supported File Formats:**
- Excel files (.xlsx, .xls)
- CSV files (.csv)

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue on GitHub.