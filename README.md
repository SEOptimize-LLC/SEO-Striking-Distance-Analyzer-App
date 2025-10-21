# SEO Striking Distance Analyzer

An AI-powered Streamlit application that intelligently analyzes and prioritizes SEO keyword opportunities based on semantic relevance, traffic potential, and ranking opportunity.

## Features

### 📊 Data Management
- **Multi-Source Support**: Upload data from Google Search Console, Ahrefs, Semrush, and Screaming Frog
- **Auto-Detection**: Automatically detects data source and column mappings
- **Multi-format Upload**: Supports Excel (.xlsx) and CSV files
- **Smart Merging**: Combine multiple keyword sources with duplicate removal

### 🔍 Analysis Capabilities
- **Web Scraping**: Extracts main content from URLs excluding navigation, footer, and sidebars
- **Flexible Criteria**: Configurable minimum clicks threshold and number of top queries to analyze
- **Impressions-Weighted Analysis**: Option to use clicks/impressions ratio for better query evaluation
- **Detailed Results**: TRUE/FALSE analysis for each URL-query combination across multiple HTML elements

### 🚀 Data Enrichment (Optional)
- **DataForSEO Integration**: Add search volume and keyword difficulty scores
- **Intelligent Batching**: Automatically batches 1,000 keywords per API request for cost efficiency ($0.075/1,000 keywords)
- **Rate Limiting**: Respects API limits (12 requests/minute)
- **Smart Caching**: Prevents duplicate API calls

### 🤖 AI-Powered Semantic Analysis (Optional)
- **Tiered AI Analysis**: Cost-optimized three-stage approach
  - **Stage 1**: Gemini 2.0 Flash for bulk filtering (keeps keywords ≥40 relevancy)
  - **Stage 2**: GPT-4o for detailed semantic scoring
  - **Stage 3**: Claude Sonnet 4.5 for deep analysis on top 20%
- **Topic Extraction**: AI-powered extraction of URL main topics
- **Relevancy Scoring**: 0-100 semantic relevancy scores for each keyword-URL pair

### 🎯 Smart Prioritization Engine
- **SEO Value Score**: Weighted composite score combining:
  - **Relevancy** (40%): AI semantic analysis or keyword presence
  - **Traffic Potential** (35%): Search volume + keyword difficulty
  - **Ranking Opportunity** (25%): Current position + improvement potential
- **Priority Classification**: Automatic categorization into High (≥75), Medium (50-74), Low (<50)
- **Top Opportunities View**: Highlights the top 10 priority keywords
- **Component Breakdown**: Detailed metrics for each scoring component

### 📈 Advanced Insights
- **Traffic Potential Analysis**: Identifies high-value opportunities (high volume + low difficulty)
- **Priority Distribution**: Shows breakdown of high/medium/low priority opportunities
- **Score Analytics**: Average scores for all components
- **Export Functionality**: Download complete results as CSV

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Standard Workflow (Recommended)

1. **Prepare your data files:**
   - **Screaming Frog Export**: Crawl your site and export meta tags (URL, Title, H1, H2s, Meta Description)
   - **Google Search Console Export**: Export performance report (URL, Query, Position, Clicks, Impressions)

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Configure optional enrichment** (in sidebar):
   - Enable **DataForSEO** for search volume & keyword difficulty
   - Enable **AI Analysis** for semantic relevancy scoring

4. **Upload both files** in the "Standard Upload" tab

5. **Click "🚀 Analyze Striking Distance"** to start the analysis

6. **Review results** and download the CSV export

### Multi-Source Upload (Alternative)

Use this when you:
- Don't have GSC access but have Ahrefs/Semrush data
- Want to merge data from multiple keyword sources
- Have historical data from different tools

**Note**: Ahrefs/Semrush don't provide clicks/impressions data. For best results, use Standard Upload with GSC + enable DataForSEO enrichment.

## Configuration Options

### Analysis Settings
- **Minimum Clicks Threshold**: Filter out queries with fewer than this number of clicks
- **Top Queries per URL**: Number of highest-performing queries to analyze per URL
- **Use Impressions-Weighted Clicks**: Calculate query performance using clicks/impressions ratio

### Data Enrichment (Optional)
- **Enrich with DataForSEO**: Add search volume and keyword difficulty scores to your analysis
  - Batches 1,000 keywords per request for cost efficiency
  - Cost: $0.075 per 1,000 keywords
  - Savings: $74,925 per million keywords vs. individual requests

### AI-Powered Analysis (Optional)
- **Enable AI Semantic Analysis**: Use tiered AI models for intelligent keyword relevancy scoring
  - **Gemini 2.0 Flash**: Bulk filtering (keeps ≥40 relevancy)
  - **GPT-4o**: Detailed semantic scoring
  - **Claude Sonnet 4.5**: Deep analysis on top 20%
  - Cost-optimized tiered approach reduces API costs by ~80%

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

#### Setting up AI Models (Optional - For Advanced Features)

1. **OpenAI (GPT-4o)**:
   - Sign up at [https://platform.openai.com/](https://platform.openai.com/)
   - Get API key from [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Pricing: $2.50/M input tokens, $10/M output tokens

2. **Anthropic (Claude Sonnet 4.5)**:
   - Sign up at [https://console.anthropic.com/](https://console.anthropic.com/)
   - Get API key from console
   - Pricing: $3/M input tokens, $15/M output tokens

3. **Google AI (Gemini 2.0 Flash)**:
   - Get API key at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
   - Pricing: $0.075/M input tokens (Gemini Flash 2.0)

4. **Add to Streamlit Secrets**:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ANTHROPIC_API_KEY = "sk-ant-..."
   GOOGLE_AI_API_KEY = "..."
   ```

5. **Cost Optimization**:
   - Tiered analysis reduces costs by ~80% vs. using only GPT-4o/Claude
   - Example: 10,000 keywords = ~$0.20-0.50 (vs. $2-3 with single model)
   - Gemini Flash filters out irrelevant keywords early
   - Claude only analyzes top 20% of filtered keywords

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

### Core Dependencies
- Python 3.9+
- Streamlit >= 1.28.0
- Pandas >= 2.0.0
- OpenPyXL >= 3.1.0 (for Excel files)
- Requests >= 2.31.0
- BeautifulSoup4 >= 4.12.0
- lxml >= 4.9.0

### Optional Dependencies (for advanced features)
- anthropic >= 0.18.0 (for Claude Sonnet 4.5)
- openai >= 1.12.0 (for GPT-4o)
- google-generativeai >= 0.3.0 (for Gemini 2.0 Flash)

**Supported File Formats:**
- Excel files (.xlsx, .xls)
- CSV files (.csv)

**Supported Data Sources:**
- Google Search Console
- Ahrefs
- Semrush
- Screaming Frog
- Custom exports (with standard columns)

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue on GitHub.