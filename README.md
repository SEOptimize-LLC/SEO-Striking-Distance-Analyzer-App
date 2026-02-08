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

### 🔗 Direct Google Search Console Integration (NEW!)
- **OAuth2 Authentication**: Connect directly to your Google Search Console account
- **No Manual Exports**: Fetch data automatically from GSC API
- **Position Filtering**: Filter by striking distance positions at API level
- **Multi-Property Support**: Access all your verified GSC properties

### 🤖 AI-Powered Semantic Analysis (Unified via OpenRouter)
- **Single API Key**: Access all AI models through OpenRouter (10-20% cost savings)
- **Tiered AI Analysis**: Cost-optimized three-stage approach
  - **Stage 1**: Gemini 2.0 Flash for bulk filtering (keeps keywords ≥40 relevancy)
  - **Stage 2**: GPT-4o for detailed semantic scoring
  - **Stage 3**: Claude Sonnet 4.5 for deep analysis on top 20%
- **Topic Extraction**: AI-powered extraction of URL main topics
- **Relevancy Scoring**: 0-100 semantic relevancy scores for each keyword-URL pair

### ✍️ AI-Powered Optimization Recommendations (NEW!)
- **Smart Keyword Filtering**: 3-stage validation (relevancy + intent matching + natural fit)
- **Title Optimization**: Get 3 AI-generated title variations with keyword incorporation
- **H1 Suggestions**: Optimized H1 headings with reasoning
- **Meta Descriptions**: 2 compelling variations under 160 characters
- **Intent Validation**: Prevents forcing irrelevant keywords that don't match page intent

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

### Recommended Workflow (GSC Direct Integration) - NEW!

1. **Run the application:**
   ```bash
   streamlit run app.py
   ```

2. **Configure optional enrichment** (in sidebar):
   - Enable **DataForSEO** for search volume & keyword difficulty
   - Enable **AI Analysis** for semantic relevancy scoring (via OpenRouter)

3. **Connect to Google Search Console** (in "Connect to GSC" tab):
   - Click "Sign in with Google"
   - Authorize the app
   - Select your website
   - Configure date range and position filters
   - Upload Screaming Frog crawl data
   - Click "Fetch GSC Data & Analyze"

4. **Generate AI-Powered Optimization Recommendations**:
   - Select URLs to optimize (up to 10)
   - Click "Generate Optimization Plans"
   - Get title, H1, H2, and meta description variations
   - Export recommendations as JSON

5. **Review results** and download exports

### Standard Workflow (Manual Upload)

Use this if you prefer manual exports or don't have OAuth set up:

1. **Prepare your data files:**
   - **Screaming Frog Export**: Crawl your site and export meta tags (URL, Title, H1, H2s, Meta Description)
   - **Google Search Console Export**: Export performance report (URL, Query, Position, Clicks, Impressions)

2. **Upload both files** in the "Standard Upload" tab

3. **Click "🚀 Analyze Striking Distance"** to start the analysis

4. **Generate optimization recommendations** (optional)

5. **Review results** and download the CSV export

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

#### Setting up OpenRouter (Optional - For AI Features)

**NEW**: All AI models now accessed through a single OpenRouter API key!

1. **Sign up for OpenRouter**:
   - Get your API key at [https://openrouter.ai/keys](https://openrouter.ai/keys)
   - One API key gives you access to Gemini, GPT-4o, and Claude
   - Pay-as-you-go pricing, often 10-20% cheaper than direct APIs

2. **Add to Streamlit Secrets**:
   ```toml
   # Single API key for all models
   OPENROUTER_API_KEY = "sk-or-v1-..."

   # Optional: Your app URL for OpenRouter rankings
   app_url = "https://your-app.streamlit.app"
   ```

3. **Cost Optimization**:
   - Tiered analysis reduces costs by ~80% vs. using only GPT-4o/Claude
   - Example: 10,000 keywords = ~$0.20-0.50 (vs. $2-3 with single model)
   - Gemini Flash filters out irrelevant keywords early
   - Claude only analyzes top 20% of filtered keywords
   - **10-20% additional savings** vs. direct API access

4. **Available Models via OpenRouter**:
   - **Gemini 2.0 Flash** (`google/gemini-2.0-flash-exp:free`) - Bulk filtering
   - **GPT-4o** (`openai/gpt-4o`) - Semantic scoring
   - **Claude Sonnet 4.5** (`anthropic/claude-sonnet-4.5`) - Deep analysis

#### Setting up Google Search Console Integration (Optional - Recommended)

**NEW**: Connect directly to GSC and eliminate manual CSV exports!

1. **Create OAuth Credentials**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
   - Create a new OAuth 2.0 Client ID
   - Add authorized redirect URI: `https://your-app.streamlit.app` (or `http://localhost:8501` for local)

2. **Add to Streamlit Secrets**:
   ```toml
   [google]
   client_id = "your-client-id.apps.googleusercontent.com"
   client_secret = "your-client-secret"
   redirect_uri = "https://your-app.streamlit.app"
   ```

3. **Benefits**:
   - ✅ No manual CSV exports
   - ✅ Always up-to-date data
   - ✅ Position filtering at API level
   - ✅ Access all verified properties
   - ✅ Faster workflow (50% time savings)

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
- Requests >= 2.31.0 (for OpenRouter API and web scraping)
- BeautifulSoup4 >= 4.12.0
- lxml >= 4.9.0

### Optional Dependencies (for advanced features)
- google-auth >= 2.23.0 (for GSC OAuth)
- google-auth-oauthlib >= 1.1.0 (for GSC OAuth)
- google-auth-httplib2 >= 0.1.1 (for GSC OAuth)
- google-api-python-client >= 2.108.0 (for GSC API)

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