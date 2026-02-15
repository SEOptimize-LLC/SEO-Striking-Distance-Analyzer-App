import requests
from bs4 import BeautifulSoup
import time
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

def scrape_urls(urls: List[str], timeout: int = 15, delay: float = 1.0) -> Dict[str, str]:
    """Scrape main content from a list of URLs.

    Args:
        urls: List of URLs to scrape
        timeout: Request timeout in seconds (increased default to 15s)
        delay: Delay between requests in seconds

    Returns:
        Dictionary mapping URLs to their scraped content
    """
    scraped_data = {}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    }

    errors = []

    for i, url in enumerate(urls, 1):
        try:
            print(f"🔍 Scraping {i}/{len(urls)}: {url}")

            # Validate URL format
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            content = scrape_single_url(url, headers, timeout)

            if content and len(content) > 100:  # At least 100 chars
                scraped_data[url] = content
                print(f"✅ Success! Scraped {len(content)} characters from {url}")
            else:
                scraped_data[url] = ""
                errors.append(f"{url}: Content too short ({len(content)} chars)")
                print(f"⚠️ Warning: Content too short for {url}")

            # Rate limiting - be a good citizen
            if i < len(urls):
                time.sleep(delay)

        except requests.exceptions.Timeout:
            error_msg = f"{url}: Timeout (>{timeout}s)"
            errors.append(error_msg)
            print(f"⏱️ {error_msg}")
            scraped_data[url] = ""
        except requests.exceptions.SSLError as e:
            error_msg = f"{url}: SSL certificate error"
            errors.append(error_msg)
            print(f"🔒 {error_msg}")
            scraped_data[url] = ""
        except requests.exceptions.ConnectionError as e:
            error_msg = f"{url}: Connection refused or DNS failure"
            errors.append(error_msg)
            print(f"🔌 {error_msg}")
            scraped_data[url] = ""
        except requests.exceptions.HTTPError as e:
            error_msg = f"{url}: HTTP {e.response.status_code}"
            errors.append(error_msg)
            print(f"❌ {error_msg}")
            scraped_data[url] = ""
        except Exception as e:
            error_msg = f"{url}: {str(e)[:100]}"
            errors.append(error_msg)
            print(f"⚠️ Unexpected error: {error_msg}")
            scraped_data[url] = ""

    successful = sum(1 for v in scraped_data.values() if v and len(v) > 100)
    print(f"\n✅ Scraping complete: {successful}/{len(urls)} URLs successful")

    if errors:
        print(f"\n⚠️ Scraping errors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")

    return scraped_data

def scrape_single_url(url: str, headers: Dict[str, str], timeout: int = 10) -> str:
    """Scrape main content from a single URL.

    Args:
        url: URL to scrape
        headers: HTTP headers to use
        timeout: Request timeout in seconds

    Returns:
        Extracted text content from the page
    """
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove unwanted elements
    unwanted_selectors = [
        'header', 'nav', 'footer', 'aside', '.sidebar', '.navigation',
        '.menu', '.footer', '.header', '.nav', '.advertisement',
        '.ads', '.social-share', '.comments', '.related-posts'
    ]

    for selector in unwanted_selectors:
        for element in soup.select(selector):
            element.decompose()

    # Try to find main content area
    main_content = None

    # Common main content selectors
    content_selectors = [
        'main', 'article', '.content', '.main-content', '.post-content',
        '.entry-content', '.article-content', '#content', '#main'
    ]

    for selector in content_selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break

    # If no main content found, use body but clean it
    if not main_content:
        main_content = soup.body or soup

    # Extract text from the main content
    text = extract_text_from_element(main_content)

    # Clean up the text
    text = clean_text(text)

    return text

def extract_text_from_element(element) -> str:
    """Extract readable text from BeautifulSoup element."""
    # Remove script and style elements
    for script in element(["script", "style"]):
        script.decompose()

    # Get text
    text = element.get_text()

    # Break into lines and remove leading/trailing space
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r'['']', "'", text)

    return text.strip()