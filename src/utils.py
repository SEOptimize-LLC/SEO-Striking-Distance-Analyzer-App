"""
Utility functions for SEO Striking Distance Analyzer
"""
import re
import pandas as pd
import numpy as np
from typing import List, Set, Tuple, Optional
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def clean_url(url: str) -> str:
    """
    Clean and standardize URLs for matching
    
    Args:
        url: Raw URL string
        
    Returns:
        Cleaned URL string
    """
    if pd.isna(url) or not url:
        return ''
    
    url = str(url).strip().lower()
    
    # Remove trailing slashes
    url = url.rstrip('/')
    
    # Remove common tracking parameters (optional)
    # You can uncomment this if you want to remove URL parameters
    # url = re.sub(r'\?.*$', '', url)
    
    # Remove fragments
    url = re.sub(r'#.*$', '', url)
    
    return url

def clean_text(text: str) -> str:
    """
    Clean text content for analysis
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text) or not text:
        return ''
    
    text = str(text)
    
    # Remove HTML tags if any
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # Convert to lowercase for matching
    text = text.lower().strip()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = ' '.join(text.split())
    
    return text

def get_ngrams(text: str, n: int) -> Set[str]:
    """
    Generate n-grams from text
    
    Args:
        text: Input text
        n: Size of n-grams
        
    Returns:
        Set of n-grams
    """
    words = text.lower().split()
    if len(words) < n:
        return set()
    
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.add(ngram)
    
    return ngrams

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using Jaccard similarity
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def check_keyword_presence(keyword: str, text: str, threshold: float = 0.7) -> Tuple[bool, float]:
    """
    Check if keyword is present in text with fuzzy matching
    
    Args:
        keyword: Keyword to search for
        text: Text to search in
        threshold: Minimum similarity threshold for partial matches
        
    Returns:
        Tuple of (is_present, match_score)
    """
    if not text or not keyword:
        return False, 0.0
    
    keyword = keyword.lower().strip()
    text = text.lower().strip()
    
    # Strategy 1: Exact match
    if keyword in text:
        return True, 1.0
    
    # Strategy 2: All words present
    keyword_words = set(keyword.split())
    text_words = set(text.split())
    
    if keyword_words.issubset(text_words):
        return True, 0.9
    
    # Strategy 3: Partial word match
    words_present = keyword_words.intersection(text_words)
    if len(keyword_words) > 0:
        match_ratio = len(words_present) / len(keyword_words)
        if match_ratio >= threshold:
            return True, match_ratio
    
    # Strategy 4: Check for phrase with minor variations
    # Remove common stop words and check
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    keyword_important = ' '.join([w for w in keyword_words if w not in stop_words])
    text_important = ' '.join([w for w in text_words if w not in stop_words])
    
    if keyword_important and keyword_important in text_important:
        return True, 0.8
    
    return False, 0.0

def extract_domain(url: str) -> str:
    """
    Extract domain from URL
    
    Args:
        url: Full URL
        
    Returns:
        Domain name
    """
    if not url:
        return ''
    
    # Remove protocol
    domain = re.sub(r'^https?://', '', url.lower())
    
    # Remove www
    domain = re.sub(r'^www\.', '', domain)
    
    # Remove path
    domain = domain.split('/')[0]
    
    return domain

def format_number(num: float, decimal_places: int = 2) -> str:
    """
    Format number for display
    
    Args:
        num: Number to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string
    """
    if pd.isna(num):
        return '0'
    
    if num >= 1000000:
        return f"{num/1000000:.{decimal_places}f}M"
    elif num >= 1000:
        return f"{num/1000:.{decimal_places}f}K"
    else:
        return str(int(num))

def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate if DataFrame has required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    df_columns = set(df.columns)
    required_set = set(required_columns)
    
    missing = list(required_set - df_columns)
    
    return len(missing) == 0, missing

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default
    
    return numerator / denominator

def normalize_series(series: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    Normalize a pandas series
    
    Args:
        series: Series to normalize
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        Normalized series
    """
    if method == 'minmax':
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series([0] * len(series), index=series.index)
        return (series - mean) / std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
