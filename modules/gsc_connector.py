"""
Google Search Console API Connector

Provides OAuth2 authentication and direct data fetching from GSC API.
Eliminates need for manual CSV exports.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# OAuth scopes required for GSC access + Google Natural Language API
SCOPES = [
    'https://www.googleapis.com/auth/webmasters.readonly',
    'https://www.googleapis.com/auth/webmasters',
    'https://www.googleapis.com/auth/cloud-language'
]


def get_auth_url() -> Optional[Tuple[str, Flow]]:
    """Generate OAuth authorization URL.

    Returns:
        Tuple of (auth_url, flow) or None if credentials not configured
    """
    try:
        # Check if we have Google OAuth credentials in secrets
        if "google" not in st.secrets:
            st.error("Google OAuth credentials not found in secrets. Please configure your secrets.toml file.")
            return None

        # Build client config from secrets
        client_config = {
            "web": {
                "client_id": st.secrets["google"]["client_id"],
                "client_secret": st.secrets["google"]["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [st.secrets["google"]["redirect_uri"]]
            }
        }

        flow = Flow.from_client_config(
            client_config,
            scopes=SCOPES,
            redirect_uri=st.secrets["google"]["redirect_uri"]
        )

        auth_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )

        return auth_url, flow

    except Exception as e:
        st.error(f"Error generating auth URL: {str(e)}")
        return None


def get_credentials_from_code(auth_code: str) -> Optional[Credentials]:
    """Exchange authorization code for credentials.

    Args:
        auth_code: Authorization code from OAuth callback

    Returns:
        Credentials object or None if exchange fails
    """
    try:
        client_config = {
            "web": {
                "client_id": st.secrets["google"]["client_id"],
                "client_secret": st.secrets["google"]["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [st.secrets["google"]["redirect_uri"]]
            }
        }

        flow = Flow.from_client_config(
            client_config,
            scopes=SCOPES,
            redirect_uri=st.secrets["google"]["redirect_uri"]
        )

        flow.fetch_token(code=auth_code)
        return flow.credentials

    except Exception as e:
        # Don't display error here - let the caller handle it
        # This prevents error messages from persisting across page loads
        return None


def save_credentials(credentials: Credentials) -> None:
    """Save credentials to session state.

    Args:
        credentials: Google OAuth credentials to save
    """
    creds_dict = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes
    }
    st.session_state["gsc_credentials"] = creds_dict


def load_credentials() -> Optional[Credentials]:
    """Load credentials from session state.

    Returns:
        Credentials object or None if not found/expired
    """
    if "gsc_credentials" not in st.session_state:
        return None

    creds_dict = st.session_state["gsc_credentials"]

    credentials = Credentials(
        token=creds_dict["token"],
        refresh_token=creds_dict["refresh_token"],
        token_uri=creds_dict["token_uri"],
        client_id=creds_dict["client_id"],
        client_secret=creds_dict["client_secret"],
        scopes=creds_dict["scopes"]
    )

    # Refresh if expired
    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
            save_credentials(credentials)  # Save refreshed credentials
        except Exception as e:
            st.error(f"Error refreshing credentials: {str(e)}")
            return None

    return credentials


def get_search_console_service(credentials: Credentials):
    """Build Search Console service.

    Args:
        credentials: Valid Google OAuth credentials

    Returns:
        Search Console service object
    """
    try:
        return build('searchconsole', 'v1', credentials=credentials)
    except Exception as e:
        st.error(f"Error building Search Console service: {str(e)}")
        return None


def get_verified_sites(service) -> List[str]:
    """Get list of verified sites from GSC.

    Args:
        service: Search Console service object

    Returns:
        List of verified site URLs
    """
    try:
        sites = service.sites().list().execute()
        return [site['siteUrl'] for site in sites.get('siteEntry', [])]
    except HttpError as e:
        st.error(f"HTTP Error getting sites: {e}")
        return []
    except Exception as e:
        st.error(f"Error getting verified sites: {str(e)}")
        return []


def get_search_console_data(
    service,
    site_url: str,
    start_date: str,
    end_date: str,
    dimensions: List[str] = None,
    filters: Optional[List[Dict]] = None,
    max_rows: int = 25000
) -> pd.DataFrame:
    """Fetch Search Console data with error handling.

    Args:
        service: Search Console service object
        site_url: Site URL to query
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        dimensions: Query dimensions (default: ['page', 'query'])
        filters: Optional dimension filters
        max_rows: Maximum rows to fetch (API limit: 25000)

    Returns:
        DataFrame with GSC data
    """
    try:
        request = {
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': dimensions or ['page', 'query'],
            'rowLimit': min(max_rows, 25000),  # API limit
            'startRow': 0
        }

        if filters:
            request['dimensionFilterGroups'] = [{'filters': filters}]

        # Execute request with error handling
        response = service.searchanalytics().query(
            siteUrl=site_url,
            body=request
        ).execute()

        # Convert to DataFrame
        if 'rows' in response:
            data = []
            for row in response['rows']:
                row_data = {}

                # Add dimension values
                if dimensions:
                    for i, dimension in enumerate(dimensions):
                        row_data[dimension] = row['keys'][i]

                # Add metrics
                row_data.update({
                    'clicks': row.get('clicks', 0),
                    'impressions': row.get('impressions', 0),
                    'ctr': row.get('ctr', 0),
                    'position': row.get('position', 0)
                })

                data.append(row_data)

            return pd.DataFrame(data)
        else:
            return pd.DataFrame()

    except HttpError as e:
        error_details = json.loads(e.content.decode())
        error_message = error_details.get('error', {}).get('message', 'Unknown error')
        st.error(f"Google Search Console API Error: {error_message}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching Search Console data: {str(e)}")
        return pd.DataFrame()


def fetch_striking_distance_data(
    service,
    site_url: str,
    start_date: str,
    end_date: str,
    min_position: float = 4.0,
    max_position: float = 20.0,
    min_clicks: int = 1
) -> pd.DataFrame:
    """Fetch GSC data filtered for striking distance opportunities.

    This is a convenience wrapper that applies position filters at the API level
    for efficiency and returns data in the format expected by the analyzer.

    Args:
        service: Search Console service object
        site_url: Site URL to query
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        min_position: Minimum average position (default: 4)
        max_position: Maximum average position (default: 20)
        min_clicks: Minimum clicks threshold (default: 1)

    Returns:
        DataFrame with columns:
        - url (renamed from 'page')
        - query
        - position (avg)
        - clicks
        - impressions
        - ctr
    """
    # Fetch data with page and query dimensions
    df = get_search_console_data(
        service=service,
        site_url=site_url,
        start_date=start_date,
        end_date=end_date,
        dimensions=['page', 'query'],
        filters=None,  # We'll filter in pandas for more flexibility
        max_rows=25000
    )

    if df.empty:
        return df

    # Apply striking distance filters
    df = df[
        (df['position'] >= min_position) &
        (df['position'] <= max_position) &
        (df['clicks'] >= min_clicks)
    ].copy()

    # Rename 'page' to 'url' to match analyzer expectations
    if 'page' in df.columns:
        df = df.rename(columns={'page': 'url'})

    # Sort by SEO value (clicks * impressions / position)
    if len(df) > 0:
        df['_score'] = (df['clicks'] * df['impressions']) / df['position']
        df = df.sort_values('_score', ascending=False)
        df = df.drop('_score', axis=1)

    return df.reset_index(drop=True)


def test_gsc_connection(service, site_url: str) -> bool:
    """Test GSC connection by fetching a small sample.

    Args:
        service: Search Console service object
        site_url: Site URL to test

    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Fetch just 1 row from last 7 days as a test
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=7)

        request = {
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d'),
            'dimensions': ['page'],
            'rowLimit': 1
        }

        response = service.searchanalytics().query(
            siteUrl=site_url,
            body=request
        ).execute()

        return True

    except Exception as e:
        st.error(f"Connection test failed: {str(e)}")
        return False
