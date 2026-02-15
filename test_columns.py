import pandas as pd
from modules.data_loader import detect_columns

# Test with sample column names that match the user's format
test_columns = pd.Index(['Query', 'Landing Page', 'Clicks', 'Impressions', 'Avg. Pos'])
df = pd.DataFrame(columns=test_columns)
detected = detect_columns(df, 'organic')
print('Organic columns detected:', detected)

# Test with variations
test_columns2 = pd.Index(['Keyword', 'Address', 'Clicks', 'Impressions', 'Average Position'])
df2 = pd.DataFrame(columns=test_columns2)
detected2 = detect_columns(df2, 'organic')
print('Organic columns (variation):', detected2)

# Test meta report
meta_columns = pd.Index(['Address', 'Title', 'H1', 'H2', 'Meta Description'])
df3 = pd.DataFrame(columns=meta_columns)
detected3 = detect_columns(df3, 'meta')
print('Meta columns detected:', detected3)

# Test with redirect url (should be excluded)
test_columns_with_redirect = pd.Index(['Query', 'Landing Page', 'Clicks', 'Impressions', 'Avg. Pos', 'Redirect URL'])
df4 = pd.DataFrame(columns=test_columns_with_redirect)
detected4 = detect_columns(df4, 'organic')
print('Organic columns (with redirect URL):', detected4)