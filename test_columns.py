import pandas as pd
from modules.data_loader import detect_columns

# Test with sample column names that match the user's format
test_columns = pd.Index(['Query', 'Landing Page', 'Clicks', 'Impressions', 'Avg. Pos'])
df = pd.DataFrame(columns=test_columns)
detected = detect_columns(df, 'organic')
print('Detected columns:', detected)

# Test with variations
test_columns2 = pd.Index(['Keyword', 'Address', 'Clicks', 'Impressions', 'Average Position'])
df2 = pd.DataFrame(columns=test_columns2)
detected2 = detect_columns(df2, 'organic')
print('Detected columns (variation):', detected2)