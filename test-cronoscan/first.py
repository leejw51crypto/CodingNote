import requests
import os
from datetime import datetime, timedelta, timezone

# Retrieve API key from environment variable
API_KEY = os.getenv('MYCRONOSCANKEY')
BASE_URL = 'https://api.cronoscan.com/api'

def get_unix_timestamp_gmt(year, month):
    """Get the Unix timestamp for the start of a month in GMT."""
    start_date = datetime(year, month, 1, tzinfo=timezone.utc)
    return int(start_date.timestamp())

def get_first_block_of_month(year, month):
    timestamp = get_unix_timestamp_gmt(year, month)

    params = {
        'apikey': API_KEY,
        'module': 'block',
        'action': 'getblocknobytime',
        'timestamp': timestamp,
        'closest': 'after'  # Use 'before' or 'after' based on your requirement
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    # Extract the block number from the response
    if data and data.get('status') == '1' and 'result' in data:
        return data['result']
    else:
        return None

# Example Usage
year = 2023
for month in range(1, 13):
    first_block = get_first_block_of_month(year, month)
    if first_block:
        print(f"First block of {year}-{month:02d}: {first_block}")
    else:
        print(f"No data available for {year}-{month:02d}")
