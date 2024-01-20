import requests
import os
from datetime import datetime, timedelta, timezone
from calendar import monthrange

# Retrieve API key from environment variable
API_KEY = os.getenv('MYCRONOSCANKEY')
SCAN_ADDRESS = os.getenv('MYCRONOSCANADDRESS')
BASE_URL = 'https://api.cronoscan.com/api'

def get_unix_timestamp_gmt(year, month, day):
    """Get the Unix timestamp for the start of a day in GMT."""
    start_date = datetime(year, month, day, tzinfo=timezone.utc)
    return int(start_date.timestamp())

def get_first_block_of_day(year, month, day):
    timestamp = get_unix_timestamp_gmt(year, month, day)

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

def daterange(start_date, end_date):
    """Generate a range of dates from start_date to end_date."""
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

# Start and end dates
start_year = 2023
start_month = 12
start_day = 1
end_date = datetime.now(timezone.utc)

# Iterate over each day from the start date to today
for single_date in daterange(datetime(start_year, start_month, start_day, tzinfo=timezone.utc), end_date):
    year = single_date.year
    month = single_date.month
    day = single_date.day
    first_block = get_first_block_of_day(year, month, day)
    if first_block:
        print(f"First block of {year}-{month:02d}-{day:02d}: {first_block}")
    else:
        print(f"No data available for {year}-{month:02d}-{day:02d}")
