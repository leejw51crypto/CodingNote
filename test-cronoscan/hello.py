import requests
import os

# Retrieve API key from environment variable
API_KEY = os.getenv('MYCRONOSCANKEY')  # Replace 'MYCRONOSCANKEY' with your environment variable name
BASE_URL = 'https://api.cronoscan.com/api'  # This URL might need to be adjusted

def get_latest_block_height():
    params = {
        'apikey': API_KEY,
        'module': 'proxy',
        'action': 'eth_blockNumber'
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if data and 'result' in data:
        return int(data['result'], 16)  # Convert hex to int
    else:
        return None

# Example Usage
print("Hello World from Cronoscan!")
latest_block_height = get_latest_block_height()
if latest_block_height is not None:
    print(f"Latest block height: {latest_block_height}")
else:
    print

