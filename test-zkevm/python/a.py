import requests
import os
import json

# Get the API key from the environment variable
api_key = os.environ.get('CRONOS_ZKEVM_MAINNET_API')
test_address = os.environ.get('CRONOS_ZKEVM_TEST_ADDRESS')

if not api_key:
    raise ValueError("API key not found in environment variables")
if not test_address:
    raise ValueError("Test address not found in environment variables")

# Base URL
base_url = "https://explorer-api.zkevm.cronos.org/api/v1"

# Module and action
module = "account"
action = "getERC20TransferByAddress"

# Parameters
params = {
    "apikey": api_key,
    "address": test_address,
    "startBlock": "156000",
    "endBlock": "156369",
    "limit": "1",  # Increased limit for efficiency
    # Remove the 'offset' parameter
}

# Construct the full URL
url = f"{base_url}/{module}/{action}"

all_results = []
page = 1

while True:
    # Make the GET request
    print(json.dumps(params, indent=2))
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        results = data['result']        
        all_results.extend(results)
        
        print(f"Fetched page {page} with {len(results)} results")
        print(json.dumps(data, indent=2))
        
        
        # Check if we've reached the last page
        pagination = data.get('pagination', {})
        if pagination.get('currentPage') == pagination.get('totalPage'):
            break
        
        # Update session for next page
        params['session'] = pagination.get('session')
        page += 1
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        break

# Print total results
print(f"\nTotal results fetched: {len(all_results)}")
print("\nTransaction Details:")
for result in all_results:
    print(f"Hash: {result['transactionHash']}")
    print(f"Block: {result['blockNumber']}")
    print(f"Timestamp: {result['timestamp']}")
    print("-" * 40)  # Separator for readability