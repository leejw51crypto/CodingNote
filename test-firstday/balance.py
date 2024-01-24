import requests
import os

def get_balance(address, apikey):
    url = f"https://explorer-api.cronos.org/mainnet/api/v1/account/getBalance?address={address}&apikey={apikey}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == '1' and data['message'] == 'OK':
            return data['result']['balance']
        else:
            raise Exception("API returned an error: " + data['message'])
    else:
        response.raise_for_status()

def get_latest_block(apikey):
    url = f"https://explorer-api.cronos.org/mainnet/api/v1/ethproxy/getBlockNumber?apikey={apikey}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "result" in data:
            return int(data["result"], 16)  # Convert hex to int
        else:
            raise Exception("Error in getting latest block: No result found")
    else:
        response.raise_for_status()

# Usage
address = os.getenv("MYETHCRONOSCANADDRESS")
apikey = os.getenv("MYCRONOSINDEXINGKEY")

if address and apikey:
    try:
        balance = get_balance(address, apikey)
        latest_block = get_latest_block(apikey)
        print(f"Balance: {balance}")
        print(f"Latest Block Number: {latest_block}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Environment variables for address or apikey are not set.")
