import requests
import os
import bech32
from datetime import datetime, timedelta, timezone
import time

API_KEY = os.getenv('MYCRONOSCANKEY')
SCAN_ADDRESS = os.getenv('MYCRONOSCANADDRESS')
BASE_URL = 'https://api.cronoscan.com/api'

def write_to_console_and_file(message, file_name="info.txt"):
    """Write the message to both the console and a file."""
    print(message)
    with open(file_name, "a") as file:
        file.write(message + "\n")

def write_command_to_file(command, file_name="command.txt"):
    """Write the command to a file."""
    with open(file_name, "a") as file:
        file.write(f"{command}\n")

def write_curl_command(myblock, myaddress) :
    command=f"curl \"https://api.cronoscan.com/api?module=account&action=balance&address=$MYETHCRONOSCANADDRESS&apikey=$MYCRONOSCANKEY&tag={myblock}\""
    write_command_to_file(command)


def convert_crc_to_eth(bech32_address):
    _, bz = bech32.bech32_decode(bech32_address)
    hexbytes=bytes(bech32.convertbits(bz, 5, 8))
    eth_address = '0x' + hexbytes.hex()
    return eth_address


def get_unix_timestamp_gmt(year, month, day):
    """Get the Unix timestamp for the start of a day in GMT."""
    start_date = datetime(year, month, day, tzinfo=timezone.utc)
    return int(start_date.timestamp())


def get_latest_block():
    """Get the latest block number."""
    params = {
        'apikey': API_KEY,
        'module': 'block',
        'action': 'getblocknobytime',
        'timestamp': int(datetime.now(timezone.utc).timestamp()),
        'closest': 'before'
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    latest_block = data.get('result') if data and data.get('status') == '1' else None
    return latest_block

def get_latest_balance(eth_address, block_number):
    """Get the latest balance for a given Ethereum address."""
    hex_first_block= hex(int(block_number))
    params = {
        'apikey': API_KEY,
        'module': 'account',
        'action': 'balance',
        'address': eth_address,
     #   'tag': hex_first_block
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    balance = data.get('result') if data and data.get('status') == '1' else None
    return balance

def get_first_block_and_balance(year, month, day):
    timestamp = get_unix_timestamp_gmt(year, month, day)

    # Fetch first block of the day
    block_params = {
        'apikey': API_KEY,
        'module': 'block',
        'action': 'getblocknobytime',
        'timestamp': timestamp,
        'closest': 'after'
    }
    block_response = requests.get(BASE_URL, params=block_params)
    block_data = block_response.json()
    first_block = block_data.get('result') if block_data and block_data.get('status') == '1' else None

    hex_first_block= hex(int(first_block))
    # Fetch balance at that block
    crc_address= SCAN_ADDRESS
    eth_address = convert_crc_to_eth(SCAN_ADDRESS)
    balance_params = {
        'apikey': API_KEY,
        'module': 'account',
        'action': 'balance',
        'tag':    hex_first_block,
        'address': eth_address
    }
    balance_response = requests.get(BASE_URL, params=balance_params)
    balance_data = balance_response.json()
    balance = balance_data.get('result') if balance_data and balance_data.get('status') == '1' else None
    return first_block, balance, crc_address, eth_address

def daterange(start_date, end_date):
    """Generate a range of dates from start_date to end_date."""
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def fetch_first_day_data():
    # Start date is now 30 days before the current date
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)

    # Iterate over each day from 30 days ago to today
    for single_date in daterange(start_date, end_date):
        year = single_date.year
        month = single_date.month
        day = single_date.day
        first_block, balance, crc_address, eth_address = get_first_block_and_balance(year, month, day)
        hex_first_block = hex(int(first_block))
        balance_eth = float(balance) / 1e18 if balance else 0
        write_curl_command(hex_first_block, eth_address)
        if first_block and balance:
            write_to_console_and_file(f"Date: {year}-{month:02d}-{day:02d}, First Block: {first_block} ({hex_first_block}), Balance: {balance_eth} ETH ({balance}) Address: {crc_address} / {eth_address}")
        else:
            write_to_console_and_file(f"No data available for {year}-{month:02d}-{day:02d}")
        time.sleep(10)



def fetch_latest():
    # Get latest block and balance
    latest_block = get_latest_block()
    eth_address = convert_crc_to_eth(SCAN_ADDRESS)
    latest_balance = get_latest_balance(eth_address,latest_block)
    latest_balance_eth = float(latest_balance) / 1e18 if latest_balance else None
    hex_latest_block= hex(int(latest_block))
    write_curl_command(hex_latest_block, eth_address)
    write_to_console_and_file("-----------------------------------------")
    if latest_block and latest_balance:
        write_to_console_and_file(f"Latest Block: {latest_block} ({hex_latest_block}), Latest Balance: {latest_balance_eth} ETH ({latest_balance}) Address: {SCAN_ADDRESS} / {eth_address}")
    else:
        write_to_console_and_file("No data available for the latest block and balance")

fetch_first_day_data()
time.sleep(10)
fetch_latest()
