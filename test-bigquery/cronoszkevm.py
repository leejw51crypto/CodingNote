from google.cloud import bigquery
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get project ID from environment variable
project_id = os.getenv("PROJECT_ID")
if not project_id:
    raise ValueError("PROJECT_ID not found in .env file")

# Initialize BigQuery client
client = bigquery.Client(project=project_id)

# SQL query to get latest block
query = f"""
SELECT
  MAX(number) as latest_block
FROM
  `{project_id}.cronos_zkevm_mainnet.blocks`
"""

# Execute the query
try:
    query_job = client.query(query)
    results = query_job.result()

    # Print the results
    for row in results:
        print(f"Latest block number: {row.latest_block}")

except Exception as e:
    print(f"Error executing query: {e}")
