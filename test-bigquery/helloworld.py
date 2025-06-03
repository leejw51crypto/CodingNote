from google.cloud import bigquery
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Get project ID from environment variable
project_id = os.getenv("PROJECT_ID")
if not project_id:
    raise ValueError("PROJECT_ID not found in .env file")

# Initialize BigQuery client
client = bigquery.Client(project=project_id)

# Define the dataset name
dataset_id = "sample_dataset"
table_id = "fruits"

# Create dataset if it doesn't exist
dataset_ref = f"{project_id}.{dataset_id}"
try:
    client.get_dataset(dataset_ref)
except Exception:
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"
    dataset = client.create_dataset(dataset, timeout=30)
    print(f"Created dataset {dataset_id}")

# Define schema for the fruits table
schema = [
    bigquery.SchemaField("fruit_id", "INTEGER"),
    bigquery.SchemaField("name", "STRING"),
    bigquery.SchemaField("color", "STRING"),
    bigquery.SchemaField("price", "FLOAT"),
    bigquery.SchemaField("last_updated", "TIMESTAMP"),
]

# Create table
table_ref = f"{dataset_ref}.{table_id}"
table = bigquery.Table(table_ref, schema=schema)
try:
    client.get_table(table)
    print("Table already exists")
except Exception:
    table = client.create_table(table)
    print(f"Created table {table_id}")

# Sample data
rows_to_insert = [
    (1, "Apple", "Red", 0.50, datetime.now()),
    (2, "Banana", "Yellow", 0.30, datetime.now()),
    (3, "Orange", "Orange", 0.60, datetime.now()),
    (4, "Grape", "Purple", 2.50, datetime.now()),
    (5, "Strawberry", "Red", 3.00, datetime.now()),
    (6, "Blueberry", "Blue", 4.00, datetime.now()),
    (7, "Mango", "Yellow", 1.50, datetime.now()),
    (8, "Pineapple", "Yellow", 2.00, datetime.now()),
    (9, "Kiwi", "Brown", 1.00, datetime.now()),
    (10, "Pear", "Green", 0.75, datetime.now()),
]

# Insert data
errors = client.insert_rows(table, rows_to_insert)
if errors == []:
    print("Successfully inserted sample data")
else:
    print("Errors occurred while inserting data:", errors)

# Query the table
query = f"""
    SELECT name, color, price
    FROM `{project_id}.{dataset_id}.{table_id}`
    ORDER BY price DESC
    LIMIT 5
"""

query_job = client.query(query)
print("\nTop 5 most expensive fruits:")
for row in query_job:
    print(f"{row.name} ({row.color}): ${row.price:.2f}")
