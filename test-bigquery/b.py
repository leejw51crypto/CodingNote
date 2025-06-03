#!/usr/bin/env python3
"""
BigQuery + OpenAI Integration
Reads user prompts, includes schema information, generates SQL with AI, and executes on BigQuery
"""

import os
import json
from openai import OpenAI
from google.cloud import bigquery
from dotenv import load_dotenv


def load_schemas():
    """Load database schemas from schemas.json file"""
    try:
        with open("schemas.json", "r") as f:
            schemas = json.load(f)
        return schemas
    except Exception as e:
        print(f"Error loading schemas.json: {e}")
        return None


def format_schema_for_prompt(schemas, project_id):
    """Format schemas into a readable string for the AI prompt"""
    schema_text = "Database Schema Information:\n\n"
    schema_text += f"Dataset: {project_id}.cronos_zkevm_mainnet\n"
    schema_text += "Available tables:\n\n"

    for table_name, table_info in schemas.items():
        full_table_name = f"`{project_id}.cronos_zkevm_mainnet.{table_name}`"
        schema_text += f"Table: {full_table_name}\n"
        schema_text += (
            f"Description: {table_info.get('description', 'No description')}\n"
        )
        schema_text += f"Rows: {table_info.get('num_rows', 'Unknown'):,}\n"
        schema_text += f"Size: {table_info.get('num_bytes', 'Unknown'):,} bytes\n"
        schema_text += "Columns:\n"

        for column in table_info.get("schema", []):
            col_name = column.get("name", "Unknown")
            col_type = column.get("type", "Unknown")
            col_mode = column.get("mode", "Unknown")
            col_desc = column.get("description", "No description")
            schema_text += f"  - {col_name} ({col_type}, {col_mode}): {col_desc}\n"

        schema_text += "\n"

    return schema_text


def get_openai_client():
    """Initialize OpenAI client"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return None

    return OpenAI(api_key=api_key)


def get_bigquery_client():
    """Initialize BigQuery client"""
    project_id = os.getenv("PROJECT_ID")
    if not project_id:
        print("Error: PROJECT_ID not found in .env file")
        return None, None

    client = bigquery.Client(project=project_id)
    return client, project_id


def generate_sql_with_ai(client, user_prompt, schema_text, project_id):
    """Use OpenAI to generate SQL query based on user prompt and schema"""
    system_prompt = f"""You are a BigQuery SQL expert. You will be given a user request and database schema information. 
Your task is to generate a valid BigQuery SQL query that answers the user's question.

Important guidelines:
1. Use proper BigQuery syntax
2. Always reference tables with the full format: `{project_id}.cronos_zkevm_mainnet.table_name`
3. Use appropriate data types and functions for BigQuery
4. Include proper filtering, aggregation, and ordering as needed
5. Return ONLY the SQL query, no explanations or markdown formatting
6. Make sure the query is optimized and follows BigQuery best practices
7. Use proper BigQuery functions for timestamps, numeric operations, etc.
8. When working with large datasets, consider using LIMIT clauses appropriately
9. IMPORTANT: If any "hash" fieldname is used, wrap it with backticks since "hash" is a reserved word in SQL (e.g., `hash`)"""

    prompt = f"""User prompt: {user_prompt}

{schema_text}

Please generate a BigQuery SQL query to answer the user's request. Remember to use the full table names with project and dataset."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.1,
        )

        sql_query = response.choices[0].message.content.strip()
        # Clean up any markdown formatting that might have slipped through
        if sql_query.startswith("```"):
            sql_query = sql_query.split("\n", 1)[1]
        if sql_query.endswith("```"):
            sql_query = sql_query.rsplit("\n", 1)[0]

        return sql_query

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def execute_bigquery(bq_client, project_id, sql_query):
    """Execute SQL query on BigQuery and return results"""
    try:
        # Ensure proper project.dataset.table format (similar to cronoszkevm.py approach)
        # Replace any incomplete table references
        tables = ["batches", "blocks", "logs", "transactions"]
        for table in tables:
            # Replace bare table names with full qualified names
            sql_query = sql_query.replace(
                f" {table} ", f" `{project_id}.cronos_zkevm_mainnet.{table}` "
            )
            sql_query = sql_query.replace(
                f" {table}\n", f" `{project_id}.cronos_zkevm_mainnet.{table}`\n"
            )
            sql_query = sql_query.replace(
                f"FROM {table}", f"FROM `{project_id}.cronos_zkevm_mainnet.{table}`"
            )
            sql_query = sql_query.replace(
                f"JOIN {table}", f"JOIN `{project_id}.cronos_zkevm_mainnet.{table}`"
            )

        print("Executing SQL Query:")
        print("-" * 50)
        print(sql_query)
        print("-" * 50)

        query_job = bq_client.query(sql_query)
        results = query_job.result()

        return results

    except Exception as e:
        print(f"Error executing BigQuery: {e}")
        print(f"SQL Query that failed: {sql_query}")
        return None


def print_results(results):
    """Print BigQuery results in a formatted way"""
    if not results:
        print("No results returned.")
        return

    print("\nQuery Results:")
    print("=" * 60)

    # Get column names from the first row
    first_row = None
    row_count = 0

    for row in results:
        if first_row is None:
            first_row = row
            # Print header
            headers = list(row.keys())
            header_str = " | ".join(f"{header:>15}" for header in headers)
            print(header_str)
            print("-" * len(header_str))

        # Print row values
        values = [str(row[key]) for key in row.keys()]
        row_str = " | ".join(f"{value:>15}" for value in values)
        print(row_str)

        row_count += 1

        # Limit output for large result sets
        if row_count >= 50:
            print(f"\n... (showing first 50 rows of {results.total_rows} total rows)")
            break

    if row_count == 0:
        print("No data returned from query.")
    else:
        print(f"\nTotal rows processed: {row_count}")


def main():
    """Main function to orchestrate the BigQuery + AI workflow"""
    # Load environment variables
    load_dotenv()

    print("BigQuery + OpenAI SQL Generator")
    print("=" * 40)

    # Get user prompt
    user_prompt = input("\nEnter your question about the database: ").strip()
    if not user_prompt:
        print("No prompt provided. Exiting.")
        return

    # Initialize BigQuery client first to get project_id
    print("Initializing BigQuery client...")
    bq_client, project_id = get_bigquery_client()
    if not bq_client:
        return

    print(f"Using project: {project_id}")
    print(f"Dataset: {project_id}.cronos_zkevm_mainnet")

    # Load schemas
    print("\nLoading database schemas...")
    schemas = load_schemas()
    if not schemas:
        return

    # Format schemas for AI prompt with project_id
    schema_text = format_schema_for_prompt(schemas, project_id)

    # Initialize OpenAI client
    print("Initializing OpenAI client...")
    openai_client = get_openai_client()
    if not openai_client:
        return

    # Generate SQL with AI
    print("Generating SQL query with AI...")
    sql_query = generate_sql_with_ai(
        openai_client, user_prompt, schema_text, project_id
    )
    if not sql_query:
        return

    # Execute BigQuery
    print("Executing query on BigQuery...")
    results = execute_bigquery(bq_client, project_id, sql_query)
    if results is None:
        return

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
