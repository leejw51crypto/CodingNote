from google.cloud import bigquery
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Get project ID from environment variable
project_id = os.getenv("PROJECT_ID")
if not project_id:
    raise ValueError("PROJECT_ID not found in .env file")

# Initialize BigQuery client
client = bigquery.Client(project=project_id)


def format_padded_table(headers, rows):
    """Format a markdown table with proper padding"""
    # Calculate column widths
    col_widths = []
    for i, header in enumerate(headers):
        max_width = len(header)
        for row in rows:
            if i < len(row):
                max_width = max(max_width, len(str(row[i])))
        col_widths.append(max_width + 2)  # Add 2 for padding

    # Format header
    header_line = "|"
    separator_line = "|"
    for i, header in enumerate(headers):
        padded_header = f" {header:<{col_widths[i]-1}}"
        header_line += padded_header + "|"
        separator_line += "-" * col_widths[i] + "|"

    # Format rows
    table_lines = [header_line, separator_line]
    for row in rows:
        row_line = "|"
        for i, cell in enumerate(row):
            if i < len(col_widths):
                padded_cell = f" {str(cell):<{col_widths[i]-1}}"
                row_line += padded_cell + "|"
        table_lines.append(row_line)

    return "\n".join(table_lines)


def print_table_schema_markdown(dataset_id, table_id):
    """Print the schema of a specific table in markdown format"""
    table_ref = client.dataset(dataset_id).table(table_id)
    table = client.get_table(table_ref)

    print(f"\n## {dataset_id}.{table_id}")
    print(f"\n**Description:** {table.description or 'No description'}")
    print(f"**Number of rows:** {table.num_rows:,}")
    print(f"**Table size:** {table.num_bytes:,} bytes")
    print(f"**Created:** {table.created}")
    print(f"**Modified:** {table.modified}")

    print("\n### Schema")

    # Prepare table data
    headers = ["Field Name", "Type", "Mode", "Description"]
    rows = []

    for field in table.schema:
        mode = field.mode if field.mode else "NULLABLE"
        description = field.description if field.description else ""
        # Escape pipe characters in descriptions for markdown table
        description = description.replace("|", "\\|") if description else ""
        rows.append(
            [f"`{field.name}`", f"`{field.field_type}`", f"`{mode}`", description]
        )

    # Print padded table
    table_output = format_padded_table(headers, rows)
    print(table_output)


def generate_markdown_file(dataset_id, tables):
    """Generate a complete markdown file with all schemas"""
    with open("schemas.md", "w") as f:
        f.write(f"# BigQuery Dataset Schema: {dataset_id}\n\n")
        f.write(
            f"Generated on: {client.query('SELECT CURRENT_DATETIME()').result().__next__()[0]}\n\n"
        )
        f.write("## Table of Contents\n\n")

        # Generate TOC
        for table_id in tables:
            f.write(f"- [{table_id}](#{table_id.lower().replace('_', '-')})\n")
        f.write("\n---\n")

        # Generate detailed schemas
        for table_id in tables:
            try:
                table_ref = client.dataset(dataset_id).table(table_id)
                table = client.get_table(table_ref)

                f.write(f"\n## {table_id}\n\n")
                f.write(f"**Description:** {table.description or 'No description'}\n\n")
                f.write(f"**Statistics:**\n")
                f.write(f"- Number of rows: {table.num_rows:,}\n")
                f.write(f"- Table size: {table.num_bytes:,} bytes\n")
                f.write(f"- Created: {table.created}\n")
                f.write(f"- Modified: {table.modified}\n\n")

                f.write("### Schema\n\n")

                # Prepare table data for file output
                headers = ["Field Name", "Type", "Mode", "Description"]
                rows = []

                for field in table.schema:
                    mode = field.mode if field.mode else "NULLABLE"
                    description = field.description if field.description else ""
                    # Escape pipe characters in descriptions for markdown table
                    description = description.replace("|", "\\|") if description else ""
                    rows.append(
                        [
                            f"`{field.name}`",
                            f"`{field.field_type}`",
                            f"`{mode}`",
                            description,
                        ]
                    )

                # Write padded table to file
                table_output = format_padded_table(headers, rows)
                f.write(table_output)
                f.write("\n\n---\n")

            except Exception as e:
                f.write(f"\n## {table_id}\n\n")
                f.write(f"❌ Error getting schema: {e}\n\n---\n")


def get_dataset_tables(dataset_id):
    """Get all tables in a dataset"""
    dataset_ref = client.dataset(dataset_id)
    tables = list(client.list_tables(dataset_ref))
    return [table.table_id for table in tables]


def main():
    dataset_id = "cronos_zkevm_mainnet"

    try:
        # Get all tables in the dataset
        print(f"# BigQuery Dataset Schema: {dataset_id}\n")
        tables = get_dataset_tables(dataset_id)

        if not tables:
            print(f"❌ No tables found in dataset {dataset_id}")
            return

        print(f"📊 Found **{len(tables)}** tables:\n")
        for table_id in tables:
            print(f"- `{table_id}`")

        print("\n---\n")

        # Print schema for each table in markdown format
        for table_id in tables:
            try:
                print_table_schema_markdown(dataset_id, table_id)
            except Exception as e:
                print(f"\n## ❌ Error: {table_id}\n")
                print(f"Could not retrieve schema: {e}\n")

        # Generate markdown file
        print(f"\n---\n")
        print("💾 Generating markdown file...")
        generate_markdown_file(dataset_id, tables)
        print("✅ Schema saved to `schemas.md`")

        # Also save schemas to JSON file for programmatic access
        schemas = {}
        for table_id in tables:
            try:
                table_ref = client.dataset(dataset_id).table(table_id)
                table = client.get_table(table_ref)
                schemas[table_id] = {
                    "description": table.description,
                    "num_rows": table.num_rows,
                    "num_bytes": table.num_bytes,
                    "created": str(table.created),
                    "modified": str(table.modified),
                    "schema": [
                        {
                            "name": field.name,
                            "type": field.field_type,
                            "mode": field.mode,
                            "description": field.description,
                        }
                        for field in table.schema
                    ],
                }
            except Exception as e:
                print(f"Error processing {table_id} for JSON: {e}")

        # Save to JSON file
        with open("schemas.json", "w") as f:
            json.dump(schemas, f, indent=2, default=str)
        print("✅ Schema also saved to `schemas.json`")

    except Exception as e:
        print(f"❌ **Error:** {e}")


if __name__ == "__main__":
    main()
