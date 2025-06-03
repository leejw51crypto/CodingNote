# BigQuery + OpenAI SQL Generator

This tool combines BigQuery with OpenAI to generate and execute SQL queries using natural language prompts. Ask questions about blockchain data in plain English, and get SQL results automatically.

## Setup

### 1. Environment Configuration
Create your environment file from the example:
```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
- `PROJECT_ID` - Your Google Cloud project ID
- `OPENAI_API_KEY` - Your OpenAI API key

### 2. Google Cloud Authentication
Authenticate with Google Cloud to access BigQuery:
```bash
gcloud auth application-default login
```

### 3. BigQuery Dataset Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to BigQuery
3. Click on "Data sharing" and import the "cronos zkevm" dataset
4. This will create the `cronos_zkevm_mainnet` dataset in your project

### 4. Download Database Schema
Generate the schema file needed for AI query generation:
```bash
python getschema.py
```
This creates `schemas.json` with table structures and column information.

## Usage

Run the main application:
```bash
python b.py
```

The program will prompt you to enter questions in natural language. Examples:
- "get latest block height"
- "get tx of hash 0x..."
- "show me the last 10 transactions"
- "what's the total gas used today?"

## Available Data

The dataset includes these tables:
- **blocks** - Block information (1.6M+ rows)
- **transactions** - Transaction details (1.9M+ rows) 
- **logs** - Event logs (15M+ rows)
- **batches** - L1 batch information (441 rows)

## How It Works

1. **Schema Loading**: Reads table structures from `schemas.json`
2. **AI Query Generation**: Uses OpenAI GPT-4 to convert your question into SQL
3. **Query Execution**: Runs the generated SQL on BigQuery
4. **Results Display**: Shows formatted results in your terminal

The AI understands the database schema and generates optimized BigQuery SQL queries automatically.