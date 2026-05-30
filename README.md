# Graph-Theoretic Research Project

A comprehensive thematic literature review system using Large Language Models (LLMs) to extract information from research papers, identify entities, and build knowledge graphs using graph-theoretic approaches.

## Project Overview

This project automates the process of:

1. Extracting information from research papers (PDFs)
2. Identifying entities and relationships using LLM APIs (Gemini, OpenAI)
3. Retrieving paper metadata from Semantic Scholar & CrossRef
4. Generating knowledge graphs from extracted entities and relationships
5. Applying network measures for analysis

## Prerequisites

- **Python**: Version 3.8 or higher
- **pip**: Python package manager (comes with Python)
- **Git**: For cloning the repository
- **API Keys** (obtain these before setup):
  - Google Gemini API key
  - OpenAI API key
  - Semantic Scholar API key (optional but recommended)

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Graph-Theoretic-Research-Project
```

### Step 2: Install Dependencies

Create a `requirements.txt` file with the following packages:

```bash
pip install -r requirements.txt
```

**Required packages:**

- `google-genai` - Google Gemini API client
- `openai` - OpenAI API client
- `python-dotenv` - Environment variable management
- `requests` - HTTP requests library
- `pymupdf` - PDF processing library
- `pymupdf4llm` - LLM-optimized PDF extraction
- `tqdm` - Progress bar utility
- `networkx` - Graph processing
- `aiohttp` - Async HTTP client

## Environment Setup

### Step 1: Create `.env` File

In the project root directory, create a `.env` file:

```bash
# For reference only, copy to .env and fill in your keys
cp .env.example .env
```

### Step 2: Add API Keys to `.env`

Edit the `.env` file in the project root and add your API keys:

```env
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Semantic Scholar API
SEMANTIC_API_KEY=your_semantic_scholar_api_key_here
```

### Step 3: Verify Setup

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('Setup complete' if all([os.getenv('GEMINI_API_KEY'), os.getenv('OPENAI_API_KEY')]) else 'Missing API keys')"
```

## Obtaining API Keys

### 1. Google Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikeys)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated API key
5. Paste it in the `.env` file as `GEMINI_API_KEY`

**Note**: Gemini offers a free tier with generous usage limits.

### 2. OpenAI API Key

1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Sign in to your OpenAI account (create one if needed)
3. Click **"Create new secret key"**
4. Copy the generated API key
5. Paste it in the `.env` file as `OPENAI_API_KEY`

**Note**: OpenAI requires billing information, but offers free trial credits.

### 3. Semantic Scholar API Key

1. Visit [Semantic Scholar API](https://www.semanticscholar.org/product/api)
2. Click **"Request an API Key"**
3. Fill out the forms
4. Request a Semantic Scholar API Key
5. Paste it in the `.env` file as `SEMANTIC_API_KEY`

**Note**: Semantic Scholar offers a free tier with standard usage limits (100 requests per 5 minutes).

## Usage

### Extract Information from Papers

```bash
python IE-Gemini.py
```

### Retrieve Paper Metadata

```bash
python scripts/FindDetails.py
```

### Generate Knowledge Graph

```bash
python scripts/NodesEdgesGeneration.py
```

### Visualize Graph (Jupyter Notebook)

```bash
jupyter notebook scripts/GenerateNetowrkX_Graph.ipynb
```

## Next Steps

1. Place your research papers in the `papers/` directory
2. Run the extraction scripts to generate the knowledge graph
3. Analyze results in the `IE-output/` and `JSON-output/` directories
4. Visualize the graph using the Jupyter notebooks

## Authors

Thenuja Sinthujan
