import requests
import json
import os
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import wraps


# Configuration
OUTPUT_DIR = Path.cwd().parent / "IE-output"
JSON_OUTPUT_DIR = Path.cwd().parent / "JSON-output"
CROSSREF_BASE_URL = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
CROSSREF_BATCH_SIZE = 10  # Parallel requests for Crossref
SEMANTIC_SCHOLAR_BATCH_SIZE = 10  # Parallel requests for Semantic Scholar
REQUEST_TIMEOUT = 15  # seconds
RATE_LIMIT_DELAY = 1.0  # seconds between batches (Crossref: 50 req/min = 1.2 sec/req)
RETRY_DELAY = 5.0  # seconds to wait before retrying on 429
MAX_RETRIES = 3  # Maximum retry attempts for rate-limited requests
USER_AGENT = "Research-Metadata-Tool/1.0 (mailto:research@example.com)"

def safe_first_str(seq: Optional[List], default: str = "") -> str:
    if not seq or not isinstance(seq, list):
        return default
    if len(seq) == 0:
        return default
    first = seq[0]
    return first if isinstance(first, str) else default

def safe_year(msg: Dict) -> str:
    for key in ("published-print", "published-online"):
        block = msg.get(key, {})
        date_parts = block.get("date-parts", [])
        if date_parts and isinstance(date_parts, list) and len(date_parts[0]) > 0:
            return str(date_parts[0][0])
    return ""

def is_blank_doi(doi: Optional[str]) -> bool:
    if doi is None:
        return True
    if not isinstance(doi, str):
        return True
    norm = doi.strip().lower()
    if norm == "":
        return True
    if norm in {"null", "none", "n/a", "na", "-", "nan"}:
        return True
    return False


# ============================================================================
# ASYNC API FUNCTIONS FOR PARALLEL REQUESTS
# ============================================================================

async def fetch_json_async(session: aiohttp.ClientSession, url: str, params: Optional[Dict] = None, timeout: int = REQUEST_TIMEOUT, retry_count: int = 0) -> Optional[Dict]:
    headers = {"User-Agent": USER_AGENT}
    
    try:
        async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 429 and retry_count < MAX_RETRIES:
                # Rate limited - implement exponential backoff
                wait_time = RETRY_DELAY * (2 ** retry_count)
                print(f"  -> Rate limited (429). Waiting {wait_time:.1f}s before retry {retry_count + 1}/{MAX_RETRIES}...")
                await asyncio.sleep(wait_time)
                return await fetch_json_async(session, url, params, timeout, retry_count + 1)
            else:
                if response.status == 429:
                    print(f"  -> Rate limited (429) after {MAX_RETRIES} retries: {url}")
                else:
                    print(f"  -> Request failed with status {response.status}: {url}")
                return None
    except asyncio.TimeoutError:
        print(f"  -> Request timeout: {url}")
        return None
    except Exception as e:
        print(f"  -> Error fetching {url}: {e}")
        return None


async def get_paper_info_from_crossref_async(doi: str, session: aiohttp.ClientSession) -> Optional[Dict]:
    url = f"{CROSSREF_BASE_URL}/{doi}"
    
    data = await fetch_json_async(session, url)
    if not data:
        return None
    
    try:
        if data.get('status') == 'ok' and 'message' in data:
            msg = data['message']
            
            # Extract authors
            authors = []
            if 'author' in msg and isinstance(msg['author'], list):
                for author in msg['author']:
                    if not isinstance(author, dict):
                        continue
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)
            
            # Extract publication year safely
            pub_year = safe_year(msg)
            
            # Extract title and venue safely
            title = safe_first_str(msg.get('title', []))
            publication = safe_first_str(msg.get('container-title', []))
            
            # Extract publisher
            publisher = msg.get('publisher', '')
            
            return {
                "PaperTitle": title,
                "Authors": authors,
                "PublicationYear": pub_year,
                "DOI": doi,
                "Publisher": publisher,
                "Publication": publication
            }
        return None
    except Exception as e:
        print(f"  -> Error processing Crossref response for {doi}: {e}")
        return None


async def fetch_multiple_papers_from_crossref(dois: List[str]) -> List[Optional[Dict]]:
    results = []
    
    async with aiohttp.ClientSession() as session:
        # Process in batches to respect rate limits (Crossref: ~50 requests per minute)
        total_batches = (len(dois) + CROSSREF_BATCH_SIZE - 1) // CROSSREF_BATCH_SIZE
        
        for batch_num, i in enumerate(range(0, len(dois), CROSSREF_BATCH_SIZE), 1):
            batch = dois[i:i + CROSSREF_BATCH_SIZE]
            print(f"  -> Processing batch {batch_num}/{total_batches}: {len(batch)} DOIs")
            
            tasks = [get_paper_info_from_crossref_async(doi, session) for doi in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            results.extend([r for r in batch_results if r is not None])
            
            # Respect rate limits between batches (spacing out requests)
            if batch_num < total_batches:
                print(f"  -> Waiting {RATE_LIMIT_DELAY:.1f}s before next batch...")
                await asyncio.sleep(RATE_LIMIT_DELAY)
    
    return results


async def check_authors_list_async(doi: str, session: aiohttp.ClientSession) -> List[str]:
    url = f"{CROSSREF_BASE_URL}/{doi}"
    
    data = await fetch_json_async(session, url)
    if not data:
        return []
    
    try:
        if data.get('status') == 'ok' and 'message' in data:
            msg = data['message']
            
            # Extract authors
            authors = []
            if 'author' in msg and isinstance(msg['author'], list):
                for author in msg['author']:
                    if not isinstance(author, dict):
                        continue
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)
            return authors
        return []
    except Exception as e:
        print(f"  -> Error processing authors for {doi}: {e}")
        return []


async def get_references_from_semantic_scholar_async(doi: str, session: aiohttp.ClientSession) -> List[Dict]:
    url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/DOI:{doi}"
    params = {"fields": "references,references.title,references.authors,references.year,references.externalIds,references.venue,references.publicationVenue"}
    
    data = await fetch_json_async(session, url, params=params)
    if not data:
        return await get_references_from_crossref_async(doi, session)
    
    try:
        references_raw = data.get('references', [])
        print(f"    -> Retrieved from Semantic Scholar: {len(references_raw)} references")
        
        if references_raw:
            references = []
            for ref in references_raw:
                ref_info = {
                    "PaperTitle": ref.get('title', ''),
                    "Authors": [author.get('name', '') for author in ref.get('authors', [])] if ref.get('authors') else [],
                    "PublicationYear": str(ref.get('year')) if ref.get('year') else "",
                    "DOI": ref.get('externalIds', {}).get('DOI') if ref.get('externalIds') else "",
                    "Publication": ref.get('venue', ''),
                    "Publisher": ref.get('publicationVenue', {}).get('name') if ref.get('publicationVenue') else "",      
                }
                
                if not is_blank_doi(ref_info["DOI"]):
                    if not ref_info["Authors"] or not isinstance(ref_info["Authors"], list):   
                        ref_info["Authors"] = await check_authors_list_async(ref_info["DOI"], session)
                    ref_of_refs = await get_ref_of_refs_from_semantic_scholar_async(ref_info["DOI"], session)
                    if ref_of_refs:
                        ref_info["References"] = ref_of_refs
                    else:
                        ref_info["References"] = []

                    references.append(ref_info)
            
            print(f"    -> Returning {len(references)} references from Semantic Scholar")
            return references
        else:
            print("    -> References not available in Semantic Scholar, trying Crossref")
            return await get_references_from_crossref_async(doi, session)
        
    except Exception as e:
        print(f"    -> Error fetching from Semantic Scholar: {e}")
        return await get_references_from_crossref_async(doi, session)


# callback to Sematic Socholar for References of References
async def get_ref_of_refs_from_semantic_scholar_async(doi: str, session: aiohttp.ClientSession) -> List[Dict]:
    url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/DOI:{doi}"
    params = {"fields": "references,references.title,references.authors,references.year,references.externalIds,references.venue,references.publicationVenue"}
    
    data = await fetch_json_async(session, url, params=params)
    if not data:
        return await get_references_from_crossref_async(doi, session)
    
    try:
        references_raw = data.get('references', [])
        print(f"    -> Retrieved from Semantic Scholar: {len(references_raw)} references")
        
        if references_raw:
            references = []
            for ref in references_raw:
                ref_info = {
                    "PaperTitle": ref.get('title', ''),
                    "Authors": [author.get('name', '') for author in ref.get('authors', [])] if ref.get('authors') else [],
                    "PublicationYear": str(ref.get('year')) if ref.get('year') else "",
                    "DOI": ref.get('externalIds', {}).get('DOI') if ref.get('externalIds') else "",
                    "Publication": ref.get('venue', ''),
                    "Publisher": ref.get('publicationVenue', {}).get('name') if ref.get('publicationVenue') else "",      
                }
                
                if not is_blank_doi(ref_info["DOI"]):
                    if not ref_info["Authors"] or not isinstance(ref_info["Authors"], list):   
                        ref_info["Authors"] = await check_authors_list_async(ref_info["DOI"], session)
                    references.append(ref_info)
            
            print(f"    -> Returning {len(references)} references from Semantic Scholar")
            return references
        else:
            print("    -> References not available in Semantic Scholar, trying Crossref")
            return await get_references_from_crossref_async(doi, session)
        
    except Exception as e:
        print(f"    -> Error fetching from Semantic Scholar: {e}")
        return await get_references_from_crossref_async(doi, session)



async def get_references_from_crossref_async(doi: str, session: aiohttp.ClientSession) -> List[Dict]:
    url = f"{CROSSREF_BASE_URL}/{doi}"
    
    data = await fetch_json_async(session, url)
    if not data:
        return []
    
    try:
        if data.get('status') == 'ok' and 'message' in data:
            msg = data['message']
            
            # Get References
            references = msg.get('reference', [])
            print(f"    -> Found {len(references)} references in Crossref")
            
            if references:
                # Filter references with DOI
                ref_items = [r for r in msg.get('reference', []) if isinstance(r, dict) and not is_blank_doi(r.get('DOI'))]
                print(f"    -> {len(ref_items)} references have DOIs")
                
                # Fetch all paper info in parallel
                ref_of_refs = await fetch_multiple_papers_from_crossref([r.get('DOI') for r in ref_items])
                print(f"    -> Found {len(ref_of_refs)} reference details")
                return ref_of_refs
        
        return []
    except Exception as e:
        print(f"  -> Error processing Crossref response for {doi}: {e}")
        return []

# Callback to Crossref for References of References
async def get_ref_of_refs_from_crossref_async(doi: str, session: aiohttp.ClientSession) -> List[Dict]:
    url = f"{CROSSREF_BASE_URL}/{doi}"
    
    data = await fetch_json_async(session, url)
    if not data:
        return []
    
    try:
        if data.get('status') == 'ok' and 'message' in data:
            msg = data['message']
            
            # Get References
            references = msg.get('reference', [])
            print(f"    -> Found {len(references)} references in Crossref")
            
            if references:
                # Filter references with DOI
                ref_items = [r for r in msg.get('reference', []) if isinstance(r, dict) and not is_blank_doi(r.get('DOI'))]
                print(f"    -> {len(ref_items)} references have DOIs")
                
                # Fetch all paper info in parallel
                ref_of_refs = await fetch_multiple_papers_from_crossref([r.get('DOI') for r in ref_items])
                print(f"    -> Found {len(ref_of_refs)} reference details")
                return ref_of_refs
        
        return []
    except Exception as e:
        print(f"  -> Error processing Crossref response for {doi}: {e}")
        return []

# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

async def process_references_async(references: List[Dict]) -> List[Dict]:
    enriched_references = []
    
    async with aiohttp.ClientSession() as session:
        # Process references sequentially with delays to respect API rate limits
        for idx, ref in enumerate(references, 1):
            print(f"\n  Processing reference {idx}/{len(references)}: {ref.get('PaperTitle', 'Unknown')[:50]}...")
            
            doi = ref.get("DOI")
            if not doi or doi.strip() == "":
                print("   Skipping reference with blank DOI.")
                continue
            
            ref["References"] = await get_references_from_semantic_scholar_async(doi, session)
            enriched_references.append(ref)
            
            # Add delay between each reference processing to avoid overwhelming APIs
            if idx < len(references):
                await asyncio.sleep(RATE_LIMIT_DELAY)
    
    return enriched_references

def clean_json_file(file_path: str) -> Optional[Dict]:
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if the file has references
        if 'References' not in data or not isinstance(data['References'], list):
            print("No references found in this file. Skipping.")
            return
        
        print(f"Found {len(data['References'])} references to process")

        refs_with_doi = list(filter(lambda r: r.get("DOI"), data['References']))

        # Update the data with enriched references
        data['References'] = asyncio.run(process_references_async(refs_with_doi))

        print(f"\n Successfully cleaned references with DOIs. Total now: {len(data['References'])}")
        return data
        
    except Exception as e:
        print(f"\n Error processing file {file_path}: {e}")

def process_save_json_file(file_path: str) -> None:
    try:
        data = clean_json_file(file_path)
        if data is None:
            print(f"\n No data to save for file {file_path}")
            return
        # Save the updated JSON file
        # output_path = file_path.replace('.json', '_enriched.json')
        output_path = os.path.join(JSON_OUTPUT_DIR, file_path.split(os.sep)[-1].replace('.json', '_enriched.json'))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"\n Successfully processed and saved to: {output_path}")
        
    except Exception as e:
        print(f"\n Error processing file {file_path}: {e}")


# ============================================================================
# LEGACY SYNCHRONOUS FUNCTIONS (KEPT FOR FALLBACK)
# ============================================================================

def check_authors_list(authors: Optional[List], doi: str) -> List[str]:
    url = f"{CROSSREF_BASE_URL}/{doi}"
    
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'ok' and 'message' in data:
            msg = data['message']
            
            # Extract authors
            authors = []
            if 'author' in msg and isinstance(msg['author'], list):
                for author in msg['author']:
                    if not isinstance(author, dict):
                        continue
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)

            return authors
                    
        else:
            return []
        
    except Exception as e:
        print(f"  -> Error fetching DOI {doi}: {e}")
        return []

def main():

    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f" Output directory '{OUTPUT_DIR}' not found!")
        return
    
    # Get all JSON files from output directory
    json_files = list(Path(OUTPUT_DIR).glob("*.json"))
    
    if not json_files:
        print(f" No JSON files found in '{OUTPUT_DIR}'")
        return
    
    print(f"Found {len(json_files)} JSON file(s) in '{OUTPUT_DIR}'")
    
    # Process each file
    for json_file in json_files:
        # Skip already enriched files
        if '_enriched' in json_file.name:
            print(f"\nSkipping already enriched file: {json_file.name}")
            continue
        
        process_save_json_file(str(json_file))
        time.sleep(1)  # Delay between files to respect rate limits
    
    print(f"\n{'='*80}")
    print("✅ All files processed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()