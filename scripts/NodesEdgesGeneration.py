import json
import pandas as pd
from pathlib import Path

DIRECTORY = Path.cwd().parent 
INPUT_FILE_NAME = DIRECTORY / "JSON-output" / "Graph_Embedding_for_Mapping_Interdisciplinary_Research_Network_gemini_enriched.json"
OUTPUT_NODES_FILE = DIRECTORY / "csv" / "knowledge_graph_nodes.csv"
OUTPUT_EDGES_FILE = DIRECTORY / "csv" / "knowledge_graph_edges.csv"


def read_json_data(file_name: str):
    # Load a plain JSON file and return its contents.
    with open(file_name, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_json_data_by_doi(data: dict) -> dict:
    """
    Process JSON data to consolidate information by DOI.
    
    For each DOI that appears multiple times in the data:
    - If any occurrence has an abstract, use it for all occurrences
    - If any occurrence has a paper title, use it for all occurrences
    
    Args:
        data: The JSON data structure containing papers and references
    
    Returns:
        Cleaned JSON data with consolidated information by DOI
    """
    # Step 1: Collect all papers from all levels and build a DOI index
    doi_index = {}
    
    def collect_papers(paper, level=0):
        """Recursively collect all papers and store best available data for each DOI"""
        doi = (paper.get("DOI") or "").strip()
        if not doi:
            return
        
        # Initialize entry if not exists
        if doi not in doi_index:
            doi_index[doi] = {
                "PaperTitle": "",
                "Abstract": "",
                "Authors": "",
                "PublicationYear": "",
            }
        
        # Update with non-empty values (prefer longer abstracts and titles)
        current_title = (paper.get("PaperTitle") or "").strip()
        current_abstract = (paper.get("Abstract") or "").strip()
        current_authors = paper.get("Authors", "")
        current_year = paper.get("PublicationYear", "")
        
        # Use the longest non-empty title
        if current_title and len(current_title) > len(doi_index[doi]["PaperTitle"]):
            doi_index[doi]["PaperTitle"] = current_title
        
        # Use the longest non-empty abstract
        if current_abstract and len(current_abstract) > len(doi_index[doi]["Abstract"]):
            doi_index[doi]["Abstract"] = current_abstract
        
        # Use non-empty authors
        if current_authors and not doi_index[doi]["Authors"]:
            doi_index[doi]["Authors"] = current_authors
        
        # Use non-empty publication year
        if current_year and not doi_index[doi]["PublicationYear"]:
            doi_index[doi]["PublicationYear"] = current_year
        
        # Recursively process references
        for ref in paper.get("References", []):
            collect_papers(ref, level + 1)
    
    # Step 2: Build the DOI index from the entire tree
    collect_papers(data)
    
    # Step 3: Apply the consolidated data back to all papers
    def apply_consolidated_data(paper):
        """Recursively apply consolidated data to all papers"""
        doi = (paper.get("DOI") or "").strip()
        if doi and doi in doi_index:
            # Update with consolidated data
            if doi_index[doi]["PaperTitle"]:
                paper["PaperTitle"] = doi_index[doi]["PaperTitle"]
            if doi_index[doi]["Abstract"]:
                paper["Abstract"] = doi_index[doi]["Abstract"]
            if doi_index[doi]["Authors"]:
                paper["Authors"] = doi_index[doi]["Authors"]
            if doi_index[doi]["PublicationYear"]:
                paper["PublicationYear"] = doi_index[doi]["PublicationYear"]
        
        # Recursively process references
        if "References" in paper:
            for ref in paper["References"]:
                apply_consolidated_data(ref)
        
        return paper
    
    # Step 4: Return cleaned data
    return apply_consolidated_data(data)


def _valid_node(title: str, doi: str) -> bool:
    # Return True when both title and doi are non-empty strings
    return bool(title and title.strip()) and bool(doi and doi.strip())


def create_graph(data: dict):
    # Create nodes and edges for seed, references, and references of references.
    nodes = []
    edges = []

    seed_title = data.get("PaperTitle", "").strip()
    seed_doi = data.get("DOI", "").strip()
    seed_year = data.get("PublicationYear", "")

    if not _valid_node(seed_title, seed_doi):
        raise ValueError("Seed paper must have both PaperTitle and DOI.")

    # Seed node
    nodes.append({
        "NodeID": seed_doi,
        "PaperTitle": seed_title,
        "Type": "Seed",
        "Authors": ", ".join(data.get("Authors", [])),
        "PublicationYear": seed_year,
        "DOI": seed_doi,
        "Abstract": data.get("Abstract", ""),
    })

    # Level 1 references
    for ref in data.get("References", []):
        ref_title = (ref.get("PaperTitle") or "").strip()
        ref_doi = (ref.get("DOI") or "").strip()
        ref_year = ref.get("PublicationYear", "")

        if not _valid_node(ref_title, ref_doi):
            continue

        nodes.append({
            "NodeID": ref_doi,
            "PaperTitle": ref_title,
            "Type": "Reference",
            "Authors": ref.get("Authors", ""),
            "PublicationYear": ref_year,
            "DOI": ref_doi,
            "Abstract": ref.get("Abstract", ""),
        })

        edges.append({
            "Source": seed_doi,
            "Target": ref_doi,
            "Relationship": "CITES",
            "PublicationYear": ref_year,
        })

        # Level 2: references of references
        for r2 in ref.get("References", []):
            r2_title = (r2.get("PaperTitle") or "").strip()
            r2_doi = (r2.get("DOI") or "").strip()
            r2_year = r2.get("PublicationYear", "")

            if not _valid_node(r2_title, r2_doi):
                continue

            nodes.append({
                "NodeID": r2_doi,
                "PaperTitle": r2_title,
                "Type": "ReferenceOfReference",
                "Authors": r2.get("Authors", ""),
                "PublicationYear": r2_year,
                "DOI": r2_doi,
                "Abstract": r2.get("Abstract", ""),
            })

            edges.append({
                "Source": ref_doi,
                "Target": r2_doi,
                "Relationship": "CITES",
                "PublicationYear": r2_year,
            })

            # Level 3: references of references of references
            for r3 in r2.get("References", []):
                r3_title = (r3.get("PaperTitle") or "").strip()
                r3_doi = (r3.get("DOI") or "").strip()
                r3_year = r3.get("PublicationYear", "")
                if not _valid_node(r3_title, r3_doi):
                    continue

                nodes.append({
                    "NodeID": r3_doi,
                    "PaperTitle": r3_title,
                    "Type": "ReferenceOfReferenceOfReference",
                    "Authors": r3.get("Authors", ""),
                    "PublicationYear": r3_year,
                    "DOI": r3_doi,
                    "Abstract": r3.get("Abstract", ""),
                })

                edges.append({
                    "Source": r2_doi,
                    "Target": r3_doi,
                    "Relationship": "CITES",
                    "PublicationYear": r3_year,
                })

    return nodes, edges


def main():
    # Load raw data
    data = read_json_data(INPUT_FILE_NAME)
    
    # Clean and consolidate data by DOI
    print("Cleaning and consolidating data by DOI...")
    cleaned_data = clean_json_data_by_doi(data)
    
    # Create graph from cleaned data
    nodes, edges = create_graph(cleaned_data)

    pd.DataFrame(nodes).drop_duplicates(subset=["NodeID"]).to_csv(OUTPUT_NODES_FILE, index=False)
    pd.DataFrame(edges).to_csv(OUTPUT_EDGES_FILE, index=False)

    print(f"Saved nodes -> {OUTPUT_NODES_FILE}")
    print(f"Saved edges -> {OUTPUT_EDGES_FILE}")

if __name__ == "__main__":
    main()
