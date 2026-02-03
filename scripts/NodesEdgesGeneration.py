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

    return nodes, edges


def main():
    data = read_json_data(INPUT_FILE_NAME)
    nodes, edges = create_graph(data)

    pd.DataFrame(nodes).drop_duplicates(subset=["NodeID"]).to_csv(OUTPUT_NODES_FILE, index=False)
    pd.DataFrame(edges).to_csv(OUTPUT_EDGES_FILE, index=False)

    print(f"Saved nodes -> {OUTPUT_NODES_FILE}")
    print(f"Saved edges -> {OUTPUT_EDGES_FILE}")

if __name__ == "__main__":
    main()
