import os
from dotenv import load_dotenv 
from google import genai
from google.genai import types
import json
from pathlib import Path
import pymupdf
from tqdm import tqdm


# Load environment variables from .env file
load_dotenv()

# set up gemini
client = genai.Client()
# model_ID = "gemini-3-pro-preview"
model_ID = "gemini-2.5-flash"

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "PaperTitle",
        "PublicationYear",
        "DOI",
        "Authors",
        "Abstract",
        "SummaryAbstract",
        "References"
    ],
    "properties": {
        "PaperTitle": {
            "type": "string",
            "minLength": 0
        },
        "PublicationYear": {
            "type": "string",
            "pattern": r"^[0-9]{4}$|^$"
        },
        "DOI": {
            "type": "string",
            "minLength": 0
        },
        "Authors": {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 1
            }
        },
        "Abstract": {
            "type": "string",
            "minLength": 0
        },
        "SummaryAbstract": {
            "type": "string",
            "minLength": 0
        },
        "References": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "PaperTitle",
                    "Authors",
                    "PublicationYear",
                    "DOI",
                    "Editors",
                    "Publisher",
                    "Publication",
                    "SemanticInformation"
                ],
                "properties": {
                    "PaperTitle": {
                        "type": "string",
                        "minLength": 0
                    },
                    "Authors": {
                        "type": "string",
                        "minLength": 0
                    },
                    "PublicationYear": {
                        "type": "string",
                        "pattern": r"^[0-9]{4}$|^$"
                    },
                    "DOI": {
                        "type": "string",
                        "minLength": 0
                    },
                    "Editors": {
                        "type": "string",
                        "minLength": 0
                    },
                    "Publisher": {
                        "type": "string",
                        "minLength": 0
                    },
                    "Publication": {
                        "type": "string",
                        "minLength": 0
                    },
                    "SemanticInformation": {
                        "type": "string",
                        "minLength": 0
                    }  
                }
            }
        }
    }
}

def clean_schema(schema: dict) -> dict:
    # Removes API-unrecognized keys from a Pydantic-generated schema.
    schema.pop('title', None)
    schema.pop('description', None)

    def strip_properties(d):
        if isinstance(d, dict):
            d.pop('additionalProperties', None)
            for key, value in d.items():
                strip_properties(value)
        elif isinstance(d, list):
            for item in d:
                strip_properties(item)

    strip_properties(schema)
    return schema


def read_prompt(prompt_path: str):
    #  Read the prompt for research paper parsing from the text file.
  with open(prompt_path, "r") as f:
    return f.read()

def extract_text_from_pdf(pdf_path: str):
    doc = pymupdf.open(pdf_path)
    num_pages = len(doc)
    pages = []
      # Process each page of the PDF
    for page_num in tqdm(range(num_pages), desc="Processing PDF pages"):
        page = doc[page_num]
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)

def clean_newlines(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ["PaperTitle", "Publisher", "Publication"] and isinstance(value, str):
                data[key] = value.replace("\n\n", "").replace("\u00ad", "")
            else:
                clean_newlines(value)
    elif isinstance(data, list):
        for item in data:
            clean_newlines(item)
    return data

def processing_pdf_paper(pdf_path: str, prompt_path: str, output_path: str = None):
    #Step 1: Extract text content from the PDF
    content = extract_text_from_pdf(pdf_path)

    #Read the prompt
    prompt_data = read_prompt(prompt_path)

    result = None

    try:
        result = client.models.generate_content(
            model=model_ID,
            contents=[content, prompt_data],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                response_schema= clean_schema(SCHEMA),
                
            )
        )
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return
    
    # --- CRITICAL: Check for None/Empty Response ---
    if not result or not result.text:
        print("\n--- API Failure ---")
        print("Error: Gemini returned a successful response object, but no generated text.")
        print("This usually means the prompt or the model configuration caused the generation to be blocked or stopped early.")
        return # Exit the function if there's no text to parse
    
    # --- 2. Parse the JSON Response ---
    try:
        response_data = json.loads(result.text)
        response_data = clean_newlines(response_data)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON response from Gemini: {result.text[:100]}...")
        return 
    

    # --- 3. Determine Output Path ---
    pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_filename = f"{pdf_base_name}_gemini.json"
    output = os.path.join(output_path, output_filename)
    os.makedirs(output_path, exist_ok=True)

    # --- 4. Save to JSON File ---
    try:
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=4)
        print(f"Successfully saved extracted data to: {output}")
        count_references_in_output(output)
    except IOError as e:
        print(f"Error saving file to {output}: {e}")



def count_references_in_output(output_file: str):
    # Count the number of reference objects in the output JSON file.
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        count = len(data["References"])
        print("Number of references objects:", count)



if __name__ == "__main__":
    pdf_path = "./papers/Graph_Embedding_for_Mapping_Interdisciplinary_Research_Network.pdf"
    prompt_path = "./prompts/information_extraction_prompt.txt"
    output_path = "./IE-output"
    processing_pdf_paper(pdf_path, prompt_path, output_path)