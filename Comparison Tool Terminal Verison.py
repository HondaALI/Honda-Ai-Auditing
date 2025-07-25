import os
import re
import fitz
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import torch # Import torch for device management

# Ensure NLTK tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Configuration ---
MODEL_PATH = "Audit nlp machine learning model"
CONTRADICTION_THRESHOLD = 0.50
TOP_N_CONTRADICTIONS = 5

# --- Helper Functions ---
def _is_valid_procedure_line(text_line):
    if not text_line:
        return False
    if len(text_line.split()) <= 5: # Skip short lines
        return False
    if re.match(r'^[A-Z]{2,}(?:\s[A-Z]{2,})*$', text_line): # Skip all-uppercase abbreviations
        return False
    if any(keyword in text_line.lower() for keyword in ["manager", "team", "department", "coordinator", "date", "page", "table of contents"]):
        return False # Skip role titles, dates, page numbers
    if re.match(r'^(page\s+\d+|[A-Z0-9]{3,}-\d{2,})$', text_line, re.IGNORECASE):
        return False # Skip page numbers or document IDs

    return True

def extract_lines_from_pdf(pdf_path):
    # This function now returns a list of dictionaries, each containing 'text' and 'source_pdf'
    procedures_with_source = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text from general page content
                page_text = page.extract_text()
                if page_text:
                    sentences = sent_tokenize(page_text)
                    for sentence in sentences:
                        cleaned_sentence = sentence.strip().replace('\xa0', ' ').replace('\n', ' ')
                        cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()

                        if _is_valid_procedure_line(cleaned_sentence):
                            procedures_with_source.append({
                                'text': cleaned_sentence,
                                'source_pdf': os.path.basename(pdf_path) # Store the filename
                            })

                # Extract text from tables
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        cleaned_row = ' '.join(cell.strip().replace('\xa0', ' ') for cell in row if cell)
                        cleaned_row = re.sub(r'\s+', ' ', cleaned_row).strip()

                        # Check if the cleaned_row (text) is already in our list to avoid duplicates
                        # This check is on the 'text' key of the dictionaries
                        is_duplicate = False
                        for p in procedures_with_source:
                            if p['text'] == cleaned_row:
                                is_duplicate = True
                                break

                        if _is_valid_procedure_line(cleaned_row) and not is_duplicate:
                            procedures_with_source.append({
                                'text': cleaned_row,
                                'source_pdf': os.path.basename(pdf_path) # Store the filename
                            })

    except fitz.FileDataError as e:
        print(f"Error opening PDF: {e}. Please ensure '{pdf_path}' is a valid PDF file.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during PDF processing: {e}")
        return []

    return procedures_with_source

def extract_procedures_from_pdfs(pdf_paths):
    all_procedures = []
    for path in pdf_paths:
        if os.path.exists(path) and path.lower().endswith(".pdf"):
            print(f"Extracting procedures from: {path}")
            procedures = extract_lines_from_pdf(path)
            all_procedures.extend(procedures)
        else:
            print(f"Invalid procedure PDF: {path}")
    return all_procedures

def extract_findings_from_pdfs(pdf_paths):
    all_findings = []
    for path in pdf_paths:
        if os.path.exists(path) and path.lower().endswith(".pdf"):
            print(f"Extracting findings from: {path}")
            findings = extract_lines_from_pdf(path) 
            all_findings.extend(findings)
        else:
            print(f"Invalid findings PDF: {path}")
    return all_findings

def find_all_contradictions_optimized(model, findings_data, procedures_data, threshold, device):
    all_contradictions = []

    # 1. Batch process all embeddings
    finding_texts = [f["text"] for f in findings_data]
    procedure_texts = [p["text"] for p in procedures_data]

    print("Encoding all finding texts...")
    finding_embeddings = model.encode(finding_texts, convert_to_tensor=True, show_progress_bar=True).to(device)
    print("Encoding all procedure texts...")
    procedure_embeddings = model.encode(procedure_texts, convert_to_tensor=True, show_progress_bar=True).to(device)

    # 2. Vectorized Cosine Similarity Calculation
    print("Calculating all-to-all cosine similarities...")
    cosine_similarity_matrix = util.cos_sim(finding_embeddings, procedure_embeddings)

    # 3. Iterate through the similarity matrix to find contradictions
    print("Identifying contradictions based on threshold...")
    for i in range(len(findings_data)): 
        for j in range(len(procedures_data)): 
            cosine_sim = cosine_similarity_matrix[i][j].item() 
            contradiction_score = 1 - cosine_sim

            if contradiction_score < threshold:
                all_contradictions.append({
                    "finding": findings_data[i]["text"],
                    "finding_pdf": findings_data[i]["source_pdf"],
                    "procedure": procedures_data[j]["text"],
                    "procedure_pdf": procedures_data[j]["source_pdf"],
                    "cosine_similarity": cosine_sim,
                    "contradiction_score": contradiction_score
                })
    return all_contradictions

# --- Main Execution ---
def main():
    procedure_pdfs = ["HNA-QMS-038_1.pdf", "QAP-AF00005_17.pdf"]
    findings_pdfs = ["Findings1.pdf"]

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at '{MODEL_PATH}'. Please ensure the path is correct.")
        return

    # Determine device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load the model directly to the determined device
        model = SentenceTransformer(MODEL_PATH, device=device)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    procedures = extract_procedures_from_pdfs(procedure_pdfs)
    findings = extract_findings_from_pdfs(findings_pdfs)

    if not procedures or not findings:
        print("No procedures or findings to compare. Exiting.")
        return

    print(f"\nTotal findings extracted: {len(findings)}")
    print(f"Total procedures extracted: {len(procedures)}")

    # Call the optimized function
    all_contradictions = find_all_contradictions_optimized(
        model,
        findings,
        procedures,
        CONTRADICTION_THRESHOLD,
        device # Pass the device to the function
    )

    # Sort the collected contradictions based on the original logic
    top_n_overall_contradictions = sorted(all_contradictions, key=lambda x: x["contradiction_score"])[:TOP_N_CONTRADICTIONS]

    print(f"\n--- Top {len(top_n_overall_contradictions)} Contradictions ---")
    if top_n_overall_contradictions:
        for i, c in enumerate(top_n_overall_contradictions):
            print(f"\nContradiction #{i+1}")
            print(f"Finding: {c['finding']}")
            print(f"From Findings PDF: {c['finding_pdf']}")
            print(f"Procedure: {c['procedure']}")
            print(f"From Procedure PDF: {c['procedure_pdf']}")
            print(f"Cosine Similarity: {c['cosine_similarity']:.2f}")
            print(f"Contradiction Score: {c['contradiction_score']:.2f}")
    else:
        print(f"\nNo contradictions found below the threshold of {CONTRADICTION_THRESHOLD:.2f}.")

if __name__ == "__main__":
    main()