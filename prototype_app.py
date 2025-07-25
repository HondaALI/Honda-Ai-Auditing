import streamlit as st
import os
import re
import fitz
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import torch
from io import BytesIO

# Ensure NLTK tokenizer is available
@st.cache_resource
def download_nltk_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
download_nltk_punkt()

# --- Configuration ---
MODEL_PATH = "HondaALI/AI-Auditing-Findings-Dectector"

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
        return False 

    return True

@st.cache_data
def extract_lines_from_pdf(uploaded_file):
    procedures_with_source = []
    try:
        uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    sentences = sent_tokenize(page_text)
                    for sentence in sentences:
                        cleaned_sentence = sentence.strip().replace('\xa0', ' ').replace('\n', ' ')
                        cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()

                        if _is_valid_procedure_line(cleaned_sentence):
                            procedures_with_source.append({
                                'text': cleaned_sentence,
                                'source_pdf': uploaded_file.name # Store the filename
                            })

                # Extract text from tables
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        cleaned_row = ' '.join(cell.strip().replace('\xa0', ' ') for cell in row if cell)
                        cleaned_row = re.sub(r'\s+', ' ', cleaned_row).strip()

                        # This check is on the 'text' key of the dictionaries
                        is_duplicate = False
                        for p in procedures_with_source:
                            if p['text'] == cleaned_row:
                                is_duplicate = True
                                break

                        if _is_valid_procedure_line(cleaned_row) and not is_duplicate:
                            procedures_with_source.append({
                                'text': cleaned_row,
                                'source_pdf': uploaded_file.name # Store the filename
                            })

    except fitz.FileDataError as e:
        st.error(f"Error opening PDF: {e}. Please ensure '{uploaded_file.name}' is a valid PDF file.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during PDF processing of '{uploaded_file.name}': {e}")
        return []

    return procedures_with_source

# Helper function to process manual input text
def process_manual_input(text_input, source_name="Manual Input"):
    lines = []
    sentences = sent_tokenize(text_input)
    for sentence in sentences:
        cleaned = sentence.strip().replace('\xa0', ' ')
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Bypass validation and include all sentences
        lines.append({
            "text": cleaned,
            "source_pdf": source_name
        })
    return lines

# --- Optimized Contradiction Finding Function ---
@st.cache_resource
def load_sentence_transformer_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Using device for model: {device}")
    
    try:
        model = SentenceTransformer(model_path, device=device)
        model.eval() # Set model to evaluation mode
        return model, device
    except Exception as e:
        st.error(f"Error loading model from '{model_path}': {e}")
        st.stop() # Stop the app if model fails to load

def find_all_contradictions_optimized(model, findings_data, procedures_data, threshold, device):
    all_contradictions = []

    # 1. Batch process all embeddings
    finding_texts = [f["text"] for f in findings_data]
    procedure_texts = [p["text"] for p in procedures_data]

    with st.spinner("Encoding finding texts..."):
        finding_embeddings = model.encode(finding_texts, convert_to_tensor=True, show_progress_bar=False).to(device)
    
    with st.spinner("Encoding procedure texts..."):
        procedure_embeddings = model.encode(procedure_texts, convert_to_tensor=True, show_progress_bar=False).to(device)

    # 2. Vectorized Cosine Similarity Calculation
    with st.spinner("Calculating all-to-all cosine similarities..."):
        cosine_similarity_matrix = util.cos_sim(finding_embeddings, procedure_embeddings)

    # 3. Iterate through the similarity matrix to find contradictions
    with st.spinner("Identifying contradictions based on threshold..."):
        for i in range(len(findings_data)): # Iterate over findings
            for j in range(len(procedures_data)): # Iterate over procedures
                cosine_sim = cosine_similarity_matrix[i][j].item() # .item() to get Python float
                contradiction_score = 1 - cosine_sim

                # Apply the original contradiction filtering logic (contradiction_score < threshold)
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

# --- Streamlit App Layout ---
st.set_page_config(page_title="Audit Contradiction Analyzer", layout="centered")

st.title("📄 Audit Contradiction Analyzer")
st.markdown("""
            This application helps identify potential contradictions between audit findings and established procedures.
            Upload your audit notes PDFs (or manually input them) and one or more procedure PDFs to start the analysis. \n\nContradiction score is
            a score calculated by the machine learning model that measures how much the sentences semantically contradict each other.
            The greater the score, the greater the contradiction. The score is not a percentage or ratio but on a nonlinear scale of 0 to 1 
            (due to how the model calculates the score).

            \n\nIn the left toolbar, you can adjust the number of maximum contradictions to show and the contradiction score threshold.
            The contradiction score threshold is the score needed for the algorithm to consider it a contradiction. The contradiction analysis
            may show less than the number of maximum contradictions selected if there are not enough contradictions identified by the model according 
            to the current threshold. 
            """)

# File Uploaders
st.header("Input Audit Notes and Procedures")

# Input method selection (PDF Upload or Manual Text)
input_method = st.radio(
    "Choose input method for Audit Notes:",
    ("Upload PDFs", "Manual Text Input"),
    key="input_method"
)

findings_uploaded_files = []
manual_findings_text = ""
MAX_FINDINGS_FILES = 5
MAX_PRCOEDURE_FILES = 15

if input_method == "Upload PDFs":
    findings_uploaded_files = st.file_uploader(
        "Upload Audit Notes PDFs (max of 5)",
        type="pdf",
        accept_multiple_files=True,
        key="findings_uploader"
    )
    if findings_uploaded_files and len(findings_uploaded_files) > MAX_FINDINGS_FILES:
        st.warning(f"⚠️ You can upload up to {MAX_FINDINGS_FILES} findings PDFs only. Please remove extra files.")

    
elif input_method == "Manual Text Input":
    manual_findings_text = st.text_area(
        "Paste or type your Audit Notes here:",
        height=300,
        key="manual_findings_text"
    )

# Procedure PDFs upload 
procedure_uploaded_files = st.file_uploader(
    "Upload Procedure PDFs (max of 15)",
    type="pdf",
    accept_multiple_files=True,
    key="procedures_uploader"
)
MAX_PROCEDURE_FILES = 15
if procedure_uploaded_files and len(procedure_uploaded_files) > MAX_PROCEDURE_FILES:
    st.warning(f"⚠️ You can upload up to {MAX_PROCEDURE_FILES} procedure PDFs only. Please remove extra files.")


# Slider for number of contradictions
st.sidebar.header("Analysis Settings")
num_contradictions_to_show = st.sidebar.slider(
    "Number of Contradictions to Show",
    min_value=1,
    max_value=20,
    value=5, 
    step=1
)

# Slider for contradiction threshold
contradiction_threshold_slider = st.sidebar.slider(
    "Contradiction Score Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50, 
    step=0.01,
    help="Lower values mean higher similarity (less contradiction). Higher values mean lower similarity (more contradiction)."
)

# Run Analysis Button
if st.button("Run Contradiction Analysis", type="primary"):
    
    if not findings_uploaded_files and not manual_findings_text:
        st.warning("Please upload at least one Audit Note PDF OR type in the Audit Notes manually to proceed.")
        st.stop()
    elif not procedure_uploaded_files:
        st.warning("Please upload at least one Procedure PDF to proceed.")
        st.stop()
    else:
        # Load model
        model, device = load_sentence_transformer_model(MODEL_PATH)

        findings_data = []

        # Process uploaded findings 
        if findings_uploaded_files:
            for file in findings_uploaded_files:
                with st.spinner(f"Extracting text from Findings PDF: {file.name}..."): 
                    findings_data.extend(extract_lines_from_pdf(file))
        
        # Process manual findings 
        if manual_findings_text.strip():
            with st.spinner("Processing manual input text..."):
                findings_data.extend(process_manual_input(manual_findings_text))

        # Process uploaded procedures files
        procedures_data = []
        for file in procedure_uploaded_files:
            with st.spinner(f"Extracting text from Procedure PDF: {file.name}..."):
                procedures_data.extend(extract_lines_from_pdf(file))

        if not findings_data:
            st.warning(f"No valid findings could be extracted from the provided Audit Notes. Cannot proceed with comparison.")
            st.stop()
        if not procedures_data:
            st.warning("No valid procedures could be extracted from the uploaded PDFs. Cannot proceed with comparison.")
            st.stop()

        st.info(f"Total findings extracted: {len(findings_data)}")
        st.info(f"Total procedures extracted: {len(procedures_data)}")

        # Perform contradiction analysis
        all_contradictions = find_all_contradictions_optimized(
            model,
            findings_data,
            procedures_data,
            contradiction_threshold_slider, # Use the dynamic threshold
            device
        )

        # Sort and display results, using the user-selected number
        top_n_overall_contradictions = sorted(all_contradictions, key=lambda x: x["contradiction_score"])[:num_contradictions_to_show]

        st.header("Analysis Results")
        if top_n_overall_contradictions:
            st.subheader(f"Top {len(top_n_overall_contradictions)} Potential Contradictions Found")
            for i, c in enumerate(top_n_overall_contradictions):
                with st.expander(f"Contradiction #{i+1} (Score: {c['cosine_similarity']:.2f})"):
                    st.write(f"**Finding:** {c['finding']}")
                    st.write(f"**From Findings PDF:** `{c['finding_pdf']}`")
                    st.write(f"**Procedure:** {c['procedure']}")
                    st.write(f"**From Procedure PDF:** `{c['procedure_pdf']}`")
                    st.write(f"**Contradiction Score:** `{c['cosine_similarity']:.2f}`")
                    st.markdown("<p style='color:red; font-weight:bold;'>Conclusion: This finding *is likely breaking* this procedure.</p>", unsafe_allow_html=True)
        else:
            st.success(f"✅ No contradictions found below the threshold of {contradiction_threshold_slider:.2f}.")