import os
import asyncio
import nest_asyncio
import numpy as np
import shutil
import fitz  # PyMuPDF

from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs

nest_asyncio.apply()

# --- CONFIGURATION ---
WORKING_DIR = "./rag_storage"
PDF_FILE = "./MLBOOK.pdf" 

# Validate API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. "
        "Please set it with: export GEMINI_API_KEY='your-api-key'"
    )

# --------------------------------------------------
# LLM function
# --------------------------------------------------
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.0-flash",
        **kwargs,
    )

# --------------------------------------------------
# Embedding function
# --------------------------------------------------
@wrap_embedding_func_with_attrs(
    embedding_dim=3072,
    max_token_size=2048,
    model_name="models/gemini-embedding-001",
)
async def embedding_func(texts: list[str], **kwargs) -> np.ndarray:
    return await gemini_embed.func(
        texts,
        api_key=GEMINI_API_KEY,
        model="models/gemini-embedding-001"
    )

# --------------------------------------------------
# Helper: Read PDF
# --------------------------------------------------
def read_pdf_text(pdf_path):
    """Extracts all text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return None

# --------------------------------------------------
# Initialize RAG
# --------------------------------------------------
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        llm_model_name="gemini-2.0-flash",
    )
    await rag.initialize_storages()
    return rag

# --------------------------------------------------
# Main Execution Logic
# --------------------------------------------------
def main():
    # 1. Create directory if it doesn't exist (Do NOT delete existing ones)
    if not os.path.exists(WORKING_DIR):
        print(f"📁 Creating new storage directory at {WORKING_DIR}...")
        os.makedirs(WORKING_DIR)

    # 2. Validate PDF file exists
    if not os.path.exists(PDF_FILE):
        raise FileNotFoundError(f"'{PDF_FILE}' not found. Please place your PDF in the folder.")

    # 3. Initialize LightRAG
    rag = asyncio.run(initialize_rag())

    # 4. Persistence Check: Only index if the GraphML file is missing
    # This file is created once the graph is built.
    index_indicator = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
    
    if os.path.exists(index_indicator):
        print("✅ Found existing index. Skipping PDF extraction and insertion.")
    else:
        print(f"📖 No index found. Extracting text from {PDF_FILE}...")
        pdf_content = read_pdf_text(PDF_FILE)

        if not pdf_content:
            print("❌ Failed to extract text from PDF. Exiting.")
            return

        print(f"✅ Extracted {len(pdf_content)} characters.")
        print("⏳ Building Graph and Vector Index (this may take a few minutes)...")
        
        # This will process the text and save it to WORKING_DIR
        rag.insert(pdf_content)
        print("✅ Indexing complete and saved to disk!")

    # 5. Complex Indirect Query (Zero Keywords)
    query = (
        "Locate the discussion regarding the expressive power of models. Identify the specific case where a single-layered arrangement of 'linear thresholds' is logically incapable of representing a parity-check function. How does the book suggest we 'expand the input space' to fix this without adding more layers?"
            )    
    # Testing different modes to see how 'Naive' fails vs others
    modes = ["naive", "local", "global", "hybrid"]
    
    for mode in modes:
        print(f"\n[{mode.upper()} SEARCH]:")
        response = rag.query(query, param=QueryParam(mode=mode))
        print(response)

if __name__ == "__main__":
    main()
