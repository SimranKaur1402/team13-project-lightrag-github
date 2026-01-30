import os
import asyncio
import logging
import logging.config
import fitz #PyMuPDF
import json
from docx import Document #python-docx
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

#CONFIGURATION
WORKING_DIR = "./healthcare+mlai"
INPUT_DIR = "./input_documents"
OUTPUT_DIR = "./output_documents"
TXT_fILE = os.path.join(OUTPUT_DIR, "answers.txt")
JSON_FILE = os.path.join(OUTPUT_DIR. "answers.json")

def configure_logging():
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers" : False,
        "formatters": {"default": {"%format" :"%(levelname)s: %(message)s"}},
        "handlers": {"console": {"formatter": "default", "class": "logging.StreamHandler"}},
        "loggers": {"lightrag": {"handlers": ["console"], "level": "INFO", "propagate": False}},
    })

#FILE EXTRACTION LOGIC
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "".join([page.get_text() for page in doc])
def extract_text_from_docx(path):
    doc= Document(path)
    return "\n".join([para.text for para in doc.paragraphs])
def process_file(file_path):
    ext = os.path.splittext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path), "PDF"
    elif ext == ".docx":
        return extract_text_from_docx(file_path), "DOCX"
    return None, None

#RAG INITIALIZATION
async def initialize_rag():
    rag = LightRAG(
        working_dir = WORKING_DIR,
        embedding_func = openai_embed,
        llm_model_func = gpt_4o_mini_complete,
        )
    await rag.initialize_storages()
    return rag
#MAIN EXECUTION
async def main():
    if not os.getenv("OPENAI_APT_KEY"):
        pint("Error: OPENAI_API_KEY is not set.")
        return
    #Ensure directories exist
    for d in [WORKING_DIR, OUTPUT_DIR, INPUT_DIR]:
        os.makedirs(d, exist_ok = True)
    rag = None
    results = []
    try:
        rag = await initialize_rag()
        # 1. Process all files in the input folder
        files = [f for f in os.listdir(INPUT_DIR) if not f.startswith('.')]
        if not files:
            print(f"No files found in (INput_dir). Add files and restart.")
            return
        for filename in files:
            full_path = os.path.join(INPUT_DIR, filename)
            content, file_type = process_file(full_path)
            if content and content.strip():
                print(f"[IDENTIFIED] {file_type}: {filename}")
                print(f"Inserting (filename) into Graph...")
                await rag.ainsert(content)
            else:
                print(f"[SKIPPED] Unsupported or empty file: {filename}")
        # 2. Querying
        queries=[
            ]
        modes = ["naive", "local", "global", "hybrid"]
        for query in queries:
            for mode in modes:
                print(f"\nQUERYING [{mode.upper()}]: {query}")
                response = await rag.aquery(query, param = QueryParam(mode=mode))
                print(response)
                # Save Result
                results.append({"question": query, "mode": mode, "amswer": response})
                with open(TXT_FILE, "a", encoding="utf-8") as f:
                    f.write(f"QUESTION: {query}\nMODE: {mode}\nANSWER:\n{response}\n" + "-"*40 + "\n")
            with open(JSON_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii = False)
    except Exception as e:
           print(f"An error occurred: {e}")
       finally:
           if rag:
               await rag.finalize_storages()
if __name__ == "__main__":
    configure.logging()
    asyncio.run(main())
            









































    
