import os
import asyncio
import logging
import logging.config
import fitz # PyMuPDF
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import logger, set_verbose_debug

# CHANGED: Use a new directory to ensure it doesn't load old Dickens data
WORKING_DIR = "./ml_lecture_04_index"
#PDF_PATH = "/home/skaur/LightRAG/MachineLearning-Lecture04.pdf"
DOC_PATH = "/home/skaur/LightRAG/large_independent_energy_report_4pages.docx."

def configure_logging():
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(levelname)s: %(message)s"},
            "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {"formatter": "default", "class": "logging.StreamHandler"},
        },
        "loggers": {
            "lightrag": {"handlers": ["console"], "level": "INFO", "propagate": False},
        },
    })

def read_pdf(path):
    """Extracts text from PDF using PyMuPDF"""
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    return rag

async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.")
        return

    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    rag = None
    try:
        # Initialize
        rag = await initialize_rag()

        # 1. Extract and Insert PDF
        print(f"Reading PDF from: {PDF_PATH}")
        pdf_text = read_pdf(PDF_PATH)
        
        if not pdf_text.strip():
            print("Error: No text found in PDF. Check if it is a scanned image.")
            return

        print("Inserting PDF content into Graph... (This creates the knowledge base)")
        await rag.ainsert(pdf_text)
        print("Insert completed successfully.")

        # 2. Perform all search modes
        modes = ["naive", "local", "global", "hybrid"]
        
        for mode in modes:
            print(f"\n{'='*20}")
            print(f"QUERY MODE: {mode.upper()}")
            print(f"{'='*20}")
            
            # We change the query to be relevant to ML instead of "the story"
            #query = "What are the core machine learning concepts and equations discussed in this lecture?"
            query = "Summarize the document and give key highlights."
            response = await rag.aquery(
                query, 
                param=QueryParam(mode=mode)
            )
            print(response)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
    print("\nProcess Finished!")

