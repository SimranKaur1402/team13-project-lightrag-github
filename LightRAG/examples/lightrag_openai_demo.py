'''import os
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import logger, set_verbose_debug
import fitz
from lightrag.rerank import jina_rerank
WORKING_DIR = "./dickens"


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status

    return rag


async def main():
    # Check if OPENAI_API_KEY environment variable exists
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set. Please set this variable before running the program."
        )
        print("You can set the environment variable by running:")
        print("  export OPENAI_API_KEY='your-openai-api-key'")
        return  # Exit the async function

    try:
        # Clear old data files
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
            "kv_store_full_entities.json",
            "kv_store_full_relations.json",
            "kv_store_entity_chunks.json"
        ]

        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleting old file:: {file_path}")

        # Initialize RAG instance
        rag = await initialize_rag()

        # Test embedding function
        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print("Test embedding function")
        print("========================")
        print(f"Test dict: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}\n\n")

        #with open("./book.txt", "r", encoding="utf-8") as f:
            #await rag.ainsert(f.read())
            
        # PyMuPDF

        def read_pdf(path):
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        # with open("./MachineLearning-Lecture04.pdf", "r", encoding="utf-8") as f:
        # await rag.ainsert(f.read())
        pdf_text = read_pdf("/home/skaur/LightRAG/MachineLearning-Lecture04.pdf")
        print("ML BELOW:")
        print(pdf_text)
        await rag.ainsert(pdf_text)
        print("Insert completed")

        # Perform naive search
        print("\n=====================")
        print("Query mode: naive")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="naive")
            )
        )

        # Perform local search
        #print("\n=====================")
        #print("Query mode: local")
        #print("=====================")
        #print(
         #   await rag.aquery(
          #      "What are the top themes in this story?", param=QueryParam(mode="local")
           # )
        #)
        res = await rag.aquery(
        "Summarize MachineLearning-Lecture04",
        param=QueryParam(
        mode="global",
        chunk_top_k=50,
        ),
        )

        print(res)

        # Perform global search
        print("\n=====================")
        print("Query mode: global")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="global"),
            )
        )

        # Perform hybrid search
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")'''
    
    
    
import os
import asyncio
import logging
import logging.config
import fitz # PyMuPDF
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import logger, set_verbose_debug

# CHANGE 1: Use a dedicated directory for ML to avoid mixing with Dickens data
WORKING_DIR = "./ml_lecture_data"

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
            "file": {
                "formatter": "detailed",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "lightrag_ml.log",
                "maxBytes": 10485760,
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "lightrag": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
        },
    })
    logger.setLevel(logging.INFO)

def read_pdf(path):
    """Extracts text from PDF and performs basic cleaning"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF file not found at: {path}")
    
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    
    # Simple cleaning to remove empty lines and extra spaces
    cleaned_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return cleaned_text

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

    # Create working directory if it doesn't exist
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    rag = None
    try:
        # Initialize RAG
        rag = await initialize_rag()

        # CHANGE 2: Extract text from your specific ML PDF path
        pdf_path = "/Users/tarinijain/LightRAG/MachineLearning-Lecture04.pdf"
        print(f"Reading PDF: {pdf_path}...")
        pdf_content = read_pdf(pdf_path)
        
        print(f"Extracted {len(pdf_content)} characters. Starting indexing...")

        # CHANGE 3: Insert the PDF content into the Knowledge Graph
        # This builds the entities and relationships specifically for the ML PDF
        await rag.ainsert(pdf_content)
        print("Indexing completed successfully.")

        # CHANGE 4: Queries updated for Machine Learning context
        print("\n=====================")
        print("QUERY: Global Summary")
        print("=====================")
        # Mode 'global' is best for high-level summaries of the entire PDF
        res_summary = await rag.aquery(
            "What are the main machine learning concepts and algorithms discussed in this lecture?",
            param=QueryParam(mode="global")
        )
        print(res_summary)

        print("\n=====================")
        print("QUERY: Specific Hybrid Search")
        print("=====================")
        # Mode 'hybrid' looks for specific details and broad context
        res_details = await rag.aquery(
            "Explain the mathematical formulas or specific examples used in this lecture.",
            param=QueryParam(mode="hybrid")
        )
        print(res_details)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
    print("\nDone!")

