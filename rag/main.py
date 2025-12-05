import os
import subprocess
import shutil
import logging
import img2pdf
import torch
import sys
import json
import time
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BATCH_SIZE = 8
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.join(SCRIPT_DIR, "../orchestrator/")
STORE_DIR = os.path.join(SCRIPT_DIR, "vector_store")
PROCESSED_DIR = os.path.join(STORE_DIR, "processed_files")
PROCESSED_DB = os.path.join(STORE_DIR, "processed_db.json")


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str, use_cuda: bool = True):
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, batch_size=BATCH_SIZE).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()


def check_dependencies():
    required = ["unoconv", "ocrmypdf"]
    missing = []
    for cmd in required:
        try:
            subprocess.run([cmd, "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(cmd)
    return missing


def read_pdf(file_path: str) -> List[Dict]:
    reader = PdfReader(file_path)
    return [
        {
            "content": page.extract_text() or "",
            "metadata": {"page": i + 1, "source": file_path},
        }
        for i, page in enumerate(reader.pages)
    ]


def read_txt(file_path: str) -> List[Dict]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return [{"content": f.read(), "metadata": {"source": file_path}}]
    except Exception as e:
        logging.error(f"Error reading {file_path}: {str(e)}")
        return []


def load_documents(folder_path: str) -> List[Document]:
    documents = []
    SUPPORTED_EXTS = {".pdf", ".txt", ".md"}

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue

        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue

        try:
            if ext == ".pdf":
                items = read_pdf(filepath)
            elif ext in (".txt", ".md"):
                items = read_txt(filepath)
            else:
                continue

            for item in items:
                if not item["content"].strip():
                    continue

                base_meta = {
                    "source": filename,
                    "file_type": ext[1:].upper(),
                    "full_path": filepath,
                    "timestamp": datetime.now().isoformat(),
                }
                base_meta.update(item["metadata"])

                documents.append(
                    Document(page_content=item["content"], metadata=base_meta)
                )
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")

    return documents


def load_processed_db():
    if os.path.exists(PROCESSED_DB):
        try:
            with open(PROCESSED_DB, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading processed file database: {str(e)}")
    return {}


def save_processed_db(db):
    os.makedirs(os.path.dirname(PROCESSED_DB), exist_ok=True)
    try:
        with open(PROCESSED_DB, "w") as f:
            json.dump(db, f)
    except Exception as e:
        logging.error(f"Error saving processed file database: {str(e)}")


def convert_and_ocr(input_dir: str, skip_ocr: bool = False):
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    OFFICE_EXTS = {
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".odt",
        ".ods",
        ".odp",
    }

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    processed_db = load_processed_db()

    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            src_path = os.path.join(root, filename)
            ext = Path(filename).suffix.lower()

            if ext not in IMAGE_EXTS | OFFICE_EXTS | {".pdf", ".txt", ".md"}:
                continue

            file_mtime = os.path.getmtime(src_path)
            if (
                src_path in processed_db
                and processed_db[src_path]["mtime"] == file_mtime
            ):
                continue

            files_to_process.append((src_path, filename, ext, file_mtime))

    if not files_to_process:
        return False

    for src_path, filename, ext, file_mtime in tqdm(
        files_to_process, desc="Processing files"
    ):
        base_name = Path(filename).stem
        dest_path = os.path.join(PROCESSED_DIR, f"{base_name}.pdf")
        dest_txt_path = os.path.join(PROCESSED_DIR, filename)

        try:
            if ext in {".txt", ".md"}:
                shutil.copy(src_path, dest_txt_path)
                processed_db[src_path] = {
                    "processed_path": dest_txt_path,
                    "mtime": file_mtime,
                    "processed_time": time.time(),
                }
                continue

            if ext in IMAGE_EXTS and not skip_ocr:
                with open(src_path, "rb") as img_file, open(
                    dest_path, "wb"
                ) as pdf_file:
                    pdf_file.write(img2pdf.convert(img_file))

            elif ext in OFFICE_EXTS and not skip_ocr:
                subprocess.run(
                    ["unoconv", "-f", "pdf", "-o", dest_path, src_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            elif ext == ".pdf":
                shutil.copy(src_path, dest_path)

            if not skip_ocr and os.path.exists(dest_path):
                subprocess.run(
                    ["ocrmypdf", dest_path, dest_path, "--force-ocr"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            if os.path.exists(dest_path):
                processed_db[src_path] = {
                    "processed_path": dest_path,
                    "mtime": file_mtime,
                    "processed_time": time.time(),
                }

        except subprocess.CalledProcessError as e:
            logging.error(f"Conversion failed for {filename}: {str(e)}")
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")

    save_processed_db(processed_db)

    return True


class RAGApplication:
    def __init__(
        self, model_name="mistral", temperature=0.5, use_sentence_transformer=False
    ):
        if use_sentence_transformer:
            self.embeddings = SentenceTransformerEmbeddings(
                model_name=EMBED_MODEL, use_cuda=torch.cuda.is_available()
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        self.llm = OllamaLLM(
            model=model_name,
            temperature=temperature,
            top_p=0.9,
        )
        self.vector_store = None

    def load_and_process_documents(self, input_dir: str, skip_ocr: bool = False):
        try:
            start_time = time.time()

            missing_deps = check_dependencies()
            if missing_deps and not skip_ocr:
                logging.warning(
                    f"Missing dependencies: {', '.join(missing_deps)}. OCR disabled."
                )
                skip_ocr = True

            new_files_processed = convert_and_ocr(input_dir, skip_ocr)

            if not new_files_processed and os.path.exists(
                os.path.join(STORE_DIR, "index.faiss")
            ):
                self.vector_store = FAISS.load_local(
                    STORE_DIR, self.embeddings, allow_dangerous_deserialization=True
                )
                logging.info("No new files to process. Loaded existing vector store.")
                return True

            documents = load_documents(PROCESSED_DIR)
            if not documents:
                logging.error("No valid documents processed")
                return False

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", "!", "?", ",", " "],
            )
            chunks = text_splitter.split_documents(documents)

            self.vector_store = FAISS.from_documents(
                chunks, self.embeddings, normalize_L2=True
            )

            os.makedirs(STORE_DIR, exist_ok=True)
            self.vector_store.save_local(STORE_DIR)

            logging.info(
                f"Processed {len(chunks)} chunks in {time.time()-start_time:.2f}s"
            )
            return True

        except Exception as e:
            logging.error(f"Document processing failed: {str(e)}")
            return False

    async def query_document(self, query: str) -> str:  # Make async
        start_time = time.time()
        try:
            if not self.vector_store:
                return "Please load documents first."

            # Retrieve relevant documents using the vector store
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)
            
            if not docs:
                return "No relevant information found in documents."
            
            # Combine document contents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create a prompt with the context and query
            prompt = f"""Based on the following context, answer the question. If you can't find the answer in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
            
            # Use the LLM to generate the answer
            response = self.llm.invoke(prompt)
            return response

        except Exception as e:
            logging.error(f"Query failed: {str(e)}")
            return f"Error: {str(e)}"

    @staticmethod
    def find_document_files(directory: str) -> List[str]:
        SUPPORTED_EXTS = {
            ".pdf",
            ".txt",
            ".md",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".odt",
            ".ods",
            ".odp",
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
        }

        documents = []
        try:
            for root, _, files in os.walk(Path(directory).resolve()):
                for file in files:
                    if Path(file).suffix.lower() in SUPPORTED_EXTS:
                        documents.append(str(Path(root) / file))
            return sorted(documents)
        except Exception as e:
            logging.error(f"File search error: {str(e)}")
            return []


async def main():
    logging.basicConfig(level=logging.INFO)

    skip_ocr = "--skip-ocr" in sys.argv

    rag = RAGApplication(use_sentence_transformer=True)

    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    print("Found documents:")
    for doc in RAGApplication.find_document_files(data_dir):
        print(f" - {doc}")

    if rag.load_and_process_documents(data_dir, skip_ocr=skip_ocr):
        print("\nDocuments processed successfully!")
    else:
        print("\nFailed to process documents")
        return

    while True:
        try:
            query = input("\nAsk a question (or 'quit'): ")
            if query.lower() in ("quit", "exit"):
                break

            start_time = time.time()
            response = rag.query_document(query)
            print(f"\nAnswer ({time.time()-start_time:.2f}s):\n{response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    import time
    import asyncio

    asyncio.run(main())
