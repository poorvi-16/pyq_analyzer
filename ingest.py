import os
import time
import json
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"
PROGRESS_FILE = "ingest_progress.json"  # tracks where we left off

def load_all_pdfs():
    documents = []
    for subject in os.listdir(DATA_PATH):
        subject_path = os.path.join(DATA_PATH, subject)
        if not os.path.isdir(subject_path):
            continue
        print(f"Loading subject: {subject}")
        for pdf_file in os.listdir(subject_path):
            if not pdf_file.endswith(".pdf"):
                continue
            pdf_path = os.path.join(subject_path, pdf_file)
            print(f"  Reading: {pdf_file}")
            filename_lower = pdf_file.lower()
            if any(word in filename_lower for word in ["note", "notes", "book", "textbook", "module", "unit"]):
                doc_type = "notes"
            else:
                doc_type = "question_paper"
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["subject"] = subject
                    doc.metadata["source_file"] = pdf_file
                    doc.metadata["doc_type"] = doc_type
                documents.extend(docs)
                print(f"  Tagged as: {doc_type}")
            except Exception as e:
                print(f"  Skipping {pdf_file}: {e}")
    print(f"\nTotal pages loaded: {len(documents)}")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def get_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f).get("last_chunk", 0)
    return 0


def save_progress(index):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_chunk": index}, f)


def create_vector_store(chunks):
    print(f"\nCreating embeddings and saving to ChromaDB...")

    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

    BATCH_SIZE = 20
    start_index = get_progress()

    if start_index > 0:
        print(f"Resuming from chunk {start_index}...")

    if not os.path.exists(CHROMA_PATH) or start_index == 0:
        first_batch = chunks[:BATCH_SIZE]
        vectorstore = Chroma.from_documents(
            documents=first_batch,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        save_progress(BATCH_SIZE)
        start_index = BATCH_SIZE
    else:
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

    total = len(chunks)
    for i in range(start_index, total, BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        end = min(i+BATCH_SIZE, total)
        print(f"  Processing chunks {i+1} to {end} of {total}...")

        while True:
            try:
                vectorstore.add_documents(batch)
                save_progress(end)
                break
            except Exception as e:
                err = str(e)
                if "RESOURCE_EXHAUSTED" in err:
                    print("  Rate limit hit. Waiting 90 seconds...")
                    time.sleep(90)
                elif "peer closed connection" in err or "RemoteProtocol" in err:
                    print("  Network error. Waiting 30 seconds and retrying...")
                    time.sleep(30)
                else:
                    raise e

        time.sleep(15)

    # Clear progress file when done
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    print(f"\nVector store saved to '{CHROMA_PATH}' folder")
    return vectorstore


if __name__ == "__main__":
    print("=== Starting PDF Ingestion ===\n")
    documents = load_all_pdfs()
    chunks = split_documents(documents)
    create_vector_store(chunks)
    print("\n=== Ingestion Complete! ===")