from pathlib import Path
from typing import Any, List, Dict
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_community.document_loaders import DirectoryLoader


## load and read the data
def load_documents(data_dir: str) -> List[Any]:
    """ 
    Load and read the documents present in the directory, convert them into langchain documents 
    structure supported: pdf, csv, excel, json
    """
    ## data directory absolute path
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data Path: {data_path}")

    ## document list
    all_documents = []

    ## access all the pdf files 
    files = list(data_path.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(files)} PDF files: {[str(f) for f in files]}")

    for doc in files:
        try:
            ## loader object
            loader = PyPDFLoader(str(doc))
            # read the data
            documents_loaded = loader.load()
            print(f"[DEBUG] Loaded {len(documents_loaded)} PDF docs from {doc}")
            ## append into the all_documents list
            all_documents.extend(documents_loaded)
        except Exception as e:
            print(f"[DEBUG] Failed to load pdf {doc}: {e}")
    return all_documents

