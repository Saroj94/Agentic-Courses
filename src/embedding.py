from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Any, List
import numpy as np
import os

## embedding class
class EmbeddingPipeline:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", chunk_size=1000, chunk_overlap=300):
        """ 
        Initialize the embedding model
        Args:
            model_name: It is a freely available embedding model.
        """
        CACHE_DIR = os.getenv("HF_HOME", "./model_cache")
        ENCODE_KWARGS ={"normalize_embeddings": True,"batch_size": 32}

        self.model = HuggingFaceEmbeddings(model_name=model_name, 
                                           cache_folder=CACHE_DIR,
                                           encode_kwargs=ENCODE_KWARGS
                                           )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Initialize embedding model: {model_name}")

    ## chunking the documents
    def chunk_document(self, documents: List[Any]) ->List[Any]:
        """ 
        Create chunks
        """
        if not documents:
            return []
        
        # Production Guardrails: Catch configuration errors before processing
        if self.chunk_size <= 0 or self.chunk_overlap < 0:
            raise ValueError("chunk_size must be positive and chunk_overlap cannot be negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        
        splitter = RecursiveCharacterTextSplitter(chunk_size = self.chunk_size,
                                              chunk_overlap = self.chunk_overlap,
                                              length_function = len,
                                              separators = ["\n##","\n###","\n\n","\n",","," "], ## explicitly use when handling markdown files/docs
                                              )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Splits {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    ## embedding documents into the vector
    def embedding_chunks(self, chunks:List[Any]) ->np.ndarray:
        ## from the above chunking method
        text_content = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Gnerating embeddings for {len(text_content)} chunks.")
        ## generate embedding vectors
        embedding_list = self.model.embed_documents(texts=text_content) ## to encode it requires list of chunk contents
        embeddings = np.asarray(embedding_list) ## more more optimize for production ready
        print(f"[INFO] Embedding shape: {embeddings.shape}")
        return embeddings

    



