import os
import faiss
import numpy as np
import pickle
from typing import Any, List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persistent_dir: str = "faiss_store", 
                 emb_model: str ="sentence-transformers/all-MiniLM-L6-v2", 
                 chunk_size=1000, 
                 chunk_overlap=300):
        ENCODE_KWARGS ={"normalize_embeddings": True,"batch_size": 32}
    
        self.persistent_dir = persistent_dir
        self.index = None
        self.metadata = []
        self.embedding_model = emb_model
        self.model = HuggingFaceEmbeddings(model_name=emb_model, 
                                           encode_kwargs=ENCODE_KWARGS
                                           )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        os.makedirs(self.persistent_dir, exist_ok=True)
        print(f"[INFO] Loaded embedding model {emb_model}")

    def build_from_documents(self, documents: List[Any]):
        """Build vector"""
        print(f"[INFO] Building vector from {len(documents)} raw documents.")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, 
                                     chunk_size=self.chunk_size, 
                                     chunk_overlap=self.chunk_overlap)
        emb_chunks = emb_pipe.chunk_document(documents=documents)
        embeddings = emb_pipe.embedding_chunks(chunks=emb_chunks)
        metadatas = [{"text": chunk.page_content} for chunk in emb_chunks]
        self.add_embeddings(embeddings, metadatas)
        self.save()
        print(f"[INFO] vector store built and saved to {self.persistent_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any]):
        dim = embeddings.shape[1] ## dimension of the vector
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors into faiss index")

    def save(self):
        faiss_path = os.path.join(self.persistent_dir, "faiss.index")
        meta_path = os.path.join(self.persistent_dir, "metadata.pkl")
        if self.index is None:
            raise ValueError("Cannot save an empty vector store")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved faiss index and metadata in {self.persistent_dir}")

    def load(self):
        faiss_path = os.path.join(self.persistent_dir, "faiss.index")
        meta_path = os.path.join(self.persistent_dir, "metadata.pkl")
        self.index  = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata=pickle.load(f)
        print(f"[INFO] Loaded faiss index and metadata from {self.persistent_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            raise ValueError("Cannot search an empty vector store")
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def user_query(self, query_text: str, top_k: int = 3):
        print(f"[INFO] Querying vector store for query: {query_text}")
        query_list = self.model.embed_documents([query_text])
        query_array = np.asarray(query_list).astype("float32")
        return self.search(query_embedding=query_array, top_k=top_k)
