import os
from dotenv import load_dotenv
from src.vector_store import FaissVectorStore
from langchain_groq import ChatGroq
from src.data_loader import load_documents

load_dotenv()

class RAGSearch:
    def __init__(self, persistent_dir: str = "faiss_store", embedder: str = "sentence-transformers/all-MiniLM-L6-v2", llm_model: str = "qwen/qwen3.8-27b"):
        self.vector_store = FaissVectorStore(persistent_dir=persistent_dir, emb_model=embedder)
        ##load or build vector store
        faiss_path = os.path.join(persistent_dir, "faiss.index")
        meta_path = os.path.join(persistent_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            docs = load_documents("data")
            self.vector_store.build_from_documents(documents=docs)
        else:
            self.vector_store.load()
        self.llm = ChatGroq(model=llm_model)
        print(f"[INFO] Groq LLM is initialize: {llm_model}")

    def search_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vector_store.user_query(query_text=query, top_k=top_k)
        texts = [r['metadata'].get("text", " ") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if context is None:
            print("No relevant information found")
        prompt = f"Summarize the following context for query: {query}, \n\nContext: {context}"
        response = self.llm.invoke([prompt])
        return str(response.content)
    