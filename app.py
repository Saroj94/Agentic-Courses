from src.data_loader import load_documents
from src.embedding import EmbeddingPipeline
from src.vector_store import FaissVectorStore
from src.search import RAGSearch



if __name__=="__main__":
    # data = load_documents(data_dir="data/text_file")
    # doc_chunks = EmbeddingManager().chunk_document(data)
    # chunkVectors = EmbeddingManager().embedding_chunks(chunks=doc_chunks)
    # print(chunkVectors)
    store = FaissVectorStore("faiss_store")
    # store.build_from_documents(data)    
    store.load()
    # print(store.user_query("AI Engineering Roadmap.", top_k=3))

    rag = RAGSearch()
    query = "AI Engineering Roadmap."
    summary = rag.search_summarize(query=query, top_k=4)
    print("summary", summary)