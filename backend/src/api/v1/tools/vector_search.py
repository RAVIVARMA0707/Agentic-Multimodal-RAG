from src.core.db import get_vector_store

def vector_query_documents(query: str, k: int = 5) -> list[dict]:

   # vector — long natural-language question
   vector_store = get_vector_store()
   docs = vector_store.similarity_search(query, k=k)
   # print(docs)
   return [{"content": doc.page_content,"chunk_id": doc.id,"metadata": doc.metadata} for doc in docs]