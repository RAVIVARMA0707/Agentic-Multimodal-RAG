import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.utilities import SQLDatabase

load_dotenv()
PG_CONNECTION = os.getenv("SQLALCHEMY_DATABASE_URL")

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model= os.environ.get("GOOGLE_EMBEDDING_MODEL"),
        api_key= os.environ.get("GOOGLE_API_KEY"),
        output_dimensionality=1536
    )

def get_vector_store(collection_name : str = "hr_support_desk"):
    return PGVector(
        collection_name=collection_name,
        connection=PG_CONNECTION,
        embeddings = get_embeddings(),
        use_jsonb=True
    )

def get_sql_database() -> SQLDatabase:
    """Return a LangChain SQLDatabase connected to the agentic_rag_db (read-only).

    Uses the rag_readonly role from sql/seed.sql — SELECT privileges only.
    Connection string is read from AGENTIC_RAG_DB_URL in the environment.
    """
    db_url = os.getenv("AGENTIC_RAG_DB_URL")
    if not db_url:
        raise ValueError("AGENTIC_RAG_DB_URL is not set. Check your .env file.")
    return SQLDatabase.from_uri(
        db_url,
        include_tables=["products", "categories", "orders", "order_items"],
        sample_rows_in_table_info=2,
    )