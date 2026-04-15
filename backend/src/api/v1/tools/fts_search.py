import os
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row

load_dotenv()

# PGVector connection string uses SQLAlchemy format: postgresql+psycopg://...
# psycopg.connect needs standard format: postgresql://...
_raw_conn = os.getenv("PG_CONNECTION_STRING", "").replace("postgresql+psycopg", "postgresql")


def fts_search(query: str, k: int = 5, collection_name: str = "insurance_claim_collection") -> list[dict]:
    """
    Keyword search against stored chunks using PostgreSQL tsvector / tsquery / ts_rank.

    Args:
        query:           User query string (plain text, any format)
        k:               Number of top results to return
        collection_name: PGVector collection to search

    Returns:
        List of dicts with 'content', 'metadata', and 'fts_rank'
    """
    sql = """
        SELECT
            e.document                                               AS content,
            e.cmetadata                                              AS metadata,
            ts_rank(
                to_tsvector('english', e.document),
                plainto_tsquery('english', %(query)s)
            )                                                        AS fts_rank
        FROM  langchain_pg_embedding  e
        JOIN  langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = %(collection)s
          AND to_tsvector('english', e.document)
              @@ plainto_tsquery('english', %(query)s)
        ORDER BY fts_rank DESC
        LIMIT %(k)s;
    """
    with psycopg.connect(_raw_conn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"query": query, "collection": collection_name, "k": k})
            rows = cur.fetchall()

    return [
        {
            "content":  row["content"],
            "metadata": row["metadata"],
            "fts_rank": round(float(row["fts_rank"]), 4),
        }
        for row in rows
    ]
