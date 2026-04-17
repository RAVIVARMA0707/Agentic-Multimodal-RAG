from pydantic import BaseModel, Field
from typing import Optional

# ---- Request ----
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    session_id: str = Field(..., description="Session ID for thread consistency")



class AIResponse(BaseModel):
    query: str = Field(description="The Given query by user must be present here")
    answer: str = Field(description="The generated response")
    policy_citations: str = Field(description="Give the Policy Citation")
    page_no: str = Field(description="The page number in the metadata")
    document_name: str = Field(description="Name of the document used")
    image_path: Optional[str] = Field(default=None, description="Path to the image ")
    sql_query_executed: Optional[str] = Field(default=None, description="The SQL query executed (for product/database queries)")

class nl2sqlResponse(BaseModel):
    nl2sql_answer: str = Field(description="The answer from executing the SQL query")
    generated_sql_query: str = Field(description="The generated SQL query")

class RoutingDecision(BaseModel):
    needs_sql: bool
    needs_rag: bool
    sql_query: str | None
    rag_query: str | None
    not_valid_query: bool