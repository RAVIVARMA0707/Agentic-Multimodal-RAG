from src.api.v1.schemas.query_schema import AIResponse, QueryRequest
from src.api.v1.agents.agent import run_rag_agent

class QueryServices():
    def run_rag_agent(request: QueryRequest)-> AIResponse:
        result = run_rag_agent(request)
        return result