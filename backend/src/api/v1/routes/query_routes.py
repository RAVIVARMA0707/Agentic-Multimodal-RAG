from fastapi import APIRouter
from src.api.v1.schemas.query_schema import QueryRequest,AIResponse
from src.api.v1.services.query_service import QueryServices 

router = APIRouter()

@router.post("/query",response_model=AIResponse)
def query_endpoint(request: QueryRequest):
    return QueryServices.run_rag_agent(request=request)