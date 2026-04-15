import os
from typing import TypedDict, List
import cohere
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START, END
from langchain_core.runnables.graph import MermaidDrawMethod
from src.api.v1.schemas.query_schema import AIResponse
from src.api.v1.tools.vector_search import vector_search
from src.api.v1.tools.fts_search import fts_search
from src.api.v1.tools.hybrid_search import hybrid_search
from src.core.db import get_sql_database

load_dotenv(override=True)

os.environ["PYPPETEER_CHROMIUM_REVISION"] = "1263111" 

llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")

class RAGState(TypedDict):
    query: List[str]
    retrieved_docs: List[Document]   # Output of Node 1 — wide retrieval (k=10)
    reranked_docs: List[Document]    # Output of Node 2 — narrowed by reranker (top_n=3)
    answer: str                      # The final answer string to return to the user
    k: int                           # Number of retrieved documents
    attempts: int                    # Number of rerank attempts
    is_sql_query:bool                # Whether the query is related to connect the db
    not_valid_query: bool            # Whether the query is valid
    no_answer_found: bool            # Whether the query resulted in a No Answer Found error


def llm_route_query(state: RAGState) -> RAGState:
    query = state["query"][-1]
    system_prompt = """
    You are a query routing agent for an agentic multimodal RAG system, designed for Reliance Financial Report and product information from the company's database.

    Classify the query into EXACTLY one label:
    VECTOR_SEARCH
    FTS_SEARCH
    HYBRID_SEARCH
    IRRELEVANT_QUERY
    SQL_QUERY

    Definitions:
    - VECTOR_SEARCH: Generic, semantic, open-ended information requests.
    - FTS_SEARCH: Exact keywords, IDs, dates, numbers, structured fields.
    - HYBRID_SEARCH: Semantic intent + exact attributes.
    - SQL_QUERY: SQL queries for product related data.
    - IRRELEVANT_QUERY: Not related to the HR domain or cannot be answered by our system.
    Rules:
    - Return ONLY the label.
    - No explanation.
    """
    

    prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "User Query:\n{query}")
        ])

    chain = prompt | llm
    response = chain.invoke({"query": query})

    decision = response.text.strip()

    #Increment attempts
    state["attempts"] += 1

    if decision == "VECTOR_SEARCH":
        return {
            **state,
            "retrieved_docs": vector_search(query, k=state["k"])
        }
    elif decision == "FTS_SEARCH":
        return {
            **state,
            "retrieved_docs": fts_search(query, state["k"])
        }
    elif decision == "HYBRID_SEARCH":
        return {
            **state,
            "retrieved_docs": hybrid_search(query, state["k"])

        }
    elif decision == "SQL_QUERY":
        return {
            **state,
            "is_sql_query": True,
        }

    elif decision == "IRRELEVANT_QUERY":
        return {
            **state,
            "not_valid_query": True,
            "retrieved_docs": []
        }
    else:
        raise ValueError(f"Invalid decision: {decision}")
    
def retrieval_gate(state: RAGState) -> str:
    if(state["not_valid_query"]):
        return "not_valid_query"
    elif(state["is_sql_query"]):
        return "nl2sql"
    elif len(state["retrieved_docs"]) > 0 :
        return "RERANK"
    elif len(state["retrieved_docs"]) == 0:
        return "rewrite_query"

def nl2sql_node(state: RAGState) -> RAGState:
    db = get_sql_database()

    # ── Step 1: Generate SQL using Gemini + live schema ─────────────────────
    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a PostgreSQL expert. Given the database schema below, 
            write a single valid SELECT query that answers the user's question.

            Rules:
            - Return ONLY the raw SQL — no explanation, no markdown fences, no backticks.
            - Use only the tables and columns present in the schema.
            - Do NOT generate INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements.
            - Always add a LIMIT clause (max 50 rows) unless the question asks for aggregates.
            - For product or text searches: NEVER search for the full multi-word phrase as one
            ILIKE pattern. Instead, split the search into individual meaningful keywords
            and OR them together across both name and description columns.
            Example — user asks "wireless headset":
                WHERE (name ILIKE '%wireless%' OR description ILIKE '%wireless%')
                OR (name ILIKE '%headset%'  OR description ILIKE '%headset%')
                OR (name ILIKE '%headphones%' OR description ILIKE '%headphones%')
            Use your knowledge of synonyms (headset/headphones, laptop/notebook, etc.)
            to cast a wider net when the exact term may not match.

            Database schema:
            {schema}"""
        ),
        ("human", "Question: {question}")
    ])

    sql_chain = sql_prompt | llm
    raw_sql = sql_chain.invoke({
        "schema": schema_info,
        "question": state["query"]
    })
    # Gemini may return content as a list of parts or a plain string
    content = raw_sql.content
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )
    generated_sql = content.strip().strip("```").strip()
    if generated_sql.lower().startswith("sql"):
        generated_sql = generated_sql[3:].strip()

    # ── Step 2: Execute SQL ──────────────────────────────────────────────────
    try:
        sql_result: str = db.run(generated_sql)
    except Exception as exc:
        sql_result = f"SQL execution error: {exc}"


    # ── Step 3: Summarise into AIResponse ────────────────────────────────────
    structured_llm = llm.with_structured_output(AIResponse)
    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful data analyst. Answer the user's question using "
            "the SQL query results below. Be concise and format numbers/lists clearly. "
            "Set policy_citations to empty string, "
            "page_no to 'N/A', and document_name to 'agentic_rag_db'."
        ),
        (
            "human",
            "Question: {query}\n\n"
            "SQL Used:\n{sql}\n\n"
            "Query Results:\n{result}"
        )
    ])

    chain = answer_prompt | structured_llm
    answer = chain.invoke({
        "query": state["query"],
        "sql": generated_sql,
        "result": sql_result
    })
    print("[nl2sql_node] Answer generated.")
    response = answer.model_dump()
    response["policy_citations"] = "N/A"
    response["sql_query_executed"] = generated_sql
    return {
        **state,
        "answer": response
    }

def rerank_node(state: RAGState) -> RAGState:
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    docs = state["retrieved_docs"]
    # print(docs)

    rerank_response = co.rerank(
        model="rerank-english-v3.0",
        query=state["query"][-1],
        documents=[doc["content"] for doc in docs],
        top_n=3
    )

    # Map Cohere result indices back to LangChain Document objects
    reranked_docs = [docs[r.index] for r in rerank_response.results]

    print(f"[rerank_node] Top {len(reranked_docs)} chunks after reranking:")
    for i, r in enumerate(rerank_response.results):
        print(f"  Rank {i+1} | Cohere score: {r.relevance_score:.4f} | original index: {r.index}")

    return {**state, "reranked_docs": reranked_docs}

def validate_reranker_node(state: RAGState) -> str:
    if len(state["reranked_docs"]) >= 1:
        return "generate_answer"
    elif len(state["reranked_docs"]) == 0 and state["attempts"] <= 3:
        return "rewrite_query"
    else:
        return "no_answer_Found"

def rewrite_node(state: RAGState) -> RAGState:
    # the node rewite a user query for better answer
    system_prompt = """
    You are a query rewriting agent for an agentic multimodal RAG system.
    The user query is below. Rewrite it to be more specific and clear,
    to help the retrieval system find better relevant documents.
    If the query is already specific, just return it as is.
    """
    query = state["query"][-1]

    prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "User Query:\n{query}")
    ])

    chain = prompt | llm
    response = chain.invoke({"query": query})

    decision = response.text.strip()

    return {
        **state,
        "query": state["query"].append(decision)
    }

    
def generate_answer_node(state: RAGState) -> RAGState:
    structured_llm = llm.with_structured_output(AIResponse)

    if(not state["is_sql_query"] and not state["not_valid_query"] and len(state["reranked_docs"]) > 0):
        context = "\n\n".join([
            f"[Source: {doc['metadata'].get('document_name', doc['metadata'].get('source', 'unknown'))} "
            f"| Page: {doc['metadata'].get('page_label', doc['metadata'].get('page', '?'))}"
            f"{f' | Image: {doc['image_path']}' if doc.get('image_path') else ''}]\n"
            f"{doc['content']}"
            for doc in state["reranked_docs"]
        ])
        prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful HR assistant. Answer the user's question directly and clearly "
            "using only the information in the context below. "
            "Never mention 'the context', 'the provided text', 'the document', or any internal source references in your answer — "
            "just answer as if you know the information. "
            "If the information is not available, say 'I don't have that information at the moment.' "
            "Always cite the source document and page number at the end."
        ),
        ("human", "Context:\n{context}\n\nQuestion: {query}")
        ])

        chain = prompt | structured_llm
        result = chain.invoke({"context": context, "query": state["query"]})

        print(f"[generate_answer_node] Answer generated.")
        return {**state, "answer": result.model_dump()}
    
    elif(state["is_sql_query"]):
         return state
    elif(state["not_valid_query"]):
        response = AIResponse(
            query=state["query"][0],
            answer="The query is not relevant to our domain. Please ask a different question.",
            policy_citations="N/A",
            page_no="N/A",
            document_name="N/A",
            sql_query_executed = None
        )
        return {**state, "answer":response}
    elif state["no_answer_found"]:
        response = AIResponse(
            query=state["query"][0],
            answer="I couldn't find an answer to your question after multiple attempts. Please try rephrasing or ask a different question.",
            policy_citations="N/A",
            page_no="N/A",
            document_name="N/A",
            sql_query_executed = None
        )
        return {**state, "answer":response}


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("tool_caller", llm_route_query)
    graph.add_node("nl2sql", nl2sql_node)
    graph.add_node("reranker", rerank_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.add_edge(START, "tool_caller")
    graph.add_conditional_edges(
    "tool_caller",
    retrieval_gate,
    {
        "RERANK": "reranker",
        "rewrite_query": "rewrite",
        "not_valid_query":"generate_answer",
        "nl2sql":"nl2sql"
    }
    )
    graph.add_edge("nl2sql","generate_answer")
    graph.add_conditional_edges(
    "reranker",
    validate_reranker_node,
    {
        "generate_answer": "generate_answer",
        "rewrite_query": "rewrite",
        "no_answer_Found": "generate_answer"
    }
    )
    graph.add_edge("rewrite", "tool_caller")
    graph.add_edge("generate_answer", END)

    return graph.compile()

def run_rag_agent(QueryRequest) -> AIResponse:
    initial_state: RAGState = {
        "query": [QueryRequest.query],
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "k": QueryRequest.k,
        "attempts": 0,
        "is_sql_query": False,
        "not_valid_query": False,
        "no_answer_found": False,
        "generated_sql": "",
        "sql_result": ""
    }

    rag_graph = build_rag_graph()
    print(rag_graph.get_graph().draw_mermaid())

    final_state = rag_graph.invoke(initial_state)
    # print(final_state)
    return final_state["answer"]

