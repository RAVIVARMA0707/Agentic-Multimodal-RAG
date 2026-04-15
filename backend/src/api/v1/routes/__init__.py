# backend/src/api/v1/routes/__init__.py

from .query_routes import router as query_router
from .upload_routes import router as upload_router

routers = [
    query_router,
    upload_router,
]
