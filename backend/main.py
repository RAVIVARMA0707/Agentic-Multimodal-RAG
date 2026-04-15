# backend/main.py

from fastapi import FastAPI
from src.api.v1.routes import routers

app = FastAPI()

for router in routers:
    app.include_router(router, prefix="/api/v1")