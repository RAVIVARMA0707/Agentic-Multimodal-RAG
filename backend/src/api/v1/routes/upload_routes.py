# backend/src/api/v1/routes/upload_routes.py

from fastapi import APIRouter, UploadFile, File
from typing import List
import pathlib
import shutil

from src.ingestion.ingestion import run_ingestion
from src.core.db import clear_all_ingested_data

router = APIRouter()

@router.post("/upload-pdf")
async def upload_multiple_pdfs(files: List[UploadFile] = File(...)):
    clear_all_ingested_data()

    upload_dir = pathlib.Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for file in files:
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = run_ingestion(str(file_path))
        results.append(result)

    return {
        "status": "success",
        "documents_ingested": len(results),
        "details": results
    }