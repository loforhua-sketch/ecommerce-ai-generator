import base64
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

from .config import Settings, get_settings
from .database import Database
from .services.html_builder import build_detail_html
from .services.openai_service import OpenAIProductService

app = FastAPI(title="Ecommerce AI Generator", version="1.0.0")


def image_data_url(path: Path) -> str:
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{media_type};base64,{encoded}"


def get_db(settings: Annotated[Settings, Depends(get_settings)]) -> Database:
    return Database(settings.database_file)


@app.on_event("startup")
def startup() -> None:
    settings = get_settings()
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    Database(settings.database_file)


settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate")
async def generate(
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    files: list[UploadFile] = File(...),
    product_name: str = Form(""),
    category: str = Form(""),
    audience: str = Form(""),
    price: str = Form(""),
    origin_price: str = Form(""),
) -> dict[str, list[dict]]:
    service = OpenAIProductService(settings.openai_api_key, settings.openai_model)
    results = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{file.filename} 不是图片文件")

        suffix = Path(file.filename or "").suffix or ".jpg"
        saved_name = f"{uuid4().hex}{suffix}"
        saved_path = settings.upload_path / saved_name
        saved_path.write_bytes(await file.read())

        analysis = service.analyze(
            saved_path,
            product_name=product_name,
            category=category,
            audience=audience,
            price=price,
            origin_price=origin_price,
        )
        html = build_detail_html(analysis, image_url=image_data_url(saved_path))
        results.append(
            db.create_generation(
                {
                    "filename": file.filename or saved_name,
                    "image_path": saved_name,
                    "product_name": analysis.get("product_name", product_name),
                    "category": analysis.get("category", category),
                    "audience": analysis.get("audience", audience),
                    "price": price,
                    "origin_price": origin_price,
                    "analysis": analysis,
                    "html": html,
                }
            )
        )
    return {"items": results}


@app.get("/api/generations")
def list_generations(db: Annotated[Database, Depends(get_db)]) -> dict[str, list[dict]]:
    return {"items": db.list_generations()}


@app.get("/api/generations/{generation_id}")
def get_generation(generation_id: int, db: Annotated[Database, Depends(get_db)]) -> dict:
    try:
        return db.get_generation(generation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="记录不存在") from exc


@app.get("/api/generations/{generation_id}/html", response_class=HTMLResponse)
def get_generation_html(generation_id: int, db: Annotated[Database, Depends(get_db)]) -> str:
    try:
        return db.get_generation(generation_id)["html"]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="记录不存在") from exc


@app.get("/api/generations/{generation_id}/export")
def export_generation_html(generation_id: int, db: Annotated[Database, Depends(get_db)]) -> Response:
    try:
        item = db.get_generation(generation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="记录不存在") from exc
    filename = f"detail-page-{generation_id}.html"
    return Response(
        item["html"],
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/uploads/{filename}")
def uploaded_file(filename: str, settings: Annotated[Settings, Depends(get_settings)]) -> Response:
    path = settings.upload_path / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return Response(path.read_bytes(), media_type=media_type)
