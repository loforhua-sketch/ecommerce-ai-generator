import base64
import io
import json
import re
from pathlib import Path
from typing import Annotated
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from .config import Settings, get_settings
from .database import Database
from .services.html_builder import PLATFORM_FILENAMES, build_platform_htmls, normalize_platform, normalize_style
from .services.ollama_service import OllamaProductService
from .services.openai_service import OpenAIProductService

app = FastAPI(title="Ecommerce AI Generator", version="1.2.0")


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
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
    settings = get_settings()
    return {"status": "ok", "provider": settings.ai_provider}


def get_product_service(settings: Settings) -> OpenAIProductService | OllamaProductService:
    if settings.ai_provider.lower() == "openai":
        return OpenAIProductService(settings.openai_api_key, settings.openai_model)
    return OllamaProductService(settings.ollama_base_url, settings.ollama_model, settings.ollama_timeout)


def safe_export_name(value: str, fallback: str) -> str:
    name = Path(value or "").stem or fallback
    name = re.sub(r'[\\/:*?"<>|\r\n]+', "-", name).strip(" .")
    return name or fallback


def generation_html_files(item: dict) -> dict[str, str]:
    html_files = item.get("analysis", {}).get("html_files")
    if isinstance(html_files, dict):
        return {key: str(value) for key, value in html_files.items() if str(value).strip()}

    raw_html = str(item.get("html") or "")
    try:
        parsed = json.loads(raw_html)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return {key: str(value) for key, value in parsed.items() if str(value).strip()}
    return {"taobao": raw_html}


def selected_html(item: dict, platform: str = "taobao") -> str:
    html_files = generation_html_files(item)
    platform = normalize_platform(platform)
    return html_files.get(platform) or html_files.get("taobao") or next(iter(html_files.values()), "")


def enrich_generation(item: dict) -> dict:
    platform = item.get("analysis", {}).get("template_platform", "taobao")
    item["html_files"] = generation_html_files(item)
    item["html"] = selected_html(item, platform)
    return item


def titles_text(items: list[dict]) -> str:
    lines = []
    for item in items:
        analysis = item.get("analysis", {})
        product_name = analysis.get("product_name") or item.get("product_name") or item.get("filename")
        lines.extend(
            [
                f"商品：{product_name}",
                f"淘宝：{analysis.get('taobao_title', '')}",
                f"拼多多：{analysis.get('pdd_title', '')}",
                f"抖店：{analysis.get('douyin_title', '')}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def selling_points_text(items: list[dict]) -> str:
    blocks = []
    for item in items:
        analysis = item.get("analysis", {})
        product_name = analysis.get("product_name") or item.get("product_name") or item.get("filename")
        points = analysis.get("selling_points") or analysis.get("core_selling_points") or []
        if not isinstance(points, list):
            points = [str(points)]
        block = [f"商品：{product_name}"]
        block.extend(f"- {point}" for point in points if str(point).strip())
        blocks.append("\n".join(block))
    return "\n\n".join(blocks).strip() + "\n"


async def generate_one(
    file: UploadFile,
    db: Database,
    settings: Settings,
    service: OpenAIProductService | OllamaProductService,
    product_name: str,
    category: str,
    audience: str,
    price: str,
    origin_price: str,
    platform: str,
    style: str,
) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise ValueError(f"{file.filename or '未命名文件'} 不是图片文件")

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
    html_files = build_platform_htmls(
        analysis,
        image_url=image_data_url(saved_path),
        style=style,
        price=price,
        origin_price=origin_price,
    )
    analysis["template_platform"] = platform
    analysis["template_style"] = style
    analysis["html_files"] = html_files

    item = db.create_generation(
        {
            "filename": file.filename or saved_name,
            "image_path": saved_name,
            "product_name": analysis.get("product_name", product_name),
            "category": analysis.get("category", category),
            "audience": analysis.get("audience", audience),
            "price": price,
            "origin_price": origin_price,
            "analysis": analysis,
            "html": html_files[platform],
        }
    )
    item["status"] = "success"
    return enrich_generation(item)


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
    platform: str = Form("taobao"),
    style: str = Form("simple"),
) -> dict[str, list[dict]]:
    service = get_product_service(settings)
    results = []
    platform = normalize_platform(platform)
    style = normalize_style(style)

    for index, file in enumerate(files, start=1):
        filename = file.filename or f"image-{index}"
        try:
            results.append(
                await generate_one(
                    file,
                    db,
                    settings,
                    service,
                    product_name,
                    category,
                    audience,
                    price,
                    origin_price,
                    platform,
                    style,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "status": "failed",
                    "filename": filename,
                    "error": str(exc),
                }
            )

    return {"items": results}


@app.get("/api/generations")
def list_generations(db: Annotated[Database, Depends(get_db)]) -> dict[str, list[dict]]:
    return {"items": [enrich_generation(item) for item in db.list_generations()]}


@app.get("/api/generations/export.zip")
def export_generations_zip(ids: str, db: Annotated[Database, Depends(get_db)]) -> StreamingResponse:
    generation_ids = []
    for raw_id in ids.split(","):
        raw_id = raw_id.strip()
        if not raw_id:
            continue
        try:
            generation_ids.append(int(raw_id))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"无效的生成记录 ID: {raw_id}") from exc

    if not generation_ids:
        raise HTTPException(status_code=400, detail="请至少提供一个生成记录 ID")

    export_items = []
    for generation_id in generation_ids:
        try:
            export_items.append(db.get_generation(generation_id))
        except KeyError:
            continue

    if not export_items:
        raise HTTPException(status_code=404, detail="没有找到可导出的详情页")

    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as zip_file:
        for platform, filename in PLATFORM_FILENAMES.items():
            html_parts = [selected_html(item, platform) for item in export_items]
            zip_file.writestr(filename, "\n\n".join(part for part in html_parts if part))
        zip_file.writestr("titles.txt", titles_text(export_items))
        zip_file.writestr("selling_points.txt", selling_points_text(export_items))

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="detail-pages.zip"'},
    )


@app.get("/api/generations/{generation_id}")
def get_generation(generation_id: int, db: Annotated[Database, Depends(get_db)]) -> dict:
    try:
        return enrich_generation(db.get_generation(generation_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="记录不存在") from exc


@app.get("/api/generations/{generation_id}/html", response_class=HTMLResponse)
def get_generation_html(
    generation_id: int,
    db: Annotated[Database, Depends(get_db)],
    platform: str = "taobao",
) -> str:
    try:
        return selected_html(db.get_generation(generation_id), platform)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="记录不存在") from exc


@app.get("/api/generations/{generation_id}/export")
def export_generation_html(
    generation_id: int,
    db: Annotated[Database, Depends(get_db)],
    platform: str = "taobao",
) -> Response:
    try:
        item = db.get_generation(generation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="记录不存在") from exc

    platform = normalize_platform(platform)
    filename = PLATFORM_FILENAMES[platform]
    return Response(
        selected_html(item, platform),
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/uploads/{filename}")
def uploaded_file(filename: str, settings: Annotated[Settings, Depends(get_settings)]) -> Response:
    path = settings.upload_path / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")

    suffix = path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
    return Response(path.read_bytes(), media_type=media_type)
