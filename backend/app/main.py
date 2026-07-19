import io
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Annotated
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import STATIC_DIR, Settings, get_settings
from .database import Database
from .services.html_builder import PLATFORM_FILENAMES, build_platform_htmls, normalize_platform, normalize_style
from .services.ollama_service import OllamaProductService
from .services.openai_service import OpenAIProductService
from .services.product_master import decode_product_image, generate_product_master
from .services.scene_image_service import generate_scene_images
from .services.scene_prompt import (
    build_scene_prompts,
    normalize_scene_style,
    scene_prompts_to_text,
    scenes_readme_text,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Ecommerce AI Generator", version="1.4.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MOCK_SCENE_IMAGES = [
    {
        "url": "/static/scenes/scene1.jpg",
        "prompt": "现代家居实拍风格场景图",
    },
    {
        "url": "/static/scenes/scene2.jpg",
        "prompt": "电商详情页展示风格场景图",
    },
    {
        "url": "/static/scenes/scene3.jpg",
        "prompt": "生活使用场景图",
    },
]


def uploaded_image_url(filename: str) -> str:
    return f"/api/uploads/{filename}"


def summarize_error(exc: Exception, limit: int = 240) -> str:
    message = re.sub(r"\s+", " ", str(exc)).strip()
    return message[:limit] or exc.__class__.__name__


def replace_embedded_image(html: str, image_url: str) -> str:
    if "data:image/" not in html:
        return html
    pattern = re.compile(
        r"""(<img\b[^>]*\bsrc=)(["'])data:image/.*?\2""",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}{image_url}{match.group(2)}", html)


def get_db(settings: Annotated[Settings, Depends(get_settings)]) -> Database:
    return Database(settings.database_file)


@app.on_event("startup")
def startup() -> None:
    settings = get_settings()
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    legacy_database = settings.legacy_database_file
    if not settings.database_file.exists() and legacy_database and legacy_database.is_file():
        settings.database_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_copy = settings.database_file.with_suffix(settings.database_file.suffix + ".migrating")
        shutil.copy2(legacy_database, temporary_copy)
        temporary_copy.replace(settings.database_file)
        logger.info("legacy_database_copied destination=%s", settings.database_file)
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


@app.post("/api/product-master")
async def create_product_master(
    file: UploadFile = File(...),
    width_px: int = Form(1300),
) -> Response:
    """Return a fixed-dimension transparent PNG product master."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")
    try:
        master = generate_product_master(
            decode_product_image(await file.read()),
            width_px=width_px,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        master.png_bytes(),
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="product-master.png"',
            "X-Master-Pixels": f"{master.width_px}x{master.height_px}",
            "X-Master-Dimensions-Cm": f"{master.width_cm:g}x{master.depth_cm:g}",
            "X-Master-Orientation-Uncertain": str(master.orientation_uncertain).lower(),
        },
    )


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
        files = {key: str(value) for key, value in html_files.items() if str(value).strip()}
    else:
        raw_html = str(item.get("html") or "")
        try:
            parsed = json.loads(raw_html)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            files = {key: str(value) for key, value in parsed.items() if str(value).strip()}
        else:
            files = {"taobao": raw_html}

    image_path = str(item.get("image_path") or "").strip()
    if image_path:
        image_url = uploaded_image_url(image_path)
        files = {key: replace_embedded_image(value, image_url) for key, value in files.items()}
    return files


def selected_html(item: dict, platform: str = "taobao") -> str:
    html_files = generation_html_files(item)
    platform = normalize_platform(platform)
    return html_files.get(platform) or html_files.get("taobao") or next(iter(html_files.values()), "")


def enrich_generation(item: dict) -> dict:
    platform = item.get("analysis", {}).get("template_platform", "taobao")
    analysis = item.get("analysis", {})
    if not analysis.get("scene_prompts"):
        analysis["scene_prompts"] = build_scene_prompts(analysis, analysis.get("scene_style", "modern"))
    item["html_files"] = generation_html_files(item)
    # Legacy records may contain three Base64-heavy HTML copies in analysis_json.
    # Keep them readable through html_files without returning the duplicate payload.
    analysis.pop("html_files", None)
    item["html"] = selected_html(item, platform)
    return item


def mock_product_analysis(
    product_name: str,
    category: str,
    audience: str,
    price: str,
    origin_price: str,
) -> dict:
    name = product_name.strip() or "Mock Product"
    product_category = category.strip() or "Ecommerce Product"
    target = audience.strip() or "Daily shoppers"
    specs = {
        "Product": name,
        "Category": product_category,
        "Audience": target,
        "Mode": "MOCK_MODE",
    }
    if price:
        specs["Price"] = price
    if origin_price:
        specs["Original price"] = origin_price

    return {
        "product_name": name,
        "category": product_category,
        "audience": target,
        "tagline": "Mock product data generated locally without calling Ollama.",
        "materials": ["Image-visible material, confirm before publishing", "Mock packaging and finish"],
        "functions": ["Product display", "Detail page preview", "Multi-platform listing draft"],
        "specifications": specs,
        "use_scenes": ["Home use", "Store display", "Gift purchase"],
        "core_keywords": [name, product_category, "mock product"],
        "long_tail_keywords": [
            f"{name} for home",
            f"{name} detail page",
            f"{product_category} mock listing",
        ],
        "core_selling_points": [
            "Local mock data, no model request required",
            "Complete fields for detail page rendering",
            "Ready for frontend and export flow testing",
        ],
        "five_point_description": [
            "Clear product summary for listing previews.",
            "Core selling points are structured for quick scanning.",
            "Specifications keep key purchase information together.",
            "Scene prompts are generated for visual planning.",
            "HTML output remains compatible with export workflows.",
        ],
        "selling_points": [
            "No Ollama call in MOCK_MODE",
            "Stable mock product content",
            "Compatible item response structure",
            "Includes generated scene image metadata",
        ],
        "taobao_title": f"{name} mock detail page product listing",
        "pdd_title": f"{name} value mock product listing",
        "douyin_title": f"{name} short video mock product pick",
        "selling_copy": [
            f"{name} is generated with local mock content for backend startup and API testing.",
            "The response keeps the same major analysis fields used by the detail page builder.",
            "Use real AI mode after configuring a reachable provider.",
        ],
        "detail_page_modules": [
            {"title": "Hero", "content": f"{name} mock detail page hero module."},
            {"title": "Selling points", "content": "Structured local copy for rendering tests."},
            {"title": "Specifications", "content": "Mock specs keep the generated page complete."},
        ],
        "detail_modules": [
            {"title": "Hero", "content": f"{name} mock detail page hero module."},
            {"title": "Selling points", "content": "Structured local copy for rendering tests."},
            {"title": "Specifications", "content": "Mock specs keep the generated page complete."},
        ],
        "buying_reasons": [
            "Fast local test response",
            "No dependency on Ollama service availability",
            "Consistent output for frontend debugging",
        ],
    }


async def generate_mock_one(
    file: UploadFile,
    db: Database,
    settings: Settings,
    product_name: str,
    category: str,
    audience: str,
    price: str,
    origin_price: str,
    platform: str,
    style: str,
    scene_style: str,
) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise ValueError(f"{file.filename or 'unnamed file'} is not an image file")

    suffix = Path(file.filename or "").suffix or ".jpg"
    saved_name = f"{uuid4().hex}{suffix}"
    saved_path = settings.upload_path / saved_name
    saved_path.write_bytes(await file.read())

    analysis = mock_product_analysis(product_name, category, audience, price, origin_price)
    html_files = build_platform_htmls(
        analysis,
        image_url=uploaded_image_url(saved_name),
        style=style,
        price=price,
        origin_price=origin_price,
    )
    analysis["template_platform"] = platform
    analysis["template_style"] = style
    analysis["scene_style"] = scene_style
    analysis["scene_prompts"] = build_scene_prompts(analysis, scene_style)
    analysis.pop("html_files", None)

    item = db.create_generation(
        {
            "filename": file.filename or saved_name,
            "image_path": saved_name,
            "product_name": analysis["product_name"],
            "category": analysis["category"],
            "audience": analysis["audience"],
            "price": price,
            "origin_price": origin_price,
            "analysis": analysis,
            "html": json.dumps(html_files, ensure_ascii=False),
        }
    )
    item["status"] = "success"
    return enrich_generation(item)


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
    scene_style: str,
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
        image_url=uploaded_image_url(saved_name),
        style=style,
        price=price,
        origin_price=origin_price,
    )
    analysis["template_platform"] = platform
    analysis["template_style"] = style
    analysis["scene_style"] = scene_style
    analysis["scene_prompts"] = build_scene_prompts(analysis, scene_style)
    analysis.pop("html_files", None)

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
            "html": json.dumps(html_files, ensure_ascii=False),
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
    scene_style: str = Form("modern"),
) -> dict[str, list[dict]]:
    results = []
    platform = normalize_platform(platform)
    style = normalize_style(style)
    scene_style = normalize_scene_style(scene_style)

    if settings.mock_mode:
        for index, file in enumerate(files, start=1):
            filename = file.filename or f"image-{index}"
            try:
                results.append(
                    await generate_mock_one(
                        file,
                        db,
                        settings,
                        product_name,
                        category,
                        audience,
                        price,
                        origin_price,
                        platform,
                        style,
                        scene_style,
                    )
                )
            except Exception as exc:
                error_summary = summarize_error(exc)
                logger.warning("generation_failed filename=%s error=%s", filename, error_summary)
                results.append(
                    {
                        "status": "failed",
                        "filename": filename,
                        "error": error_summary,
                    }
                )

        scene_images = MOCK_SCENE_IMAGES
        for item in results:
            item["scene_images"] = scene_images

        response = {
            "scene_images": scene_images,
            "items": results,
        }
        for item in results:
            logger.info(
                "generation_finished task_id=%s filename=%s status=%s",
                item.get("id", "-"),
                item.get("filename", "-"),
                item.get("status", "failed"),
            )
        return response

    service = get_product_service(settings)

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
                    scene_style,
                )
            )
        except Exception as exc:
            error_summary = summarize_error(exc)
            logger.warning("generation_failed filename=%s error=%s", filename, error_summary)
            results.append(
                {
                    "status": "failed",
                    "filename": filename,
                    "error": error_summary,
                }
            )

    scene_images = generate_scene_images(
        results[0].get("product_name", "") if results else product_name,
        scene_style,
    )
    for item in results:
        item["scene_images"] = scene_images
    for item in results:
        logger.info(
            "generation_finished task_id=%s filename=%s status=%s",
            item.get("id", "-"),
            item.get("filename", "-"),
            item.get("status", "failed"),
        )

    return {
        "scene_images": scene_images,
        "items": results,
    }

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
        zip_file.writestr("scene_prompts.txt", scene_prompts_to_text(export_items))
        zip_file.writestr("scenes/README.txt", scenes_readme_text())

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
    path = (settings.upload_path / Path(filename).name).resolve()
    if not path.is_file() and settings.legacy_upload_path:
        legacy_path = (settings.legacy_upload_path / Path(filename).name).resolve()
        if legacy_path.parent == settings.legacy_upload_path.resolve() and legacy_path.is_file():
            path = legacy_path
    allowed_parents = {settings.upload_path.resolve()}
    if settings.legacy_upload_path:
        allowed_parents.add(settings.legacy_upload_path.resolve())
    if path.parent not in allowed_parents or not path.is_file():
        raise HTTPException(status_code=404, detail="图片不存在")

    suffix = path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
    return Response(path.read_bytes(), media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
