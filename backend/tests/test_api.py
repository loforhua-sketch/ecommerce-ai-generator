from __future__ import annotations

import os
from io import BytesIO
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient


os.environ["MOCK_MODE"] = "true"

from backend.app.config import get_settings  # noqa: E402
from backend.app.main import app  # noqa: E402


SAMPLE_IMAGE = b"\x89PNG\r\n\x1a\n" + b"test-image"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "app.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


@pytest.fixture()
def generated(client: TestClient) -> dict:
    response = client.post(
        "/api/generate",
        files={"files": ("sample.png", SAMPLE_IMAGE, "image/png")},
        data={
            "product_name": "测试商品",
            "category": "家居",
            "audience": "家庭用户",
            "price": "99",
            "origin_price": "129",
            "platform": "taobao",
            "style": "simple",
            "scene_style": "modern",
        },
    )
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["status"] == "success"
    return item


def test_health(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_and_mock_generation(client: TestClient, generated: dict):
    assert generated["image_path"].endswith(".png")
    assert "data:image/" not in generated["html"]
    assert "/api/uploads/" in generated["html"]
    upload = client.get(f"/api/uploads/{generated['image_path']}")
    assert upload.status_code == 200
    assert upload.content == SAMPLE_IMAGE


def test_history(client: TestClient, generated: dict):
    response = client.get("/api/generations")
    assert response.status_code == 200
    items = response.json()["items"]
    assert any(item["id"] == generated["id"] for item in items)


def test_single_html_export(client: TestClient, generated: dict):
    response = client.get(f"/api/generations/{generated['id']}/export?platform=pdd")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "pdd.html" in response.headers["content-disposition"]
    assert b"data:image/" not in response.content


def test_zip_export(client: TestClient, generated: dict):
    response = client.get(f"/api/generations/export.zip?ids={generated['id']}")
    assert response.status_code == 200
    with ZipFile(BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {
            "taobao.html",
            "pdd.html",
            "douyin.html",
            "titles.txt",
            "selling_points.txt",
            "scene_prompts.txt",
            "scenes/README.txt",
        }
        assert b"data:image/" not in archive.read("taobao.html")
