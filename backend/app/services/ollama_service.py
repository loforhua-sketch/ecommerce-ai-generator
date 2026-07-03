import base64
import ast
import json
import re
from pathlib import Path
from typing import Any
from urllib import error, request

from .openai_service import SYSTEM_PROMPT, _fallback, _merge_with_fallback


_JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        quote = ""
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    in_string = False
                continue

            if char in {'"', "'"}:
                in_string = True
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)
    return None


def _parse_json_text(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    without_trailing_commas = re.sub(r",\s*([}\]])", r"\1", text.strip())
    if without_trailing_commas != candidates[0]:
        candidates.append(without_trailing_commas)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        if isinstance(value, dict):
            return value
    return None


def _parse_model_content(content: str) -> dict[str, Any] | None:
    candidates = [content]
    candidates.extend(match.group(1) for match in _JSON_CODE_BLOCK_RE.finditer(content))

    for candidate in candidates:
        parsed = _parse_json_text(candidate)
        if parsed is not None:
            return parsed

        json_object = _extract_first_json_object(candidate)
        if json_object:
            parsed = _parse_json_text(json_object)
            if parsed is not None:
                return parsed
    return None


def _ollama_response_content(response_body: str) -> str:
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        json_object = _extract_first_json_object(response_body)
        parsed = _parse_json_text(json_object) if json_object else None

    if isinstance(parsed, dict):
        response = parsed.get("response")
        if isinstance(response, str):
            return response
    return response_body


class OllamaProductService:
    def __init__(self, base_url: str, model: str, timeout: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def analyze(
        self,
        image_path: Path,
        product_name: str = "",
        category: str = "",
        audience: str = "",
        price: str = "",
        origin_price: str = "",
    ) -> dict[str, Any]:
        fallback = _fallback(product_name, category, audience, price, origin_price)
        user_prompt = f"""
{SYSTEM_PROMPT}

补充信息：
- 商品名称：{product_name or "请从图片识别"}
- 商品品类：{category or "请从图片识别"}
- 目标人群：{audience or "请根据图片推断"}
- 售价：{price or "未知"}
- 原价：{origin_price or "未知"}

请分析上传的商品图片，返回严格 JSON，字段必须包含：
product_name, category, audience, tagline, materials, specifications, use_scenes, selling_points,
taobao_title, pdd_title, douyin_title, selling_copy, detail_modules, buying_reasons。

要求：
1. selling_points 生成 5-8 条短句。
2. use_scenes 生成 3-6 个使用场景。
3. materials 识别材质、工艺、颜色、外观特征，无法确定时用“图片可见/建议复核”的谨慎表述。
4. specifications 使用对象结构，包含品类、材质、颜色、尺寸/容量/型号等可识别规格。
5. detail_modules 为数组，每项包含 title 和 content，覆盖首屏主图、核心卖点、场景展示、参数介绍、购买理由、结尾营销模块。
6. 淘宝标题偏搜索关键词，拼多多标题偏实惠和场景，抖店标题偏短视频种草，但都要自然可读。
7. 不要输出 Markdown，不要用代码块包裹 JSON。
"""
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "images": [_image_base64(image_path)],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.4},
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
            print(f"Ollama raw response (first 500 chars): {response_body[:500]}")

            content = _ollama_response_content(response_body)
            data = _parse_model_content(content)
            if data is None:
                data = {"selling_copy": [content or response_body]}
        except (OSError, error.URLError, TimeoutError) as exc:
            print(f"Ollama request failed: {exc}")
            data = {}

        return _merge_with_fallback(data, fallback)
