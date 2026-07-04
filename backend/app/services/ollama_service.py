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
product_name, category, audience, tagline, materials, functions, specifications, use_scenes,
core_keywords, long_tail_keywords, core_selling_points, five_point_description, selling_points,
taobao_title, pdd_title, douyin_title, selling_copy, detail_page_modules, detail_modules, buying_reasons。

要求：
1. product_name 输出真实商品名称，不输出图片描述；如果识别为楼梯垫/楼梯踏步垫，商品名称必须是“防滑楼梯垫”。
2. audience 输出真实购买人群，不输出图片场景、拍摄场景或“家居场景”；如果是楼梯垫，必须输出“复式家庭、老人家庭、儿童家庭、养宠家庭”。
3. materials 自动提取材质、工艺、颜色、外观特征；无法确定时用“图片可见/建议复核”的谨慎表述。
4. functions 自动提取商品功能，例如防滑、防水、收纳、保暖、装饰、保护、清洁等，只写商品真实功能。
5. use_scenes 输出 3-6 个使用场景，例如家庭楼梯、玄关、厨房、办公室、车内等，不要和 audience 混淆。
6. core_keywords 输出 3-8 个核心关键词，long_tail_keywords 输出 5-10 个长尾关键词，必须围绕商品名称、材质、功能、场景。
7. core_selling_points 输出 3-5 条核心卖点；five_point_description 必须输出 5 条五点描述；selling_points 输出 5-8 条短句。
8. specifications 使用对象结构，包含品类、材质、颜色、尺寸/容量/型号、适用场景等可识别规格。
9. detail_page_modules 和 detail_modules 都为数组，每项包含 title 和 content，覆盖首屏主张、核心卖点、材质工艺、功能场景、参数说明、购买理由、结尾转化模块。
10. 淘宝标题偏搜索关键词和属性组合；拼多多标题偏实惠、家用、套装/多件等真实购买语境；抖店标题偏短视频种草和痛点解决。三类标题都要自然可读，像真实商家上架标题，不能写成品牌广告语。
11. 严格 JSON：只能输出一个 JSON 对象，双引号包裹 key 和字符串，不要 Markdown，不要代码块，不要解释文字。
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
