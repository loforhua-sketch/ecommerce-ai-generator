import base64
import json
from pathlib import Path
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """你是资深中国电商运营、视觉识别和详情页策划。
请根据商品图片和补充信息识别商品，并生成可直接用于淘宝、拼多多、抖店上架的商品标题、卖点文案和详情页模块文案。
标题必须符合真实中国电商风格：像淘宝、拼多多、抖店商家真实上架标题，不写广告口号式标题，不堆砌无关词。
输出必须是严格 JSON，只能输出一个 JSON 对象；不要输出 Markdown，不要用代码块包裹 JSON，不要包含注释或额外解释。
不要包含虚假极限词、医疗功效承诺、绝对化承诺，或无法从图片/补充信息合理推断的参数。"""


REQUIRED_FIELDS = {
    "product_name": "",
    "category": "",
    "audience": "",
    "tagline": "",
    "materials": [],
    "functions": [],
    "specifications": {},
    "use_scenes": [],
    "core_keywords": [],
    "long_tail_keywords": [],
    "core_selling_points": [],
    "five_point_description": [],
    "selling_points": [],
    "taobao_title": "",
    "pdd_title": "",
    "douyin_title": "",
    "selling_copy": [],
    "detail_page_modules": [],
    "detail_modules": [],
    "buying_reasons": [],
}


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _fallback(product_name: str, category: str, audience: str, price: str = "", origin_price: str = "") -> dict[str, Any]:
    name = product_name or "AI 识别商品"
    product_category = category or "电商商品"
    target = audience or "日常消费人群"
    specs = {
        "商品名称": name,
        "商品品类": product_category,
        "适用人群": target,
    }
    if price:
        specs["售价"] = price
    if origin_price:
        specs["参考原价"] = origin_price

    data = {
        "product_name": name,
        "category": product_category,
        "audience": target,
        "tagline": f"围绕{target}的真实需求，生成更适合上架转化的商品详情页。",
        "materials": ["请结合实物图片复核材质", "支持在生成后手动补充规格"],
        "functions": ["满足日常使用需求", "适合电商详情页展示"],
        "specifications": specs,
        "use_scenes": ["居家日用", "送礼自用", "店铺新品上架"],
        "core_keywords": [name, product_category],
        "long_tail_keywords": [f"{name}家用", f"{name}实用款", f"{product_category}新品上架"],
        "core_selling_points": ["主体清晰，适合详情页首屏展示", "卖点结构完整，便于多平台上架", "参数和场景模块可继续编辑"],
        "five_point_description": [
            "商品主体清晰，便于用户快速识别。",
            "围绕材质、功能和使用场景组织卖点。",
            "标题适配淘宝、拼多多、抖店不同平台。",
            "参数信息可用于详情页基础展示。",
            "详情页模块支持导出后继续编辑。",
        ],
        "selling_points": ["主体清晰，适合详情页首屏展示", "卖点结构完整，便于多平台上架", "参数和场景模块可继续编辑", "支持一键导出 HTML 详情页"],
        "taobao_title": f"{name} 实用高颜值商品 家用场景适配 新品推荐",
        "pdd_title": f"{name} 实惠好物 家用实用 多件可选 新品上架",
        "douyin_title": f"{name} 今日好物推荐 高颜值实用款",
        "selling_copy": [
            f"{name}聚焦外观、功能和使用场景，帮助用户快速理解商品价值。",
            "从核心卖点到参数说明完整呈现，适合用于商品详情页基础稿。",
            "标题按淘宝、拼多多、抖店不同平台的搜索和展示习惯生成。",
        ],
        "detail_modules": [
            {"title": "首屏主张", "content": f"{name}，为{target}打造的实用商品。"},
            {"title": "核心卖点", "content": "围绕外观、功能、场景和规格组织信息，降低用户决策成本。"},
            {"title": "场景展示", "content": "适合日常使用、店铺陈列、活动上新等常见电商场景。"},
            {"title": "参数介绍", "content": "商品名称、品类、适用人群、价格等信息清晰展示。"},
        ],
        "buying_reasons": ["信息完整，方便快速上架", "多平台标题可直接复制", "详情页 HTML 可下载二次编辑"],
    }
    data["detail_page_modules"] = data["detail_modules"]
    return data


def _normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_modules(value: Any) -> list[dict[str, str]]:
    modules: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                title = str(item.get("title", "")).strip()
                content = str(item.get("content", "")).strip()
                if title or content:
                    modules.append({"title": title or "详情模块", "content": content})
            elif str(item).strip():
                modules.append({"title": "详情模块", "content": str(item).strip()})
    return modules


def _merge_with_fallback(data: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = {**REQUIRED_FIELDS, **fallback, **data}
    merged["materials"] = _normalize_list(merged.get("materials")) or fallback["materials"]
    merged["functions"] = _normalize_list(merged.get("functions")) or fallback["functions"]
    merged["use_scenes"] = _normalize_list(merged.get("use_scenes")) or fallback["use_scenes"]
    merged["core_keywords"] = _normalize_list(merged.get("core_keywords")) or fallback["core_keywords"]
    merged["long_tail_keywords"] = _normalize_list(merged.get("long_tail_keywords")) or fallback["long_tail_keywords"]
    merged["core_selling_points"] = _normalize_list(merged.get("core_selling_points")) or fallback["core_selling_points"]
    merged["five_point_description"] = _normalize_list(merged.get("five_point_description")) or fallback["five_point_description"]
    merged["selling_points"] = _normalize_list(merged.get("selling_points")) or fallback["selling_points"]
    merged["selling_copy"] = _normalize_list(merged.get("selling_copy")) or fallback["selling_copy"]
    merged["buying_reasons"] = _normalize_list(merged.get("buying_reasons")) or fallback["buying_reasons"]
    merged["detail_modules"] = _normalize_modules(merged.get("detail_modules")) or fallback["detail_modules"]
    merged["detail_page_modules"] = _normalize_modules(merged.get("detail_page_modules")) or merged["detail_modules"]
    if not isinstance(merged.get("specifications"), dict):
        merged["specifications"] = fallback["specifications"]
    _apply_product_rules(merged)
    return merged


def _apply_product_rules(data: dict[str, Any]) -> None:
    searchable = " ".join(
        str(data.get(key, ""))
        for key in ("product_name", "category", "taobao_title", "pdd_title", "douyin_title")
    )
    if "楼梯垫" not in searchable and "楼梯踏步垫" not in searchable:
        return

    data["product_name"] = "防滑楼梯垫"
    data["audience"] = "复式家庭、老人家庭、儿童家庭、养宠家庭"
    data["core_keywords"] = ["防滑楼梯垫", "楼梯垫", "楼梯踏步垫", "楼梯防滑垫"]
    data["long_tail_keywords"] = [
        "家用防滑楼梯垫",
        "自粘楼梯踏步垫",
        "复式楼梯防滑垫",
        "老人儿童楼梯防滑垫",
        "养宠家庭楼梯垫",
    ]


class OpenAIProductService:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key.strip()
        self.model = model
        has_real_key = bool(self.api_key) and self.api_key != "sk-your-openai-api-key"
        self.client = OpenAI(api_key=self.api_key) if has_real_key else None

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
        if self.client is None:
            return fallback

        user_prompt = f"""
补充信息：
- 商品名称：{product_name or "请从图片识别"}
- 商品品类：{category or "请从图片识别"}
- 目标人群：{audience or "请根据图片推断"}
- 售价：{price or "未知"}
- 原价：{origin_price or "未知"}

请返回 JSON，字段必须包含：
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
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                    ],
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {}
        return _merge_with_fallback(data, fallback)
