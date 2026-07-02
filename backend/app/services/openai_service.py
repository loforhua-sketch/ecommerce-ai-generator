import base64
import json
from pathlib import Path
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """你是资深中国电商运营和商品详情页策划。
请根据商品图片和补充信息，识别商品品类、外观、材质、可能用途和核心卖点。
输出必须是 JSON，不要输出 Markdown。标题需要符合对应平台风格，但不要包含虚假极限词。"""


def _image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _fallback(product_name: str, category: str, audience: str) -> dict[str, Any]:
    name = product_name or "图片识别商品"
    return {
        "product_name": name,
        "category": category or "电商商品",
        "audience": audience or "日常消费人群",
        "tagline": f"面向{audience or '目标用户'}的高转化商品详情页文案。",
        "selling_points": ["图片主体清晰", "适合多平台上架", "详情页表达完整", "支持批量生成"],
        "taobao_title": f"{name} 高颜值实用款 官方同款 商品详情页推荐",
        "pdd_title": f"{name} 实惠好物 多件优惠 家用实用 新品",
        "douyin_title": f"{name} 今日推荐 高颜值实用好物",
        "detail_copy": [
            f"{name}围绕真实使用场景组织卖点，让用户快速理解商品价值。",
            "从外观、功能、材质和适用人群出发，生成可直接用于详情页的介绍文案。",
            "适合淘宝、拼多多、抖店等渠道快速搭建商品页并二次编辑。",
        ],
        "specs": {"品类": category or "待识别", "适用人群": audience or "目标用户"},
    }


class OpenAIProductService:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else None

    def analyze(
        self,
        image_path: Path,
        product_name: str = "",
        category: str = "",
        audience: str = "",
        price: str = "",
        origin_price: str = "",
    ) -> dict[str, Any]:
        if self.client is None:
            return _fallback(product_name, category, audience)

        user_prompt = f"""
补充信息：
- 商品名：{product_name or "请从图片识别"}
- 品类：{category or "请从图片识别"}
- 目标人群：{audience or "请根据图片推断"}
- 售价：{price or "未知"}
- 原价：{origin_price or "未知"}

请返回以下 JSON 字段：
product_name, category, audience, tagline, selling_points, taobao_title,
pdd_title, douyin_title, detail_copy, specs。
selling_points 为 4-8 个短句；detail_copy 为 3-5 段；specs 为对象。
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
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(image_path)},
                        },
                    ],
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = _fallback(product_name, category, audience)

        fallback = _fallback(product_name, category, audience)
        return {**fallback, **data}

