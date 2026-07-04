from html import escape
from typing import Any


PLATFORMS = {
    "taobao": "淘宝",
    "pdd": "拼多多",
    "douyin": "抖店",
}

STYLES = {
    "simple": "简约",
    "premium": "高端",
    "promotion": "促销",
    "scene": "场景化",
}

PLATFORM_FILENAMES = {
    "taobao": "taobao.html",
    "pdd": "pdd.html",
    "douyin": "douyin.html",
}


def normalize_platform(value: str | None) -> str:
    return value if value in PLATFORMS else "taobao"


def normalize_style(value: str | None) -> str:
    return value if value in STYLES else "simple"


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _list_items(values: list[str]) -> str:
    return "\n".join(f"<li>{escape(item)}</li>" for item in values)


def _paragraphs(values: list[str]) -> str:
    return "\n".join(f"<p>{escape(item)}</p>" for item in values)


def _module_blocks(modules: Any) -> str:
    if not isinstance(modules, list):
        return ""
    blocks = []
    for module in modules:
        if not isinstance(module, dict):
            continue
        title = str(module.get("title", "详情模块"))
        content = str(module.get("content", ""))
        blocks.append(
            f"""
      <article class="module-card">
        <h3>{escape(title)}</h3>
        <p>{escape(content)}</p>
      </article>"""
        )
    return "\n".join(blocks)


def _spec_rows(specs: Any) -> str:
    if not isinstance(specs, dict):
        return ""
    return "\n".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in specs.items()
    )


def _style_tokens(style: str) -> dict[str, str]:
    tokens = {
        "simple": {
            "bg": "#edf1f5",
            "surface": "#ffffff",
            "ink": "#17202a",
            "muted": "#64717f",
            "accent": "#0f766e",
            "accent2": "#2563eb",
            "soft": "#f8fafc",
        },
        "premium": {
            "bg": "#f1eee9",
            "surface": "#fffdf8",
            "ink": "#1d1a16",
            "muted": "#6f665b",
            "accent": "#8a5a18",
            "accent2": "#243b53",
            "soft": "#f7f1e8",
        },
        "promotion": {
            "bg": "#fff3e8",
            "surface": "#ffffff",
            "ink": "#231815",
            "muted": "#7a5145",
            "accent": "#e11d48",
            "accent2": "#f97316",
            "soft": "#fff7ed",
        },
        "scene": {
            "bg": "#eef5ef",
            "surface": "#ffffff",
            "ink": "#17211b",
            "muted": "#5f7066",
            "accent": "#16724f",
            "accent2": "#a16207",
            "soft": "#f6fbf7",
        },
    }
    return tokens[normalize_style(style)]


def _platform_copy(platform: str, analysis: dict[str, Any], price: str, origin_price: str) -> dict[str, Any]:
    product = str(analysis.get("product_name") or "商品详情页")
    category = str(analysis.get("category") or "精选商品")
    tagline = str(analysis.get("tagline") or "")
    keywords = _as_list(analysis.get("core_keywords")) + _as_list(analysis.get("long_tail_keywords"))
    selling_points = _as_list(analysis.get("core_selling_points") or analysis.get("selling_points"))
    use_scenes = _as_list(analysis.get("use_scenes"))
    materials = _as_list(analysis.get("materials"))
    selling_copy = _as_list(
        analysis.get("five_point_description") or analysis.get("selling_copy") or analysis.get("detail_copy")
    )
    buying_reasons = _as_list(analysis.get("buying_reasons"))
    modules = analysis.get("detail_page_modules") or analysis.get("detail_modules")

    if platform == "pdd":
        title = str(analysis.get("pdd_title") or product)
        hero = "实惠好物，价格利益点一眼看清"
        badges = [item for item in [f"到手价 {price}" if price else "", "多件更划算", "家用实用款"] if item]
        sections = {
            "points": "为什么划算",
            "scene": "适合这些日常场景",
            "specs": "规格参数，买前看清",
            "copy": "实惠卖点说明",
            "reasons": "下单理由",
            "closing": "限时转化模块",
        }
    elif platform == "douyin":
        title = str(analysis.get("douyin_title") or product)
        hero = "短视频种草节奏，3 秒抓住核心卖点"
        badges = [item for item in [f"现在 {price}" if price else "", "高频卖点", "场景化种草"] if item]
        sections = {
            "points": "先看爆点",
            "scene": "镜头场景",
            "specs": "关键信息",
            "copy": "种草脚本式卖点",
            "reasons": "立刻想买的理由",
            "closing": "直播间收口",
        }
    else:
        title = str(analysis.get("taobao_title") or product)
        hero = "关键词丰富、参数清晰、详情模块完整"
        badges = [item for item in [category, "搜索关键词覆盖", "参数完整展示"] if item]
        sections = {
            "points": "核心卖点",
            "scene": "使用场景",
            "specs": "参数介绍",
            "copy": "商品卖点文案",
            "reasons": "购买理由",
            "closing": "详情页模块",
        }

    if origin_price and platform == "pdd":
        badges.append(f"参考价 {origin_price}")

    return {
        "product": product,
        "category": category,
        "title": title,
        "hero": hero,
        "tagline": tagline,
        "badges": badges,
        "keywords": keywords,
        "selling_points": selling_points,
        "use_scenes": use_scenes,
        "materials": materials,
        "selling_copy": selling_copy,
        "buying_reasons": buying_reasons,
        "modules": modules,
        "sections": sections,
    }


def build_detail_html(
    analysis: dict[str, Any],
    image_url: str,
    platform: str = "taobao",
    style: str = "simple",
    price: str = "",
    origin_price: str = "",
) -> str:
    platform = normalize_platform(platform)
    style = normalize_style(style)
    tokens = _style_tokens(style)
    copy = _platform_copy(platform, analysis, price, origin_price)
    keyword_html = _list_items(copy["keywords"][:12])
    badge_html = "".join(f"<span>{escape(item)}</span>" for item in copy["badges"])
    style_name = STYLES[style]
    platform_name = PLATFORMS[platform]
    product = copy["product"]
    title = copy["title"]
    category = copy["category"]
    hero = copy["hero"]
    tagline = copy["tagline"]
    sections = copy["sections"]

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{escape(platform_name)}详情页 - {escape(product)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: {tokens["ink"]}; font-family: "PingFang SC", "Microsoft YaHei", Arial, sans-serif; background: {tokens["bg"]}; }}
    .page {{ max-width: 980px; margin: 0 auto; background: {tokens["surface"]}; }}
    .hero {{ display: grid; grid-template-columns: 1fr 1fr; min-height: 560px; background: {tokens["ink"]}; color: #fff; }}
    .hero-copy {{ padding: 62px 52px; display: flex; flex-direction: column; justify-content: center; }}
    .hero img {{ width: 100%; height: 100%; min-height: 360px; object-fit: cover; display: block; background: #e6edf2; }}
    .eyebrow {{ color: {tokens["accent2"]}; font-weight: 900; letter-spacing: 0; }}
    h1 {{ margin: 18px 0; font-size: 42px; line-height: 1.14; letter-spacing: 0; }}
    h2 {{ margin: 0 0 20px; font-size: 30px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 10px; font-size: 20px; letter-spacing: 0; }}
    p {{ margin: 0; line-height: 1.85; color: {tokens["muted"]}; }}
    .hero p {{ color: #eef2f6; font-size: 17px; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }}
    .badges span {{ padding: 8px 12px; border-radius: 8px; background: {tokens["accent"]}; color: #fff; font-weight: 800; }}
    section {{ padding: 50px 52px; border-top: 1px solid #e3e9ef; }}
    .platform-strip {{ display: flex; justify-content: space-between; gap: 14px; background: {tokens["soft"]}; color: {tokens["muted"]}; font-weight: 800; }}
    .point-grid, .scene-grid, .reason-grid, .module-grid, .keyword-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; padding: 0; margin: 0; list-style: none; }}
    .point-grid li, .scene-grid li, .reason-grid li, .module-card, .keyword-grid li {{ padding: 18px; border: 1px solid #dbe3ea; border-radius: 8px; background: {tokens["soft"]}; }}
    .point-grid li {{ font-weight: 900; color: {tokens["ink"]}; border-left: 5px solid {tokens["accent"]}; }}
    .keyword-grid li {{ font-weight: 800; color: {tokens["accent"]}; }}
    .copy-stack {{ display: grid; gap: 12px; }}
    .material-list {{ margin: 18px 0 0; padding-left: 20px; color: {tokens["muted"]}; line-height: 1.8; }}
    table {{ width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 8px; }}
    th, td {{ border: 1px solid #dbe3ea; padding: 15px; text-align: left; line-height: 1.7; }}
    th {{ width: 170px; background: {tokens["soft"]}; color: {tokens["muted"]}; }}
    .closing {{ background: {tokens["soft"]}; }}
    .closing h2 {{ color: {tokens["accent"]}; }}
    @media (max-width: 760px) {{
      .hero {{ grid-template-columns: 1fr; }}
      .hero-copy, section {{ padding: 34px 24px; }}
      h1 {{ font-size: 31px; }}
      .point-grid, .scene-grid, .reason-grid, .module-grid, .keyword-grid {{ grid-template-columns: 1fr; }}
      .platform-strip {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="platform-strip">
      <span>{escape(platform_name)}详情页模板</span>
      <span>{escape(style_name)}风格</span>
    </section>
    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">{escape(category)}</div>
        <h1>{escape(title)}</h1>
        <p>{escape(hero)}。{escape(tagline)}</p>
        <div class="badges">{badge_html}</div>
      </div>
      <img src="{escape(image_url)}" alt="{escape(product)}" />
    </section>

    <section>
      <h2>{escape(sections["points"])}</h2>
      <ul class="point-grid">{_list_items(copy["selling_points"])}</ul>
    </section>

    <section>
      <h2>{escape(sections["scene"])}</h2>
      <ul class="scene-grid">{_list_items(copy["use_scenes"])}</ul>
    </section>

    <section>
      <h2>搜索关键词</h2>
      <ul class="keyword-grid">{keyword_html}</ul>
    </section>

    <section>
      <h2>{escape(sections["specs"])}</h2>
      <table>{_spec_rows(analysis.get("specifications") or analysis.get("specs"))}</table>
      <ul class="material-list">{_list_items(copy["materials"])}</ul>
    </section>

    <section>
      <h2>{escape(sections["copy"])}</h2>
      <div class="copy-stack">{_paragraphs(copy["selling_copy"])}</div>
    </section>

    <section>
      <h2>{escape(sections["reasons"])}</h2>
      <ul class="reason-grid">{_list_items(copy["buying_reasons"])}</ul>
    </section>

    <section class="closing">
      <h2>{escape(sections["closing"])}</h2>
      <div class="module-grid">{_module_blocks(copy["modules"])}</div>
    </section>
  </main>
</body>
</html>"""


def build_platform_htmls(
    analysis: dict[str, Any],
    image_url: str,
    style: str = "simple",
    price: str = "",
    origin_price: str = "",
) -> dict[str, str]:
    return {
        platform: build_detail_html(analysis, image_url, platform, style, price, origin_price)
        for platform in PLATFORMS
    }
