from html import escape
from typing import Any


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


def build_detail_html(analysis: dict[str, Any], image_url: str) -> str:
    product = str(analysis.get("product_name") or "商品详情页")
    category = str(analysis.get("category") or "精选商品")
    tagline = str(analysis.get("tagline") or "")
    selling_points = _as_list(analysis.get("selling_points"))
    use_scenes = _as_list(analysis.get("use_scenes"))
    materials = _as_list(analysis.get("materials"))
    selling_copy = _as_list(analysis.get("selling_copy") or analysis.get("detail_copy"))
    buying_reasons = _as_list(analysis.get("buying_reasons"))

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{escape(product)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: #16202a; font-family: "PingFang SC", "Microsoft YaHei", Arial, sans-serif; background: #eef2f6; }}
    .page {{ max-width: 980px; margin: 0 auto; background: #fff; }}
    .hero {{ display: grid; grid-template-columns: 1fr 1fr; min-height: 560px; background: #101820; color: #fff; }}
    .hero-copy {{ padding: 64px 52px; display: flex; flex-direction: column; justify-content: center; }}
    .hero img {{ width: 100%; height: 100%; min-height: 360px; object-fit: cover; display: block; background: #e6edf2; }}
    .eyebrow {{ color: #7dd3c7; font-weight: 800; letter-spacing: .04em; }}
    h1 {{ margin: 18px 0; font-size: 44px; line-height: 1.12; letter-spacing: 0; }}
    h2 {{ margin: 0 0 20px; font-size: 30px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 10px; font-size: 20px; letter-spacing: 0; }}
    p {{ margin: 0; line-height: 1.85; color: #485666; }}
    .hero p {{ color: #d7e1ea; font-size: 17px; }}
    section {{ padding: 50px 52px; border-top: 1px solid #e3e9ef; }}
    .point-grid, .scene-grid, .reason-grid, .module-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; padding: 0; margin: 0; list-style: none; }}
    .point-grid li, .scene-grid li, .reason-grid li, .module-card {{ padding: 18px; border: 1px solid #dbe3ea; border-radius: 8px; background: #f8fafc; }}
    .point-grid li {{ font-weight: 800; color: #18212b; }}
    .copy-stack {{ display: grid; gap: 12px; }}
    .material-list {{ margin: 18px 0 0; padding-left: 20px; color: #485666; line-height: 1.8; }}
    table {{ width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 8px; }}
    th, td {{ border: 1px solid #dbe3ea; padding: 15px; text-align: left; line-height: 1.7; }}
    th {{ width: 170px; background: #f8fafc; color: #64717f; }}
    .closing {{ background: #f7efe7; }}
    .closing h2 {{ color: #9a4f12; }}
    @media (max-width: 760px) {{
      .hero {{ grid-template-columns: 1fr; }}
      .hero-copy, section {{ padding: 34px 24px; }}
      h1 {{ font-size: 32px; }}
      .point-grid, .scene-grid, .reason-grid, .module-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">{escape(category)}</div>
        <h1>{escape(product)}</h1>
        <p>{escape(tagline)}</p>
      </div>
      <img src="{escape(image_url)}" alt="{escape(product)}" />
    </section>

    <section>
      <h2>核心卖点</h2>
      <ul class="point-grid">{_list_items(selling_points)}</ul>
    </section>

    <section>
      <h2>场景展示</h2>
      <ul class="scene-grid">{_list_items(use_scenes)}</ul>
    </section>

    <section>
      <h2>参数介绍</h2>
      <table>{_spec_rows(analysis.get("specifications") or analysis.get("specs"))}</table>
      <ul class="material-list">{_list_items(materials)}</ul>
    </section>

    <section>
      <h2>商品卖点文案</h2>
      <div class="copy-stack">{_paragraphs(selling_copy)}</div>
    </section>

    <section>
      <h2>购买理由</h2>
      <ul class="reason-grid">{_list_items(buying_reasons)}</ul>
    </section>

    <section class="closing">
      <h2>结尾营销模块</h2>
      <div class="module-grid">{_module_blocks(analysis.get("detail_modules"))}</div>
    </section>
  </main>
</body>
</html>"""
