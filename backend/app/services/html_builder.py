from html import escape
from typing import Any


def _items(values: list[str]) -> str:
    return "\n".join(f"<li>{escape(item)}</li>" for item in values)


def build_detail_html(analysis: dict[str, Any], image_url: str) -> str:
    product = analysis.get("product_name", "商品详情页")
    tagline = analysis.get("tagline", "")
    selling_points = analysis.get("selling_points", [])
    paragraphs = analysis.get("detail_copy", [])
    specs = analysis.get("specs", {})

    spec_rows = "\n".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in specs.items()
    )
    copy_blocks = "\n".join(f"<p>{escape(text)}</p>" for text in paragraphs)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{escape(product)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: #17202a; font-family: "PingFang SC", "Microsoft YaHei", Arial, sans-serif; background: #f3f6f8; }}
    .page {{ max-width: 980px; margin: 0 auto; background: #fff; }}
    .hero {{ display: grid; grid-template-columns: 1fr 1fr; min-height: 520px; }}
    .hero-copy {{ padding: 64px 48px; display: flex; flex-direction: column; justify-content: center; }}
    .hero img {{ width: 100%; height: 100%; object-fit: cover; display: block; background: #e8eef2; }}
    .eyebrow {{ color: #0f766e; font-weight: 700; }}
    h1 {{ margin: 16px 0; font-size: 42px; line-height: 1.15; }}
    h2 {{ margin: 0 0 18px; font-size: 28px; }}
    p {{ line-height: 1.8; color: #435363; }}
    section {{ padding: 46px 48px; border-top: 1px solid #e4e9ee; }}
    ul {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; padding: 0; list-style: none; }}
    li {{ padding: 16px; border: 1px solid #d9e1e8; border-radius: 8px; font-weight: 700; background: #f8fafc; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #d9e1e8; padding: 14px; text-align: left; }}
    th {{ width: 160px; background: #f8fafc; color: #64717f; }}
    @media (max-width: 760px) {{ .hero {{ grid-template-columns: 1fr; }} ul {{ grid-template-columns: 1fr; }} .hero-copy, section {{ padding: 32px 24px; }} h1 {{ font-size: 32px; }} }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">{escape(analysis.get("category", "精选商品"))}</div>
        <h1>{escape(product)}</h1>
        <p>{escape(tagline)}</p>
      </div>
      <img src="{escape(image_url)}" alt="{escape(product)}" />
    </section>
    <section>
      <h2>核心卖点</h2>
      <ul>{_items(selling_points)}</ul>
    </section>
    <section>
      <h2>详情页文案</h2>
      {copy_blocks}
    </section>
    <section>
      <h2>商品参数</h2>
      <table>{spec_rows}</table>
    </section>
  </main>
</body>
</html>"""

