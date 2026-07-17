from typing import Any


SCENE_STYLES = {
    "nordic": "北欧风",
    "modern": "现代风",
    "light_luxury": "轻奢风",
    "maternal_baby": "母婴风",
    "home_real": "家居实拍风",
}

STYLE_DESCRIPTIONS = {
    "nordic": "北欧原木家居空间，浅木色、白色和柔和织物搭配，干净自然",
    "modern": "现代家居空间，简洁线条、中性色搭配，自然光，高级感",
    "light_luxury": "轻奢家居空间，精致软装、金属细节、低饱和高级配色",
    "maternal_baby": "母婴友好家居空间，柔和采光，安全温馨，亲子生活氛围",
    "home_real": "真实家居实拍空间，生活化陈设，自然光，电商详情页摄影质感",
}

DEFAULT_SCENES = ["客厅", "卧室", "厨房", "玄关"]


def normalize_scene_style(value: str | None) -> str:
    return value if value in SCENE_STYLES else "modern"


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _pick_scenes(analysis: dict[str, Any]) -> list[str]:
    scenes = _as_list(analysis.get("use_scenes"))
    if not scenes:
        scenes = DEFAULT_SCENES
    scenes = scenes[:4]
    if len(scenes) == 1:
        scenes.append(DEFAULT_SCENES[0] if scenes[0] != DEFAULT_SCENES[0] else DEFAULT_SCENES[1])
    return scenes[:4]


def _join_features(values: list[str], limit: int = 3) -> str:
    return "、".join(values[:limit])


def build_scene_prompts(analysis: dict[str, Any], scene_style: str = "modern") -> list[dict[str, str]]:
    scene_style = normalize_scene_style(scene_style)
    product = str(analysis.get("product_name") or "商品").strip()
    category = str(analysis.get("category") or "").strip()
    style_name = SCENE_STYLES[scene_style]
    style_description = STYLE_DESCRIPTIONS[scene_style]
    materials = _as_list(analysis.get("materials"))
    selling_points = _as_list(analysis.get("core_selling_points") or analysis.get("selling_points"))
    functions = _as_list(analysis.get("functions"))
    feature_bits = _join_features(materials + functions + selling_points)
    feature_text = f"，突出{feature_bits}" if feature_bits else ""

    prompts: list[dict[str, str]] = []
    for index, scene in enumerate(_pick_scenes(analysis), start=1):
        product_label = f"{category}{product}" if category and category not in product else product
        prompt = (
            f"一张真实家居场景摄影图，{style_description}，{scene}场景，"
            f"{product_label}自然摆放并清晰展示{feature_text}，自然光，画面干净，高级感，"
            "真实材质细节，电商详情页风格，适合商品主图和详情页场景图，"
            "无文字，无水印，无夸张变形"
        )
        prompts.append(
            {
                "index": str(index),
                "style": style_name,
                "scene": scene,
                "prompt": prompt,
            }
        )
    return prompts


def scene_prompts_to_text(items: list[dict[str, Any]]) -> str:
    blocks = []
    for item in items:
        analysis = item.get("analysis", {})
        product_name = analysis.get("product_name") or item.get("product_name") or item.get("filename")
        prompts = analysis.get("scene_prompts") or build_scene_prompts(
            analysis,
            analysis.get("scene_style", "modern"),
        )
        lines = [f"商品：{product_name}"]
        for prompt_item in prompts:
            if isinstance(prompt_item, dict):
                label = f"{prompt_item.get('index', '')}. {prompt_item.get('style', '')} / {prompt_item.get('scene', '')}".strip()
                prompt = str(prompt_item.get("prompt", "")).strip()
            else:
                label = ""
                prompt = str(prompt_item).strip()
            if prompt:
                lines.append(f"{label}\n{prompt}" if label else prompt)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks).strip() + "\n"


def scenes_readme_text() -> str:
    return (
        "scenes/ 为 v1.3 场景图占位目录。\n"
        "当前版本不接入真实 AI 生图 API，因此 ZIP 内不会生成实际图片。\n"
        "请使用根目录 scene_prompts.txt 中的 Prompt，在后续生图工具中生成场景图后放入本目录。\n"
        "建议命名：scene_01.jpg、scene_02.jpg、scene_03.jpg、scene_04.jpg。\n"
    )
