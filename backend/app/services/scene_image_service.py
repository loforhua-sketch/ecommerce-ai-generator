def generate_scene_images(product_name: str, scene_style: str):
    base_url = "/static/scenes/"

    return [
        {
            "url": base_url + "scene1.jpg",
            "prompt": f"{product_name} - {scene_style} 场景1"
        },
        {
            "url": base_url + "scene2.jpg",
            "prompt": f"{product_name} - {scene_style} 场景2"
        },
        {
            "url": base_url + "scene3.jpg",
            "prompt": f"{product_name} - {scene_style} 场景3"
        }
    ]
