# AI 电商详情页生成器

基于 Vue 3 + Vite + Element Plus + FastAPI + SQLite 的电商商品详情页生成工具。上传商品图片后，后端调用本地 Ollama 或 OpenAI Vision 识别商品信息，并生成淘宝、拼多多、抖店三套详情页 HTML、平台标题、卖点文案和场景图 Prompt。

## v1.3 功能

- 新增 AI 场景图 Prompt 生成：根据商品识别结果自动生成 2-4 条可用于后续 AI 生图的场景 Prompt。
- 支持场景图风格：北欧风、现代风、轻奢风、母婴风、家居实拍风。
- 前端新增“场景图 Prompt”展示区域，可直接复制每条 Prompt。
- ZIP 导出新增 `scene_prompts.txt` 和 `scenes/README.txt`。
- 当前版本不接入真实生图 API，只生成 Prompt 和场景图占位说明文件。
- 保持 v1.2 的淘宝、拼多多、抖店模板功能正常。

## v1.2 功能

- 新增平台模板引擎：淘宝详情页、拼多多详情页、抖店详情页。
- 前端支持平台选择：淘宝 / 拼多多 / 抖店。
- 前端支持风格选择：简约 / 高端 / 促销 / 场景化。
- 后端按平台生成不同 HTML：
  - 淘宝：关键词丰富、参数清晰、详情模块完整。
  - 拼多多：价格利益点突出、实惠感强。
  - 抖店：短视频种草风格、卖点强、节奏快。
- 每次成功生成都会保存三个平台的独立 HTML。
- 单个 HTML 导出会按当前选择的平台下载 `taobao.html`、`pdd.html` 或 `douyin.html`。
- ZIP 导出固定包含：
  - `taobao.html`
  - `pdd.html`
  - `douyin.html`
  - `titles.txt`
  - `selling_points.txt`
  - `scene_prompts.txt`
  - `scenes/README.txt`
- 保持 v1.1 批量生成功能：一次上传多张图片、逐张生成、单张失败不影响后续图片。

## 目录结构

```text
backend/             FastAPI 后端
backend/app/         API、配置、数据库、Ollama/OpenAI 服务、HTML 模板引擎
frontend/            Vue 3 前端
requirements.txt     后端依赖
docker-compose.yml   Docker 编排
.env.example         环境变量示例
```

## 环境变量

复制示例文件：

```bash
copy .env.example .env
```

填写 `.env`：

```env
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5vl:7b
OLLAMA_TIMEOUT=180
OPENAI_API_KEY=你的 OpenAI API Key
OPENAI_MODEL=gpt-4o-mini
BACKEND_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
DATABASE_PATH=backend/data/app.db
UPLOAD_DIR=backend/uploads
```

默认使用本地 Ollama。切换到 OpenAI 时设置 `AI_PROVIDER=openai` 并填写 `OPENAI_API_KEY`。

## Ollama 准备

确保本机 Ollama 已启动，并拉取视觉模型：

```bash
ollama pull qwen2.5vl:7b
ollama serve
```

## 本地启动

启动后端：

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r ..\requirements.txt
python start.py
```

也可以从项目根目录直接启动后端（不依赖当前工作目录解析数据和静态文件）：

```bash
python -m backend.app.main
```

启动前端：

```bash
cd frontend
npm install
npm run dev
```

访问地址：

- 前端：http://localhost:5173
- 后端文档：http://localhost:8000/docs

## 使用方法

1. 在左侧上传一张或多张商品图片。
2. 填写商品名称、品类、目标人群、售价和原价。
3. 选择平台：淘宝 / 拼多多 / 抖店。
4. 选择详情页风格：简约 / 高端 / 促销 / 场景化。
5. 选择场景图风格：北欧风 / 现代风 / 轻奢风 / 母婴风 / 家居实拍风。
6. 点击生成详情页或批量生成。
7. 在右侧查看商品图片、平台标题、卖点、场景图 Prompt 和 HTML 详情页预览。
8. 点击导出 HTML 下载当前平台的独立 HTML。
9. 点击下载全部 ZIP 导出三平台 HTML、标题文本、卖点文本、场景 Prompt 和场景图占位目录说明。

## 主要接口

- `POST /api/generate`：上传一张或多张图片并生成识别结果、标题、卖点、场景图 Prompt 和三平台 HTML。
- `GET /api/generations`：查看历史记录。
- `GET /api/generations/{id}`：查看单条记录。
- `GET /api/generations/{id}/html?platform=taobao`：查看指定平台 HTML。
- `GET /api/generations/{id}/export?platform=taobao`：下载指定平台 HTML。
- `GET /api/generations/export.zip?ids=1,2,3`：下载 ZIP，包含三平台 HTML、`titles.txt`、`selling_points.txt`、`scene_prompts.txt` 和 `scenes/README.txt`。
- `GET /api/uploads/{filename}`：访问上传图片。

## 验证命令

```bash
python -m compileall backend\app
cd frontend
npm.cmd run build
```

## Docker 启动

```bash
copy .env.example .env
docker compose up --build
```

Docker 前端使用 Nginx 提供生产构建文件，并将 `/api` 和 `/static` 反向代理到后端。
容器内的 Ollama 地址由 Compose 设置为 `http://host.docker.internal:11434`。

## 存储清理（默认只读）

以下命令只报告无引用上传文件和超过 20 MB 的旧记录，不会删除任何内容：

```bash
python -m backend.scripts.cleanup_storage
```

确需清理时必须显式提供删除范围、`--apply` 和 `--yes`。删除大记录前脚本会先备份数据库：

```bash
python -m backend.scripts.cleanup_storage --delete-orphans --apply --yes
python -m backend.scripts.cleanup_storage --delete-large-records --max-record-mb 20 --vacuum --apply --yes
```

## 自动化测试

```bash
pip install -r requirements-dev.txt
python -m pytest backend/tests -q
```
