# AI 电商详情页生成器

基于 Vue 3 + Vite + Element Plus + FastAPI + SQLite 的电商商品详情页生成工具。上传商品图片后，后端默认调用本地 Ollama `qwen2.5vl:7b` 识别商品信息，并自动生成淘宝、拼多多、抖店标题、卖点文案、详情页模块文案和可下载的 HTML 详情页。项目仍保留 OpenAI Vision 配置，可通过环境变量切换。

## 功能

- 上传单张或多张商品图片
- 本地 Ollama `qwen2.5vl:7b` 默认识别商品名称、品类、卖点、使用场景、材质和规格
- 保留 OpenAI Vision 支持，可按需切换
- 自动生成淘宝标题、拼多多标题、抖店标题
- 自动生成商品卖点文案和详情页模块文案
- 自动生成 HTML 详情页，包含首屏主图、核心卖点、场景展示、参数介绍、购买理由、结尾营销模块
- 前端支持加载动画、生成进度、一键复制标题、复制 HTML、导出 HTML、下载详情页
- SQLite 保存历史生成记录

## 目录结构

```text
backend/             FastAPI 后端
backend/app/         API、配置、数据库、Ollama/OpenAI 服务、HTML 生成器
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

如果 Ollama 服务不可用、模型未拉取或 OpenAI 未配置，后端会返回本地降级示例内容，便于验证项目是否可运行。

## Ollama 准备

确保本机 Ollama 已启动，并拉取视觉模型：

```bash
ollama pull qwen2.5vl:7b
ollama serve
```

后端会调用 `http://localhost:11434/api/generate`，把上传的商品图片传给 `qwen2.5vl:7b` 分析。

## 本地启动

启动后端：

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r ..\requirements.txt
python start.py
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

## Docker 启动

```bash
copy .env.example .env
docker compose up --build
```

## 主要接口

- `POST /api/generate`：上传图片并生成识别结果、标题、文案和 HTML
- `GET /api/generations`：查看历史记录
- `GET /api/generations/{id}`：查看单条记录
- `GET /api/generations/{id}/export`：下载 HTML 详情页
- `GET /api/uploads/{filename}`：访问上传图片

## 使用流程

1. 上传商品图片。
2. 可选填写商品名称、品类、目标人群、售价、原价。
3. 点击“生成详情页”。
4. 在结果区复制三平台标题，查看识别信息、卖点文案和 HTML 预览。
5. 点击“导出 HTML”或“下载详情页”保存生成结果。
