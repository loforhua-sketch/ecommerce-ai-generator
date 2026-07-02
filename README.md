# 电商 AI 标题与详情页生成器

完整项目包含 Vue3 + Vite + Element Plus 前端、FastAPI 后端、SQLite 数据库。支持单张或批量上传商品图片，调用 OpenAI API 识别商品与卖点，生成淘宝、拼多多、抖店标题、详情页文案和可导出的 HTML 详情页。

## 目录结构

```text
frontend/          Vue3 + Vite + Element Plus 前端
backend/           Python + FastAPI 后端
requirements.txt   后端依赖
docker-compose.yml Docker 编排
.env.example       环境变量示例
```

## 本地启动

1. 复制环境变量：

```bash
copy .env.example .env
```

2. 编辑 `.env`，填入 `OPENAI_API_KEY`。

3. 启动后端：

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r ..\requirements.txt
python start.py
```

4. 启动前端：

```bash
cd frontend
npm install
npm run dev
```

前端地址：http://localhost:5173
后端 API：http://localhost:8000/docs

## Docker 启动

```bash
copy .env.example .env
docker compose up --build
```

## 功能

- 上传产品图片
- OpenAI API 识别商品图片和卖点
- 自动生成淘宝标题
- 自动生成拼多多标题
- 自动生成抖店标题
- 自动生成详情页文案
- 自动生成 HTML 详情页
- 导出 HTML 文件
- 批量生成并保存历史记录

