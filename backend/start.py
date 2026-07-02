import uvicorn
from pathlib import Path


if __name__ == "__main__":
    app_dir = str(Path(__file__).resolve().parent)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, app_dir=app_dir)
