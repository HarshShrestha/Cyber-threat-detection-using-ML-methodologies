from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend.routes.api import router

ROOT = Path(__file__).resolve().parent
app = FastAPI(title='Network Flow Lab', version='1.0.0', description='Experimental intrusion detection from NetFlow features.')
app.include_router(router)
app.mount('/assets', StaticFiles(directory=ROOT / 'frontend' / 'static'), name='assets')

@app.get('/', include_in_schema=False)
def home():
    return FileResponse(ROOT / 'frontend' / 'templates' / 'index.html')
