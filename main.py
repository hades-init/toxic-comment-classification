from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from src.api.app import app

app.mount('/', StaticFiles(directory='web', html=True), name='static')

if __name__  == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
