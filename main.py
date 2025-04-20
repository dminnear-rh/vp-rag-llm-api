from fastapi import FastAPI
from router import router

from config import AppConfig

app = FastAPI()
config = AppConfig.load()
app.state.config = config

app.include_router(router)
