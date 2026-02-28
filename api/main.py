from fastapi import FastAPI
from api.routes import router
from database.models import create_tables

app = FastAPI()

create_tables()

app.include_router(router)