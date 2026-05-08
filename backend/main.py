import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from backend.agent.loop import AgentLoop
from backend.agent.providers.openai_provider import OpenAIProvider
from backend.agent.tools.tools import build_default_tool_registry
from backend.utils.helpers import ensure_dir
from db.session import async_engine
import models.energy
import models.building
import models.weather
from db.base import Base
from api.routes.upload import router as upload_router
from api.routes.chat import router as chat_router
from api.routes.search import router as search_router
from api.routes.report import router as report_router
# from api.routes.forecast import router as forecast_router
from dotenv import load_dotenv
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    workspace = Path(__file__).resolve().parent / "workspace"
    ensure_dir(workspace)
    app.state.chat_sessions = {}

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")
    model = os.environ.get("OPENAI_MODEL")

    if api_key:
        provider = OpenAIProvider(
            api_key=api_key,
            api_base=api_base if api_base else None,
            default_model=model or "deepseek-chat",
        )
        app.state.agent_loop = AgentLoop(
            provider=provider,
            workspace=workspace,
            tools=build_default_tool_registry(),
        )
    else:
        app.state.agent_loop = None
        warnings.warn("未设置 OPENAI_API_KEY  POST /api/v1/chat 将返回 503")

    yield


app = FastAPI(
    title="建筑能源智能管理系统" , 
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(report_router, prefix="/api/v1")
# app.include_router(forecast_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "欢迎使用建筑能源智能管理系统"}

if __name__ == '__main__':
    uvicorn.run(
        "main:app" , 
        host="0.0.0.0" , 
        port=8000 , 
        reload=False
    )