from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ 目录（本文件在 backend/core/ 下）
_BACKEND_DIR = Path(__file__).resolve().parent.parent
# 项目根目录（A08project/）
_PROJECT_ROOT = _BACKEND_DIR.parent

# 无论从哪启动，都优先读这两个位置的 .env（先 backend，再根目录）
_ENV_FILES = tuple(
    p
    for p in (
        _BACKEND_DIR / ".env",
        _PROJECT_ROOT / ".env",
    )
    if p.is_file()
)


class Settings(BaseSettings):
    PROJECT_NAME: str = "建筑能源智能管理系统"
    DATABASE_URL: str
    SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    # MCP 子进程用 stdout 传 JSON-RPC，echo=True 会把 SQL 打到 stdout 导致协议错乱
    SQLALCHEMY_ECHO: bool = False

    model_config = SettingsConfigDict(
        env_file=_ENV_FILES if _ENV_FILES else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()