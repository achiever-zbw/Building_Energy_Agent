# 搭建异步数据库会话
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from core.config import settings

# 创建异步引擎
database_url = settings.DATABASE_URL
# 不打印 SQL 语句
async_engine = create_async_engine(database_url, echo=settings.SQLALCHEMY_ECHO)

# 创建异步会话工厂
async_session = async_sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)