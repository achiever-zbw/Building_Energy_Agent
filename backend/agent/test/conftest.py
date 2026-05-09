"""pytest 固定装置：仓库根目录加入 sys.path（与其它 backend 测试一致）。"""
from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_ROOT = _REPO_ROOT / "backend"
# 先于用例读取环境变量（pytest 默认不会加载 .env）
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_BACKEND_ROOT / ".env")

for _p in (_REPO_ROOT, _BACKEND_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: 真实 HTTP / LLM（例如压缩归档）；依赖 .env 或环境中的 OPENAI_API_KEY / DEEPSEEK_API_KEY",
    )


@pytest.fixture
def agent_workspace() -> Path:
    """可写 workspace（建在仓库内，避免 pytest /tmp 在沙箱下不可写）。"""
    uid = uuid.uuid4().hex[:12]
    base = _REPO_ROOT / ".pytest_backend_ws" / uid
    base.mkdir(parents=True)
    ws = base / "workspace"
    ws.mkdir()
    (ws / "memory").mkdir(parents=True)
    try:
        yield ws
    finally:
        shutil.rmtree(base, ignore_errors=True)
