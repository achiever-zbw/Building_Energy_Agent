from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator
from backend.utils.helpers import ensure_dir
from contextlib import suppress
import json
from loguru import logger
from backend.utils.helpers import strip_think


class MemoryStore:

    _DEFAULT_MAX_HISTORY= 1000

    def __init__(self, workspace: Path, max_history_entries: int = _DEFAULT_MAX_HISTORY) :
        self.workspace = workspace
        self.max_history_entries = max_history_entries              # 保留最近的最大历史信息
        self.memory_dir = ensure_dir(workspace / 'memory')
        self.memory_file = self.memory_dir / "MEMORY.md"            # 长期记忆文件
        self.history_file = self.memory_dir / "history.jsonl"       # 对话历史文件
        self.legacy_history_file = self.memory_dir / "HISTORY.md"
        self.soul_file = workspace / "SOUL.md"                      # Agent 的长期风格、原则
        self.user_file = workspace / "USER.md"                      # 用户偏好风格
        self._cursor_file = self.memory_dir / ".cursor"

    @staticmethod
    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    # --- MEMORY.md
    def read_memory(self) -> str:
        return self.read_file(path = self.memory_file)

    def write_memory(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    # --- SOUL.md
    def read_soul(self) -> str:
        return self.read_file(path = self.soul_file)

    def write_soul(self, content: str) -> None:
        self.soul_file.write_text(content, encoding="utf-8")

    # --- USER.md
    def read_user(self) -> str:
        return self.read_file(path = self.user_file)

    def write_user(self, content: str) -> None:
        self.user_file.write_text(content, encoding="utf-8")

    # --- 历史记录 jsonl
    def append_history(self, entry: str) -> int:
        # 获取下一个 cursor
        cursor = self._next_cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        raw = entry.rstrip()
        # 清洗后的内容
        content = strip_think(raw)

        if raw and not content:
            logger.debug(
                "history entry {} stripped to empty (likely template leak); "
                "persisting empty content to avoid re-polluting context",
                cursor,
            )
        # 构建一条完整的信息
        record = {
            "cursor": cursor,
            "timestamp": ts,
            "content": content,
        }

        # 写入这一条历史信息
        with open(self.history_file, "a" , encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._cursor_file.write_text(str(cursor), encoding="utf-8")
        return cursor

    def _next_cursor(self) -> int:
        """ 获取下一个 cursor
        1. 最快 : 直接读取 .cursor
        2. .cursor 不能使用，可以读取最后一条 history 信息的 cursor 再 + 1
        3. 都不行，遍历所有有效记录找到最大的 cursor + 1
        """
        # 优先读 cursor 文件
        if self._cursor_file.exists():
            with suppress(ValueError, OSError):
                # 获取下一个编号,存储的是 str 类型，要转成 int
                return int(self._cursor_file.read_text(encoding="utf-8").strip()) + 1
        # cursor 文件不存在
        last_history = self._read_last_entry() or {}
        # 一条历史信息的形式 : { "cursor": 2 , "content":"xxx" } 根据 "cursor" 字段获取
        cursor = self._valid_cursor(last_history.get("cursor"))

        if cursor is not None:
            return cursor + 1
        return max((c for _, c in self._iter_valid_entries()), default=0) + 1


    def _read_last_entry(self) -> dict[str, Any] | None:
        """ 从 JSONL 文件(history.jsonl) 读取最有一条 history 信息 """
        try:
            with open(self.history_file, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return None
                read_size = min(size , 4096)
                f.seek(size - read_size)

                data = f.read().decode("utf-8")

                lines = [l for l in data.split("\n") if l.strip()]
                if not lines:
                    return None
                return json.loads(lines[-1])
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    @staticmethod
    def _valid_cursor(value: Any) -> int | None:
        """Int cursors only — reject bool (``isinstance(True, int)`` is True)."""
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    def _iter_valid_entries(self) -> Iterator[tuple[dict[str, Any], int]]:
        """ 过滤掉无效的历史条目 """
        poisoned: Any = None
        # 遍历所有历史条目
        for entry in self._read_entries():
            raw = entry.get("cursor")
            if raw is None:
                continue
            cursor = self._valid_cursor(raw)
            # 监测到非法的 cursor 值
            if cursor is None:
                poisoned = raw
                continue
            yield entry, cursor


    def _read_entries(self) -> list[dict[str, Any]]:
        """ 从 history.jsonl 中读取所有历史信息 """
        entries: list[dict[str, Any]] = []
        with suppress(FileNotFoundError):
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        return entries

    def get_memory_context(self) -> str:
        """供 ContextBuilder 注入：内容由 ``memory/MEMORY.md`` 提供。
        ``build_system_prompt`` 外层已有 ``# Memory``，此处不再重复加 ``## Long-term Memory``，避免出现双层同名标题。
        """
        text = self.read_memory().strip()
        return text


    def get_recent_history(self, limit: int) -> list[dict[str, Any]]:
        # 获取最后 limit 条的历史信息
        if limit <= 0:
            return []
        return [e for e, c in self._iter_valid_entries()][-limit:]