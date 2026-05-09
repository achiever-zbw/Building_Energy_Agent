"""Dream / 维护任务用的受限工作区文件工具（读任意相对路径；写仅限 MEMORY/SOUL/USER）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.agent.tools.base import Tool, tool_parameters


def resolve_workspace_file(workspace: Path, relative_path: str) -> Path:
    ws = workspace.resolve()
    raw = (relative_path or "").strip().replace("\\", "/")
    rel = Path(raw)
    if rel.is_absolute() or raw.startswith("/"):
        raise ValueError("path_must_be_relative")
    full = (ws / rel).resolve()
    try:
        full.relative_to(ws)
    except ValueError as exc:
        raise ValueError("path_outside_workspace") from exc
    return full


# Phase 2 只允许改动这三处长期记忆文件（ posix 形式便于校验）
_DREAM_EDITABLE = frozenset(
    {
        "memory/MEMORY.md",
        "SOUL.md",
        "USER.md",
    },
)


def _normalized_rel(workspace: Path, relative_path: str) -> str:
    full = resolve_workspace_file(workspace, relative_path)
    return full.relative_to(workspace.resolve()).as_posix()


@tool_parameters(
    {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "相对 workspace 根的路径，例如 memory/MEMORY.md",
            },
        },
        "required": ["path"],
    },
)
class ReadWorkspaceFileTool(Tool):
    concurrency_safe = False

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "read_workspace_file"

    @property
    def description(self) -> str:
        return (
            "读取 workspace 内 UTF-8 文本文件的全部内容。"
            "用于在写入前先查看 memory/MEMORY.md、SOUL.md、USER.md 等。"
        )

    async def execute(self, *, path: str, **_: Any) -> str:
        full = resolve_workspace_file(self._workspace, path)
        if not full.is_file():
            return f"Error: file not found: {path}"
        try:
            return full.read_text(encoding="utf-8")
        except OSError as exc:
            return f"Error: {type(exc).__name__}: {exc}"


@tool_parameters(
    {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "仅允许 memory/MEMORY.md、SOUL.md、USER.md",
            },
            "old_string": {
                "type": "string",
                "description": "要被替换的原始片段（需与文件中完全一致，默认唯一匹配）",
            },
            "new_string": {
                "type": "string",
                "description": "替换后的文本",
            },
            "replace_all": {
                "type": "boolean",
                "description": "为 true 时替换所有匹配（慎用）",
                "default": False,
            },
        },
        "required": ["path", "old_string", "new_string"],
    },
)
class EditWorkspaceFileTool(Tool):
    concurrency_safe = False

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "edit_workspace_file"

    @property
    def description(self) -> str:
        return (
            "在允许的长期记忆文件中用 new_string 替换 old_string。"
            "必须先 read_workspace_file 确认当前内容；保持增量小步修改。"
        )

    async def execute(
        self,
        *,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **_: Any,
    ) -> str:
        key = _normalized_rel(self._workspace, path)
        if key not in _DREAM_EDITABLE:
            return (
                "Error: edit_workspace_file 仅允许修改 "
                + ", ".join(sorted(_DREAM_EDITABLE))
            )
        full = resolve_workspace_file(self._workspace, path)
        if not full.is_file():
            return f"Error: file not found: {path}"
        try:
            text = full.read_text(encoding="utf-8")
        except OSError as exc:
            return f"Error: {type(exc).__name__}: {exc}"

        if old_string == new_string:
            return "no-op: old_string equals new_string"

        count = text.count(old_string)
        if count == 0:
            return "Error: old_string not found in file"
        if not replace_all and count != 1:
            return f"Error: old_string matches {count} times; must be unique or set replace_all=true"
        if replace_all:
            updated = text.replace(old_string, new_string)
        else:
            updated = text.replace(old_string, new_string, 1)

        try:
            full.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return f"Error: {type(exc).__name__}: {exc}"

        return f"ok: wrote {full.relative_to(self._workspace.resolve()).as_posix()} ({len(updated)} chars)"


def build_dream_workspace_tools(workspace: Path) -> tuple[ReadWorkspaceFileTool, EditWorkspaceFileTool]:
    ws = Path(workspace)
    return ReadWorkspaceFileTool(ws), EditWorkspaceFileTool(ws)
