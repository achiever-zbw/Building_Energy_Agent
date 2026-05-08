"""本地知识库：backend/data/kb 下 .md/.txt，无需 RAGFlow。"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB_ROOT = _BACKEND_ROOT / "data" / "kb"


def _tokenize(s: str) -> set[str]:
    """
    为兼顾中英文：英文/数字取长度>=2 的连续段；中文连续汉字拆成单字与相邻二字组。
    否则整段中文会被当成一个 token，与问句无法求交集。
    """
    if not (s or "").strip():
        return set()
    s = (s or "").strip().lower()
    tokens: set[str] = set()
    for m in re.finditer(r"[a-z0-9]{2,}", s):
        tokens.add(m.group())
    for m in re.finditer(r"[\u4e00-\u9fff]+", s):
        chunk = m.group()
        tokens.update(chunk)
        if len(chunk) >= 2:
            for i in range(len(chunk) - 1):
                tokens.add(chunk[i : i + 2])
    return tokens


def search_kb(
    question: str,
    *,
    kb_root: Path | None = None,
    top_k: int = 5,
    max_chars: int = 6000,
) -> dict:
    root = kb_root or DEFAULT_KB_ROOT
    if not root.is_dir():
        return {
            "answer": "",
            "sources": [],
            "note": f"本地知识库目录不存在: {root}",
        }

    q_tokens = _tokenize(question)
    if not q_tokens:
        return {"answer": "", "sources": [], "note": "问题为空"}

    scored: list[tuple[int, str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix.lower() not in (".md", ".txt", ".markdown"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        paras = re.split(r"\n\s*\n+", text)
        rel = str(path.relative_to(root))
        for para in paras:
            para = para.strip()
            if len(para) < 12:
                continue
            tset = _tokenize(para)
            score = len(q_tokens & tset)
            if score > 0:
                scored.append((score, rel, para))

    scored.sort(key=lambda x: -x[0])
    picked = scored[:top_k]

    parts: list[str] = []
    sources: list[dict] = []
    total = 0
    for score, rel, para in picked:
        block = f"【{rel}】\n{para}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        sources.append({"file": rel, "score": score})
        total += len(block)

    if not parts:
        return {
            "answer": "",
            "sources": [],
            "note": f"本地目录 {root} 下无匹配段落，请添加 .md/.txt",
        }

    return {
        "answer": "\n\n---\n\n".join(parts),
        "sources": sources,
    }
