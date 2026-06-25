"""中文/中英混合文本分词，供 SQLite FTS5 关键词索引使用（见 :mod:`dejaread.keyword.store`）。

FTS5 默认 tokenizer（``unicode61``）按空格/标点切分，无法正确处理无空格的中文文本，
所以索引写入前先用 jieba 把文本切成词，再以空格拼接成 FTS5 能正确检索的形式。索引和
查询必须共用同一套清洗逻辑（:func:`segment`），否则会出现切词口径不一致导致搜不到的问题。
"""

from __future__ import annotations

import re

import jieba

_PUNCT_RE = re.compile(r"^[\s\W]+$", re.UNICODE)


def segment(text: str) -> list[str]:
    """切分文本为清洗后的小写 token 列表。

    中文按词切分，英文/数字保持整词（jieba 不会拆成单字母），过滤纯空白/标点 token。
    """
    tokens = []
    for tok in jieba.cut(text):
        tok = tok.strip()
        if not tok or _PUNCT_RE.match(tok):
            continue
        tokens.append(tok.lower())
    return tokens


def segment_for_index(text: str) -> str:
    """供写入 FTS5：分词结果用空格拼接成一段文本。"""
    return " ".join(segment(text))


def segment_for_query(query_text: str) -> str:
    """供检索：分词后用 OR 连接并加引号，构造 FTS5 MATCH 查询串。

    每个 token 加双引号，避免裸词撞上 FTS5 查询语法关键字（如 NEAR/AND）导致解析出错。
    """
    tokens = segment(query_text)
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)
