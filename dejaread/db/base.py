"""数据库连接与会话管理（SQLite + SQLAlchemy）。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..config import get_config


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


_engine = None
_SessionFactory: sessionmaker | None = None


def init_db(db_url: str | None = None, *, echo: bool | None = None) -> None:
    """初始化数据库引擎并创建所有表（若不存在）。

    ``db_url`` / ``echo`` 未传入时使用 ``config/config.yaml`` 中的 ``database`` 配置。
    可重复调用以切换数据库（主要用于测试，例如传入内存库 "sqlite:///:memory:"）。
    """
    global _engine, _SessionFactory

    db_config = get_config().database
    db_url = db_url if db_url is not None else db_config.url
    echo = echo if echo is not None else db_config.echo

    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    _engine = create_engine(db_url, echo=echo, connect_args=connect_args)
    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)

    # 模型必须先被 import 才能在 Base.metadata 中注册。
    from . import models  # noqa: F401

    Base.metadata.create_all(_engine)


def get_session() -> Session:
    """获取一个新的 Session。调用方负责 commit/close（或使用 session_scope）。"""
    if _SessionFactory is None:
        init_db()
    assert _SessionFactory is not None
    return _SessionFactory()


@contextmanager
def session_scope() -> Iterator[Session]:
    """提供一个自动 commit/rollback/close 的 Session 上下文。"""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
