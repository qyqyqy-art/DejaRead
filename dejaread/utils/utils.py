import os
import re
import ast
import json
import time
import logging
from typing import Any, Optional
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from fastapi import Depends, Header, HTTPException


def _try_json_or_python_literal(text: str) -> Any:
    """
    先尝试按 JSON 解析，如果失败再尝试按 Python 字面量解析。
    仅在结果是常见 JSON 兼容类型时返回解析结果，否则返回 None。
    """
    if isinstance(text, str):
        text = text.strip()

    # 1) 尝试 JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2) 尝试 Python 字面量（支持 {'a': 1}, True/False/None 等）
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
            return obj
    except (ValueError, SyntaxError):
        pass

    return None


def parse_json(content: str) -> Any:
    """
    尝试从模型输出中尽可能稳健地提取 JSON（支持纯 JSON / 代码块 / 混杂文本 / Python 字面量风格）。
    """

    # 策略 1: 直接解析（JSON 或 Python 字面量）
    result = _try_json_or_python_literal(content)
    if result is not None:
        # 如果解析结果本身还是字符串，可能是外层 JSON/字符串包了一层 Python dict/JSON
        if isinstance(result, str):
            nested = _try_json_or_python_literal(result)
            if nested is not None:
                return nested
        return result

    # 策略 2: 清理 markdown 代码块后再解析
    cleaned = re.sub(
        r'^```(?:json)?\s*|\s*```$',
        '',
        content.strip(),
        flags=re.MULTILINE,
    )

    result = _try_json_or_python_literal(cleaned)
    if result is not None:
        if isinstance(result, str):
            nested = _try_json_or_python_literal(result)
            if nested is not None:
                return nested
        return result

    # 策略 3: 尝试提取 JSON / Python 风格对象（{...}）
    try:
        json_match = re.search(
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            content,
            re.DOTALL,
        )
        if json_match:
            candidate = json_match.group(0)
            result = _try_json_or_python_literal(candidate)
            if result is not None:
                return result
    except Exception:
        pass

    # 策略 4: 尝试提取 JSON / Python 风格数组（[...]）
    try:
        array_match = re.search(
            r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',
            content,
            re.DOTALL,
        )
        if array_match:
            candidate = array_match.group(0)
            result = _try_json_or_python_literal(candidate)
            if result is not None:
                return result
    except Exception:
        pass

    # 全部失败则返回空 dict
    return {}


def setup_logger(log_dir: str = "logs", logger_name: str = "chatbot_logger") -> logging.Logger:
    """
    Set two log handlers: info level and error level, and record them to different files.
    """
    log_dir = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent logs from bubbling up to root logger (avoids duplicate terminal output)

    if not logger.handlers:
        # Info Log
        info_handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, "app.log"), # Always store the current log
            # when="m",
            when="midnight",
            interval=1,
            backupCount=0,
            encoding="utf-8",
            utc=False
        )
        info_handler.suffix = "%Y-%m-%d-%H:%M:%S" # Store the expired log with the storage suffix, such as app.log.2025-04-28-16:22:43
        info_handler.setLevel(logging.INFO)
        info_formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        info_handler.setFormatter(info_formatter)

        # Error Log
        error_handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, "error.log"),
            when="midnight",
            interval=1,
            backupCount=0,
            encoding="utf-8",
            utc=False
        )
        error_handler.suffix = "%Y-%m-%d-%H:%M:%S"
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        error_handler.setFormatter(error_formatter)

        logger.addHandler(info_handler)
        logger.addHandler(error_handler)

    return logger


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()

    @property
    def cost(self):
        return self.end - self.start


def format_time(timer: Optional[Timer] = None) -> Optional[str]:
    return "{:.2f}s".format(timer.cost) if timer else None


def get_env() -> str:
    return os.getenv("ENV", "prod")
