"""PDF 解析：提取标题/作者等元数据，并按页输出文本（含图注、表格的原始文本）。

对应设计文档 3.1 节 "PDF 解析" 步骤。技术选型表中给出的是 marker ，但它是一个较
重的依赖（依赖本地模型权重），因此这里用一个 PDFParser 接口把解析逻辑抽象出来：

- :class: MarkerPDFParser  —— 生产环境首选，封装 marker 库（按需 import）。
- :class: PaddleOCRPDFParser  —— 调用 PaddleOCR-VL，输出 Markdown。
- :class: PyPDFParser —— 轻量回退方案，基于 pypdf 做纯文本提取，无需额外模型，
  方便在没有安装 marker / 没有 GPU 的环境下开发和测试。

这些解析器都实现同一个接口，可以在 :class: ~dejaread.ingestion.pipeline.IngestionPipeline
中互换使用。
"""

from __future__ import annotations

import json
import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import get_config
from ..utils.utils import setup_logger

logger = setup_logger(log_dir="logs/log_ingestion_parser", logger_name="ingestion_parser")


class ParsedSection(BaseModel):
    """
    解析出的一段文本（粒度为页或章节）。
    """

    page_number: int
    text: str
    heading: str | None = None


class ParsedPaper(BaseModel):
    """
    PDF 解析结果：论文元数据 + 按页/章节切分的文本。
    """

    title: str
    authors: str | None = None
    venue: str | None = None
    year: int | None = None
    sections: list[ParsedSection] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(section.text for section in self.sections)


class PDFParser(ABC):
    """
    PDF 解析器接口。
    """

    @abstractmethod
    def parse(self, pdf_path: str | Path) -> ParsedPaper:
        """
        解析一个 PDF 文件，返回结构化的 :class:`ParsedPaper`。
        """


class PyPDFParser(PDFParser):
    """
    基于 pypdf 的轻量解析器，逐页提取文本。

    不做版面分析，无法精确识别双栏排版下的阅读顺序，但足以支撑分块和向量化。
    标题默认取文件名（去掉后缀），可在入库时由调用方覆盖。
    """

    def parse(self, pdf_path: str | Path) -> ParsedPaper:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的提示
            raise ImportError(
                "PyPDFParser 需要安装 pypdf：pip install pypdf"
            ) from exc

        path = Path(pdf_path)
        reader = PdfReader(str(path))

        metadata = reader.metadata or {}
        title = (metadata.get("/Title") or "").strip() or path.stem

        sections = [
            ParsedSection(page_number=i + 1, text=(page.extract_text() or "").strip())
            for i, page in enumerate(reader.pages)
        ]
        return ParsedPaper(title=title, sections=sections)


class MarkerPDFParser(PDFParser):
    """
    基于 marker 的解析器，对学术论文双栏排版、公式、表格支持更好。

    marker 依赖本地模型权重，体积较大，因此放在 import 时按需加载，避免影响
    不需要它的代码路径（如单元测试）。
    """

    def parse(self, pdf_path: str | Path) -> ParsedPaper:
        try:
            from marker.convert import convert_single_pdf  # type: ignore
            from marker.models import load_all_models  # type: ignore
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的提示
            raise ImportError(
                "MarkerPDFParser 需要安装 marker-pdf：pip install marker-pdf"
            ) from exc

        path = Path(pdf_path)
        model_lst = load_all_models()
        full_text, _images, out_meta = convert_single_pdf(str(path), model_lst)

        title = out_meta.get("title") or path.stem
        # marker 输出的是整篇 Markdown 文本，这里按双换行粗粒度切页/段，
        # 真正的章节切分交给 Chunker 处理。
        paragraphs = [p for p in full_text.split("\n\n") if p.strip()]
        sections = [
            ParsedSection(page_number=i + 1, text=p.strip())
            for i, p in enumerate(paragraphs)
        ]
        return ParsedPaper(title=title, sections=sections)


class PaddleOCRPDFParser(PDFParser):
    """
    基于 PaddleOCR-VL 的 PDF 解析器。

    调用方式沿用 test_paddle_ocr.py：实例化 PaddleOCRVL，通过已部署的
    vLLM server 完成 VL 识别，然后使用 PaddleOCR 结果对象导出 Markdown/JSON。
    """

    def __init__(
        self,
        output_dir: str | Path = "pdf_parser",
        pipeline_version: str | None = None,
        vl_rec_backend: str | None = None,
        vl_rec_server_url: str | None = None,
        timeout: float | None = None,
        **pipeline_kwargs: Any,
    ) -> None:
        paddleocr_config = get_config().paddleocr
        self.pipeline_version = pipeline_version or paddleocr_config.pipeline_version
        self.vl_rec_backend = vl_rec_backend or paddleocr_config.vl_rec_backend
        self.vl_rec_server_url = vl_rec_server_url or paddleocr_config.vl_rec_server_url
        self.cuda_visible_devices = paddleocr_config.cuda_visible_devices or os.environ.get(
            "PADDLEOCR_CUDA_VISIBLE_DEVICES"
        )
        self.timeout = timeout
        self.output_dir = Path(output_dir)
        self.pipeline_kwargs = pipeline_kwargs
        self._pipeline: Any | None = None
        self._last_artifacts: dict[str, Any] | None = None

    def parse(self, pdf_path: str | Path) -> ParsedPaper:
        path = Path(pdf_path)
        logger.info("PaddleOCR-VL 解析开始：pdf_path=%s", path)
        try:
            output = list(self._get_pipeline().predict(str(path)))
        except Exception:
            logger.exception("PaddleOCR-VL 解析请求失败：pdf_path=%s", path)
            raise
        if not output:
            logger.error("PaddleOCR-VL 返回空结果：pdf_path=%s", path)
            raise RuntimeError(f"PaddleOCR-VL returned no results for {path}")

        predict_output_dir = self._save_predict_output(output, path)
        sections = self._extract_markdown_sections(predict_output_dir, path)
        markdown_text = "\n\n".join(section.text for section in sections)
        parsed = ParsedPaper(
            title=self._extract_title(markdown_text, path),
            sections=sections,
        )
        self._last_artifacts = {
            "pdf_path": str(path),
            "predict_output_dir": str(predict_output_dir),
            "markdown_text": markdown_text,
            "pipeline_settings": self._pipeline_settings(),
        }
        logger.info("PaddleOCR-VL 解析完成：pdf_path=%s title=%r sections=%d", path, parsed.title, len(sections))
        return parsed

    def save_artifacts(self, paper_name: str, paper_id: str) -> Path:
        """
        补充保存最近一次 PaddleOCR 识别结果的合并 Markdown 和入库元数据。
        """
        if self._last_artifacts is None:
            raise RuntimeError("No PaddleOCR artifacts to save; call parse() first")

        target_dir = Path(self._last_artifacts["predict_output_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)

        (target_dir / "parsed.md").write_text(
            self._last_artifacts["markdown_text"],
            encoding="utf-8",
        )
        metadata = {
            "pdf_path": self._last_artifacts["pdf_path"],
            "paper_name": paper_name,
            "paper_id": paper_id,
            "pipeline_settings": self._last_artifacts["pipeline_settings"],
        }
        (target_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target_dir

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        
        if self.cuda_visible_devices is not None and sys.platform.startswith("linux"):
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的提示
            raise ImportError(
                "PaddleOCRPDFParser 需要安装 paddleocr：pip install paddleocr"
            ) from exc

        self._pipeline = PaddleOCRVL(**self._pipeline_settings())
        return self._pipeline

    def _pipeline_settings(self) -> dict[str, Any]:
        return {
            "pipeline_version": self.pipeline_version,
            "vl_rec_backend": self.vl_rec_backend,
            "vl_rec_server_url": self.vl_rec_server_url,
            **self.pipeline_kwargs,
        }

    def _save_predict_output(self, output: list[Any], pdf_path: Path) -> Path:
        target_dir = self.output_dir / f"{self._safe_path_component(pdf_path.stem)}_output"
        target_dir.mkdir(parents=True, exist_ok=True)
        for result in output:
            result.save_to_json(save_path=str(target_dir))
            result.save_to_markdown(save_path=str(target_dir))
        return target_dir

    @classmethod
    def _extract_markdown_sections(
        cls, markdown_dir: Path, pdf_path: Path
    ) -> list[ParsedSection]:
        markdown_paths = sorted(
            markdown_dir.glob("*.md"),
            key=lambda md_path: cls._page_sort_key(md_path, pdf_path),
        )
        if not markdown_paths:
            raise RuntimeError("PaddleOCR-VL did not export any markdown files")

        sections = []
        for index, markdown_path in enumerate(markdown_paths):
            if markdown_path.name == "parsed.md":
                continue
            text = markdown_path.read_text(encoding="utf-8").strip()
            if text:
                sections.append(
                    ParsedSection(
                        page_number=cls._page_number(markdown_path, index),
                        text=text,
                    )
                )
        if not sections:
            raise RuntimeError("PaddleOCR-VL returned empty markdown text")
        return sections

    @staticmethod
    def _page_sort_key(markdown_path: Path, pdf_path: Path) -> tuple[int, str]:
        match = re.search(r"_(\d+)$", markdown_path.stem)
        if match:
            return int(match.group(1)), markdown_path.name
        if markdown_path.stem == pdf_path.stem:
            return 0, markdown_path.name
        return 10**9, markdown_path.name

    @staticmethod
    def _page_number(markdown_path: Path, fallback_index: int) -> int:
        match = re.search(r"_(\d+)$", markdown_path.stem)
        if match:
            return int(match.group(1)) + 1
        return fallback_index + 1

    @staticmethod
    def _safe_path_component(value: str) -> str:
        value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value)
        value = re.sub(r"\s+", "_", value).strip("._ ")
        return value[:120] or "paper"

    @staticmethod
    def _extract_title(markdown_text: str, path: Path) -> str:
        h1_titles = []
        for line in markdown_text.splitlines():
            match = re.match(r"^#\s+(.+?)\s*$", line)
            if match:
                h1_titles.append(match.group(1).strip())
                if len(h1_titles) >= 2:
                    break
            elif h1_titles and line.strip():
                break
        if not h1_titles:
            return path.stem
        if len(h1_titles) == 1:
            return h1_titles[0]
        separator = " " if h1_titles[0].endswith(":") else " - "
        return separator.join(h1_titles)
