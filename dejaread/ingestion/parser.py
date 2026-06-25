"""PDF 解析：提取标题/作者等元数据，并按页输出文本（含图注、表格的原始文本）。

对应设计文档 3.1 节 "PDF 解析" 步骤。技术选型表中给出的是 ``marker``，但它是一个较
重的依赖（依赖本地模型权重），因此这里用一个 ``PDFParser`` 接口把解析逻辑抽象出来：

- :class:`MarkerPDFParser` —— 生产环境首选，封装 ``marker`` 库（按需 import）。
- :class:`PyPDFParser` —— 轻量回退方案，基于 ``pypdf`` 做纯文本提取，无需额外模型，
  方便在没有安装 marker / 没有 GPU 的环境下开发和测试。

两者都实现同一个接口，可以在 :class:`~dejaread.ingestion.pipeline.IngestionPipeline`
中互换使用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedSection:
    """解析出的一段文本（粒度为页或章节）。"""

    page_number: int
    text: str
    heading: str | None = None


@dataclass
class ParsedPaper:
    """PDF 解析结果：论文元数据 + 按页/章节切分的文本。"""

    title: str
    authors: str | None = None
    venue: str | None = None
    year: int | None = None
    sections: list[ParsedSection] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(section.text for section in self.sections)


class PDFParser(ABC):
    """PDF 解析器接口。"""

    @abstractmethod
    def parse(self, pdf_path: str | Path) -> ParsedPaper:
        """解析一个 PDF 文件，返回结构化的 :class:`ParsedPaper`。"""


class PyPDFParser(PDFParser):
    """基于 ``pypdf`` 的轻量解析器，逐页提取文本。

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
    """基于 ``marker`` 的解析器，对学术论文双栏排版、公式、表格支持更好。

    ``marker`` 依赖本地模型权重，体积较大，因此放在 import 时按需加载，避免影响
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
