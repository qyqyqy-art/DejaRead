# DejaRead

DejaRead 是一个面向论文阅读的本地知识助手原型。它把 PDF 论文解析成 Markdown，切分为可检索的 chunk，写入 SQLite、向量库和关键词索引，并提供选词标注、笔记和问答能力。

当前项目主要覆盖：

- **论文入库**：PDF 解析、Markdown 保存、智能分块、embedding、向量索引、关键词索引、元数据入库。
- **选词标注**：针对论文中的术语生成语境化解释，并发现跨论文概念关联。
- **笔记系统**：每篇论文维护一份 Markdown 笔记，并同步索引笔记 section。
- **问答预览**：基于当前论文 chunk 和笔记做混合检索，再调用 LLM 生成回答。

## 项目结构

```text
dejaread/
  ingestion/   # PDF 解析、分块、入库管线
  embedding/   # embedding 客户端和向量库抽象
  keyword/     # SQLite FTS5 关键词索引
  retrieval/   # 向量 + 关键词混合检索
  concepts/    # 选词标注和概念关联
  notes/       # Markdown 笔记读写与索引
  qa/          # 问答服务
  db/          # SQLAlchemy ORM 模型
config/
  config.yaml  # 本地配置文件，建议不要提交真实密钥
app.py         # Gradio 演示入口
```

## 安装

建议使用 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[full]"
```

如果只安装基础依赖：

```bash
pip install -r requirements.txt
```

PaddleOCR-VL 相关依赖和模型服务需要按你的运行环境单独准备。

## 配置

项目通过 `config/config.yaml` 读取数据库、embedding、向量库、LLM 和 PaddleOCR-VL 配置。不要把真实 API key 提交到仓库。

示例：

```yaml
database:
  url: "sqlite:///dejaread.db"

paddleocr:
  pipeline_version: "v1.6"
  vl_rec_backend: "vllm-server"
  vl_rec_server_url: "http://127.0.0.1:8080/v1"

embedding:
  base_url: "https://your-embedding-api/v1"
  model: "your-embedding-model"
  api_key: "YOUR_API_KEY"

vector_store:
  backend: "chroma"  # memory | chroma
  persist_directory: "./chroma_data"

llm:
  model: "your-chat-model"
  base_url: "https://your-llm-api/v1"
  api_key: "YOUR_API_KEY"
```

## 运行

启动 Gradio demo：

```bash
CUDA_VISIBLE_DEVICES=5 python app.py
```

默认会在 `0.0.0.0:9000` 启动。

页面包含：

1. **论文入库**：上传 PDF，解析并写入索引。
2. **选词标注**：输入论文中的术语，生成解释和跨论文关联。
3. **笔记**：加载、编辑、保存每篇论文的 Markdown 笔记。
4. **问答（预览）**：基于当前论文和笔记进行检索增强问答。

## PaddleOCR-VL

项目提供了一个 PaddleOCR-VL vLLM server 的 Docker 启动脚本：

```bash
bash dejaread/ingestion/paddleocr.sh
```

脚本内容等价于：

```bash
docker run \
  --rm \
  --gpus all \
  --network host \
  -v /path/to/PaddleOCR-VL-1.6:/path/to/PaddleOCR-VL-1.6 \
  -e CUDA_VISIBLE_DEVICES=5 \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
  paddleocr genai_server \
    --model_name PaddleOCR-VL-1.6-0.9B \
    --model_dir /path/to/PaddleOCR-VL-1.6/PaddlePaddle/PaddleOCR-VL-1___6 \
    --host 0.0.0.0 \
    --port 8080 \
    --backend vllm
```

使用前需要根据本机环境修改脚本中的模型目录挂载路径、`--model_dir` 和 `CUDA_VISIBLE_DEVICES`。服务启动后，客户端地址通常是：

```yaml
paddleocr:
  vl_rec_backend: "vllm-server"
  vl_rec_server_url: "http://127.0.0.1:8080/v1"
```

当前默认解析器是 `PaddleOCRPDFParser`，调用方式与 `test_paddle_ocr.py` 类似：

```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    pipeline_version="v1.6",
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://127.0.0.1:8080/v1",
)
output = pipeline.predict("paper.pdf")
```

解析时会保存 PaddleOCR-VL 的中间结果：

```text
pdf_parser/{PDF文件名}_output/
  *.md
  *_res.json
  imgs/
  parsed.md
  metadata.json
```

## 当前状态

这是一个原型项目。论文入库、选词标注、笔记和基础问答链路已经打通；完整的 QA Agent 和长期记忆模块仍在规划中。

接下来比较自然的方向：

- 完善 QA 引用展示和检索范围控制。
- 接入概念图谱检索，支持概念比较和溯源。
- 增加 episodic / semantic / procedural memory。
- 做 query rewrite 和 self-reflection。

## 开发

运行语法检查：

```bash
python -m compileall dejaread
```

运行测试：

```bash
pytest
```
