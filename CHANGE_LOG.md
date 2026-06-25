# Changelog

本文件记录 DejaRead 项目的显著变更，遵循 [Keep a Changelog](https://keepachangelog.com/) 的格式约定：按时间倒序排列，每条记录只写对用户/后续开发者有意义的变化（新增能力、行为变更、修复的问题），不是 commit log 的复述。

## 2026-06-25

### Added

- 关键词索引：`dejaread/keyword/` 新增 `SQLiteFTSStore`（基于 SQLite FTS5 + jieba 中文分词），为 `chunks` 和 `concepts` 两个集合提供关键词检索能力，与原有的向量检索（ChromaDB/InMemory）并列。
- 混合检索：`dejaread/retrieval/HybridRetriever`，用 RRF（Reciprocal Rank Fusion）融合向量检索和关键词检索的结果，供未来 QA Agent 复用。
- `IngestionPipeline` 和 `ConceptAnnotationService` 入库/标注时同步写入关键词索引。
- 配置新增 `keyword_store.db_path`，默认复用主 SQLite 数据库文件。

### Fixed

- 修复选词标注（`ConceptAnnotationService.annotate()`）报错 `database is locked`：关键词索引写入被移到 ORM 事务 `commit()` 之后执行，避免两个独立的 SQLite 连接同时争用同一个数据库文件的写锁。
