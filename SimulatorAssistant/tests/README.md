# 测试（tests）

本目录用于存放单元测试与集成测试。当前为规划阶段，尚未接入测试框架。

## 建议覆盖

- **RagEngine（单元）**
  - `stripXmlText`：XML 标签去除、实体反转义、空白折叠正确性。
  - `splitChunks`：块大小上限、块间重叠、不被边界切断关键信息。
  - `embed`：输出向量维度 = 模型 `n_embd`，且已 L2 归一化（模长≈1）。
  - `loadKnowledgeBase` + `retrieve`：建索引后，相关 query 能召回正确来源分块（混合检索 top-k 顺序合理）。
  - `retrieve` 相关性门槛：全局最佳向量分低于阈值时 `used=false`、不注入参考资料（抑制幻觉来源）。
  - `.docx` 解析：与人工提取的正文逐段一致。

- **并发 / 线程安全（集成）**
  - 热重载（`rebuildIndexAsync` 后台 `QtConcurrent` 重建索引）期间持续检索，不崩溃、不读到半成品、结果一致。

- **LLMWorker（集成）**
  - 增量解码：多轮对话上下文不串扰；容量超限后按整轮丢弃并重填。

## 运行方式（待定）

优先接入 **Qt Test**（`QT += testlib`），或在 CI 中用 GoogleTest 驱动。
需要本机已准备 bge 嵌入模型与一份样例知识库（`knowledge_base/` 即可复用）。
