# 架构说明（ARCHITECTURE）

## 总览

```
用户输入
  │
  ▼
MainWindow（UI 线程）
  │  on_pushButton_clicked → invokeMethod(chat)
  ▼
LLMWorker（推理线程，QThread）
  │  chat():
  │    1) 若 RAG 启用 → RagEngine::retrieve(query, topK)（混合检索：向量 top-k + BM25 重排）
  │       取回【参考资料】文本，拼到 user 内容前
  │    2) 增量解码（只喂本轮新增，KV 缓存保留）
  │    3) 流式 token → tokenReady 信号
  ▼
MainWindow → textEdit 打字机追加
```

RAG 的注入点只在「拼 prompt 之前改喂入文字」，**不动** KV 缓存、采样、生成循环，
因此模型权重与推理逻辑完全不受知识库影响。

## 模块职责

### `module/Engine/` — 推理引擎（.h/.cpp 平铺）

- **`LLMWorker`**：继承 `QObject`，`moveToThread` 到常驻子线程，提供异步流式推理。
  - 上下文 `n_ctx=4096`、`n_batch=n_ubatch=512`；**KV 缓存已量化 `GGML_TYPE_Q8_0`**（默认 F16→Q8，内存约减半、质量几乎无损，纯 CPU 也受益）。
  - 增量解码：每轮只喂增量，KV 缓存保留；容量不足时按**整轮**丢弃最早对话后重建。
  - 采样链：`top_k → top_p → min_p → temp → dist`（无 greedy，否则采样参数失效）。
  - 通过 `tokenReady` / `replyFinished` / `errorOccurred` 信号与 UI 通信。
  - 持有 `RagEngine*` 指针（`setRagEngine`），在 `chat()` 中按需检索。
  - 增量解码：每轮只喂增量，KV 缓存保留；容量不足时按**整轮**丢弃最早对话后重建。
  - 采样链：`top_k → top_p → min_p → temp → dist`（无 greedy，否则采样参数失效）。
  - 通过 `tokenReady` / `replyFinished` / `errorOccurred` 信号与 UI 通信。
  - 持有 `RagEngine*` 指针（`setRagEngine`），在 `chat()` 中按需检索。

### `module/Rag/` — RAG 引擎（.h/.cpp 平铺）

- **`RagEngine`**：
  - 加载 bge 嵌入模型（encoder-only，用 `llama_encode` + `[CLS]` 池化 + L2 归一化）。
  - 建索引拆分为两步，便于后台线程运行：
    - `buildIndex(dir, onProgress)`：递归读取 `.txt/.md/.docx/.doc` → **语义边界切块** → 收集分块后调用 `embedBatch()` **批量编码**（多句打包进同一次 `llama_encode`，按 `n_ctx` 贪心分组），返回分块（不替换当前索引）。命中向量缓存时直接加载、跳过编码。
    - `installIndex(idx)`：加锁整体替换索引，并构建不可变 `IndexSnapshot`（分块 + 预计算 BM25 词频）。
    - `loadKnowledgeBase(dir)`：组合上面两步，作为同步兜底 API。
    - `.docx`：用 Qt 私有头 `QZipReader` 解压 `word/document.xml` 提取正文（无额外 zlib 依赖）。
    - `.doc`：老版二进制 OLE2 格式，无依赖「尽力而为」扫描 UTF-16LE 文本流提取正文（丢格式/结构，保真度低于 .docx）。
  - `retrieve(query, topK, threshold)`：**混合检索**——
    1. 向量余弦取 top-k 候选；
    2. 候选上做 BM25 词法分，与向量分归一化后按 `α·向量 + (1-α)·BM25` 融合重排；
    3. 全局最佳向量分低于 `threshold` 则整体丢弃（视为知识库未覆盖，不注入参考资料，抑制幻觉来源）。
    返回 `RagResult{context, sources, scores, bestVectorScore, latencyMs, used}`。
  - 线程安全：`m_enabled` 为 `std::atomic<bool>`；`embed()` 加 `m_encodeMutex`（bge 上下文不可重入）；
    索引以 `shared_ptr<IndexSnapshot>` 原子替换，检索时抓取快照，与后台重建互不干扰。

### `app/SimulatorAssistant/` — 应用层 / 界面（.h/.cpp 平铺 + form/）

- **`MainWindow`**：主窗口（入口 `main.cpp` 同目录）。菜单栏「RAG」提供：
  - RAG 开关（`ragAct` checkable，后台索引完成前禁用，显示「RAG：索引中…」）
  - 「重新加载知识库」（`rebuildIndexAsync`：后台 `QtConcurrent` 线程建索引 + 状态栏进度 `N/M` → 完成后默认开启）
  - 「选择知识库目录…」（`QFileDialog` + `QSettings` 持久化 `rag/kbDir` + 立即热重载）
  - 监听知识库目录（`QFileSystemWatcher` + 800ms 去抖）自动重建索引

## 数据流：RAG 检索

```
query ──embed()──▶ vec_q
                     │
                     ▼
       for each chunk: sim = dot(vec_q, chunk.vec)   // 均已 L2 归一化
                     │
                     ▼
       partial_sort 向量 top-k 候选 ──▶ BM25 词法分融合重排 ──▶ top-k
                              │
                              ▼
       "【参考资料】\n1. (来源)\n...\n2. ..."
                              │
                              ▼
       effectiveInput = 参考资料 + 用户问题 ──▶ 喂给 LLMWorker::chat
```

## 构建与依赖解析

- `.pro` 通过环境变量 `LLAMA_DIR` 定位 llama.cpp（默认 `D:/CAI/llama/llama.cpp`），
  拼接 `include/`、`ggml/include` 到 `INCLUDEPATH`，拼接 `build/.../Release/*.lib` 到 `LIBS`。
- Qt 私有头（解压 `.docx` 用）通过 qmake 内置变量 `$$[QT_INSTALL_HEADERS]/QtGui/6.5.3` 引用，跨 Qt 安装可移植。
- 中间产物（moc / ui / obj）归入 `build/`，源码树保持干净。
