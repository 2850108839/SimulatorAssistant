# SimulatorAssistant — 供大模型阅读理解的项目上下文（AI_CONTEXT）

> 本文档由一次重构 + 优化会话沉淀而成，目标是让**另一个大模型**在接手本项目时，无需回溯完整对话即可准确理解：项目是什么、目录为什么是现在这样、已做了哪些优化、还有哪些坑和可优化项、以及如何验证改动。
> 所有结论均基于实际读代码与编译验证，非臆测。最后更新：2026-10-02。

---

## 0. 一句话概览

`SimulatorAssistant` 是一个 **Qt 6.5.3 (MSVC2019) + llama.cpp** 的 Windows 桌面应用：本地大模型对话 + RAG 知识库。基于 `Qwen2.5-7B` 的 GGUF 量化模型做生成，基于 `bge-small-zh` 的 GGUF 量化模型做知识库向量检索，检索结果以纯文本拼进 prompt（不改模型权重）。

---

## 1. 技术栈与关键事实（先记住这些，避免猜错）

| 项 | 值 | 备注 |
|---|---|---|
| 构建 | Qt 6.5.3 / MSVC2019 64-bit，qmake 工程 | 真机用 Qt Creator 打开 `SimulatorAssistant.pro` |
| 推理后端 | llama.cpp，build 11039（约 0.4.1 dev） | 需先编出 `llama.lib` / `ggml.lib` |
| 生成模型 | `Qwen2.5-7B` GGUF（Q4 档，约 4.6 GB） | 权重路径由 `LLMWorker::loadModel` 入参传入 |
| 嵌入模型 | `bge-small-zh-v1.5-q8_0.gguf`，512 维 | RAG 检索用，CLS 池化 |
| 知识库 | `knowledge_base/`（.txt/.md/.docx/.doc） | 目录由 `QSettings` 持久化 |
| llama.cpp 头 | `D:/CAI/llama/llama.cpp/include/llama.h` | 该头**已 `#include "ggml.h"`**，故 `GGML_TYPE_*` 可直接用 |

### llama.cpp build 11039 的 API 要点（易踩坑，务必核对）
- **`n_gpu_layers` 在 `llama_model_params`（不是 `llama_context_params`）**。加载模型时设 `mp.n_gpu_layers` 才生效；不设这个字段则默认 `0` = **全 CPU 推理**。
- **KV 缓存精度**用 `llama_context_params.type_k` / `type_v`（`enum ggml_type`，实验字段）。`GGML_TYPE_F16=1`、`GGML_TYPE_Q8_0=8`。本项目已设为 `GGML_TYPE_Q8_0`。
- **Flash Attention**用枚举 `llama_context_params.flash_attn_type`：`LLAMA_FLASH_ATTN_TYPE_AUTO=-1` / `DISABLED=0` / `ENABLED=1`。需构建带 flash-attn 后端才有效。
- 编码器（bge）用 `llama_encode`（不是 `llama_decode`）；取向量用 `llama_get_embeddings_seq(ctx, seq_id)`。
- KV 缓存串行化：`llama_memory_clear(llama_get_memory(ctx), false)` 可在每次 embed 前清掉，避免串扰。

---

## 2. 目录结构：为什么是现在这样（决策史）

最终结构（**平铺**，无 `include/` `src/` 子目录）：

```
SimulatorAssistant/
├── SimulatorAssistant.pro          # 单 exe 工程（TEMPLATE 默认 app，非 subdirs）
├── README.md
├── .gitignore
├── module/
│   ├── Engine/   LLMWorker.h  LLMWorker.cpp    # 推理引擎（异步流式）
│   └── Rag/      RagEngine.h  RagEngine.cpp     # RAG：切块/嵌入/混合检索/索引缓存
├── app/
│   └── SimulatorAssistant/
│       ├── mainwindow.h  main.cpp  mainwindow.cpp
│       └── form/  mainwindow.ui                   # .ui 单独放 form/
├── knowledge_base/                 # 运行期数据（被 .gitignore 忽略）
├── docs/  tools/  tests/  build/   # 文档 / 离线脚本 / 黄金集 / 构建产物
```

决策演进（用户逐步指令，最终落到「平铺」）：
1. 初始：`src/{app,engine,rag}` 各自含 `.h/.cpp`。
2. 用户：同层同类工程同文件夹、内分 `header/`+`src/` → 改为 `app/ engine/ rag/` 各含 `header/`+`src/`。
3. 用户：`header` 应叫 `include` → 改名 `include/`。
4. 用户：结构「类似于 `D:/work/code/2025006-dispatchclient/trunk/DispatchSys/DispatchClient`」。
   - 参考工程规范：顶层 `module/`(各模块是独立 lib 子工程) + `app/<工程名>/`(含 `src/` 混放 `.h/.cpp`、`form/` 放 `.ui`)；`.h/.cpp` **混放在 `src/`**，无 `include/` 拆分。
   - 用户当时确认两点：**① 顶层拆 `module/`+`app/`，但保持单 exe（不拆 lib、不加 `preconfig.pri`）；② 模块内保留 `include/`+`src/` 拆分**。
5. 用户：「去掉 `include` 和 `src`」→ **最终平铺**：`.h/.cpp` 直接放模块根，仅 `.ui` 留 `form/`。

> 给接手者的提示：目录是用户按 DispatchClient 风格主观定的，不是技术强制。若再改动，注意 `.pro` 的 `INCLUDEPATH` / `SOURCES` / `HEADERS` / `FORMS` 必须同步；`#include "Xxx.h"` 写法保持不变（靠 INCLUDEPATH 解析）。

---

## 3. QSettings 持久化在哪（常见问题）

- 用默认 `QSettings("CAI", "SimulatorAssistant")`（`NativeFormat`）→ **写入 Windows 注册表，不是文件**。
- 路径：`HKEY_CURRENT_USER\Software\CAI\SimulatorAssistant\rag`，值 `kbDir`（知识库目录）。
- 重启还在；`regedit` 删 `SimulatorAssistant` 项即恢复默认 `knowledge_base/`。
- 别与磁盘文件混淆：`rag_metrics.jsonl` 和 `.rag_cache_*.bin` 才是落盘文件（在知识库父目录，已被 `.gitignore` 忽略）。

---

## 4. 推理 / 内存优化：现状分析与已落地项

### 4.1 分析阶段发现的问题（按收益排序）
| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| 1 | `n_gpu_layers` 未设（默认 0）→ 全 CPU | 最大速度瓶颈（7B 可能仅 5–10 tok/s） | **未做**（机器纯 CPU，无独显） |
| 2 | KV 缓存默认 F16 | 7B@4096 约占 1.6 GB | **已做**（→ Q8，约 0.8 GB） |
| 3 | Flash Attention 未设 | 长上下文显存/速度 | **未做**（需构建带后端） |
| 4 | `LlmEngine` 同步版是死代码，且其 `chat()` 每 token 新建/释放采样链 | 反面范式 + 混淆 | **已做**（删除） |
| 5 | 建索引逐块串行 `embed()` | 大库建库慢 | **已做**（批量编码） |
| 6 | 检索全量 O(N) 向量余弦扫描 | 超大库才明显 | 未做（当前规模不急，可换 ANN） |

### 4.2 已落地的三项（用户确认：纯 CPU，先做安全项）
**① KV 缓存量化** — `module/Engine/LLMWorker.cpp` 的 `loadModel`：
```cpp
llama_context_params cp = llama_context_default_params();
cp.n_ctx    = kNCtx;        // 4096
cp.n_batch  = kBatch;       // 512
cp.n_ubatch = kBatch;       // 512
cp.type_k = GGML_TYPE_Q8_0; // KV 量化 F16→Q8，内存约减半、质量几乎无损
cp.type_v = GGML_TYPE_Q8_0;
```
> 回退方式：把两行改回 `GGML_TYPE_F16`（或删掉）即可。

**② 删除 `LlmEngine` 死代码**：同步版 `chat()` 已不被调用（mainwindow 里 `// m_engine = new LlmEngine();` 被注释）。已删除 `module/Engine/LlmEngine.h` 与 `.cpp`，并清理 `mainwindow.h/.cpp`、`.pro`、`README.md`、`ARCHITECTURE.md` 全部引用。应用实际只用 `LLMWorker`（异步流式）。

**③ 索引批量编码** — `module/Rag/RagEngine.cpp` 新增 `embedBatch()`：
- 把多句打包进**同一次** `llama_encode`（不同 `seq_id`），按 `n_ctx` 贪心分组（单句不跨批拆开）。
- `buildIndex` 改为：先收集全部 `(来源, 分块文本)`，再一次性 `embedBatch` 编码。
- 检索路径仍用单句 `embed()`（`retrieve()` 中），RAG 逻辑零改动。
- 建库编码调用次数约降一个数量级。

### 4.3 未做项（机器换独显 / 超大知识库时再议）
- GPU 卸载：`mp.n_gpu_layers = 99`（或 -1 全卸载），在 `LLMWorker::loadModel` / `RagEngine::loadModel` 的 `llama_model_params mp` 里设。
- Flash Attention：`cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;`
- 缩 `n_ctx`（如 2048）：KV 线性下降，但要权衡 RAG【参考资料】占的 prompt 长度。
- 检索 ANN：N 很大时把全量 O(N) 扫描换成 faiss/hnsw。

---

## 5. 架构要点（速记，细节见 `ARCHITECTURE.md`）

- **`LLMWorker`**：`QObject` + `moveToThread` 常驻子线程，异步流式。KV 缓存「增量追加」，溢出时按整轮丢弃最早对话后重建；采样链 `top_k → top_p → min_p → temp → dist`（以 `dist` 结尾，否则采样参数失效）。
- **`RagEngine`**：加载 bge，`buildIndex` 切块→编码→`installIndex` 原子替换不可变 `IndexSnapshot`；`retrieve` 做「向量 top-k 候选 + BM25 词法融合重排」混合检索，`bestVectorScore < threshold` 则整体丢弃（抑制幻觉来源）。`embed()` 持 `m_encodeMutex` 串行化（bge 上下文不可重入）。
- **数据流**：`query → RagEngine::retrieve → 【参考资料】+ 用户输入 → LLMWorker::chat`，只改喂入文字，不动 KV/采样/生成循环。

---

## 6. 如何验证改动（无需真机 Qt Creator 也能快速验编译）

本项目用 MinGW 做语法级校验（真机仍是 MSVC 构建）。先 uic 生成 ui 头，再 `-fsyntax-only`：

```bash
# 生成 ui 头（否则 mainwindow.cpp 找不到 ui_mainwindow.h）
/c/Qt/6.5.3/mingw_64/bin/uic.exe \
    app/SimulatorAssistant/form/mainwindow.ui -o build/ui/ui_mainwindow.h

# 语法检查四个源文件
GPP=/c/Qt/Tools/mingw1120_64/bin/g++.exe
INCS="-Iapp/SimulatorAssistant -Imodule/Engine -Imodule/Rag \
      -I/c/Qt/6.5.3/mingw_64/include \
      -I/c/Qt/6.5.3/mingw_64/include/QtCore \
      -I/c/Qt/6.5.3/mingw_64/include/QtGui \
      -I/c/Qt/6.5.3/mingw_64/include/QtWidgets \
      -I/c/Qt/6.5.3/mingw_64/include/QtConcurrent \
      -I/c/Qt/6.5.3/mingw_64/include/QtCore/6.5.3 \
      -I/c/Qt/6.5.3/mingw_64/include/QtGui/6.5.3 \
      -I/c/Qt/6.5.3/mingw_64/include/QtWidgets/6.5.3 \
      -I/c/Qt/6.5.3/mingw_64/include/QtCore/6.5.3/QtCore/private \
      -I/c/Qt/6.5.3/mingw_64/include/QtGui/6.5.3/QtGui/private \
      -I/d/CAI/llama/llama.cpp/include -I/d/CAI/llama/llama.cpp/ggml/include \
      -Ibuild/ui"
DEFS="-DQT_CORE_LIB -DQT_GUI_LIB -DQT_WIDGETS_LIB -DQT_CONCURRENT_LIB"
FLAGS="-fsyntax-only -std=c++17 -fpermissive -w"
for f in module/Rag/RagEngine.cpp module/Engine/LLMWorker.cpp \
         app/SimulatorAssistant/mainwindow.cpp app/SimulatorAssistant/main.cpp; do
  $GPP $FLAGS $DEFS $INCS $f
done
# 期望：四个文件均无 error 输出（RESULT_OK）
```
> 注意：路径随第 2 节目录调整而变化；若目录再改，本校验命令的 `-I` 与 `for` 列表也要同步。

---

## 7. 已知限制
- 老版二进制 `.doc` 用无依赖「尽力而为」文本提取（扫 UTF-16LE 文本流），丢格式/结构、可能混入页眉页脚噪声；高保真需求先转 `.docx`。
- KV 量化（Q8）属轻微精度取舍；若回答质量可感知下降，回退到 `GGML_TYPE_F16`。
- 当前为单 exe 工程，未采用 DispatchClient 的 `subdirs` + 每模块独立 lib 形态（用户当时选择最小改动）。
