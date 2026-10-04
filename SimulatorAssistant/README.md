# SimulatorAssistant

本地大模型对话桌面应用，内置 **RAG 知识库**（支持 `.txt` / `.md` / `.docx` / `.doc`）。可在界面直接选择知识库目录并**热重载**，无需重启程序。

## 特性

- 基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的本地大模型推理（Qwen2.5-7B 等 GGUF 模型）
- **RAG**：`bge-small-zh` 嵌入 + 余弦相似度检索，检索结果以纯文本拼入 prompt（不改变模型本身）
- 知识库支持 `.txt` / `.md` / **`.docx`**（OOXML，解压 `word/document.xml`）/ **`.doc`**（老版二进制，无依赖「尽力而为」提取正文）
- **热加载**：菜单「RAG → 重新加载知识库 / 选择知识库目录」，不重启即可重建索引
- 知识库目录通过 `QSettings` 持久化，下次启动自动加载

## 目录结构

```
SimulatorAssistant/
├── SimulatorAssistant.pro          # qmake 工程文件（单 exe，对齐 DispatchClient 的 module/app 分层）
├── README.md
├── .gitignore
├── module/                         # 业务模块（对齐 DispatchClient：同层同类工程各占一文件夹）
│   ├── Engine/                     #   推理引擎（.h/.cpp 平铺）
│   │   ├── LLMWorker.h
│   │   └── LLMWorker.cpp           #   异步流式推理（独立线程）
│   └── Rag/                        #   RAG 模块（.h/.cpp 平铺）
│       ├── RagEngine.h
│       └── RagEngine.cpp           #   切块 / 嵌入 / 混合检索 / 索引缓存
├── app/                            # 应用层 / 界面（对齐 DispatchClient：app/<工程名>/）
│   └── SimulatorAssistant/
│       ├── mainwindow.h
│       ├── main.cpp                #   程序入口
│       ├── mainwindow.cpp
│       └── form/
│           └── mainwindow.ui       #   主窗口 + 菜单（RAG 开关/热加载/后台索引）
├── knowledge_base/                 # RAG 知识库数据（.txt / .md / .docx / .doc，运行期）
├── docs/                           # 设计文档
│   └── ARCHITECTURE.md
├── tools/                          # 离线工具 / 脚本（评测、缓存诊断）
│   ├── eval_retrieval.py           #   Golden Set 检索评测（BM25 词法腿 + 缓存体检）
│   └── dump_rag_cache.py           #   索引缓存诊断
├── tests/                          # 测试 / 黄金集
│   ├── golden_qa.json              #   检索黄金集（评测输入）
│   └── README.md
└── build/                          # 构建中间产物（moc / ui / obj，git 忽略）
```

## 依赖

- **Qt 6.5.3**（MSVC2019 64-bit）
- **llama.cpp**（build 11039，0.4.1 dev）：需先编译出 `llama.lib` / `ggml.lib`
- **bge-small-zh-v1.5-q8_0.gguf** 嵌入模型（512 维）

## 构建

1. 用 Qt Creator 打开 `SimulatorAssistant.pro`，选择 **MSVC2019 64-bit** kit。
2. 若 llama.cpp 不在默认路径 `D:/CAI/llama/llama.cpp`，设置环境变量 `LLAMA_DIR` 指向其根目录后再打开 / 构建：
   ```bat
   set LLAMA_DIR=D:/path/to/llama.cpp
   ```
3. 构建并运行。moc / ui / 目标文件等中间产物自动归入 `build/`，保持源码树干净。

## 使用

菜单栏「**RAG**」：

- 勾选「RAG」开关 → 启用检索（未建索引时显示「RAG：不可用」）
- 「重新加载知识库」→ 按当前目录重建索引
- 「选择知识库目录…」→ 浏览并切换知识库（自动持久化 + 热重载）

知识库目录默认 `knowledge_base/`，可在「选择知识库目录…」中更改。

## 已知限制

- 老版二进制 `.doc` 采用无依赖的「尽力而为」文本提取（丢格式/结构，可能混入页眉页脚等噪声）；高保真需求建议先转 `.docx`。
