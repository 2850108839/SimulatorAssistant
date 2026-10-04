# 技术版本清单（SimulatorAssistant）

> 记录日期：2026-10-04　用途：① 面试被问"用的什么版本"直接答；② 换机器/重建环境的依据。
> 每个版本均在本机实测核验过（llama_version()、pip import、GitHub Release 资产名）。

## 一、开发与运行环境

| 项 | 版本 | 备注 |
|---|---|---|
| 操作系统 | Windows（用户本机） | 64 位 |
| Qt | 6.5.3（MSVC2019_64 kit） | qmake 工程，`.pro` 头注释标注 |
| 编译器 | MSVC 2019（x64） | Qt 官方 kit；链接 llama.lib / onnxruntime.lib |
| C++ 标准 | C++17 | `CONFIG += c++17` |

## 二、推理引擎

| 项 | 版本 | 位置 | 核验方式 |
|---|---|---|---|
| llama.cpp | **0.4.1-dev** | D:/CAI/llama/llama.cpp | `llama_version()` 输出 0.4.1-dev |
| ONNX Runtime | **1.30.0（win-x64 CPU）** | D:/CAI/llama/onnxruntime/onnxruntime-win-x64-1.30.0 | GitHub Release v1.30.0；VERSION_NUMBER=1.30.0 |

## 三、模型文件（GGUF）

| 模型 | 文件 | 参数 | 实测 |
|---|---|---|---|
| Qwen2.5-7B-Instruct | qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf（~4.7GB，分片 1/2） | Q4_K_M 量化 | 本地 CPU 13.3 t/s |
| bge-small-zh-v1.5 | bge-small-zh-v1.5-q8_0.gguf（26.5MB） | Q8_0 量化，embedding 维度 512（CompendiumLabs 出品） | llama.cpp CLS pooling 已对齐 |

模型目录：D:/CAI/llama/llama.cpp/models/

## 四、Python 与导出依赖（bge → ONNX 导出链）

| 项 | 版本 | 说明 |
|---|---|---|
| Python | **3.14.7（64-bit）** | Anaconda 根目录 D:\ProgramData\anaconda3，MSC v.1944 |
| pip | 25.3 | 源：华为云镜像 mirrors.huaweicloud.com |
| transformers | 5.18.0 | 加载 bge 模型、tokenizer |
| optimum | 2.3.0 | ORTModelForFeatureExtraction 导出 |
| onnx | 1.23.1 | 模型图操作 / 校验 |
| onnxruntime | 1.30.0 | 导出脚本验证、双后端对比（cp314 wheel） |
| sentencepiece | 0.2.2 | bge tokenizer 依赖 |
| torch | 2.14.1 | optimum 导出后端 |
| numpy | 2.5.3 | 向量运算 |
| huggingface-hub | 1.33.0 | 模型下载（若本地无缓存） |
| tokenizers | 0.23.2 | transformers 依赖 |
| safetensors | 0.8.0 | 权重格式 |

> 注：pip 安装时 transformers 依赖的 huggingface-hub 曾先解析到 2.1.1、后回落为 1.33.0，最终以 1.33.0 生效（pip 自动解决）。

## 五、配置与发布要点

- `.pro` 环境变量覆盖：`LLAMA_DIR`（默认 D:/CAI/llama/llama.cpp）、`ORT_DIR`（默认 D:/CAI/llama/onnxruntime/onnxruntime-win-x64-1.30.0），未设置时回退默认路径。
- 链接：`llama.lib`（build/src/Release）+ `ggml.lib`（build/ggml/src/Release）+ `onnxruntime.lib`（$$ORT_DIR/lib）。
- **运行时 DLL（发布必带）**：llama.dll（如有）、ggml.dll（如有）、**onnxruntime.dll**、**onnxruntime_providers_shared.dll** 需与 exe 同目录。
- Python 3.14 属较新版本：onnxruntime/transformers 均已提供 cp314 wheel，兼容性已实测通过。
