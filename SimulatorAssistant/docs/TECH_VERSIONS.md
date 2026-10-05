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
| transformers | 4.57.6 | 加载 bge 模型、tokenizer（optimum[onnxruntime] 解析后的兼容版本） |
| optimum | 2.1.0 | ORTModelForFeatureExtraction 导出（需额外装 extras，见下） |
| optimum-onnx | 0.1.0 | optimum 的 ONNX Runtime 独立集成包（`pip install "optimum[onnxruntime]"` 引入） |
| onnx | 1.23.1 | 模型图操作 / 校验 |
| onnxruntime | 1.30.0 | 导出脚本验证、双后端对比（cp314 wheel） |
| sentencepiece | 0.2.2 | bge tokenizer 依赖 |
| torch | 2.14.1 | optimum 导出后端 |
| numpy | 2.5.3 | 向量运算 |
| huggingface-hub | 0.36.2 | 模型下载（optimum[onnxruntime] 解析后的兼容版本） |
| tokenizers | 0.22.2 | transformers 依赖（同上） |
| safetensors | 0.8.0 | 权重格式 |

> 注 1：optimum 2.x 的 ONNX Runtime 集成拆到独立 extras，直接 `pip install optimum` 不含 `optimum.onnxruntime` 模块，必须 `pip install "optimum[onnxruntime]"`；该 extras 会把 transformers/optimum/huggingface-hub 解析为上述兼容组合（曾先装到 transformers 5.18.0/optimum 2.3.0/hub 1.33.0，后被 extras 回落）。
> 注 2：导出产物 `D:/CAI/llama/llama.cpp/models/bge_onnx/`：model.onnx（FP32，90.4MB）、model_int8.onnx（INT8 动态量化，22.8MB）、vocab.txt（109KB）；ORT 模型输入 input_ids/attention_mask/token_type_ids（int64），输出 last_hidden_state [batch, seq, 512]（已用 tools/verify_bge_onnx.py 实测通过）。

## 五、配置与发布要点

- `.pro` 环境变量覆盖：`LLAMA_DIR`（默认 D:/CAI/llama/llama.cpp）、`ORT_DIR`（默认 D:/CAI/llama/onnxruntime/onnxruntime-win-x64-1.30.0），未设置时回退默认路径。
- 链接：`llama.lib`（build/src/Release）+ `ggml.lib`（build/ggml/src/Release）+ `onnxruntime.lib`（$$ORT_DIR/lib）。
- **运行时 DLL（发布必带，都在 `$$ORT_DIR\lib\` 下，不是顶层）**：**onnxruntime.dll**（16.4MB）、**onnxruntime_providers_shared.dll** 需与 exe 同目录；llama.dll / ggml.dll 如有同样处理。
- 踩坑：exe 目录缺少 onnxruntime.dll 时，Windows 会从 PATH 找到系统残留的旧版本（实测加载到 1.17.1 导致 "requested API version [30] not available"）——务必让 exe 同目录的 DLL 版本与链接的 lib 一致。
- Python 3.14 属较新版本：onnxruntime/transformers 均已提供 cp314 wheel，兼容性已实测通过。
