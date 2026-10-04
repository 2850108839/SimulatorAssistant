# -*- coding: utf-8 -*-
"""
tools/export_bge_onnx.py
方案 A 第 1 步：把 bge-small-zh-v1.5 导出为 ONNX（FP32 + INT8 动态量化），供 ONNX Runtime 后端使用。

用法（Windows PowerShell）：
    pip install transformers optimum onnx onnxruntime sentencepiece
    python tools/export_bge_onnx.py --model BAAI/bge-small-zh-v1.5 --output_dir D:/CAI/llama/llama.cpp/models/bge_onnx

产出：
    model.onnx          FP32 模型（推理用）
    model_int8.onnx     INT8 动态量化版（推荐实际加载这个）
    vocab.txt           WordPiece 词典（C++ OnnxEmbedder 自实现 tokenizer 用）
    tokenizer.json / special_tokens_map.json 等（备用）

说明：
    1) optimum 的 ORTModelForFeatureExtraction 会做图优化，输出只剩 last_hidden_state，
       形状 [batch, seq, hidden=512]，池化在 C++ 侧做（取 CLS 位 + L2 归一化，与 llama.cpp 的 CLS pooling 对齐）。
    2) 动态量化对 embedding 质量影响小、无需校准集，先用它；想进一步压缩可再研究静态量化。
"""
import argparse
import os

def main():
    ap = argparse.ArgumentParser(description="Export bge-small-zh-v1.5 to ONNX (FP32 + INT8)")
    ap.add_argument("--model", default="BAAI/bge-small-zh-v1.5",
                    help="HuggingFace 模型名（首次运行自动下载，需能访问 HF）")
    ap.add_argument("--output_dir", default="D:/CAI/llama/llama.cpp/models/bge_onnx")
    ap.add_argument("--no_int8", action="store_true", help="跳过 INT8 量化")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 导出 FP32 ONNX（optimum 会自动做算子融合/图优化）
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    print("==> 导出 FP32 ONNX ...")
    model = ORTModelForFeatureExtraction.from_pretrained(args.model, export=True)
    model.save_pretrained(args.output_dir)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.save_pretrained(args.output_dir)  # 含 vocab.txt
    fp32 = os.path.join(args.output_dir, "model.onnx")
    print("==> FP32 完成:", fp32, f"({os.path.getsize(fp32)/1024/1024:.1f} MB)")

    # 2) INT8 动态量化（weight 量化为主，无需校准集）
    if not args.no_int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        print("==> INT8 动态量化 ...")
        int8 = os.path.join(args.output_dir, "model_int8.onnx")
        quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
        print("==> INT8 完成:", int8, f"({os.path.getsize(int8)/1024/1024:.1f} MB)")

    print("==> 全部完成。vocab.txt 路径:", os.path.join(args.output_dir, "vocab.txt"))

if __name__ == "__main__":
    main()
