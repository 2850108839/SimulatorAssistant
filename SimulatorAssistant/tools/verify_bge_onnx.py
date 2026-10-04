# -*- coding: utf-8 -*-
"""
tools/verify_bge_onnx.py
验证导出的 bge ONNX 模型可用性：加载 → 前向 → 检查输出形状/维度。
用于方案 A 双后端对比前的快速 sanity check。
"""
import onnxruntime as ort
import numpy as np

def check(path: str, label: str):
    print(f"==> {label}: {path}")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in sess.get_inputs()}
    print("   inputs:", [(n, i.shape, i.type) for n, i in inputs.items()])
    # 用 token ids 模拟一个 4-token 的输入
    ids = np.array([[101, 2057, 102, 0]], dtype=np.int64)  # [1, 4]
    feed = {name: (ids if name == "input_ids" else np.ones_like(ids)) for name in inputs}
    out = sess.run(None, feed)[0]
    print("   output shape:", out.shape, "-> embedding dim =", out.shape[-1])
    assert out.shape[-1] == 512, "维度不是 512！与 llama.cpp bge (dim=512) 不一致"
    print("   OK: 512 维，与 llama.cpp 端对齐")

if __name__ == "__main__":
    import os
    base = "D:/CAI/llama/llama.cpp/models/bge_onnx"
    check(os.path.join(base, "model.onnx"), "FP32")
    if os.path.exists(os.path.join(base, "model_int8.onnx")):
        check(os.path.join(base, "model_int8.onnx"), "INT8")
