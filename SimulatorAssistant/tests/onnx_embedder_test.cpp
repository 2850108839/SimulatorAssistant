// tests/onnx_embedder_test.cpp —— 验证 ONNX Runtime 嵌入后端（方案 A）
// 用法（命令行）：onnx_embedder_test.exe [model.onnx]
// 验证点：模型加载、embed 出 512 维向量、L2 归一化为 1、两个句子余弦相似度
//（vocab.txt 约定与模型同目录，自动探测）
#include "../module/Rag/OnnxEmbedder.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

static double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

int main(int argc, char** argv) {
    const std::string modelPath = argc > 1 ? argv[1]
        : "D:/CAI/llama/llama.cpp/models/bge_onnx/model_int8.onnx";

    OnnxEmbedder emb;
    if (!emb.loadModel(modelPath)) {
        std::cerr << "[FAIL] loadModel 失败: " << modelPath << std::endl;
        return 1;
    }
    std::cout << "[OK] 模型加载成功" << std::endl;

    std::vector<float> v1, v2;
    if (!emb.embed("如何新建一个数据采集任务？", v1, 512)) { std::cerr << "[FAIL] embed1 失败" << std::endl; return 1; }
    if (!emb.embed("数据采集任务的创建步骤", v2, 512))     { std::cerr << "[FAIL] embed2 失败" << std::endl; return 1; }

    std::cout << "[OK] v1 dim=" << v1.size() << " first8=";
    for (int i = 0; i < 8 && i < (int)v1.size(); ++i) std::cout << v1[i] << " ";
    std::cout << std::endl;

    double n1 = 0, n2 = 0;
    for (float v : v1) n1 += v * v;
    for (float v : v2) n2 += v * v;
    std::cout << "[CHECK] L2 norm v1=" << std::sqrt(n1) << " v2=" << std::sqrt(n2) << " (应为 1)" << std::endl;

    std::cout << "[CHECK] 相似句余弦=" << cosine(v1, v2) << " (应 > 0.5)" << std::endl;
    return 0;
}
