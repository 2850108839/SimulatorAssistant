// tests/dual_backend_test.cpp —— 双后端一致性验证（方案 A 验收项）
// 同一批文本分别用 llama.cpp（bge Q8_0 GGUF）与 ONNX Runtime（bge INT8）编码，
// 对比两后端向量的余弦相似度：同模型不同量化/框架，余弦应 > 0.95 视为一致。
#include "../module/Rag/LlamaEmbedder.h"
#include "../module/Rag/OnnxEmbedder.h"
#include "llama.h"
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
    // llama.cpp 嵌入/推理前必须最先初始化后端（不调则模型加载失败）
    llama_backend_init();

    const std::string ggufPath = argc > 1 ? argv[1]
        : "D:/CAI/llama/llama.cpp/models/bge-small-zh-v1.5-q8_0.gguf";
    const std::string onnxPath = argc > 2 ? argv[2]
        : "D:/CAI/llama/llama.cpp/models/bge_onnx/model_int8.onnx";

    LlamaEmbedder llamaEmb;
    OnnxEmbedder  onnxEmb;
    if (!llamaEmb.loadModel(ggufPath)) { std::cerr << "[FAIL] llama 后端加载失败" << std::endl; return 1; }
    if (!onnxEmb.loadModel(onnxPath))  { std::cerr << "[FAIL] onnx 后端加载失败" << std::endl; return 1; }
    std::cout << "[OK] llama 后端 dim=" << llamaEmb.dim()
              << " | onnx 后端 dim=" << onnxEmb.dim() << std::endl;

    const std::vector<std::string> sents = {
        "如何新建一个数据采集任务？",
        "数据采集任务的创建步骤",
        "模拟器支持哪些采集协议？",
        "通道配置与采样频率设置",
        "软件安装与许可证激活",
        "How to configure the device?",
        "知识库未找到相关内容时的通用建议",
        "历史数据导出与报表生成操作说明"
    };

    int pass = 0;
    for (const auto& s : sents) {
        std::vector<float> a, b;
        if (!llamaEmb.embed(s, a) || !onnxEmb.embed(s, b)) {
            std::cout << "[FAIL] " << s << " 编码失败" << std::endl;
            continue;
        }
        const double c = cosine(a, b);
        const bool ok = c > 0.95;
        pass += ok ? 1 : 0;
        std::cout << (ok ? "[PASS]" : "[FAIL]") << " 余弦=" << c
                  << "  " << s << std::endl;
    }
    std::cout << "--- 双后端一致性：" << pass << "/" << (int)sents.size()
              << " 通过（阈值 >0.95）---" << std::endl;
    return pass == (int)sents.size() ? 0 : 2;
}
