// LlamaEmbedder.h —— llama.cpp 嵌入后端（IEmbedder 实现）
// 封装 bge GGUF 模型的加载与编码，接口与 OnnxEmbedder 对齐，供 RagEngine 双后端切换。
#pragma once
#include "OnnxEmbedder.h"   // IEmbedder
#include "llama.h"
#include <string>
#include <vector>
#include <mutex>

class LlamaEmbedder : public IEmbedder {
public:
    LlamaEmbedder() = default;
    ~LlamaEmbedder() override;

    // modelPath: bge 的 GGUF 路径
    bool loadModel(const std::string& modelPath) override;
    bool isLoaded() const override { return m_model != nullptr; }
    int  dim() const override { return m_nEmbd; }
    bool embed(const std::string& text, std::vector<float>& vec, int dim = 512) override;
    // 批量编码：贪心打包进单次 llama_encode（多 seq_id），建索引性能关键路径
    std::vector<std::vector<float>> embedBatch(const std::vector<std::string>& texts, int dim = 512) override;

private:
    static void normalize(std::vector<float>& v);

    llama_model   * m_model = nullptr;
    llama_context * m_ctx   = nullptr;
    const llama_vocab * m_vocab = nullptr;
    int m_nEmbd = 0;
    std::mutex m_encodeMutex;   // bge 上下文不可重入，编码必须串行
};
