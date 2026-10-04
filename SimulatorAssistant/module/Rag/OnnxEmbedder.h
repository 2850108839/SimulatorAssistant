// OnnxEmbedder.h —— 方案 A 第 2 步：ONNX Runtime 嵌入推理后端（骨架，可编译扩展）
// 依赖：onnxruntime_cxx_api.h（onnxruntime.dll 从 GitHub Release 下载 Windows x64，加入 LIBS/INCLUDEPATH）
// 放工程位置：module/Rag/OnnxEmbedder.h（同目录放 OnnxEmbedder.cpp）
#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

// 统一嵌入接口（RagEngine 通过它切换后端）
class IEmbedder {
public:
    virtual ~IEmbedder() = default;
    virtual bool embed(const std::string& text, std::vector<float>& vec, int dim = 512) = 0;
};

// ONNX Runtime 后端：模型 + 自实现 WordPiece tokenizer
class OnnxEmbedder : public IEmbedder {
public:
    OnnxEmbedder() = default;
    ~OnnxEmbedder() override = default;

    // modelPath: model_int8.onnx；vocabPath: 导出的 vocab.txt
    bool load(const std::string& modelPath, const std::string& vocabPath);
    bool embed(const std::string& text, std::vector<float>& vec, int dim = 512) override;

    // 线程配置（JD 常考点：intra_op 控制单算子内线程）
    void setThreads(int intra, int inter = 1);

private:
    // ---- WordPiece（BERT 中文分词，自实现，ORT 不管 tokenizer）----
    bool loadVocab(const std::string& vocabPath);          // vocab.txt → id 表
    std::vector<int> tokenize(const std::string& text, int maxSeq = 512) const;
    std::vector<std::string> splitBasic(const std::string& text) const; // 中文按字、英文按空格
    int lookupWithSubword(const std::string& token) const;             // 最长匹配 + ## 后缀
    std::string norm(const std::string& s) const;                      // 小写化等

    // ---- ORT session ----
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    Ort::MemoryInfo m_memInfo{nullptr};
    std::vector<std::string> m_inNames;   // 输入名：input_ids / attention_mask
    std::vector<std::string> m_outNames;  // 输出名：last_hidden_state

    // ---- 词表 ----
    std::unordered_map<std::string, int> m_vocab;
    int m_clsId = 101;   // [CLS]
    int m_sepId = 102;   // [SEP]
    int m_padId = 0;     // [PAD]
};
