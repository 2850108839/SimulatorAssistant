// OnnxEmbedder.h —— 方案 A 第 2 步：ONNX Runtime 嵌入推理后端（骨架，可编译扩展）
// 依赖：onnxruntime_cxx_api.h（onnxruntime.dll 从 GitHub Release 下载 Windows x64，加入 LIBS/INCLUDEPATH）
// 放工程位置：module/Rag/OnnxEmbedder.h（同目录放 OnnxEmbedder.cpp）
#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

// 统一嵌入接口（RagEngine 通过它切换后端：llama.cpp / ONNX Runtime）
class IEmbedder {
public:
    virtual ~IEmbedder() = default;
    // 加载模型：llama 后端传 GGUF 路径；onnx 后端传 .onnx 路径（vocab.txt 约定在同目录）
    virtual bool loadModel(const std::string& modelPath) = 0;
    virtual bool isLoaded() const = 0;
    virtual int  dim() const = 0;                                   // 嵌入维度（bge = 512）
    virtual bool embed(const std::string& text, std::vector<float>& vec, int dim = 512) = 0;
    // 批量编码：默认逐句循环；llama 后端覆写为贪心打包单次 encode，性能更高
    virtual std::vector<std::vector<float>> embedBatch(const std::vector<std::string>& texts, int dim = 512) {
        std::vector<std::vector<float>> out(texts.size());
        for (size_t i = 0; i < texts.size(); ++i)
            if (!embed(texts[i], out[i], dim)) out[i].clear();
        return out;
    }
};

// ONNX Runtime 后端：模型 + 自实现 WordPiece tokenizer
class OnnxEmbedder : public IEmbedder {
public:
    OnnxEmbedder() = default;
    ~OnnxEmbedder() override = default;

    // modelPath: model_int8.onnx；vocab.txt 约定在模型同目录（export_bge_onnx.py 产物）
    bool loadModel(const std::string& modelPath) override;
    bool isLoaded() const override { return m_session != nullptr; }
    int  dim() const override { return m_dim; }
    bool embed(const std::string& text, std::vector<float>& vec, int dim = 512) override;

    // 线程配置（JD 常考点：intra_op 控制单算子内线程）
    // 必须在 loadModel() 之前调用才生效；默认 intra=4 / inter=1
    void setThreads(int intra, int inter = 1);

private:
    // ---- WordPiece（BERT 中文分词，自实现，ORT 不管 tokenizer）----
    bool loadVocab(const std::string& vocabPath);          // vocab.txt → id 表
    std::vector<int> tokenize(const std::string& text, int maxSeq = 512) const;
    std::vector<std::string> splitBasic(const std::string& text) const; // 中文按字、英文按空格+标点
    // 完整 WordPiece 子词切分（与 transformers BertWordPieceTokenizer 对齐）：
    // 整体查表失败则贪心最长匹配拆前缀，剩余部分递归带 "##" 前缀继续拆；拆不出返回 [UNK]
    std::vector<int> wordPiece(const std::string& token) const;
    std::string norm(const std::string& s) const;                      // 小写化等

    // ---- ORT session ----
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    Ort::MemoryInfo m_memInfo{nullptr};
    std::vector<std::string> m_inNames;   // 输入名：input_ids / attention_mask / token_type_ids
    std::vector<std::string> m_outNames;  // 输出名：last_hidden_state
    int m_dim = 512;                      // 嵌入维度（bge-small-zh = 512）
    int m_intraOp = 4;                    // intra_op 线程（load 前经 setThreads 修改）
    int m_interOp = 1;                    // inter_op 线程

    // ---- 词表 ----
    std::unordered_map<std::string, int> m_vocab;
    int m_clsId = 101;   // [CLS]
    int m_sepId = 102;   // [SEP]
    int m_padId = 0;     // [PAD]
};
