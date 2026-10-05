// OnnxEmbedder.cpp —— 方案 A 第 2 步实现骨架
// 关键点：CLS pooling + L2 归一化（与 llama.cpp 端 CLS pooling 对齐，保证两后端向量可比）
#include "OnnxEmbedder.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
// Windows 上 ORTCHAR_T = wchar_t：模型路径必须转 UTF-16（同时支持中文路径）
static std::wstring utf8ToWide(const std::string& s) {
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}
#endif

// ------------------------------------------------------------------
// ORT session 加载
// ------------------------------------------------------------------
bool OnnxEmbedder::load(const std::string& modelPath, const std::string& vocabPath) {
    if (!loadVocab(vocabPath)) return false;

    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rag-onnx");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(4);      // 单算子内并行线程
    so.SetInterOpNumThreads(1);      // 算子间串行即可
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); // 图优化全开
    so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

#ifdef _WIN32
    const std::wstring wpath = utf8ToWide(modelPath);
    m_session = std::make_unique<Ort::Session>(*m_env, wpath.c_str(), so);
#else
    m_session = std::make_unique<Ort::Session>(*m_env, modelPath.c_str(), so);
#endif
    const auto nIn  = m_session->GetInputCount();
    const auto nOut = m_session->GetOutputCount();

    m_inNames.clear();  m_outNames.clear();
    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < nIn; ++i) {
        auto name = m_session->GetInputNameAllocated(i, alloc);
        m_inNames.emplace_back(name.get());
    }
    for (size_t i = 0; i < nOut; ++i) {
        auto name = m_session->GetOutputNameAllocated(i, alloc);
        m_outNames.emplace_back(name.get());
    }
    m_memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return !m_inNames.empty() && !m_outNames.empty();
}

void OnnxEmbedder::setThreads(int intra, int inter) {
    // 骨架：如需运行时改线程，可在 load 前配置 SessionOptions；此处留占位
    (void)intra; (void)inter;
}

// ------------------------------------------------------------------
// embed：tokenize → 推理 → CLS pooling → L2 归一化 → 512 维
// ------------------------------------------------------------------
bool OnnxEmbedder::embed(const std::string& text, std::vector<float>& vec, int dim) {
    if (!m_session) return false;

    const int maxSeq = 512;
    std::vector<int> ids = tokenize(text, maxSeq);
    if (ids.size() < 2) return false;

    const int seq = (int)ids.size();
    std::vector<int64_t> inputIds(ids.begin(), ids.end());
    std::vector<int64_t> attMask(seq, 1);          // 无 padding，全 1
    std::vector<int64_t> segIds(seq, 0);           // token_type_ids：单句全 0（模型要求 3 输入）

    std::array<int64_t, 2> shape{1, (int64_t)seq};
    Ort::Value inIds  = Ort::Value::CreateTensor<int64_t>(m_memInfo, inputIds.data(), inputIds.size(), shape.data(), shape.size());
    Ort::Value inMask = Ort::Value::CreateTensor<int64_t>(m_memInfo, attMask.data(), attMask.size(), shape.data(), shape.size());
    Ort::Value inSeg  = Ort::Value::CreateTensor<int64_t>(m_memInfo, segIds.data(), segIds.size(), shape.data(), shape.size());

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(std::move(inIds));
    inputs.emplace_back(std::move(inMask));
    inputs.emplace_back(std::move(inSeg));

    std::vector<const char*> inNames, outNames;
    for (auto& n : m_inNames)  inNames.push_back(n.c_str());
    for (auto& n : m_outNames) outNames.push_back(n.c_str());

    auto outputs = m_session->Run(Ort::RunOptions{nullptr}, inNames.data(), inputs.data(), inputs.size(),
                                  outNames.data(), outNames.size());
    if (outputs.empty()) return false;

    // last_hidden_state: [1, seq, hidden]
    auto& out = outputs[0];
    const auto* data = out.GetTensorData<float>();
    const int64_t hidden = (int64_t)dim;
    const int64_t total = out.GetTensorTypeAndShapeInfo().GetElementCount();
    if (total < hidden) return false;

    // CLS pooling：取位置 0 的向量（llama.cpp bge 默认 CLS pooling，两后端对齐）
    vec.assign(data, data + hidden);

    // L2 归一化
    double sum = 0.0;
    for (float v : vec) sum += (double)v * v;
    if (sum < 1e-12) return false;
    const double inv = 1.0 / std::sqrt(sum);
    for (float& v : vec) v = (float)(v * inv);
    return true;
}

// ------------------------------------------------------------------
// WordPiece 分词（中文按字 + 英文按词 + 最长匹配 + ## 后缀）
// ------------------------------------------------------------------
bool OnnxEmbedder::loadVocab(const std::string& vocabPath) {
    std::ifstream f(vocabPath);
    if (!f.is_open()) return false;
    m_vocab.clear();
    std::string line;
    int id = 0;
    while (std::getline(f, line)) {
        // vocab.txt 每行一个 token；去掉 \r（Windows 行尾）
        if (!line.empty() && line.back() == '\r') line.pop_back();
        m_vocab[line] = id++;
    }
    // 特殊 token 兜底（若词表含 [CLS] 则取实际 id）
    auto it = m_vocab.find("[CLS]"); if (it != m_vocab.end()) m_clsId = it->second;
    it = m_vocab.find("[SEP]"); if (it != m_vocab.end()) m_sepId = it->second;
    it = m_vocab.find("[PAD]"); if (it != m_vocab.end()) m_padId = it->second;
    return !m_vocab.empty();
}

std::string OnnxEmbedder::norm(const std::string& s) const {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) {
        return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : (char)c;
    });
    return r;
}

std::vector<std::string> OnnxEmbedder::splitBasic(const std::string& text) const {
    // 中文（UTF-8 3 字节，0xE4-0xE9 起始）逐字切；ASCII 按空白分词
    std::vector<std::string> out;
    std::string cur;
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = (unsigned char)text[i];
        if (c >= 0xE4 && c <= 0xE9 && i + 2 < text.size()) { // 中文字符
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            out.emplace_back(text.substr(i, 3));
            i += 3;
        } else if (c <= 0x20) { // 空白
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            ++i;
        } else {
            cur += (char)c;
            ++i;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

int OnnxEmbedder::lookupWithSubword(const std::string& token) const {
    auto it = m_vocab.find(token);
    if (it != m_vocab.end()) return it->second;
    // 最长匹配 + ## 后缀（WordPiece 子词切分）
    for (size_t len = token.size(); len >= 1; --len) {
        std::string prefix = token.substr(0, len);
        auto pit = m_vocab.find(prefix);
        if (pit != m_vocab.end()) {
            std::string rest = "##" + token.substr(len);
            auto rit = m_vocab.find(rest);
            if (rit != m_vocab.end()) return rit->second; // 简化：只拆一层，够中文场景用
        }
    }
    return m_vocab.count("[UNK]") ? m_vocab.at("[UNK]") : 100; // [UNK] 兜底
}

std::vector<int> OnnxEmbedder::tokenize(const std::string& text, int maxSeq) const {
    std::vector<int> ids;
    ids.push_back(m_clsId);
    for (const auto& piece : splitBasic(text)) {
        std::string t = norm(piece);
        if (t.empty()) continue;
        ids.push_back(lookupWithSubword(t));
        if ((int)ids.size() >= maxSeq - 1) break;
    }
    ids.push_back(m_sepId);
    if ((int)ids.size() > maxSeq) ids.resize(maxSeq);
    return ids;
}
