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
bool OnnxEmbedder::loadModel(const std::string& modelPath) {
    // vocab.txt 约定与模型同目录（export_bge_onnx.py 产物）
    const size_t slash = modelPath.find_last_of("/\\");
    std::string vocabPath = (slash == std::string::npos)
        ? "vocab.txt" : modelPath.substr(0, slash + 1) + "vocab.txt";
    if (!loadVocab(vocabPath)) return false;

    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rag-onnx");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(m_intraOp);  // 单算子内并行线程（load 前 setThreads 可配）
    so.SetInterOpNumThreads(m_interOp);  // 算子间串行即可
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
    // 输出 last_hidden_state 最后一维 = 嵌入维度（bge-small-zh = 512）
    const auto outShape = m_session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (!outShape.empty()) m_dim = (int)outShape.back();
    return !m_inNames.empty() && !m_outNames.empty();
}

void OnnxEmbedder::setThreads(int intra, int inter) {
    // 必须在 loadModel() 前调用：SessionOptions 在建 session 时固化线程数
    m_intraOp = intra > 0 ? intra : 1;
    m_interOp = inter > 0 ? inter : 1;
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
    // 对齐 BERT BasicTokenizer：
    //  - 中文（UTF-8 多字节，0xE4-0xE9 起始）逐字切，全角标点整体保留；
    //  - ASCII 按空白分词，字母/数字累积，标点独立成 token（"device?" -> "device" + "?"，
    //    否则 "device?" 整体查词表落 [UNK]，向量与 llama.cpp 端严重偏离）
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
        } else if (c < 0x80) {  // ASCII
            if (std::isalnum(c)) {
                cur += (char)c;
            } else {
                if (!cur.empty()) { out.push_back(cur); cur.clear(); }
                out.emplace_back(1, (char)c);      // 标点独立 token
            }
            ++i;
        } else {                // 其他多字节（全角标点等）：整体累积
            cur += (char)c;
            ++i;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::vector<int> OnnxEmbedder::wordPiece(const std::string& token) const {
    // 整体命中直接返回
    auto it = m_vocab.find(token);
    if (it != m_vocab.end()) return {it->second};

    // 标准 WordPiece：从左到右最长匹配，首段不带 ##，后续段带 ## 前缀
    std::vector<int> ids;
    size_t start = 0;
    while (start < token.size()) {
        size_t end = token.size();
        std::string piece;
        bool found = false;
        while (start < end) {
            std::string cand = token.substr(start, end - start);
            if (start > 0) cand = "##" + cand;
            auto cit = m_vocab.find(cand);
            if (cit != m_vocab.end()) { piece = cand; found = true; break; }
            --end;
        }
        if (!found) {                       // 拆不出合法子词：整体 [UNK]
            int unk = m_vocab.count("[UNK]") ? m_vocab.at("[UNK]") : 100;
            return {unk};
        }
        ids.push_back(m_vocab.at(piece));
        start = end;
    }
    return ids;
}

std::vector<int> OnnxEmbedder::tokenize(const std::string& text, int maxSeq) const {
    std::vector<int> ids;
    ids.push_back(m_clsId);
    for (const auto& piece : splitBasic(text)) {
        std::string t = norm(piece);
        if (t.empty()) continue;
        const std::vector<int> sub = wordPiece(t);
        for (int id : sub) {
            ids.push_back(id);
            if ((int)ids.size() >= maxSeq - 1) break;
        }
        if ((int)ids.size() >= maxSeq - 1) break;
    }
    ids.push_back(m_sepId);
    if ((int)ids.size() > maxSeq) ids.resize(maxSeq);
    return ids;
}
