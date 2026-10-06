// LlamaEmbedder.cpp —— llama.cpp 嵌入后端（逻辑自 RagEngine 抽出，行为不变）
#include "LlamaEmbedder.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace {
constexpr int kNCtxEmb = 512;   // bge 单句最大 token 数（模型训练长度）
// llama.cpp 0.4.1-dev：n_ctx = n_seq_max × 每序列长度。要支持批量打包多句且每句不截断，
// 需 n_ctx = nSeqMax × 512；n_ubatch 是单次 graph 上限，批量打包不能超过它。
constexpr int kSeqMax = 8;      // 贪心打包最多同批句子数
constexpr int kNBatch = kNCtxEmb * kSeqMax;   // 4096：总上下文
constexpr int kNUBatch = 2048;  // 单次 encode 最大 token 数（一批实际远小于此）
}

LlamaEmbedder::~LlamaEmbedder() {
    if (m_ctx)   llama_free(m_ctx);
    if (m_model) llama_model_free(m_model);
}

bool LlamaEmbedder::loadModel(const std::string& modelPath) {
    if (m_model) return true;

    llama_model_params mp = llama_model_default_params();
    m_model = llama_model_load_from_file(modelPath.c_str(), mp);
    if (!m_model) return false;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = kNBatch;                  // 4096 = kSeqMax × 512（每序列全容量）
    cp.n_batch      = kNBatch;
    cp.n_ubatch     = kNUBatch;
    cp.n_seq_max    = kSeqMax;                  // 批量建索引：贪心打包按句分 seq_id，单序列会 encode 失败
    cp.embeddings   = true;                      // 关键：开启嵌入输出
    cp.pooling_type = LLAMA_POOLING_TYPE_CLS;   // bge 用 [CLS] 池化（与模型默认一致）
    m_ctx = llama_init_from_model(m_model, cp);
    if (!m_ctx) { llama_model_free(m_model); m_model = nullptr; return false; }

    m_vocab = llama_model_get_vocab(m_model);
    m_nEmbd = llama_model_n_embd_out(m_model);
    return true;
}

bool LlamaEmbedder::embed(const std::string& text, std::vector<float>& vec, int dim) {
    // bge 上下文不可重入：建索引（后台线程）与检索（worker 线程）的编码必须串行
    std::lock_guard<std::mutex> encLock(m_encodeMutex);
    if (!m_ctx || !m_model) return false;
    vec.clear();

    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(m_vocab, text.c_str(), (int32_t)text.size(),
                           toks.data(), (int32_t)toks.size(), true, false);
    if (n < 0) {                              // 缓冲区不足，扩容重试
        toks.resize(-n + 16);
        n = llama_tokenize(m_vocab, text.c_str(), (int32_t)text.size(),
                           toks.data(), (int32_t)toks.size(), true, false);
        if (n < 0) return false;
    }
    toks.resize(n);

    // 必要时补 EOS（与官方 retrieval 范例一致）
    const llama_token eos = llama_vocab_eos(m_vocab);
    if (eos >= 0 && (toks.empty() || toks.back() != eos)) toks.push_back(eos);

    // 单序列 batch，所有 token 都要求输出（logits=true）
    llama_batch batch = llama_batch_init((int32_t)toks.size(), 0, 1);
    for (int i = 0; i < (int)toks.size(); ++i) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }
    batch.n_tokens = (int32_t)toks.size();

    llama_memory_clear(llama_get_memory(m_ctx), false);   // 嵌入不需要 KV 缓存，清空避免串扰
    // bge 是编码器模型：必须用 llama_encode（不是 llama_decode）
    if (llama_encode(m_ctx, batch) != 0) {
        llama_batch_free(batch);
        return false;
    }

    const float* emb = llama_get_embeddings_seq(m_ctx, 0);   // CLS 池化向量
    if (emb) {
        vec.assign(emb, emb + m_nEmbd);
        normalize(vec);
    }
    llama_batch_free(batch);
    return !vec.empty();
}

std::vector<std::vector<float>> LlamaEmbedder::embedBatch(const std::vector<std::string>& texts, int dim) {
    std::lock_guard<std::mutex> encLock(m_encodeMutex);
    std::vector<std::vector<float>> out(texts.size());
    if (texts.empty() || !m_ctx) return out;

    const int nCtx = (int)llama_n_ctx(m_ctx);

    // 1) 每句 tokenize（与 embed 同款：加 BOS、尾部补 EOS）
    std::vector<std::vector<llama_token>> toks(texts.size());
    const llama_token eos = llama_vocab_eos(m_vocab);
    for (size_t i = 0; i < texts.size(); ++i) {
        const std::string& s = texts[i];
        std::vector<llama_token> t(s.size() + 16);
        int n = llama_tokenize(m_vocab, s.c_str(), (int32_t)s.size(),
                               t.data(), (int32_t)t.size(), true, false);
        if (n < 0) {
            t.resize(-n + 16);
            n = llama_tokenize(m_vocab, s.c_str(), (int32_t)s.size(),
                               t.data(), (int32_t)t.size(), true, false);
        }
        if (n < 0) continue;                     // 失败：该句留空向量
        t.resize(n);
        if (eos >= 0 && (t.empty() || t.back() != eos)) t.push_back(eos);
        toks[i] = std::move(t);
    }

    // 2) 贪心分组：每批累计 token 数不超过 n_ctx，单句不跨批拆开
    size_t i = 0;
    while (i < toks.size()) {
        if (toks[i].empty()) { ++i; continue; }
        std::vector<int> grp;
        int used = 0;
        while (i < toks.size() && !toks[i].empty()) {
            const int len = (int)toks[i].size();
            // 同时受 n_ctx、n_ubatch、n_seq_max 三重约束（0.4.1-dev 下超任一即失败/断言）
            if (!grp.empty() && (used + len > nCtx || used + len > kNUBatch || (int)grp.size() >= kSeqMax)) break;
            grp.push_back((int)i);
            used += len;
            ++i;
            if (used >= nCtx) break;
        }
        if (grp.empty()) { ++i; continue; }

        int totalTok = 0;
        for (int g : grp) totalTok += (int)toks[g].size();

        llama_batch batch = llama_batch_init(totalTok, 0, (int)grp.size());
        int base = 0;
        for (int gi = 0; gi < (int)grp.size(); ++gi) {
            const int idx = grp[gi];
            const int L = (int)toks[idx].size();
            for (int j = 0; j < L; ++j) {
                batch.token[base + j]     = toks[idx][j];
                batch.pos[base + j]       = j;
                batch.n_seq_id[base + j]  = 1;
                batch.seq_id[base + j][0] = gi;   // 每个句子一个独立序列
                batch.logits[base + j]    = 1;
            }
            base += L;
        }
        batch.n_tokens = totalTok;

        llama_memory_clear(llama_get_memory(m_ctx), false);
        if (llama_encode(m_ctx, batch) == 0) {
            for (int gi = 0; gi < (int)grp.size(); ++gi) {
                const int idx = grp[gi];
                const float* emb = llama_get_embeddings_seq(m_ctx, gi);
                if (emb) {
                    std::vector<float> v(emb, emb + m_nEmbd);
                    normalize(v);
                    out[idx] = std::move(v);
                }
            }
        }
        llama_batch_free(batch);
    }
    return out;
}

void LlamaEmbedder::normalize(std::vector<float>& v) {
    float n = 0;
    for (float x : v) n += x * x;
    n = std::sqrt(n);
    if (n > 1e-9f) for (float& x : v) x /= n;
}
