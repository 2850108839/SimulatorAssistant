#include "LlmEngine.h"
#include <QDebug>

LlmEngine::LlmEngine() {
    llama_backend_init();
}

LlmEngine::~LlmEngine()
{
    if (m_ctx)   llama_free(m_ctx);
    if (m_model) llama_model_free(m_model);
    llama_backend_free();
}

bool LlmEngine::loadModel(const QString &modelPath) {

    llama_model_params mp = llama_model_default_params();
    m_model = llama_model_load_from_file(modelPath.toStdString().c_str(), mp);
    if (!m_model) return false;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;                              // 上下文窗口，和验证时一致
    m_ctx = llama_new_context_with_model(m_model, cp);
    return m_ctx != nullptr;
}

QString LlmEngine::chat(const QString &input) {
    if (!m_ctx || !m_model) return QStringLiteral("模型未加载");

    const llama_vocab *vocab = llama_model_get_vocab(m_model);

    QString prompt = m_history
                     + QStringLiteral("<|im_start|>user\n") + input
                     + QStringLiteral("<|im_end|>\n<|im_start|>assistant\n");

    std::string text = prompt.toStdString();
    std::vector<llama_token> toks(text.size() + 8);
    int n_tok = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                               toks.data(), (int32_t)toks.size(),
                               true, true);
    if (n_tok < 0) return QStringLiteral("tokenize 失败");
    toks.resize(n_tok);

    const int n_batch = 512;
    int n_past = 0;
    for (int i = 0; i < (int)toks.size(); i += n_batch) {
        int n = std::min(n_batch, (int)toks.size() - i);
        llama_batch b = llama_batch_get_one(toks.data() + i, n);
        if (llama_decode(m_ctx, b) != 0) return QStringLiteral("推理出错");
        n_past += n;
    }

    const llama_token eos = llama_token_eos(vocab);
    std::string im_end_str = "<|im_end|>";
    std::vector<llama_token> im_end_toks(im_end_str.size() + 8);
    int n_im = llama_tokenize(vocab, im_end_str.c_str(), (int32_t)im_end_str.size(),
                              im_end_toks.data(), (int32_t)im_end_toks.size(),
                              false, true);
    const llama_token im_end = (n_im == 1) ? im_end_toks[0] : llama_token(-1);

    QString result;
    const int max_gen = 512;

    // 关键：llama_batch_init 出来的 batch 必须显式把 n_seq_id[0]=1，
    // 否则 llama_decode 读到 0 个序列就崩（这是最常见的崩溃点）
    llama_batch gen = llama_batch_init(n_batch, 0, 1);
    gen.n_tokens = 1;
    gen.n_seq_id[0] = 1;      // ← 必须加这行
    gen.seq_id[0][0] = 0;

    for (int step = 0; step < max_gen; ++step) {
        llama_sampler *smpl =
            llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        llama_token id = llama_sampler_sample(smpl, m_ctx, -1);
        llama_sampler_free(smpl);

        if (id == eos || id == im_end) break;

        char buf[16];
        int len = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
        result += QString::fromUtf8(buf, len);

        gen.token[0] = id;
        gen.pos[0] = n_past++;
        if (llama_decode(m_ctx, gen) != 0) break;
    }
    llama_batch_free(gen);

    if (m_history.size() > 8000)
        m_history = m_history.mid(m_history.size() - 4000);
    m_history = prompt + result + QStringLiteral("<|im_end|>\n");
    return result;
}