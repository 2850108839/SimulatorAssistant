#include "LLMWorker.h"
#include "RagEngine.h"
#include <vector>
#include <algorithm>
#include <QByteArray>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QStringConverter>

namespace {
constexpr int kNCtx   = 4096;   // 上下文容量，7B Q4 下显存/内存开销可接受
constexpr int kBatch  = 512;    // prefill 分块大小
}

QString LLMWorker::systemTurn(const QString &prompt) {
    if (prompt.isEmpty()) return QString();
    return QStringLiteral("<|im_start|>system\n") + prompt + QStringLiteral("<|im_end|>\n");
}
QString LLMWorker::userTurn(const QString &text) {
    return QStringLiteral("<|im_start|>user\n") + text + QStringLiteral("<|im_end|>\n");
}
QString LLMWorker::assistOpen() {
    return QStringLiteral("<|im_start|>assistant\n");
}
QString LLMWorker::closeTag() {
    return QStringLiteral("<|im_end|>\n");
}

namespace {
// 不用 QStringList::join，避免 Qt 6 各小版本间 join 重载差异
QString joinAll(const QStringList &parts) {
    QString out;
    for (const QString &p : parts) out += p;
    return out;
}

// 返回从头开始"完整"的 UTF-8 字节数（结尾不完整的序列会被排除）。
// 用于流式输出：一个中文/多字节字符可能被切成多个 token，
// 必须等字节齐了再解码，否则 QString::fromUtf8 会把半截字节变成 "◆" 之类替换符。
int utf8CompleteLen(const QByteArray &b) {
    const int n = b.size();
    int i = 0;
    while (i < n) {
        const uchar c = uchar(b[i]);
        int need;
        if (c < 0x80)               need = 1;   // ASCII
        else if ((c & 0xE0) == 0xC0) need = 2;  // 2 字节
        else if ((c & 0xF0) == 0xE0) need = 3;  // 3 字节（汉字多在此）
        else if ((c & 0xF8) == 0xF0) need = 4;  // 4 字节
        else return i;                          // 非法起始字节：交给兜底 flush
        if (i + need > n) break;                // 结尾序列不完整，保留等下一 token
        i += need;
    }
    return i;
}
}

LLMWorker::LLMWorker(QObject *parent) : QObject(parent) {
    m_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
}

LLMWorker::~LLMWorker() {
    if (m_sampler) llama_sampler_free(m_sampler);
    if (m_ctx)     llama_free(m_ctx);
    if (m_model)   llama_model_free(m_model);
}

void LLMWorker::loadModel(const QString &modelPath) {
    if (m_model && m_ctx) { emit modelLoaded(true); return; }
    m_modelPath = modelPath;          // 会话恢复时校验用

    llama_model_params mp = llama_model_default_params();
    m_model = llama_model_load_from_file(modelPath.toStdString().c_str(), mp);
    if (!m_model) { emit modelLoaded(false); return; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = kNCtx;
    cp.n_batch  = kBatch;
    cp.n_ubatch = kBatch;
    // KV 缓存量化：默认 F16 → Q8_0，显存/内存约减半，质量几乎无损（纯 CPU 也受益）
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
    m_ctx = llama_init_from_model(m_model, cp);   // 新接口，llama_new_context_with_model 已废弃
    if (!m_ctx) { emit modelLoaded(false); return; }

    m_turns.clear();
    m_cached.clear();
    m_needClose = false;
    m_ready     = true;
    emit modelLoaded(true);
}

int LLMWorker::usedTokens() const {
    if (!m_ctx) return 0;
    return (int)(llama_memory_seq_pos_max(llama_get_memory(m_ctx), 0) + 1);
}

bool LLMWorker::decodeTokens(const std::vector<llama_token> &toks) {
    if (toks.empty()) return true;
    // get_one 的 pos 为 NULL，llama.cpp 会从「缓存当前最大位置 + 1」继续写。
    // 因为我们喂的正是增量，这个语义恰好是对的 —— 千万别拿它喂全量历史。
    for (size_t i = 0; i < toks.size(); i += kBatch) {
        int n = std::min<int>(kBatch, (int)(toks.size() - i));
        llama_batch b = llama_batch_get_one(const_cast<llama_token *>(toks.data()) + i, n);
        if (llama_decode(m_ctx, b) != 0) return false;
    }
    return true;
}

void LLMWorker::buildSampler() {
    if (m_sampler) { llama_sampler_free(m_sampler); m_sampler = nullptr; }
    m_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(m_sampler, llama_sampler_init_top_k(m_topK));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_top_p(m_topP, 1));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_temp(m_temperature));
    // 关键：以 dist 结尾，前面的 temp / top_p / top_k 才会真正影响结果。
    // 用 greedy 结尾的话永远取 argmax，采样参数全部失效。
    llama_sampler_chain_add(m_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

bool LLMWorker::rebuild(const QString &userPart) {
    const llama_vocab *vocab = llama_model_get_vocab(m_model);
    const int n_ctx = (int)llama_n_ctx(m_ctx);

    // 按「轮次」丢弃最早的对话，直到放得下（保结构完整，不会切出半句话）
    while (!m_turns.isEmpty()) {
        const QString cand = joinAll(m_turns) + userPart;
        const std::string s = cand.toStdString();
        const int need = -llama_tokenize(vocab, s.c_str(), (int32_t)s.size(),
                                         nullptr, 0, true, true);
        if (need > 0 && need + m_maxTokens <= n_ctx) break;
        m_turns.removeFirst();
    }

    llama_memory_clear(llama_get_memory(m_ctx), true);

    // 重建时同样要把 system 前缀放回最前面（m_turns 里不含 system）
    const QString full = systemTurn(m_systemPrompt) + joinAll(m_turns) + userPart;
    const std::string s = full.toStdString();
    std::vector<llama_token> toks(s.size() + 16);
    const int n = llama_tokenize(vocab, s.c_str(), (int32_t)s.size(),
                                 toks.data(), (int32_t)toks.size(), true, true);
    if (n < 0) { emit errorOccurred(QStringLiteral("tokenize 失败")); return false; }
    toks.resize(n);

    if (!decodeTokens(toks)) { emit errorOccurred(QStringLiteral("重建上下文失败")); return false; }
    m_cached = full;
    return true;
}

bool LLMWorker::ensureRoom(int needTokens) {
    return (usedTokens() + needTokens) <= (int)llama_n_ctx(m_ctx);
}

void LLMWorker::setRagEngine(RagEngine *rag) { m_rag = rag; }
void LLMWorker::setRagEnabled(bool on) { if (m_rag) m_rag->setEnabled(on); }
void LLMWorker::setSystemPrompt(const QString &prompt) { m_systemPrompt = prompt; }

void LLMWorker::chat(const QString &input) {
    if (!m_ready || !m_ctx || !m_model) { emit errorOccurred(QStringLiteral("模型未加载")); return; }
    m_stopRequested = false;

    const llama_vocab *vocab = llama_model_get_vocab(m_model);
    const int n_ctx = (int)llama_n_ctx(m_ctx);

    // RAG 检索：在拼 prompt 之前，先从知识库混合检索 top-k 片段，拼成「参考资料」前缀。
    // 这一步只改喂给模型的文字，不动 KV 缓存、采样链、生成循环。
    // 低于相关性阈值的命中会被 retrieve() 整体丢弃（res.used=false），避免注入噪声/幻觉来源。
    QString effectiveInput = input;
    if (m_rag && m_rag->enabled()) {
        RagResult res = m_rag->retrieve(input, 3);
        if (res.used) {
            effectiveInput = res.context + input;
            emit ragRetrieved(res.sources);   // 让界面明确提示本次用到了哪些知识库文件
        }
    }

    // 本轮要喂进去的「增量」：
    //   closePart —— 上一轮 assistant 结尾的 <|im_end|>\n（生成时在 eog 前就停了，没收进缓存）
    //   userPart  —— 本轮 user 提问 + assistant 开头
    const QString closePart = m_needClose ? closeTag() : QString();
    const QString userPart  = userTurn(effectiveInput) + assistOpen();
    // 首轮（缓存为空）把 system 前缀一并喂入并进缓存；后续轮次不再重复，避免污染增量
    const bool    firstTurn = (usedTokens() == 0);
    const QString sysPart   = firstTurn ? systemTurn(m_systemPrompt) : QString();
    const QString delta     = closePart + sysPart + userPart;

    // tokenize：只有缓存为空（第一轮）才加 BOS；parse_special 打开才能把 <|im_end|> 识别成特殊 token
    const bool addBos = firstTurn;
    std::string text = delta.toStdString();
    std::vector<llama_token> toks(text.size() + 16);
    int n_tok = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                               toks.data(), (int32_t)toks.size(), addBos, true);
    if (n_tok < 0) { emit errorOccurred(QStringLiteral("tokenize 失败")); return; }
    toks.resize(n_tok);

    // 容量检查：放不下就丢最早的轮次并重建（重建内容里已含 userPart，不再需要 closePart）
    if (!ensureRoom(n_tok + m_maxTokens)) {
        if (!rebuild(userPart)) {
            m_cached.clear();
            m_needClose = false;
            return;
        }
    } else {
        if (!decodeTokens(toks)) { emit errorOccurred(QStringLiteral("推理出错")); return; }
        m_cached += delta;
    }
    m_needClose = false;

    buildSampler();

    // 生成循环：每轮 1 个 token，emit tokenReady → 界面打字机
    // 用字节缓冲累积：多字节字符可能被切到两个 token 里，等字节完整再解码，
    // 否则 QString::fromUtf8 会把半截字节解成 "◆" 之类替换符
    QString result;
    QByteArray pending;
    for (int step = 0; step < m_maxTokens; ++step) {
        if (m_stopRequested.load()) break;                       // 用户点了停止
        if (usedTokens() >= n_ctx) break;                        // 缓存已满，体面收场

        llama_token id = llama_sampler_sample(m_sampler, m_ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;                // <|im_end|> / eos

        char buf[256];
        const int len = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (len <= 0) break;
        pending.append(buf, len);

        const int complete = utf8CompleteLen(pending);           // 只解码完整序列
        if (complete > 0) {
            const QString piece = QString::fromUtf8(pending.constData(), complete);
            result += piece;
            emit tokenReady(piece);                              // ← 流式核心
            pending.remove(0, complete);
        } else if (pending.size() > 1024) {
            // 极端兜底：首字节非法导致无法前进时强制冲刷，避免无限累积
            const QString piece = QString::fromUtf8(pending);
            result += piece;
            emit tokenReady(piece);
            pending.clear();
        }

        if (llama_decode(m_ctx, llama_batch_get_one(&id, 1)) != 0) break;
    }
    if (!pending.isEmpty()) {                                    // 冲刷尾部残余（不完整序列）
        const QString piece = QString::fromUtf8(pending);
        result += piece;
        emit tokenReady(piece);
    }

    // 记录本轮（整轮保存，便于以后溢出重建）；缓存里的文本同步累加
    m_turns.append(userTurn(input) + assistOpen() + result + closeTag());
    m_cached += result;
    m_needClose = true;      // 生成在 eog 前就停了，收尾符留到下一轮补进缓存

    emit replyFinished(result);
}

void LLMWorker::stop() { m_stopRequested.store(true); }

void LLMWorker::setParams(float temperature, float top_p, int top_k, int max_tokens) {
    m_temperature = temperature;
    m_topP        = top_p;
    m_topK        = top_k;
    m_maxTokens   = max_tokens;
}

void LLMWorker::reset() {
    m_turns.clear();
    m_cached.clear();
    m_needClose = false;
    if (m_ctx) llama_memory_clear(llama_get_memory(m_ctx), true);
    m_stopRequested = true;
}

// ===== 会话持久化：保存/恢复 =====
// 面试点：KV 是模型层的运行时状态，保存它 = 记住"上下文计算到哪了"；
//         恢复后新问题从 usedTokens() 继续，增量语义自然延续，不用重算历史。
void LLMWorker::saveSession(const QString &dirPath) {
    if (!m_ready || !m_ctx || !m_model) {
        emit sessionSaved(false, QStringLiteral("模型未加载，无法保存"));
        return;
    }
    QDir dir(dirPath);
    if (!dir.exists() && !dir.mkpath(QStringLiteral("."))) {
        emit sessionSaved(false, QStringLiteral("目录不存在且创建失败：%1").arg(dirPath));
        return;
    }

    // 1) 构造与 KV 缓存严格一一对应的 token 序列：
    //    m_cached 是"已进缓存文本"，m_needClose 表示最后那个 <|im_end|> 还没进，
    //    保存时必须补上，恢复时才能精确重建。
    const QString kvText = m_cached + (m_needClose ? closeTag() : QString());
    const llama_vocab *vocab = llama_model_get_vocab(m_model);
    const std::string s = kvText.toStdString();
    std::vector<llama_token> toks(s.size() + 16);
    const int n = llama_tokenize(vocab, s.c_str(), (int32_t)s.size(),
                                 toks.data(), (int32_t)toks.size(), false, true);
    if (n < 0) { emit sessionSaved(false, QStringLiteral("tokenize 失败")); return; }
    toks.resize(n);

    // 2) 落盘 KV（含采样 RNG）。返回 0 表示失败。
    const QString statePath = dirPath + QStringLiteral("/session.state");
    if (llama_state_save_file(m_ctx, statePath.toStdString().c_str(),
                              toks.data(), toks.size()) == 0) {
        emit sessionSaved(false, QStringLiteral("state 写入失败：%1").arg(statePath));
        return;
    }

    // 3) 元数据：第1行模型路径；第2行 needClose(0/1)；第3行 m_turns（0x1F 分隔）
    QFile meta(dirPath + QStringLiteral("/session.meta"));
    if (!meta.open(QIODevice::WriteOnly | QIODevice::Text)) {
        emit sessionSaved(false, QStringLiteral("meta 写入失败"));
        return;
    }
    QTextStream ts(&meta);
    ts.setEncoding(QStringConverter::Utf8);
    ts << m_modelPath << '\n'
       << (m_needClose ? QLatin1Char('1') : QLatin1Char('0')) << '\n'
       << m_turns.join(QString(QChar(0x1F))) << '\n';
    meta.close();

    emit sessionSaved(true, QStringLiteral("会话已保存（%1 轮，%2 tokens）")
                     .arg(m_turns.size()).arg((int)toks.size()));
}

// 恢复会话：先校验模型路径，再按 token 精确匹配加载 KV；
// 任一校验不过就安全回退为新会话，绝不把错位 KV 当历史用。
void LLMWorker::loadSession(const QString &dirPath) {
    if (!m_ready || !m_ctx || !m_model) {
        emit sessionLoaded(false, 0, QStringLiteral("模型未加载，无法恢复会话"));
        return;
    }

    // 1) 读元数据
    QFile meta(dirPath + QStringLiteral("/session.meta"));
    if (!meta.open(QIODevice::ReadOnly | QIODevice::Text)) {
        emit sessionLoaded(false, 0, QStringLiteral("未找到会话文件"));
        return;
    }
    QTextStream ts(&meta);
    ts.setEncoding(QStringConverter::Utf8);
    const QString savedModel = ts.readLine().trimmed();
    const bool needClose = ts.readLine().trimmed() == QLatin1String("1");
    QStringList turns = ts.readLine().split(QChar(0x1F), Qt::SkipEmptyParts);
    meta.close();

    // 2) 模型强绑定校验：KV 是模型层权重+词表的产物，换模型/换路径即失效
    if (savedModel != m_modelPath) {
        emit sessionLoaded(false, 0, QStringLiteral("模型路径与上次保存不一致，已放弃恢复"));
        return;
    }

    // 3) 从元数据重建 KV 对应文本：system 前缀 + 全部轮次；
    //    needClose=true 说明最后那个 closeTag 尚未进缓存，要从重建文本里减掉。
    QString kvText = systemTurn(m_systemPrompt) + joinAll(turns);
    if (needClose && !turns.isEmpty()) kvText.chop(closeTag().length());

    const llama_vocab *vocab = llama_model_get_vocab(m_model);
    const std::string s = kvText.toStdString();
    std::vector<llama_token> toks(s.size() + 16);
    const int n = llama_tokenize(vocab, s.c_str(), (int32_t)s.size(),
                                 toks.data(), (int32_t)toks.size(), false, true);
    if (n < 0) { emit sessionLoaded(false, 0, QStringLiteral("tokenize 失败")); return; }
    toks.resize(n);

    // 4) 加载 KV；nOut = 实际匹配的 token 数
    const QString statePath = dirPath + QStringLiteral("/session.state");
    size_t nOut = 0;
    if (llama_state_load_file(m_ctx, statePath.toStdString().c_str(),
                              toks.data(), toks.size(), &nOut) == 0) {
        emit sessionLoaded(false, 0, QStringLiteral("state 读取失败"));
        return;
    }

    // 5) 精确匹配校验：不匹配说明 n_ctx / 系统提示 / 词表已变，硬用会重蹈"位置错位"
    if (nOut != toks.size()) {
        llama_memory_clear(llama_get_memory(m_ctx), true);
        turns.clear();
        m_cached.clear();
        m_needClose = false;
        emit sessionLoaded(false, 0, QStringLiteral("会话状态不匹配（n_ctx/系统提示可能已变），已开启新会话"));
        return;
    }

    m_turns = turns;
    m_cached = kvText;        // 恢复后 m_cached 与 KV 严格一致，增量语义无缝续接
    m_needClose = needClose;
    emit sessionLoaded(true, (int)turns.size(),
                       QStringLiteral("会话已恢复（%1 轮，%2 tokens）")
                       .arg(turns.size()).arg((int)toks.size()));
}
