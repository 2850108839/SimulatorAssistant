#ifndef LLMWorker_H
#define LLMWorker_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <atomic>
#include <vector>
#include "llama.h"

class RagEngine;   // 前向声明，避免头文件耦合

// 运行在子线程的推理 worker。
// KV 缓存策略（对齐 llama.cpp 官方 simple-chat 的做法）：
//   1) 缓存是「增量追加」的：每轮只把新增文本喂进去，历史不重算。
//   2) 每步 decode 前显式检查剩余容量，绝不等底层报错。
//   3) 空间不足时按「轮次」丢弃最早的对话并重建（保结构完整，不是按字符硬切）。
//   4) 采样链以 dist 结尾，temperature / top_p / top_k 才真正生效。
class LLMWorker : public QObject {
    Q_OBJECT
public:
    explicit LLMWorker(QObject *parent = nullptr);
    ~LLMWorker();

public slots:   // 这些槽都会跑在子线程
    void loadModel(const QString &modelPath);
    void chat(const QString &input);
    void stop();                       // 停止生成（原子量，线程安全）
    void setParams(float temperature, float top_p, int top_k, int max_tokens);
    void reset();                      // 清空对话历史与 KV 缓存

    void saveSession(const QString &dirPath);   // 保存 KV + 元数据到目录
    void loadSession(const QString &dirPath);   // 恢复会话（模型强绑定校验）

    void setRagEngine(RagEngine *rag); // 注入 RAG 引擎（MainWindow 持有）
    void setRagEnabled(bool on);       // 开/关 RAG 检索
    void setSystemPrompt(const QString &prompt);   // 设置系统提示（强制依据参考资料作答）

signals:
    void modelLoaded(bool ok);
    void tokenReady(const QString &piece);          // 每生成 1 个 token 发一次
    void replyFinished(const QString &fullReply);
    void errorOccurred(const QString &message);
    void ragRetrieved(const QStringList &sources);   // 本轮从知识库检索到的命中来源（空=未使用）
    void sessionSaved(bool ok, const QString &msg);
    void sessionLoaded(bool ok, int turns, const QString &msg);

private:
    // ---- 上下文容量管理 ----
    int  usedTokens() const;                       // 当前缓存已占用的 token 数
    bool ensureRoom(int needTokens);               // 空间不够就丢最早的轮次并重建
    bool rebuild(const QString &userPart);         // 清空缓存 + 按保留的轮次全量重填
    bool decodeTokens(const std::vector<llama_token> &toks);   // 分块 prefill
    void buildSampler();                           // 按当前参数重建采样链

    static QString systemTurn(const QString &prompt); // <|im_start|>system\n...<|im_end|>\n
    static QString userTurn(const QString &text);   // <|im_start|>user\n...<|im_end|>\n
    static QString assistOpen();                    // <|im_start|>assistant\n
    static QString closeTag();                      // <|im_end|>\n

    llama_model   *m_model   = nullptr;
    llama_context *m_ctx     = nullptr;
    llama_sampler *m_sampler = nullptr;

    QStringList m_turns;        // 已完成的轮次（整轮文本，用于溢出时重建）
    QString     m_systemPrompt; // 系统提示；首轮与重建时作为前缀进缓存
    QString     m_cached;       // 与 KV 缓存内容严格一一对应的文本
    bool        m_needClose = false;   // 上一轮 assistant 结尾的 <|im_end|>\n 还没进缓存
    bool        m_ready     = false;   // 模型与上下文是否就绪

    float  m_temperature = 0.7f;
    float  m_topP        = 0.9f;
    int    m_topK        = 40;
    int    m_maxTokens   = 512;
    std::atomic<bool> m_stopRequested{false};

    RagEngine *m_rag = nullptr;       // 非拥有；由 MainWindow 管理生命周期

    QString m_modelPath;   // loadModel 时记录，会话恢复时做模型强绑定校验
};

#endif // LLMWorker_H
