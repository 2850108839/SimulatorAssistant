#pragma once
#include <QString>
#include <QStringList>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <QFileInfo>
#include <QCryptographicHash>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDateTime>
#include "OnnxEmbedder.h"   // IEmbedder 接口（llama.cpp / ONNX Runtime 双后端）

// 知识库中的一个分块
struct RagChunk {
    QString source;              // 来源文件名
    QString text;                // 分块原文
    std::vector<float> vec;      // bge 编码后的向量（已 L2 归一化）
};

// 检索结果：含拼好的参考资料文本、来源溯源、质量指标（用于界面提示与离线评测）
struct RagResult {
    QString context;            // 拼好的【参考资料】文本；未达相关性阈值则为空
    QStringList sources;        // 命中来源文件名（按相关度，已去重）
    std::vector<float> scores;  // 各命中片段的综合相似度（向量 + BM25 融合，已归一化）
    float bestVectorScore = 0.f; // top-1 向量余弦（相关性门槛判定的依据）
    float latencyMs = 0.f;       // 检索耗时（毫秒）
    bool used = false;           // 是否达到相关性阈值（达到才应注入 prompt，避免注入噪声）
};

// 索引快照：分块 + BM25 词频统计，构建一次后不可变，检索时以 shared_ptr 原子抓取，
// 避免每次查询复制全部向量，也避免后台重建与检索之间的数据竞争。
struct IndexSnapshot {
    std::vector<RagChunk> chunks;
    std::unordered_map<QString,int> termDf;                       // 全局词项文档频率
    std::vector<std::unordered_map<QString,int>> docTf;           // 每文档词项频率
    std::vector<int> docLen;                                      // 每文档词项数
    int nDocs = 0;
    float avgdl = 0.f;
};

// 轻量 RAG 引擎：基于 llama.cpp 加载 bge 嵌入模型，对本地知识库做
// 切块 -> 编码建索引 -> 查询时混合检索（向量 top-k 候选 + BM25 词法融合重排）top-k 片段。
// 检索结果以纯文本形式返回，由调用方拼进 prompt（不改变模型本身）。
class RagEngine {
public:
    RagEngine();
    ~RagEngine();

    // 加载 bge 嵌入模型。后端由调用方决定：
    //  loadModel()    —— llama.cpp 后端（bge GGUF，默认）
    //  loadOnnxModel()—— ONNX Runtime 后端（bge .onnx + 同目录 vocab.txt，方案 A）
    // 二者互斥，后加载者替换前者。成功返回 true
    bool loadModel(const QString &modelPath);
    bool loadOnnxModel(const QString &modelPath);

    // 同步建索引并安装（保留旧 API，供测试/兜底使用）
    int loadKnowledgeBase(const QString &dir);

    // 纯构建：只读取目录、编码、返回分块，不替换当前索引。供后台线程调用。
    // onProgress(done,total)：每处理完一个文件回调一次（在后台线程调用，调用方需自行切回 GUI 线程）。
    std::vector<RagChunk> buildIndex(const QString &dir,
                                     const std::function<void(int,int)> &onProgress = {});

    // 原子安装：加锁整体替换索引 + 重建 BM25 统计。应在主线程调用。
    void installIndex(std::vector<RagChunk> &&idx);

    // 检索：向量 top-k 候选 + BM25 词法融合重排（混合检索），低于阈值的命中整体丢弃。
    RagResult retrieve(const QString &query, int topK = 3, float threshold = 0.22f);

    bool isModelLoaded() const { return m_embedder && m_embedder->isLoaded(); }
    bool isIndexed()     const;
    int  chunkCount()    const;

    void setEnabled(bool on) { m_enabled.store(on); }
    bool enabled() const { return m_enabled.load(); }

    void setRetrievalThreshold(float t) { m_threshold = t; }
    void setHybridAlpha(float a)        { m_alpha = a; }
    // 设置检索指标日志路径（JSONL，每行一条）；置空关闭
    void setMetricsLogPath(const QString &path) { m_metricsPath = path; }

    // 索引缓存（向量库持久化）：把 encode 后的向量 + 分块原文落到磁盘，
    // 重启时命中缓存即可跳过整库 bge 编码，大幅缩短启动时间。
    // 缓存文件放在知识库目录的【父目录】下（隐藏文件），避免被目录监听器误触发重建。
    bool saveIndexCache(const QString &dir, const std::vector<RagChunk> &index, const QString &sig) const;
    bool loadIndexCache(const QString &dir, std::vector<RagChunk> &out, const QString &sig) const;

private:
    // 单句编码：tokenize -> llama_encode(embedding) -> 取 [CLS] 池化向量 -> L2 归一化
    std::vector<float> embed(const QString &text);

    // 批量编码：把多句打包进同一次 llama_encode（不同 seq_id），减少调用次数。
    // 检索路径仍用单句 embed()；此处只服务建索引，不改变检索逻辑。
    // 自动按 n_ctx 贪心分组，避免单批超出 bge 最大序列长度。
    std::vector<std::vector<float>> embedBatch(const QStringList &texts);

    static float dot(const std::vector<float> &a, const std::vector<float> &b);

    // 语义边界分块：按句号/问号/感叹号/分号/换行切句，再贪心合并成 ≤ kChunkChars 的块
    static QStringList splitChunks(const QString &text);
    // BM25 中文词项切分（字级基线）：CJK 每字一词项，Latin 连续字母数字一词项
    static QStringList tokenizeTerms(const QString &text);
    // 建快照 + 预计算 BM25 统计（建索引时只算一次，检索时直接查表，O(候选) 而非 O(全库)）
    static IndexSnapshot buildSnapshot(std::vector<RagChunk> &&chunks);
    static float bm25Score(const IndexSnapshot &s, int docId, const QStringList &qterms);

    // 从 .docx 抽取纯文本（解压 word/document.xml 并去标签）
    static QString extractDocxText(const QString &filePath);
    // 从老版 .doc（二进制 OLE2）尽力而为地提取正文（无外部依赖）
    static QString extractDocBinaryText(const QString &filePath);
    static QString stripXmlText(const QString &xml);

    // 知识库内容签名：文件清单(路径|大小|修改时间)的 Sha1，用于判断缓存是否失效
    static QString computeKbSignature(const QFileInfoList &files);
    // 缓存文件落点：放在 kbDir 父目录下，文件名带 kbDir 的 Sha1，互不干扰且不被监听
    static QString cachePathFor(const QString &dir);

    void logMetrics(const QString &query, const RagResult &r) const;

    std::unique_ptr<IEmbedder> m_embedder;   // 嵌入后端（llama.cpp / ONNX Runtime 二选一）
    int m_nEmbd = 0;                         // 嵌入维度（缓存维度一致性校验用）
    std::atomic<bool>   m_enabled{false};

    std::shared_ptr<IndexSnapshot> m_snapshot;   // 当前索引（不可变快照，原子替换）
    std::mutex m_mutex;        // 保护 m_snapshot 的替换
    std::mutex m_encodeMutex;  // 串行化 embed()（嵌入后端不可重入）

    float m_threshold = 0.22f; // 向量余弦门槛：低于则视为"知识库未覆盖"，不注入参考资料
    float m_alpha = 0.6f;      // 混合检索融合权重：向量 0.6 + BM25 0.4
    QString m_metricsPath;     // 检索指标日志（JSONL），空=不记录
};
