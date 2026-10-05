#include "RagEngine.h"
#include "LlamaEmbedder.h"
#include <cmath>
#include <algorithm>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFileInfoList>
#include <QTextStream>
#include <QRegularExpression>
#include <QDataStream>
#include <QJsonArray>
#include <QtGui/private/qzipreader_p.h>
#include <QDebug>

namespace {
constexpr int     kChunkChars = 240; // 每块约 240 字（中文安全低于 512 token）
constexpr int     kOverlap  = 40;   // 块间重叠，避免关键信息被边界切断
constexpr quint32 kCacheVersion = 4; // 索引缓存格式版本；提取/分块/检索逻辑变更时必须 +1 以失效旧缓存
const QStringView kQueryPrefix = u"为这个句子生成表示以用来检索相关文章："; // bge-zh 官方检索指令前缀

// .doc 二进制提取文本的专用清理：只折叠空白，**绝不做 <...> 去标签**。
// stripXmlText 的 <[^>]*> 是给 XML 用的；二进制乱码里散落的 '<' '>' 会远距离配对，
// 把中间大段正文整段删除（实测吞掉 48% 内容，含"转发监控"等关键词）。
QString normalizeExtractedText(const QString &s) {
    QString t = s;
    t.replace(QRegularExpression(QStringLiteral("[ \\t\\x0B\\f\\r]+")), QStringLiteral(" "));
    t.replace(QRegularExpression(QStringLiteral("\\n{3,}")), QStringLiteral("\n\n"));
    return t.trimmed();
}

// --- 老版 .doc（二进制 OLE2）的"尽力而为"文本提取辅助 ---
// 主路径：扫描 UTF-16LE 文本流（Word 正文基本是 UTF-16LE），把连续可打印字符抠出来
bool isDocTextChar(quint16 c) {
    if (c < 0x20) return false;                   // 控制字符
    if (c == 0x0A || c == 0x0D || c == 0x0B || c == 0x0C) return false; // 换行类当分隔
    if (c <= 0x7E) return true;                   // ASCII 可打印
    if (c >= 0x00A0 && c <= 0x00FF) return true;  // Latin-1 补充（CP1252 单字节在 UTF-16LE 中）
    if (c >= 0x2000 && c <= 0x206F) return true;  // 一般标点符号
    if (c >= 0x3000 && c <= 0x303F) return true;  // CJK 标点
    // 0x3400-0x4DBF（CJK 扩展 A）故意不放行：老 .doc 二进制流被误读成 UTF-16LE 时，
    // 乱码大量落在该区，放行会让近半正文变成垃圾块；现代软件手册基本不用扩展 A 汉字。
    if (c >= 0x4E00 && c <= 0x9FFF) return true;  // CJK 基本汉字
    if (c >= 0xFF00 && c <= 0xFFEF) return true;  // 全角字符
    return false;
}

QString extractUtf16LeRuns(const QByteArray &data) {
    const uchar *p = reinterpret_cast<const uchar *>(data.constData());
    QString out, run;
    // 最小连续串 2→3：丢掉二进制散落的 1~2 字符孤立噪声（真实内容多以更长串出现）
    auto flush = [&]() { if (run.length() >= 3) { out += run; out += QChar('\n'); } run.clear(); };
    for (int i = 0; i + 1 < data.size(); i += 2) {
        const quint16 c = quint16(p[i]) | (quint16(p[i + 1]) << 8); // 显式小端拼装，跨平台正确
        if (isDocTextChar(c)) run += QChar(c);
        else flush();
    }
    flush();
    return out;
}

QString extractLatin1Runs(const QByteArray &data) {
    QString out, run;
    auto flush = [&]() { if (run.length() >= 4) { out += run; out += QChar('\n'); } run.clear(); };
    for (char ch : data) {
        const uchar b = uchar(ch);
        if ((b >= 0x20 && b < 0x7F) || b >= 0xA0) run += QChar(b);
        else flush();
    }
    flush();
    return out;
}
}

RagEngine::RagEngine() {}
RagEngine::~RagEngine() {
    m_embedder.reset();   // 后端析构（llama 模型 / ORT session）
}

bool RagEngine::loadModel(const QString &modelPath) {
    auto emb = std::make_unique<LlamaEmbedder>();
    if (!emb->loadModel(modelPath.toStdString())) {
        qWarning() << "[RAG] llama.cpp 嵌入模型加载失败:" << modelPath;
        return false;
    }
    m_nEmbd = emb->dim();
    m_embedder = std::move(emb);
    qDebug() << "[RAG] 后端 = llama.cpp，bge 已加载，维度 =" << m_nEmbd;
    return true;
}

bool RagEngine::loadOnnxModel(const QString &modelPath) {
    auto emb = std::make_unique<OnnxEmbedder>();
    if (!emb->loadModel(modelPath.toStdString())) {
        qWarning() << "[RAG] ONNX Runtime 嵌入模型加载失败:" << modelPath;
        return false;
    }
    m_nEmbd = emb->dim();
    m_embedder = std::move(emb);
    qDebug() << "[RAG] 后端 = ONNX Runtime，bge 已加载，维度 =" << m_nEmbd;
    return true;
}

std::vector<float> RagEngine::embed(const QString &text) {
    if (!m_embedder || !m_embedder->isLoaded()) return {};
    // 嵌入后端不可重入：建索引（后台线程）与检索（worker 线程）的编码必须串行
    std::lock_guard<std::mutex> encLock(m_encodeMutex);
    std::vector<float> out;
    m_embedder->embed(text.toStdString(), out, m_nEmbd);
    return out;
}

std::vector<std::vector<float>> RagEngine::embedBatch(const QStringList &texts) {
    if (!m_embedder || !m_embedder->isLoaded()) return {};
    std::lock_guard<std::mutex> encLock(m_encodeMutex);
    std::vector<std::string> t;
    t.reserve((size_t)texts.size());
    for (const QString &s : texts) t.push_back(s.toStdString());
    return m_embedder->embedBatch(t, m_nEmbd);
}

// 语义边界分块：先按句子结束符切成最小语义单元，再贪心合并成不超过 kChunkChars 的块，
// 块间保留 kOverlap 字符的重叠，避免关键信息被块边界切断。相比纯"按字数切行"更贴合语义。
QStringList RagEngine::splitChunks(const QString &text) {
    QStringList out;
    QStringList sents = text.split(
        QRegularExpression(QStringLiteral("[。！？!?；;\\r\\n]+")), Qt::SkipEmptyParts);
    QString cur;
    for (const QString &raw : sents) {
        QString piece = raw.trimmed();
        if (piece.isEmpty()) continue;
        if (!cur.isEmpty() && cur.length() + piece.length() > kChunkChars) {
            out << cur.trimmed();
            cur = cur.right(kOverlap) + piece;   // 下一块带重叠前缀
        } else {
            cur = cur.isEmpty() ? piece : cur + piece;
        }
    }
    if (!cur.trimmed().isEmpty()) out << cur.trimmed();
    return out;
}

// BM25 中文词项切分（字级基线）：CJK / 全角 / CJK 标点每字一个词项；Latin 连续字母数字作为一个词项。
// 这是工业 RAG 里"混合检索"词法那一腿的离线可用子集（不依赖分词器即可对中文生效）。
QStringList RagEngine::tokenizeTerms(const QString &text) {
    QStringList out;
    const int n = text.length();
    QString lat;
    auto flushLat = [&]() { if (!lat.isEmpty()) { out << lat.toLower(); lat.clear(); } };
    for (int i = 0; i < n; ++i) {
        const QChar ch = text.at(i);
        const ushort u = ch.unicode();
        if ((u >= 0x4E00 && u <= 0x9FFF) ||
            (u >= 0x3400 && u <= 0x4DBF) ||
            (u >= 0x3000 && u <= 0x303F) ||
            (u >= 0xFF00 && u <= 0xFFEF)) {
            flushLat();
            out << QString(ch);                  // 每个 CJK 字作为一个词项
        } else if (ch.isLetterOrNumber()) {
            lat += ch;                           // 累积 Latin 词
        } else {
            flushLat();                          // 其它字符（标点/空白）作分隔
        }
    }
    flushLat();
    return out;
}

IndexSnapshot RagEngine::buildSnapshot(std::vector<RagChunk> &&chunks) {
    IndexSnapshot s;
    s.chunks = std::move(chunks);
    s.nDocs = (int)s.chunks.size();
    s.docTf.reserve(s.nDocs);
    long long totalLen = 0;
    for (auto &rc : s.chunks) {
        const QStringList terms = tokenizeTerms(rc.text);
        std::unordered_map<QString,int> tf;
        for (const QString &t : terms) tf[t]++;
        s.docTf.push_back(tf);
        s.docLen.push_back((int)terms.size());
        totalLen += (long long)terms.size();
        for (const auto &kv : tf) s.termDf[kv.first]++;
    }
    s.avgdl = s.nDocs ? (float)((double)totalLen / (double)s.nDocs) : 0.f;
    return s;
}

float RagEngine::bm25Score(const IndexSnapshot &s, int docId, const QStringList &qterms) {
    const auto &df = s.docTf[docId];
    const int dl = s.docLen[docId];
    const float avgdl = s.avgdl > 0.f ? s.avgdl : 1.f;
    const float k1 = 1.5f, b = 0.75f;
    float score = 0.f;
    for (const QString &t : qterms) {
        auto it = s.termDf.find(t);
        if (it == s.termDf.end()) continue;          // 语料未出现的词项无贡献
        auto dit = df.find(t);
        if (dit == df.end()) continue;               // 该文档不含此词
        const int f = dit->second;
        const float idf = std::log(1.f + (s.nDocs - it->second + 0.5f) / (it->second + 0.5f));
        score += idf * (f * (k1 + 1.f)) / (f + k1 * (1.f - b + b * ((float)dl / avgdl)));
    }
    return score;
}

std::vector<RagChunk> RagEngine::buildIndex(const QString &dir, const std::function<void(int,int)> &onProgress) {
    std::vector<RagChunk> newIndex;
    if (!m_embedder || !m_embedder->isLoaded()) { qWarning() << "[RAG] 模型未加载，无法建索引"; return newIndex; }

    QDir root(dir);
    if (!root.exists()) { qWarning() << "[RAG] 知识库目录不存在:" << dir; return newIndex; }

    QStringList filters; filters << QStringLiteral("*.txt") << QStringLiteral("*.md")
                   << QStringLiteral("*.docx") << QStringLiteral("*.doc");
    QFileInfoList files = root.entryInfoList(filters, QDir::Files | QDir::Readable);
    // 递归进入子目录
    for (const QFileInfo &d : root.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot))
        files += QDir(d.absoluteFilePath()).entryInfoList(filters, QDir::Files | QDir::Readable);

    // 优先命中向量缓存：知识库未变动时直接加载分块+向量，跳过整库 bge 编码（启动加速关键）
    const QString sig = computeKbSignature(files);
    if (loadIndexCache(dir, newIndex, sig)) {
        qDebug() << "[RAG] 命中索引缓存，跳过编码，分块数 =" << newIndex.size();
        return newIndex;
    }

    // 未命中缓存：先建到临时索引，最后再整体替换，避免检索线程读到半成品
    newIndex.clear();

    // 收集 (来源, 分块文本)，最后统一批量编码，把 llama_encode 调用次数降一个数量级
    struct Item { QString source; QString text; };
    std::vector<Item> items;
    items.reserve(files.size());
    const int total = (int)files.size();
    int done = 0;
    for (const QFileInfo &fi : files) {
        QString content;
        const QString suffix = fi.suffix().toLower();
        if (suffix == QStringLiteral("docx")) {
            content = extractDocxText(fi.absoluteFilePath());        // Word 新格式（OOXML）
        } else if (suffix == QStringLiteral("doc")) {
            content = extractDocBinaryText(fi.absoluteFilePath());    // Word 老版二进制（尽力而为）
        } else {
            QFile f(fi.absoluteFilePath());
            if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) { ++done; if (onProgress) onProgress(done, total); continue; }
            QTextStream ts(&f); ts.setAutoDetectUnicode(true);
            content = ts.readAll();
            f.close();
        }
        if (!content.trimmed().isEmpty()) {
            for (const QString &c : splitChunks(content)) {
                if (c.trimmed().isEmpty()) continue;
                items.push_back({fi.fileName(), c.trimmed()});
            }
        }
        ++done;
        if (onProgress) onProgress(done, total);
    }

    // 批量编码：每批贪心塞满 bge 的 n_ctx，整库分块数 >> 批次数，建库明显提速
    if (!items.empty()) {
        QStringList texts;
        texts.reserve((int)items.size());
        for (const auto &it : items) texts.append(it.text);
        std::vector<std::vector<float>> vecs = embedBatch(texts);
        for (size_t k = 0; k < items.size() && k < vecs.size(); ++k) {
            if (vecs[k].empty()) continue;          // 编码失败的分块跳过
            RagChunk rc;
            rc.source = items[k].source;
            rc.text   = items[k].text;
            rc.vec    = std::move(vecs[k]);
            newIndex.push_back(std::move(rc));
        }
    }

    // 重建成功：落盘向量缓存，下次重启直接命中（仅当确有内容时写，避免空缓存）
    if (!newIndex.empty()) saveIndexCache(dir, newIndex, sig);
    qDebug() << "[RAG] 知识库建索引完成，分块数 =" << newIndex.size();
    return newIndex;
}

int RagEngine::loadKnowledgeBase(const QString &dir) {
    auto idx = buildIndex(dir);
    installIndex(std::move(idx));
    return chunkCount();
}

void RagEngine::installIndex(std::vector<RagChunk> &&idx) {
    auto snap = std::make_shared<IndexSnapshot>(buildSnapshot(std::move(idx)));
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_snapshot = std::move(snap);
    }
}

QString RagEngine::computeKbSignature(const QFileInfoList &files) {
    QCryptographicHash h(QCryptographicHash::Sha1);
    QFileInfoList sorted = files;
    std::sort(sorted.begin(), sorted.end(),
              [](const QFileInfo &a, const QFileInfo &b) {
                  return a.absoluteFilePath() < b.absoluteFilePath();
              });
    for (const QFileInfo &fi : sorted) {
        const QString entry = fi.absoluteFilePath() + QLatin1Char('|')
                            + QString::number(fi.size()) + QLatin1Char('|')
                            + QString::number(fi.lastModified().toMSecsSinceEpoch());
        h.addData(entry.toUtf8());
    }
    return QString::fromLatin1(h.result().toHex());
}

QString RagEngine::cachePathFor(const QString &dir) {
    const QByteArray h = QCryptographicHash::hash(dir.toUtf8(), QCryptographicHash::Sha1).toHex();
    // 放在 kbDir 的父目录下（隐藏文件），避免被监听 kbDir 的 QFileSystemWatcher 误触发重建
    return QFileInfo(dir).absolutePath() + QStringLiteral("/.rag_cache_") + QString::fromLatin1(h) + QStringLiteral(".bin");
}

bool RagEngine::saveIndexCache(const QString &dir, const std::vector<RagChunk> &index, const QString &sig) const {
    const QString path = cachePathFor(dir);
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly)) { qWarning() << "[RAG] 索引缓存写入失败:" << path; return false; }
    QDataStream out(&f);
    out.setByteOrder(QDataStream::LittleEndian);
    out.setVersion(QDataStream::Qt_6_5);
    const char magic[4] = {'R','A','G','C'};
    out.writeRawData(magic, 4);
    out << kCacheVersion << qint32(m_nEmbd);
    const QByteArray sigBa = sig.toUtf8();
    out << qint32(sigBa.size());
    out.writeRawData(sigBa.constData(), sigBa.size());
    out << qint32(qint32(index.size()));
    for (const RagChunk &rc : index) {
        const QByteArray src = rc.source.toUtf8();
        out << qint32(src.size()); out.writeRawData(src.constData(), src.size());
        const QByteArray txt = rc.text.toUtf8();
        out << qint32(txt.size()); out.writeRawData(txt.constData(), txt.size());
        out.writeRawData(reinterpret_cast<const char *>(rc.vec.data()),
                         qint32(rc.vec.size() * sizeof(float)));
    }
    f.close();
    qDebug() << "[RAG] 索引缓存已保存:" << path << "分块数 =" << index.size();
    return true;
}

bool RagEngine::loadIndexCache(const QString &dir, std::vector<RagChunk> &out, const QString &sig) const {
    const QString path = cachePathFor(dir);
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return false;
    QDataStream in(&f);
    in.setByteOrder(QDataStream::LittleEndian);
    in.setVersion(QDataStream::Qt_6_5);
    char magic[4] = {0};
    if (in.readRawData(magic, 4) != 4) return false;
    if (magic[0] != 'R' || magic[1] != 'A' || magic[2] != 'G' || magic[3] != 'C') {
        qWarning() << "[RAG] 索引缓存 magic 不匹配，忽略"; return false;
    }
    quint32 ver; in >> ver;
    if (ver != kCacheVersion) { qWarning() << "[RAG] 索引缓存版本不匹配，忽略"; return false; }
    qint32 nEmbd; in >> nEmbd;
    if (nEmbd != m_nEmbd) { qWarning() << "[RAG] 索引缓存维度不匹配，忽略"; return false; }
    qint32 sigLen; in >> sigLen;
    if (sigLen < 0 || sigLen > 1024 * 1024) return false;
    QByteArray sigBa(sigLen, 0);
    if (in.readRawData(sigBa.data(), sigLen) != sigLen) return false;
    if (QString::fromUtf8(sigBa) != sig) {
        qDebug() << "[RAG] 索引缓存签名不匹配（知识库已变动），重建";
        return false;
    }
    qint32 count; in >> count;
    if (count <= 0 || count > 5000000) { qWarning() << "[RAG] 索引缓存分块数异常，忽略"; return false; }
    out.clear(); out.reserve(count);
    for (qint32 i = 0; i < count; ++i) {
        RagChunk rc;
        qint32 l; QByteArray b;
        if (in >> l, l < 0 || l > 64 * 1024 * 1024) return false;
        b.resize(l); if (in.readRawData(b.data(), l) != l) return false;
        rc.source = QString::fromUtf8(b);
        if (in >> l, l < 0 || l > 64 * 1024 * 1024) return false;
        b.resize(l); if (in.readRawData(b.data(), l) != l) return false;
        rc.text = QString::fromUtf8(b);
        rc.vec.resize(nEmbd);
        const int vecBytes = nEmbd * qint32(sizeof(float));
        if (in.readRawData(reinterpret_cast<char *>(rc.vec.data()), vecBytes) != vecBytes) return false;
        out.push_back(std::move(rc));
    }
    if (in.status() != QDataStream::Ok) { qWarning() << "[RAG] 索引缓存读取不完整，忽略"; return false; }
    return true;
}

QString RagEngine::extractDocxText(const QString &filePath) {
    QZipReader zip(filePath);
    if (!zip.exists()) { qWarning() << "[RAG] 无法打开 docx:" << filePath; return QString(); }
    QByteArray xml = zip.fileData(QStringLiteral("word/document.xml"));
    zip.close();
    if (xml.isEmpty()) return QString();
    return stripXmlText(QString::fromUtf8(xml));
}

QString RagEngine::stripXmlText(const QString &xml) {
    QString s = xml;
    s.replace(QStringLiteral("</w:p>"), QStringLiteral("\n"));   // 段落换行
    s.replace(QStringLiteral("<w:br/>"),  QStringLiteral("\n"));
    s.remove(QRegularExpression(QStringLiteral("<[^>]*>")));      // 去掉所有 XML 标签
    s.replace(QStringLiteral("&amp;"),  QStringLiteral("&"));
    s.replace(QStringLiteral("&lt;"),   QStringLiteral("<"));
    s.replace(QStringLiteral("&gt;"),   QStringLiteral(">"));
    s.replace(QStringLiteral("&quot;"), QStringLiteral("\""));
    s.replace(QStringLiteral("&apos;"), QStringLiteral("'"));
    s.replace(QStringLiteral("&nbsp;"), QStringLiteral(" "));
    s.replace(QRegularExpression(QStringLiteral("[ \\t\\x0B\\f\\r]+")), QStringLiteral(" ")); // 折叠空白为单空格
    s.replace(QRegularExpression(QStringLiteral("\\n{3,}")), QStringLiteral("\n\n"));          // 折叠多余空行
    return s.trimmed();
}

QString RagEngine::extractDocBinaryText(const QString &filePath) {
    QFile f(filePath);
    if (!f.open(QIODevice::ReadOnly)) return QString();
    const QByteArray data = f.readAll();
    f.close();
    if (data.isEmpty()) return QString();

    // 主路径：UTF-16LE 扫描；提得很少时兜底单字节（CP1252 / Latin-1）扫描
    QString out = extractUtf16LeRuns(data);
    if (out.trimmed().length() < 20) {
        const QString alt = extractLatin1Runs(data);
        if (alt.trimmed().length() > out.trimmed().length()) out = alt;
    }
    qDebug() << "[RAG] .doc 尽力而为提取:" << filePath << "字符数 =" << out.trimmed().length();
    return normalizeExtractedText(out);   // 不能用 stripXmlText：会把正文当标签删掉
}

RagResult RagEngine::retrieve(const QString &query, int topK, float threshold) {
    RagResult r;
    if (!m_enabled) return r;

    const auto t0 = std::chrono::steady_clock::now();

    // bge-zh 官方要求 query 加检索指令前缀（建索引的文档不加），显著提升检索命中率
    const QString qText = kQueryPrefix.toString() + query;
    std::vector<float> qv = embed(qText);          // 内部已加编码锁
    if (qv.empty()) return r;

    // 原子抓取当前索引快照（后台重建期间检索读旧快照，互不干扰）
    std::shared_ptr<IndexSnapshot> snap;
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        snap = m_snapshot;
    }
    if (!snap || snap->chunks.empty()) return r;

    const int N = (int)snap->chunks.size();

    // 1) 全量向量余弦，找全局最佳（用于相关性门槛判定）
    std::vector<std::pair<int,float>> sims;
    sims.reserve(N);
    float bestVector = -1.f;
    for (int i = 0; i < N; ++i) {
        const float s = dot(qv, snap->chunks[i].vec);
        if (s > bestVector) bestVector = s;
        sims.emplace_back(i, s);
    }
    r.bestVectorScore = bestVector;

    // 相关性门槛：全局最佳都低于阈值 → 视为"知识库未覆盖"，不注入任何资料（抑制噪声/幻觉来源）
    if (bestVector < threshold) {
        const auto t1 = std::chrono::steady_clock::now();
        r.latencyMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
        logMetrics(query, r);
        return r;
    }
    r.used = true;

    // 2) 取向量 top-K 候选（候选池放大，给 BM25 重排留出空间）
    const int candK = std::min(N, std::max(topK * 4, 20));
    std::partial_sort(sims.begin(), sims.begin() + candK, sims.end(),
        [](const std::pair<int,float> &a, const std::pair<int,float> &b) { return a.second > b.second; });

    // 3) BM25 词法分（混合检索的另一半）
    const QStringList qterms = tokenizeTerms(query);
    struct Cand { int idx; float v; float b; };
    std::vector<Cand> cands;
    cands.reserve(candK);
    float vMin = 1e9f, vMax = -1e9f, bMin = 1e9f, bMax = -1e9f;
    for (int k = 0; k < candK; ++k) {
        const int idx = sims[k].first;
        const float v = sims[k].second;
        const float b = bm25Score(*snap, idx, qterms);
        cands.push_back({idx, v, b});
        if (v < vMin) vMin = v; if (v > vMax) vMax = v;
        if (b < bMin) bMin = b; if (b > bMax) bMax = b;
    }

    // 4) 归一化后融合（α·向量 + (1-α)·BM25），按融合分重排
    for (auto &c : cands) {
        const float vN = (vMax > vMin) ? (c.v - vMin) / (vMax - vMin) : 0.f;
        const float bN = (bMax > bMin) ? (c.b - bMin) / (bMax - bMin) : 0.f;
        c.b = m_alpha * vN + (1.f - m_alpha) * bN;   // 复用 b 字段存融合分
    }
    std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) { return a.b > b.b; });

    // 5) 取最终 topK，拼参考资料 + 来源溯源
    const int k = std::min(topK, (int)cands.size());
    QString block = QStringLiteral("【参考资料】\n");
    for (int i = 0; i < k; ++i) {
        const RagChunk &rc = snap->chunks[cands[i].idx];
        block += QStringLiteral("%1. (来源：%2)\n%3\n\n").arg(i + 1).arg(rc.source).arg(rc.text);
        r.sources.append(rc.source);
        r.scores.push_back(cands[i].b);
    }
    r.context = block;

    const auto t1 = std::chrono::steady_clock::now();
    r.latencyMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    logMetrics(query, r);
    return r;
}

void RagEngine::logMetrics(const QString &query, const RagResult &r) const {
    if (m_metricsPath.isEmpty()) return;
    QFile f(m_metricsPath);
    if (!f.open(QIODevice::Append | QIODevice::Text)) return;
    QJsonObject o;
    o["ts"] = QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
    o["query"] = query;
    o["used"] = r.used;
    o["best_vector_score"] = r.bestVectorScore;
    o["latency_ms"] = r.latencyMs;
    o["chunks"] = (int)r.sources.size();
    o["sources"] = QJsonArray::fromStringList(r.sources);
    QJsonDocument doc(o);
    f.write(doc.toJson(QJsonDocument::Compact));
    f.putChar('\n');
}

bool RagEngine::isIndexed() const {
    std::lock_guard<std::mutex> lk(const_cast<std::mutex &>(m_mutex));
    return m_snapshot && !m_snapshot->chunks.empty();
}

int RagEngine::chunkCount() const {
    std::lock_guard<std::mutex> lk(const_cast<std::mutex &>(m_mutex));
    return m_snapshot ? (int)m_snapshot->chunks.size() : 0;
}

float RagEngine::dot(const std::vector<float> &a, const std::vector<float> &b) {
    float s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
