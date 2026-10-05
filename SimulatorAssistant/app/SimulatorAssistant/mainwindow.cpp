#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QScrollBar>

#include <QDebug>
#include <QDir>
#include <QFileDialog>
#include <QSettings>
#include <QMetaObject>
#include <QtConcurrent/QtConcurrent>
#include <QFuture>
#include <QCoreApplication>
#include <QFileInfo>

// ---- 配置文件（exe 同目录 config.ini）：模型路径可编辑，换机无需重新编译 ----
static QString configFilePath() {
    return QCoreApplication::applicationDirPath() + QStringLiteral("/config.ini");
}
// 读配置；键缺失或值为空时用默认值
static QString readModelConfig(const QString &key, const QString &def) {
    QSettings s(configFilePath(), QSettings::IniFormat);
    const QString v = s.value(key, def).toString().trimmed();
    return v.isEmpty() ? def : v;
}
// 首次运行自动生成默认配置；删除该文件即恢复默认
static void ensureConfigFile() {
    if (QFileInfo::exists(configFilePath())) return;
    QSettings s(configFilePath(), QSettings::IniFormat);
    s.setValue(QStringLiteral("model/model_path"),
               QStringLiteral("D:/CAI/llama/llama.cpp/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"));
    s.setValue(QStringLiteral("model/bge_path"),
               QStringLiteral("D:/CAI/llama/llama.cpp/models/bge-small-zh-v1.5-q8_0.gguf"));
    // 嵌入后端：llama（默认，llama.cpp 加载 GGUF）| onnx（ONNX Runtime 加载 .onnx，方案 A）
    s.setValue(QStringLiteral("model/embed_backend"), QStringLiteral("llama"));
    s.setValue(QStringLiteral("model/bge_onnx_path"),
               QStringLiteral("D:/CAI/llama/llama.cpp/models/bge_onnx/model_int8.onnx"));
    s.sync();
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 模型路径从配置文件读取（exe 同目录 config.ini，首次运行自动生成）
    ensureConfigFile();
    const QString modelPath = readModelConfig(QStringLiteral("model/model_path"),
        QStringLiteral("D:/CAI/llama/llama.cpp/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"));

    qDebug()<<llama_version();
    llama_backend_init();
    //ui->textEdit->setText("开始");


    // 1) 建 worker 并 moveToThread（QThread 正确姿势）
    m_worker = new LLMWorker;
    m_worker->moveToThread(&m_thread);
    m_thread.start();

    // 系统提示：强制模型依据 RAG 检索到的【参考资料】作答并标注来源。
    // 缺这一步，模型会把参考资料当普通聊天上下文，反问用户"请提供更多信息"。
    const QString sysPrompt = QStringLiteral(
        "你是一名专业的技术支持助手，负责依据知识库资料准确回答用户问题。\n"
        "回答规则：\n"
        "1. 当用户消息中提供了【参考资料】时，必须严格依据参考资料作答，不得编造或臆测；\n"
        "2. 回答末尾用「依据：<来源文件名>」标注所引用的资料；\n"
        "3. 若参考资料未覆盖该问题，先明确说明「知识库中未找到相关内容」，再给出通用性建议；\n"
        "4. 使用简洁、条理清晰的中文回答。");
    QMetaObject::invokeMethod(m_worker, "setSystemPrompt", Qt::QueuedConnection,
                              Q_ARG(QString, sysPrompt));

    // 2) 启动 RAG 引擎：加载 bge 嵌入模型（同步，很快），建索引放到后台线程（避免卡 UI）
    //    嵌入后端由 config.ini 的 model/embed_backend 决定（llama | onnx）
    m_rag = new RagEngine;
    const QString embedBackend = readModelConfig(QStringLiteral("model/embed_backend"),
                                                 QStringLiteral("llama"));
    bool ragLoaded = false;
    if (embedBackend.compare(QStringLiteral("onnx"), Qt::CaseInsensitive) == 0) {
        const QString onnxPath = readModelConfig(QStringLiteral("model/bge_onnx_path"),
            QStringLiteral("D:/CAI/llama/llama.cpp/models/bge_onnx/model_int8.onnx"));
        ragLoaded = m_rag->loadOnnxModel(onnxPath);
        if (ragLoaded)
            qDebug() << "[RAG] 嵌入后端 = ONNX Runtime";
    } else {
        const QString bgePath = readModelConfig(QStringLiteral("model/bge_path"),
            QStringLiteral("D:/CAI/llama/llama.cpp/models/bge-small-zh-v1.5-q8_0.gguf"));
        ragLoaded = m_rag->loadModel(bgePath);
        if (ragLoaded)
            qDebug() << "[RAG] 嵌入后端 = llama.cpp";
    }
    // 知识库目录：优先读持久化配置（QSettings），否则用默认目录
    QSettings settings(QStringLiteral("CAI"), QStringLiteral("SimulatorAssistant"));
    m_kbDir = settings.value(QStringLiteral("rag/kbDir"),
                             QStringLiteral("D:/CAI/code/SimulatorAssistant/knowledge_base")).toString();
    if (ragLoaded) {
        m_worker->setRagEngine(m_rag);
        // 检索指标日志（JSONL）：每次提问记录向量最佳分/耗时/命中来源，用于离线分析检索质量
        m_rag->setMetricsLogPath(QFileInfo(m_kbDir).absolutePath()
                                 + QStringLiteral("/rag_metrics.jsonl"));
        rebuildIndexAsync();   // 后台建索引 + 完成后默认开启 RAG（状态栏显示进度）
    } else {
        qWarning() << "[RAG] bge 模型加载失败，RAG 将不生效";
        statusBar()->showMessage(QStringLiteral("RAG 不可用：bge 模型加载失败"));
    }

    // 3) RAG 开关 + 知识库管理（菜单栏，避免改动 .ui）
    QMenu *ragMenu = menuBar()->addMenu(QStringLiteral("RAG"));
    QAction *ragAct = ragMenu->addAction(QStringLiteral("RAG：索引中…"));
    ragAct->setCheckable(true);
    ragAct->setEnabled(false);     // 后台索引完成前不可手动开关
    m_ragAct = ragAct;
    connect(ragAct, &QAction::toggled, this, [this, ragAct](bool on) {
        if (m_rag && m_rag->isIndexed()) {
            m_rag->setEnabled(on);
            ragAct->setText(on ? QStringLiteral("RAG：开") : QStringLiteral("RAG：关"));
        } else {
            ragAct->setChecked(false);
            ragAct->setText(QStringLiteral("RAG：不可用"));
        }
    });

    // 热重载闭包：触发后台重建索引（默认开启 + 状态栏进度），全程不重启程序
    auto reloadKb = [this]() {
        rebuildIndexAsync();
    };

    // "重新加载知识库"：不重启即按当前目录重建索引
    QAction *reloadAct = ragMenu->addAction(QStringLiteral("重新加载知识库"));
    connect(reloadAct, &QAction::triggered, this, reloadKb);

    // "选择知识库目录…"：浏览目录并持久化，随后立即热重载
    QAction *selectDirAct = ragMenu->addAction(QStringLiteral("选择知识库目录…"));
    connect(selectDirAct, &QAction::triggered, this, [this, reloadKb]() {
        const QString dir = QFileDialog::getExistingDirectory(
            this, QStringLiteral("选择知识库目录"), m_kbDir);
        if (dir.isEmpty()) return;               // 用户取消
        // 切换监听路径：先把旧目录从 watcher 移除，再监听新目录
        if (m_kbWatcher && !m_kbDir.isEmpty() && m_kbWatcher->directories().contains(m_kbDir)) {
            m_kbWatcher->removePath(m_kbDir);
        }
        m_kbDir = dir;
        if (m_kbWatcher) m_kbWatcher->addPath(m_kbDir);
        QSettings s(QStringLiteral("CAI"), QStringLiteral("SimulatorAssistant"));
        s.setValue(QStringLiteral("rag/kbDir"), dir);   // 持久化
        s.sync();
        reloadKb();                              // 选完即重建
    });

    // 自动热重载：监听知识库目录，文件增删改时自动重建索引（每次更新只重建一次）
    // 用去抖计时器避免编辑器连续写入触发多次重建
    m_kbWatcher = new QFileSystemWatcher(this);
    m_kbDebounce = new QTimer(this);
    m_kbDebounce->setSingleShot(true);
    if (!m_kbDir.isEmpty() && QDir(m_kbDir).exists()) {
        m_kbWatcher->addPath(m_kbDir);
    }
    connect(m_kbWatcher, &QFileSystemWatcher::directoryChanged, this, [this]() {
        if (m_kbDebounce) m_kbDebounce->start(800);   // 重启计时器，连写只触发最后一次
    });
    connect(m_kbDebounce, &QTimer::timeout, this, reloadKb);
    qDebug() << "[RAG] 已启动知识库目录自动监听:" << m_kbDir;

    // ===== 会话持久化菜单（评审第 1 项补丁）=====
    QMenu *sessionMenu = menuBar()->addMenu(QStringLiteral("会话"));
    QAction *saveAct = sessionMenu->addAction(QStringLiteral("保存当前会话…"));
    connect(saveAct, &QAction::triggered, this, [this]() {
        const QString dir = QFileDialog::getExistingDirectory(
            this, QStringLiteral("选择会话保存目录"));
        if (dir.isEmpty()) return;
        QMetaObject::invokeMethod(m_worker, "saveSession", Qt::QueuedConnection,
                                  Q_ARG(QString, dir));
    });
    QAction *loadAct = sessionMenu->addAction(QStringLiteral("恢复会话…"));
    connect(loadAct, &QAction::triggered, this, [this]() {
        const QString dir = QFileDialog::getExistingDirectory(
            this, QStringLiteral("选择会话目录"));
        if (dir.isEmpty()) return;
        QMetaObject::invokeMethod(m_worker, "loadSession", Qt::QueuedConnection,
                                  Q_ARG(QString, dir));
    });
    connect(m_worker, &LLMWorker::sessionSaved, this, [this](bool ok, const QString &msg) {
        statusBar()->showMessage((ok ? QStringLiteral("[会话] ") : QStringLiteral("[会话失败] ")) + msg, 6000);
    });
    connect(m_worker, &LLMWorker::sessionLoaded, this, [this](bool ok, int, const QString &msg) {
        statusBar()->showMessage((ok ? QStringLiteral("[会话] ") : QStringLiteral("[会话失败] ")) + msg, 6000);
    });

    // 2) worker 的销毁放在析构里（线程退出后直接 delete），
    //    不挂 finished→deleteLater：线程事件循环已停，deleteLater 不会被执行，徒增泄漏

    // 3) 流式信号 → textEdit 追加（思考过程 + 结果都在这里），打字机 + 自动滚底
    connect(m_worker, &LLMWorker::tokenReady, this, [this](const QString &piece) {
        if (m_thinking) {                        // 首个 token 抵达：思考结束，清状态栏
            statusBar()->clearMessage();
            m_thinking = false;
        }
        ui->textEdit->insertPlainText(piece);
        QScrollBar *sb = ui->textEdit->verticalScrollBar();
        sb->setValue(sb->maximum());
    });

    // 检索提示：明确告诉用户本次回答是否用到了知识库，并列出来源文件（解决"好像没找到知识库"的困惑）
    connect(m_worker, &LLMWorker::ragRetrieved, this, [this](const QStringList &sources) {
        QString msg = QStringLiteral("[已引用知识库 %1 段]").arg(sources.size());
        if (!sources.isEmpty()) {
            QStringList uniq = sources; uniq.removeDuplicates();
            msg += QStringLiteral(" 来源：") + uniq.join(QStringLiteral("、"));
        }
        ui->textEdit->append(msg);
    });

    // 4) 回复结束 → 恢复输入；出错 → 提示
    connect(m_worker, &LLMWorker::replyFinished, this, [this] {
        if (m_thinking) { statusBar()->clearMessage(); m_thinking = false; }
        ui->lineEdit->setEnabled(true);
        //ui->sendBtn->setEnabled(true);
        //ui->stopBtn->setEnabled(false);   // 停止按钮代码先留着，后续加
        m_busy = false;
    });
    connect(m_worker, &LLMWorker::errorOccurred, this,
            [this](const QString &msg) {
                if (m_thinking) { statusBar()->clearMessage(); m_thinking = false; }
                ui->textEdit->append("[错误] " + msg);
                // 出错路径不会发 replyFinished，这里必须自己解锁，否则界面永久卡住
                ui->lineEdit->setEnabled(true);
                m_busy = false;
            });

    // 5) 异步加载模型（不卡界面）
    connect(m_worker, &LLMWorker::modelLoaded, this, [this](bool ok) {
        //statusBar()->showMessage(ok ? "模型已加载" : "模型加载失败");
    });
    QMetaObject::invokeMethod(m_worker, "loadModel", Qt::QueuedConnection,
                              Q_ARG(QString, modelPath));
}

void MainWindow::rebuildIndexAsync() {
    if (m_indexing) return;                       // 已有重建在跑（防抖 + 并发保护）
    if (!m_rag || !m_rag->isModelLoaded()) {
        qWarning() << "[RAG] 模型未加载，无法建索引";
        return;
    }
    if (m_kbDir.isEmpty() || !QDir(m_kbDir).exists()) {
        qWarning() << "[RAG] 知识库目录不存在，建索引取消:" << m_kbDir;
        return;
    }

    m_indexing = true;
    m_rag->setEnabled(false);                     // 重建期间冻结检索，避免读到半成品
    if (m_ragAct) { m_ragAct->setEnabled(false); m_ragAct->setText(QStringLiteral("RAG：索引中…")); }
    statusBar()->showMessage(QStringLiteral("正在重建知识库索引…"));

    auto *watcher = new QFutureWatcher<std::vector<RagChunk>>(this);
    m_indexWatcher = watcher;
    connect(watcher, &QFutureWatcher<std::vector<RagChunk>>::finished, this,
            [this, watcher]() {
        std::vector<RagChunk> idx = watcher->result();
        m_indexing = false;
        m_indexWatcher = nullptr;
        watcher->deleteLater();

        if (!idx.empty()) {
            m_rag->installIndex(std::move(idx)); // 原子替换 + 重建 BM25 统计
            m_rag->setEnabled(true);
            if (m_ragAct) {                       // 同步菜单（setChecked 会触发 toggled，但幂等）
                m_ragAct->setEnabled(true);
                m_ragAct->setChecked(true);
                m_ragAct->setText(QStringLiteral("RAG：开"));
            }
            statusBar()->showMessage(QStringLiteral("知识库索引已就绪，已默认开启检索"), 4000);
            qDebug() << "[RAG] 后台索引完成，分块数 =" << m_rag->chunkCount();
        } else {
            m_rag->setEnabled(false);
            if (m_ragAct) { m_ragAct->setEnabled(false); m_ragAct->setText(QStringLiteral("RAG：不可用")); }
            statusBar()->showMessage(QStringLiteral("RAG 不可用：知识库为空或解析失败"));
        }
    });

    // 进度回调在后台线程触发，通过 QueuedConnection 切回 GUI 线程更新状态栏
    auto progressCb = [this](int done, int total) {
        QMetaObject::invokeMethod(this, [this, done, total]() {
            statusBar()->showMessage(QStringLiteral("正在重建知识库索引… %1/%2").arg(done).arg(total));
        }, Qt::QueuedConnection);
    };

    QFuture<std::vector<RagChunk>> f = QtConcurrent::run([this, progressCb]() {
        return m_rag->buildIndex(m_kbDir, progressCb);
    });
    watcher->setFuture(f);
}

MainWindow::~MainWindow()
{
    // 若后台索引还在跑，先断开其信号并等其结束，避免它回调已销毁的 this / m_rag
    if (m_indexWatcher) {
        disconnect(m_indexWatcher, nullptr, this, nullptr);
        m_indexWatcher->waitForFinished();
        delete m_indexWatcher;
        m_indexWatcher = nullptr;
    }

    if (m_worker) {
        m_worker->stop();            // 原子量，主线程直接置位
        m_thread.quit();
        // 生成中单次 decode 可能较久，等足 30 秒；
        // 一定要等线程真正退出再 llama_backend_free()，否则子线程还在用后端就崩
        if (!m_thread.wait(30000)) {
            qWarning() << "worker thread did not finish within 30s";
            m_thread.terminate();
            m_thread.wait();
        }
        delete m_worker;
        m_worker = nullptr;
    }

    delete m_rag;
    m_rag = nullptr;

    llama_backend_free();
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    // QString text = ui->lineEdit->text();
    // ui->lineEdit->setText("");
    // ui->textEdit->append(text);

    // QString result = m_engine->chat(text);
    // qDebug()<<result;
    // ui->textEdit->append(result);

    QString input = ui->lineEdit->text().trimmed();
    if (input.isEmpty() || m_busy) return;
    m_busy = true;
    ui->textEdit->append("我: " + input);
    ui->textEdit->append("AI: ");
    // 推理前先给明确反馈：RAG 检索 + 7B 模型 prefill 期间可能要等几秒
    m_thinking = true;
    statusBar()->showMessage(QStringLiteral("思考中…（RAG 检索 + 模型推理）"));
    ui->lineEdit->clear();
    //ui->sendBtn->setEnabled(false);
    //ui->stopBtn->setEnabled(true);      // 后续加了按钮，这里就生效

    QMetaObject::invokeMethod(m_worker, "chat", Qt::QueuedConnection,
                              Q_ARG(QString, input));
}

// void MainWindow::onStopBtnClicked() {    // 后续 UI 加上停止按钮后，connect 到它
//     QMetaObject::invokeMethod(m_worker, "stop", Qt::QueuedConnection);
// }

// void MainWindow::onParamsChanged() {   // 参数面板 valueChanged 接到这里（可选）
//     QMetaObject::invokeMethod(m_worker, "setParams", Qt::QueuedConnection,
//                               Q_ARG(float, ui->tempSpin->value()),
//                               Q_ARG(float, ui->topPSpin->value()),
//                               Q_ARG(int,   ui->topKSpin->value()),
//                               Q_ARG(int,   ui->maxLenSpin->value()));
// }


