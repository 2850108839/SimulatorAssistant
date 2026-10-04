#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include "LLMWorker.h"
#include "RagEngine.h"
#include <QThread>
#include <QFileSystemWatcher>
#include <QTimer>
#include <QFutureWatcher>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void on_pushButton_clicked();
    //void onParamsChanged();

private:
    // 后台建索引：buildIndex 在 QtConcurrent 线程跑，finished 时 installIndex + 更新菜单 + 状态栏进度
    void rebuildIndexAsync();

    Ui::MainWindow *ui;
    QThread    m_thread;         // 常驻子线程（成员对象）
    LLMWorker *m_worker = nullptr;
    RagEngine *m_rag = nullptr;   // RAG 引擎（RAG 知识库）
    QString   m_kbDir;            // 知识库目录（QSettings 持久化）
    QFileSystemWatcher *m_kbWatcher = nullptr; // 监听知识库目录变动，自动重建索引
    QTimer   *m_kbDebounce = nullptr;           // 去抖：编辑器连写只触发一次重建
    QAction  *m_ragAct = nullptr;               // RAG 开关菜单项（后台索引完成后更新文案）
    QFutureWatcher<std::vector<RagChunk>> *m_indexWatcher = nullptr; // 后台索引 future 监视器
    bool m_indexing = false;     // 是否有索引重建在跑（防抖 + 并发保护）
    bool m_busy = false;
    bool m_thinking = false;   // 模型推理中（prefill + 生成首个 token 之前），用于状态栏提示
};
#endif // MAINWINDOW_H
