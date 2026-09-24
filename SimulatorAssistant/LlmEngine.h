#ifndef LLMENGINE_H
#define LLMENGINE_H

#include <QString>
#include "llama.h"

class LlmEngine
{
public:
    LlmEngine();
    ~LlmEngine();
    bool loadModel(const QString &modelPath);   // 加载 GGUF 模型
    QString chat(const QString &input);          // 同步对话，先不做流式
private:
    QString m_history;
    llama_model   *m_model = nullptr;
    llama_context *m_ctx   = nullptr;
};

#endif // LLMENGINE_H
