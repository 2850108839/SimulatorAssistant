# =============================================================================
# SimulatorAssistant — 本地大模型对话 + RAG 知识库
# Qt 6.5.3 / MSVC2019  ·  qmake 工程
# =============================================================================
QT += core-private gui-private widgets concurrent

CONFIG += c++17

# 禁止已弃用 API 编译（需要时取消注释）
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000

# -----------------------------------------------------------------------------
# 外部依赖：llama.cpp（嵌入模型 + 推理后端）
# 可用环境变量 LLAMA_DIR 覆盖默认路径；未设置时回退到下方默认目录。
# -----------------------------------------------------------------------------
LLAMA_DIR = $$(LLAMA_DIR)
isEmpty(LLAMA_DIR): LLAMA_DIR = D:/CAI/llama/llama.cpp
INCLUDEPATH += $$LLAMA_DIR/include $$LLAMA_DIR/ggml/include
LIBS += $$LLAMA_DIR/build/src/Release/llama.lib \
        $$LLAMA_DIR/build/ggml/src/Release/ggml.lib

# -----------------------------------------------------------------------------
# 外部依赖：ONNX Runtime（方案 A —— 嵌入推理第二后端）
# 可用环境变量 ORT_DIR 覆盖默认路径；未设置时回退到下方默认目录。
# 运行时需将 onnxruntime.dll、onnxruntime_providers_shared.dll 拷贝到 exe 同目录。
# -----------------------------------------------------------------------------
ORT_DIR = $$(ORT_DIR)
isEmpty(ORT_DIR): ORT_DIR = D:/CAI/llama/onnxruntime/onnxruntime-win-x64-1.30.0
INCLUDEPATH += $$ORT_DIR/include
LIBS += $$ORT_DIR/lib/onnxruntime.lib

# Qt 私有头（qzipreader_p.h，用于解压 .docx）
# - 通过 core-private / gui-private 模块自动引入私有包含路径（正规做法）
# - 同时显式加入版本化目录 QtCore/6.5.3、QtGui/6.5.3，
#   以解析 <QtCore/private/...>、<QtGui/private/...> 这类传递包含，跨 Qt 安装可移植
INCLUDEPATH += $$[QT_INSTALL_HEADERS]/QtCore/6.5.3 $$[QT_INSTALL_HEADERS]/QtGui/6.5.3

# 各模块头文件目录加入包含路径，保持 #include "Xxx.h" 的写法不变
# 布局对齐 DispatchClient：业务模块放 module/（.h/.cpp 平铺），界面放 app/SimulatorAssistant/
INCLUDEPATH += \
    app/SimulatorAssistant \
    module/Engine \
    module/Rag

# -----------------------------------------------------------------------------
# 源文件（每个模块内 .h/.cpp 平铺；界面 .ui 单独放 form/）
# -----------------------------------------------------------------------------
SOURCES += \
    app/SimulatorAssistant/main.cpp \
    module/Engine/LLMWorker.cpp \
    module/Rag/RagEngine.cpp \
    module/Rag/OnnxEmbedder.cpp \
    module/Rag/LlamaEmbedder.cpp \
    app/SimulatorAssistant/mainwindow.cpp

HEADERS += \
    module/Engine/LLMWorker.h \
    module/Rag/RagEngine.h \
    module/Rag/OnnxEmbedder.h \
    module/Rag/LlamaEmbedder.h \
    app/SimulatorAssistant/mainwindow.h

FORMS += \
    app/SimulatorAssistant/form/mainwindow.ui

# -----------------------------------------------------------------------------
# 构建中间产物（moc / 目标文件 / ui / rcc）集中到 build/，保持源码树干净
# -----------------------------------------------------------------------------
MOC_DIR     = build/moc
OBJECTS_DIR = build/obj
RCC_DIR     = build/rcc
UI_DIR      = build/ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
