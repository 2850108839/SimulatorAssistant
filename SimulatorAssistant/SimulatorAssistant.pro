QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += D:/CAI/llama/llama.cpp/include
INCLUDEPATH += D:/CAI/llama/llama.cpp/ggml/include

LIBS += D:/CAI/llama/llama.cpp/build/src/Release/llama.lib
LIBS += D:/CAI/llama/llama.cpp/build/ggml/src/Release/ggml.lib

SOURCES += \
    LlmEngine.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    LlmEngine.h \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
