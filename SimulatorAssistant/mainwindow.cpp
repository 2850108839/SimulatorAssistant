#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>
const char* modelPath = "D:/CAI/llama/llama.cpp/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf";

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->textEdit->setText("开始");
    m_engine = new LlmEngine();
    m_engine->loadModel(modelPath);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    QString text = ui->lineEdit->text();
    ui->lineEdit->setText("");
    ui->textEdit->append(text);

    QString result = m_engine->chat(text);
    qDebug()<<result;
    ui->textEdit->append(result);
}

