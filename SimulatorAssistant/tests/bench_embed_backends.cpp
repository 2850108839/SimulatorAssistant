// tests/bench_embed_backends.cpp —— 双后端嵌入性能对比（llama.cpp vs ONNX Runtime）
// 用法：bench_embed_backends.exe
// 对比口径（面试要能讲清）：
//   1) 加载耗时：GGUF(Q8_0,26.5MB) vs ONNX(INT8 动态量化,22.8MB)
//   2) 单句嵌入延迟（检索 query 场景，两后端逐句，10 次取平均/最快/最慢）
//   3) 批量建索引：llama 走贪心打包单次 encode（embedBatch）；onnx 无覆写，走基类逐句循环
//   注意：两后端量化格式不同，数字是"该后端+该量化"的端到端表现，不是纯引擎对比。
#include "../module/Rag/LlamaEmbedder.h"
#include "../module/Rag/OnnxEmbedder.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>

using Clock = std::chrono::steady_clock;
static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// 检索场景 query（与 dual_backend_test 同源）
static const std::vector<std::string> kQueries = {
    "如何新建一个数据采集任务？",
    "模拟器支持哪些采集协议？",
    "通道配置与采样频率在哪里设置？",
    "历史数据导出与报表生成的操作步骤",
    "软件安装与许可证激活方法",
};

// 模拟知识库 chunk（性能测试文本，非真实知识库内容）
static const std::vector<std::string> kDocs = {
    "数据采集任务的新建入口在主界面左侧任务列表，点击新建任务按钮后填写任务名称、采集通道与采样频率。",
    "模拟器支持 Modbus RTU、Modbus TCP、OPC UA 三种采集协议，协议参数在任务配置页中设置。",
    "通道配置页可批量设置采样频率，范围 1Hz 到 1000Hz，修改后点击应用按钮生效。",
    "历史数据默认保存在安装目录下的 data 文件夹，导出支持 CSV 与 Excel 两种格式。",
    "软件安装完成后需要输入许可证密钥激活，激活入口在帮助菜单下的关于对话框。",
    "报表生成需要先选择时间范围，再点击生成报表按钮，系统自动汇总所选时间段内的统计数据。",
    "设备连接失败时请检查网络配置与设备地址，Modbus 设备的默认端口为 502。",
    "数据回放功能支持按时间轴拖动定位，回放速度可在 0.5 倍到 8 倍之间调节。",
    "告警规则支持阈值告警与变化率告警两种模式，告警记录可导出为文本文件。",
    "多台采集设备可同时接入，设备列表按采集通道分组显示，状态灯绿色表示在线。",
    "任务暂停后采集数据不会丢失，恢复任务时自动续采，并在数据文件头部标记暂停段。",
    "采样数据的存储格式为二进制分块文件，每块 4KB，文件头部包含时间戳与通道索引。",
    "系统时间同步支持 NTP 服务器自动校时，校时周期默认 24 小时，可在系统设置中调整。",
    "报表模板支持自定义表头与单位，模板文件存放在 report_templates 目录下。",
    "许可证密钥分为单机版与网络版，网络版需要配置许可证服务器地址与端口。",
    "数据导出时可以选择是否包含原始时间戳，导出文件按通道分工作表存储。",
    "采集任务运行状态可在主界面状态栏查看，包括已采集点数、错误计数与运行时长。",
    "数据库备份功能支持定时自动备份，备份文件默认保留最近 30 天。",
    "模拟量通道支持量程设置与线性校准，校准参数保存后立即生效。",
    "软件支持中英文界面切换，切换后需要重启程序才能完全生效。",
};

static void benchSingle(IEmbedder& e, const std::string& name, int rounds) {
    std::vector<float> vec;
    // 预热 2 次（首次含图构建/内存分配）
    e.embed(kQueries[0], vec, 512);
    e.embed(kQueries[0], vec, 512);

    double total = 0, best = 1e9, worst = 0;
    std::vector<double> all;
    for (int r = 0; r < rounds; ++r) {
        for (const auto& q : kQueries) {
            auto t0 = Clock::now();
            e.embed(q, vec, 512);
            double d = msSince(t0);
            all.push_back(d);
            total += d;
            best = std::min(best, d);
            worst = std::max(worst, d);
        }
    }
    std::sort(all.begin(), all.end());
    double med = all[all.size() / 2];
    std::cout << "  " << std::left << std::setw(22) << name
              << " 单句x" << std::setw(4) << rounds * (int)kQueries.size()
              << " 平均=" << std::fixed << std::setprecision(2) << std::setw(7) << total / all.size()
              << "ms  中位=" << std::setw(7) << med
              << "ms  最快=" << std::setw(7) << best
              << "ms  最慢=" << std::setw(7) << worst << "ms\n";
}

int main() {
    std::cout << "=== 双后端嵌入性能对比（llama.cpp Q8_0 vs ONNX INT8 动态量化）===\n\n";

    LlamaEmbedder llama;
    auto t0 = Clock::now();
    bool okL = llama.loadModel("D:/CAI/llama/llama.cpp/models/bge-small-zh-v1.5-q8_0.gguf");
    double loadL = msSince(t0);
    std::cout << "llama 后端加载: " << (okL ? "OK" : "FAIL") << "  " << loadL << " ms  dim=" << llama.dim() << "\n";

    OnnxEmbedder onnx;
    t0 = Clock::now();
    bool okO = onnx.loadModel("D:/CAI/llama/llama.cpp/models/bge_onnx/model_int8.onnx");
    double loadO = msSince(t0);
    std::cout << "onnx 后端加载: " << (okO ? "OK" : "FAIL") << "  " << loadO << " ms  dim=" << onnx.dim() << "\n\n";
    if (!okL || !okO) { std::cerr << "模型加载失败\n"; return 1; }

    std::cout << "[1] 单句嵌入延迟（检索场景，5 query x 10 轮）\n";
    benchSingle(llama, "llama.cpp", 10);
    benchSingle(onnx, "onnx-runtime", 10);

    std::cout << "\n[2] 批量建索引（20 条 chunk）\n";
    std::vector<std::vector<float>> outL, outO;
    t0 = Clock::now();
    outL = llama.embedBatch(kDocs, 512);
    double batchL = msSince(t0);
    t0 = Clock::now();
    outO = onnx.embedBatch(kDocs, 512);   // 基类默认：逐句循环
    double batchO = msSince(t0);
    int nonEmptyL = (int)std::count_if(outL.begin(), outL.end(), [](const auto& v){ return !v.empty(); });
    int nonEmptyO = (int)std::count_if(outO.begin(), outO.end(), [](const auto& v){ return !v.empty(); });
    std::cout << "  " << std::left << std::setw(22) << "llama.cpp"
              << " 总耗时=" << std::fixed << std::setprecision(2) << std::setw(7) << batchL
              << "ms  每条=" << std::setw(6) << batchL / kDocs.size() << "ms（贪心打包单次 encode）"
              << "  有效=" << nonEmptyL << "/20\n";
    std::cout << "  " << std::left << std::setw(22) << "onnx-runtime"
              << " 总耗时=" << std::fixed << std::setprecision(2) << std::setw(7) << batchO
              << "ms  每条=" << std::setw(6) << batchO / kDocs.size() << "ms（基类逐句循环）"
              << "  有效=" << nonEmptyO << "/20\n";

    std::cout << "\n口径说明：量化不同（Q8_0 26.5MB vs INT8 22.8MB）；批量路径 llama 有贪心打包优化，"
                 "onnx 走逐句。数字为单机端到端表现。\n";
    return 0;
}
