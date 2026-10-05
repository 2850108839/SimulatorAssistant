@echo off
rem tools/build_dual_backend_test.bat - build & run dual-backend consistency test (MSVC)
rem Prereq: llama.cpp build outputs (llama.lib/ggml.lib/DLLs) + onnxruntime-win-x64-1.30.0
call "E:\newDeskTop\Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set LLAMA=D:/CAI/llama/llama.cpp
set ORT=D:/CAI/llama/onnxruntime/onnxruntime-win-x64-1.30.0
set ROOT=D:\CAI\code\SimulatorAssistant
if not exist "%ROOT%\build\tests" mkdir "%ROOT%\build\tests"
cl /nologo /std:c++17 /EHsc /I "%LLAMA%/include" /I "%LLAMA%/ggml/include" /I "%ORT%/include" "%ROOT%\tests\dual_backend_test.cpp" "%ROOT%\module\Rag\LlamaEmbedder.cpp" "%ROOT%\module\Rag\OnnxEmbedder.cpp" /link "%LLAMA%/build/src/Release/llama.lib" "%LLAMA%/build/ggml/src/Release/ggml.lib" "%ORT%/lib/onnxruntime.lib" /out:"%ROOT%\build\tests\dual_backend_test.exe"
rem Runtime DLLs must sit next to the exe (avoid PATH hijack by old versions)
copy /y "%LLAMA%\build\bin\Release\llama.dll" "%ROOT%\build\tests\" >nul
copy /y "%LLAMA%\build\bin\Release\ggml.dll" "%ROOT%\build\tests\" >nul
copy /y "%ORT%\lib\onnxruntime.dll" "%ROOT%\build\tests\" >nul
copy /y "%ORT%\lib\onnxruntime_providers_shared.dll" "%ROOT%\build\tests\" >nul
"%ROOT%\build\tests\dual_backend_test.exe"
