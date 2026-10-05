@echo off
rem tools/build_onnx_embedder_test.bat - build & run ORT embedder test (MSVC)
rem Prereq: onnxruntime-win-x64-1.30.0 unzipped under D:\CAI\llama\onnxruntime\
call "E:\newDeskTop\Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set ORT=D:/CAI/llama/onnxruntime/onnxruntime-win-x64-1.30.0
set ROOT=D:\CAI\code\SimulatorAssistant
if not exist "%ROOT%\build\tests" mkdir "%ROOT%\build\tests"
cl /nologo /std:c++17 /EHsc /I "%ORT%/include" "%ROOT%\tests\onnx_embedder_test.cpp" "%ROOT%\module\Rag\OnnxEmbedder.cpp" /link "%ORT%/lib/onnxruntime.lib" /out:"%ROOT%\build\tests\onnx_embedder_test.exe"
rem DLLs live under %ORT%\lib\ (NOT the top level) - must sit next to the exe
copy /y "%ORT%\lib\onnxruntime.dll" "%ROOT%\build\tests\" >nul
copy /y "%ORT%\lib\onnxruntime_providers_shared.dll" "%ROOT%\build\tests\" >nul
"%ROOT%\build\tests\onnx_embedder_test.exe"
