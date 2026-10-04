# -*- coding: utf-8 -*-
"""解析 RagEngine 的索引缓存（.rag_cache_*.bin），用于诊断分块内容/检索质量。
格式（QDataStream LittleEndian）:
  magic 'RAGC'(4) | quint32 version | qint32 nEmbd | qint32 sigLen | sig[bytes]
  qint32 count | 每块: qint32 srcLen|src | qint32 txtLen|txt | nEmbd*float32 vec
"""
import struct, sys, collections, re

path = sys.argv[1] if len(sys.argv) > 1 else r"D:/CAI/code/SimulatorAssistant/.rag_cache_dd777079eb47121209357d0b0626c8b36f2771b2.bin"
kw = sys.argv[2] if len(sys.argv) > 2 else "转发"

data = open(path, "rb").read()
off = 0
def take(n):
    global off
    b = data[off:off+n]; off += n; return b
def i32():
    global off
    v = struct.unpack_from("<i", data, off)[0]; off += 4; return v
def u32():
    global off
    v = struct.unpack_from("<I", data, off)[0]; off += 4; return v

magic = take(4)
print("magic =", magic)
ver = u32(); nEmbd = i32(); sigLen = i32(); sig = take(sigLen).decode("utf-8", "replace")
count = i32()
print("version =", ver, "| nEmbd =", nEmbd, "| sig =", sig[:16], "| count =", count)

chunks = []
for i in range(count):
    sl = i32(); src = take(sl).decode("utf-8", "replace")
    tl = i32(); txt = take(tl).decode("utf-8", "replace")
    off += nEmbd * 4  # 跳过向量
    chunks.append((src, txt))

print("\n=== 各来源分块数 ===")
for src, n in collections.Counter(s for s, _ in chunks).most_common():
    print(f"  {n:6d}  {src}")

hits = [(i, s, t) for i, (s, t) in enumerate(chunks) if kw in t]
print(f"\n=== 含 '{kw}' 的分块共 {len(hits)} 个，前 15 个预览 ===")
for i, s, t in hits[:15]:
    print(f"\n--- chunk#{i} 来源={s} 长度={len(t)} ---")
    print(t[:300].replace("\n", " / "))

print(f"\n=== 前 3 个分块样式（看提取质量） ===")
for i, (s, t) in enumerate(chunks[:3]):
    print(f"\n--- chunk#{i} 来源={s} 长度={len(t)} ---")
    print(repr(t[:300]))

print(f"\n文本总长度 = {sum(len(t) for _, t in chunks)}")
