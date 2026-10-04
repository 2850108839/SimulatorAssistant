#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 检索离线评测脚本（纯标准库，无需第三方依赖）。

做什么：
  解析 SimulatorAssistant 的向量索引缓存 .rag_cache_<sha1>.bin（与 RagEngine.cpp
  的二进制格式完全一致），在缓存分块上用 BM25 词法检索复现"混合检索"中
  词法那一腿，并对 Golden Set（tests/golden_qa.json）逐题评估：

    - 来源召回：期望来源文件名是否出现在 top-k 命中里
    - 关键词覆盖：期望关键词是否出现在 top-k 命中文本中
    - 命中率 / recall@k / 必须命中失败数

  同时做缓存完整性体检：各来源分块数、是否含某些关键词（调试提取质量很有用）。

为什么只跑 BM25（词法）而不跑 bge 向量检索：
  向量检索需要在本机用 llama.cpp 跑 bge 编码，Python 侧没有现成绑定；
  而 BM25 词法腿是"混合检索"里可离线复现、且对中文分块/提取质量最敏感的部分。
  应用运行时跑的是「向量 top-k 候选 + BM25 融合重排」的完整混合检索；本脚本
  作为回归/冒烟测试，专门盯住词法腿与缓存/抽取的正确性。

用法：
  python tools/eval_retrieval.py --kbdir D:/CAI/code/SimulatorAssistant/knowledge_base
  python tools/eval_retrieval.py --cache D:/CAI/code/SimulatorAssistant/.rag_cache_xxxx.bin --golden tests/golden_qa.json --topk 3

退出码：0=通过（无 must 失败且来源命中率 >= --min-recall），1=有必须命中失败或低于阈值。
"""

import argparse
import hashlib
import json
import math
import os
import struct
import sys


# ---- 与 RagEngine.cpp 完全一致的缓存二进制格式 ----
CACHE_MAGIC = b"RAGC"


def cache_path_for(kb_dir: str) -> str:
    """复刻 C++ cachePathFor()：kbDir 父目录下 .rag_cache_<sha1>.bin。"""
    h = hashlib.sha1(kb_dir.encode("utf-8")).hexdigest()
    parent = os.path.dirname(os.path.abspath(kb_dir))
    return os.path.join(parent, ".rag_cache_" + h + ".bin")


def parse_cache(path: str):
    """返回 (version, nEmbd, chunks)，chunks=[(source, text, vec)]。"""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != CACHE_MAGIC:
        raise ValueError("缓存 magic 不匹配（不是 RAGC）")
    off = 4
    (version,) = struct.unpack_from("<I", data, off); off += 4
    (nEmbd,) = struct.unpack_from("<i", data, off); off += 4
    (sig_len,) = struct.unpack_from("<i", data, off); off += 4
    off += sig_len  # 跳过签名
    (count,) = struct.unpack_from("<i", data, off); off += 4
    chunks = []
    for _ in range(count):
        (src_len,) = struct.unpack_from("<i", data, off); off += 4
        source = data[off:off + src_len].decode("utf-8", "replace"); off += src_len
        (txt_len,) = struct.unpack_from("<i", data, off); off += 4
        text = data[off:off + txt_len].decode("utf-8", "replace"); off += txt_len
        vec = struct.unpack_from("<%df" % nEmbd, data, off); off += nEmbd * 4
        chunks.append((source, text, vec))
    return version, nEmbd, chunks


# ---- 与 RagEngine::tokenizeTerms 一致的中文词项切分（字级基线）----
def tokenize(text: str):
    terms = []
    lat = []
    def flush():
        if lat:
            terms.append("".join(lat).lower())
            lat.clear()
    for ch in text:
        o = ord(ch)
        if (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF) or \
           (0x3000 <= o <= 0x303F) or (0xFF00 <= o <= 0xFFEF):
            flush()
            terms.append(ch)              # 每个 CJK 字一个词项
        elif ch.isalnum():
            lat.append(ch)                # 累积 Latin 词
        else:
            flush()                       # 其它字符作分隔
    flush()
    return terms


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_terms = [tokenize(d) for d in docs]
        self.df = {}
        for terms in self.doc_terms:
            for t in set(terms):
                self.df[t] = self.df.get(t, 0) + 1
        self.doc_len = [len(t) for t in self.doc_terms]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0
        self.n = len(docs)

    def score(self, doc_id, qterms):
        dl = self.doc_len[doc_id]
        tf = {}
        for t in self.doc_terms[doc_id]:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in qterms:
            if t not in self.df:
                continue
            f = tf.get(t, 0)
            if f == 0:
                continue
            idf = math.log(1.0 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
            s += idf * (f * (self.k1 + 1.0)) / (f + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl)))
        return s

    def search(self, query, topk):
        qterms = tokenize(query)
        scored = [(i, self.score(i, qterms)) for i in range(self.n)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(i, sc) for i, sc in scored[:topk] if sc > 0.0]


def main():
    ap = argparse.ArgumentParser(description="RAG 检索离线评测（BM25 词法腿 + 缓存体检）")
    ap.add_argument("--kbdir", help="知识库目录（自动定位 .rag_cache_*.bin）")
    ap.add_argument("--cache", help="直接指定缓存 bin 路径")
    ap.add_argument("--golden", default=os.path.join(os.path.dirname(__file__), "..", "tests", "golden_qa.json"))
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--min-recall", type=float, default=0.5, help="来源命中率门槛（must 失败始终判失败）")
    args = ap.parse_args()

    cache_path = args.cache or (cache_path_for(args.kbdir) if args.kbdir else None)
    if cache_path:
        cache_path = os.path.abspath(cache_path)   # 归一化（兼容 /tmp 这类前导斜杠写法）
    if not cache_path or not os.path.exists(cache_path):
        print("错误：找不到缓存文件。请先运行一次程序建索引，或用 --cache 指定。", file=sys.stderr)
        return 2

    version, nEmbd, chunks = parse_cache(cache_path)
    print("缓存版本=%d  nEmbd=%d  分块数=%d" % (version, nEmbd, len(chunks)))
    if not chunks:
        print("缓存为空，无可评测内容。", file=sys.stderr)
        return 2

    # 缓存完整性体检
    by_src = {}
    for src, text, _ in chunks:
        by_src.setdefault(src, 0)
        by_src[src] += 1
    print("\n[缓存体检] 各来源分块数：")
    for src in sorted(by_src):
        print("  %-30s %d" % (src, by_src[src]))

    docs = [text for _, text, _ in chunks]
    sources = [src for src, _, _ in chunks]
    bm25 = BM25(docs)

    # 评测
    golden_path = os.path.abspath(args.golden)
    if not os.path.exists(golden_path):
        print("\n未找到 golden 集：%s（仅做了缓存体检）" % golden_path)
        return 0
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)
    topk = golden.get("topk", args.topk)
    questions = golden.get("questions", [])

    print("\n[评测] top-k=%d  题目数=%d" % (topk, len(questions)))
    if not questions:
        print("golden 集为空，跳过评测（仅缓存体检）。")
        return 0

    src_hit = 0
    kw_hit = 0
    must_fail = 0
    for i, q in enumerate(questions):
        qtext = q.get("q", "")
        expect_src = set(q.get("expect_sources", []))
        expect_kw = q.get("expect_keywords", [])
        must = q.get("must", False)

        hits = bm25.search(qtext, topk)
        hit_sources = [sources[i] for i, _ in hits]
        hit_text = " ".join(docs[i] for i, _ in hits)

        src_ok = bool(expect_src & set(hit_sources)) if expect_src else True
        kw_ok = all(kw in hit_text for kw in expect_kw) if expect_kw else True
        passed = src_ok and kw_ok
        if src_ok:
            src_hit += 1
        if kw_ok:
            kw_hit += 1
        if must and not passed:
            must_fail += 1

        print("\n  Q%d: %s" % (i + 1, qtext))
        print("    命中来源: %s" % (", ".join(hit_sources) if hit_sources else "(无)"))
        print("    期望来源: %s  -> %s" % (", ".join(sorted(expect_src)) or "(无)",
                                          "✓" if src_ok else "✗"))
        if expect_kw:
            miss = [kw for kw in expect_kw if kw not in hit_text]
            print("    关键词: %s  -> %s" % (", ".join(expect_kw),
                                            "✓" if not miss else "✗ 缺 " + ", ".join(miss)))
        if must and not passed:
            print("    [必须命中失败]")

    recall = src_hit / len(questions)
    kw_rate = kw_hit / len(questions)
    print("\n[汇总] 来源命中率=%.2f (%d/%d)  关键词覆盖率=%.2f  必须命中失败=%d"
          % (recall, src_hit, len(questions), kw_rate, must_fail))

    if must_fail > 0:
        print("结果：FAIL（有必须命中失败）")
        return 1
    if recall < args.min_recall:
        print("结果：FAIL（来源命中率 %.2f < 门槛 %.2f）" % (recall, args.min_recall))
        return 1
    print("结果：PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
