#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 混合检索 Hybrid Search Hello World BM25(稀疏) + Dense向量(稠密) + RRF融合 """
import os
# ========= 新增：HuggingFace国内镜像，必须放在import transformers/sentence_transformers之前 =========
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ========== 1. 模拟医疗文档集 ==========
DOCS = [
    "患者白细胞计数(WBC)升高，提示细菌感染可能，建议进一步检查降钙素原(PCT)。",
    "WBC异常增高，中性粒细胞比例上升，考虑脓毒症，需监测血乳酸。",
    "血小板(PLT)减少，凝血功能障碍，警惕弥散性血管内凝血(DIC)。",
    "降钙素原(PCT)显著升高，严重细菌感染指标，可用于脓毒症早期诊断。",
    "NT-proBNP升高提示心力衰竭，脓毒症心肌病患者常明显升高。",
    "血肌酐(Cr)升高，急性肾损伤(AKI)，需评估SOFA肾脏评分。",
    "平均动脉压(MAP)低于65mmHg，提示循环障碍，需液体复苏。",
    "C反应蛋白(CRP)升高，急性期反应蛋白，非特异性炎症指标。",
]

# ========== 2. 构建 BM25 稀疏索引 ==========
tokenized_docs = [list(jieba.cut(doc)) for doc in DOCS]
bm25 = BM25Okapi(tokenized_docs)

# ========== 3. 构建稠密向量索引 ==========
encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
doc_embeddings = encoder.encode(DOCS, normalize_embeddings=True)

# ========== 4. 三路检索函数 ==========
def bm25_search(query, top_k=5):
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return {idx: float(scores[idx]) for idx in top_idx if scores[idx] > 0}

def dense_search(query, top_k=5):
    query_emb = encoder.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(query_emb, doc_embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    return {idx: float(scores[idx]) for idx in top_idx}

def hybrid_search_rrf(query, top_k=5, k=60):
    """
    RRF (Reciprocal Rank Fusion) 倒数排名融合
    score = Σ 1/(k + rank)
    """
    bm25_results = bm25_search(query, top_k=top_k * 2)
    dense_results = dense_search(query, top_k=top_k * 2)

    # 获取各自路内的排名（按分数降序）
    bm25_ranked = sorted(bm25_results.keys(), key=lambda x: bm25_results[x], reverse=True)
    dense_ranked = sorted(dense_results.keys(), key=lambda x: dense_results[x], reverse=True)

    all_docs = set(bm25_ranked) | set(dense_ranked)
    rrf_scores = {}

    for doc_id in all_docs:
        score = 0.0
        if doc_id in bm25_ranked:
            rank = bm25_ranked.index(doc_id) + 1
            score += 1.0 / (k + rank)
        if doc_id in dense_ranked:
            rank = dense_ranked.index(doc_id) + 1
            score += 1.0 / (k + rank)
        rrf_scores[doc_id] = score

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

# ========== 5. 测试：对比三路检索 ==========
QUERIES = [
    "白细胞高是什么原因",      # 同义表达，Dense强，BM25可能漏"WBC"
    "PCT升高代表什么",         # 专业缩写，BM25强
    "心脏指标异常",            # 宽泛语义，Dense强
    "脓毒症休克血压低",        # 综合场景，需要两者互补
]

for q in QUERIES:
    print(f"\n{'='*60}")
    print(f"查询: 「{q}」")
    print(f"{'='*60}")

    print("\n>>> BM25 稀疏检索:")
    for idx, score in sorted(bm25_search(q, 3).items(), key=lambda x: x[1], reverse=True):
        print(f"  文档{idx} | 得分:{score:.4f} | {DOCS[idx]}")

    print("\n>>> Dense 稠密检索:")
    for idx, score in sorted(dense_search(q, 3).items(), key=lambda x: x[1], reverse=True):
        print(f"  文档{idx} | 得分:{score:.4f} | {DOCS[idx]}")

    print("\n>>> Hybrid RRF 混合检索:")
    for idx, score in hybrid_search_rrf(q, 3):
        print(f"  文档{idx} | RRF:{score:.4f} | {DOCS[idx]}")
