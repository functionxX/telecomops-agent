"""RAG 评测指标：Recall@K / MRR。

rankings: 每个查询返回的 doc_id 有序列表
expected:  每个查询的期望 doc_id 集合
"""



def recall_at_k(rankings: list[list[str]], expected: list[list[str]], k: int) -> float:
    """Recall@K：前 K 个结果中命中期望文档的比例（逐查询平均）。"""
    if not rankings:
        return 0.0
    hits = 0
    for ranked, exp in zip(rankings, expected, strict=False):
        top = set(ranked[:k])
        hits += len(top & set(exp)) / max(len(set(exp)), 1)
    return round(hits / len(rankings), 4)


def mrr(rankings: list[list[str]], expected: list[list[str]]) -> float:
    """MRR：首个命中期望文档的排名倒数的平均。"""
    if not rankings:
        return 0.0
    total = 0.0
    for ranked, exp in zip(rankings, expected, strict=False):
        exp_set = set(exp)
        for i, doc_id in enumerate(ranked, start=1):
            if doc_id in exp_set:
                total += 1.0 / i
                break
    return round(total / len(rankings), 4)
