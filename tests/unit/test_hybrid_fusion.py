"""Hybrid Fusion 单元测试：归一化与加权融合（纯函数，无 DB）。"""

from app.rag.hybrid_search import _minmax


def test_minmax_normalizes():
    scores = [("a", 1.0), ("b", 0.5), ("c", 0.0)]
    norm = _minmax(scores)
    assert norm["a"] == 1.0
    assert norm["c"] == 0.0
    assert 0 < norm["b"] < 1


def test_minmax_flat_returns_one():
    norm = _minmax([("a", 0.7), ("b", 0.7)])
    assert norm == {"a": 1.0, "b": 1.0}


def test_minmax_empty():
    assert _minmax([]) == {}


def test_bigram_tokenize():
    from app.rag.keyword_search import bigram_tokenize

    tokens = bigram_tokenize("国际漫游")
    assert tokens == "国际 际漫 漫游"

    # 标点被剔除
    tokens2 = bigram_tokenize("5G，套餐！")
    assert "，" not in tokens2 and "！" not in tokens2

    # 单字符输入
    assert bigram_tokenize("流") == "流"
    assert bigram_tokenize("!@#") == ""
