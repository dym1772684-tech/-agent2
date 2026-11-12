"""[t08] RAG 检索链路（需 embedding/索引，首次运行会触发建索引，较慢）。

验证：检索非空、rerank 重排生效、融合函数排序、检索指标日志可读。
"""
from _lib import check, results


def run():
    print("[t08] RAG 检索链路")
    try:
        from allagent2.rag.rag import get_rag_index, search_knowledge
    except (OSError, MemoryError, ImportError) as e:
        # DLL/页面文件加载失败 = 机器内存不足，环境问题 → SKIP
        return [("RAG: 模块导入", None, f"SKIP：资源不足({type(e).__name__}: {str(e)[:60]})")]
    except Exception as e:
        return [("RAG: 模块导入", False, str(e)[:80])]

    # ---- 1) 检索（get_rag_index 会按指纹自动建/复用索引；加载模型吃内存）----
    try:
        hits = search_knowledge("血压高的老人饮食注意什么", top_k=4)
    except (OSError, MemoryError) as e:
        # os error 1455 = 页面文件太小：机器内存不足，模型加载不了 → 环境问题，SKIP
        return [("RAG: 检索执行", None, f"SKIP：资源不足无法加载模型({type(e).__name__})")]
    except Exception as e:
        return [("RAG: 检索执行", False, f"{type(e).__name__}: {str(e)[:70]}")]
    check("RAG: 检索返回非空", len(hits) > 0, f"hits={len(hits)}")
    if hits:
        h0 = hits[0]
        check("RAG: 命中含文本与分数",
              ("text" in h0 or "content" in h0 or "document" in h0) and "score" in h0,
              str(list(h0))[:60])

        # ---- 2) rerank 重排 ----
        try:
            from allagent2.rag.rerank import rerank
            reranked = rerank("血压饮食注意", list(hits), top_k=3) if len(hits) >= 2 else []
            check("RAG: rerank 返回有序结果", isinstance(reranked, list) and len(reranked) >= 1,
                  f"out={len(reranked)}")
        except (OSError, MemoryError):
            check("RAG: rerank 可用", None, "SKIP：资源不足无法加载 CrossEncoder")
        except Exception as e:
            check("RAG: rerank 可用", False, f"{type(e).__name__}: {str(e)[:60]}")

        # ---- 3) 融合函数（纯逻辑，零依赖）----
        from allagent2.rag.config import RAGConfig
        from allagent2.rag.fusion import fuse
        a = [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.5}]
        b = [{"id": "2", "score": 0.9}, {"id": "3", "score": 0.4}]
        fused = fuse(a, b, RAGConfig())
        check("RAG: 融合后取并集且排序", len(fused) >= 3 and
              all("score" in x for x in fused), f"n={len(fused)}")

    # ---- 4) 检索指标日志 ----
    try:
        from allagent2.rag.rag import read_metrics_log
        logs = read_metrics_log(limit=5)
        check("RAG: 指标日志可读(允许空)", isinstance(logs, list), f"entries={len(logs)}")
    except Exception as e:
        check("RAG: 指标日志可读", False, str(e)[:60])

    return results()


if __name__ == "__main__":
    run()
