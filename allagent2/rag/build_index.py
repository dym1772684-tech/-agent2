"""RAG 索引离线预构建脚本（B3 持久化配套）。

为什么要有它：
    在线路径 get_rag_index() 已改为持久化复用——磁盘上有指纹匹配的索引
    就跳过重嵌入。本脚本把「首次构建」这一步提前到部署期离线做：
    服务启动 / 首次对话直接命中磁盘索引，零重嵌入、embedding 模型延迟到
    首次查询才加载。

什么情况要重新跑：
    1. elder_care.md 知识库内容更新（指纹含文档内容，自动识别）
    2. 切分参数 / embedding 模型 / HNSW 参数变更（指纹含这些项）
    3. 想强制重建：python build_index.py --rebuild

用法：
    E:/python/python.exe build_index.py             # 构建（命中则跳过）
    E:/python/python.exe build_index.py --rebuild   # 强制删除重建
    E:/python/python.exe build_index.py --check     # 只查指纹是否命中，不构建
"""
import argparse
import sys
import types
from pathlib import Path

# 与 e2e_rag_eval 同款接线：allagent2 目录注册成包（本脚本位于包内部，
# 直接跑时父目录默认不在 sys.path）
AGENT2_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT2_DIR.parent))
_pkg = types.ModuleType("allagent2")
_pkg.__path__ = [str(AGENT2_DIR)]
sys.modules.setdefault("allagent2", _pkg)

from allagent2.guance.logging_helper import logger  # noqa: E402
from allagent2.rag.config import rag_config  # noqa: E402
from allagent2.rag.rag import _index_fingerprint, build_index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG 索引离线预构建")
    ap.add_argument("--rebuild", action="store_true", help="强制删除旧索引重建")
    ap.add_argument("--check", action="store_true", help="只检查指纹是否命中")
    args = ap.parse_args()

    if args.check:
        from allagent2.rag.splitter import split_document
        from allagent2.rag.vectorstore import ChromaStore

        split = split_document("", child_tokens=rag_config.CHILD_TOKENS,
                               parent_ratio=rag_config.PARENT_RATIO)
        fp = _index_fingerprint(rag_config, split.children)
        store = ChromaStore(rag_config, persistent=True, fingerprint=fp)
        status = "命中（可直接启动服务，无需构建）" if store.reused else "未命中（需要构建）"
        print(f"指纹={fp[:8]}…  磁盘条数={store.count()}  当前文档子块数={len(split.children)}")
        print(f"状态：{status}")
        return 0 if store.reused else 1

    idx = build_index(rag_config, persist=True, force_rebuild=args.rebuild)
    reused = getattr(idx.store, "reused", False)
    print(f"\n完成：子块={len(idx.children)} 父块={len(idx.parents)} "
          f"落盘目录={rag_config.PERSIST_DIR} {'(命中复用)' if reused else '(已重建)'}")
    if reused and not args.rebuild:
        logger.info("磁盘索引已是指纹匹配的最新版，无需重建")
    return 0


if __name__ == "__main__":
    sys.exit(main())
