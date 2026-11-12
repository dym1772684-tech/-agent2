"""RAG 参数中心：所有可调系数集中一处。

============================================================
【与 allagent2/config 的关系 —— 避免踩第二套配置的坑】
------------------------------------------------------------
本项目有两套配置对象，职责不同、**不要互相修改对方的默认值**：

  1) 本文件 `rag_config`（RAGConfig）：RAG 链路的**唯一运行参数源**，
     rag.py / rerank.py / fusion.py / lexical.py 全部从它读。
     环境变量前缀混用 RAG_*（切分/召回）与 HYDE_*/W_HYDE（第三路召回）。
  2) allagent2/config/schema.py 的 `Settings`：全局运行时配置快照
     （agent 压缩阈值、memory 反思门槛等），其中的 rag.* / hyde.* 字段
     只是为了"把生效配置整体打出来做快照"，rag 代码并不读它。

两处读的是**同一批环境变量**、默认值目前一致（HYDE_ENABLED=True /
HYDE_TOP_K=20 / W_HYDE=0.3）。改默认值时两边都要改，否则快照会骗人。
长期方案是删掉 schema.py 里的 rag.* 段、只保留本文件——属于技术债，未合并。

============================================================
为什么要这个文件
------------------------------------------------------------
原来 k1=1.5 / b=0.75 / rrf_k=10 全部硬编码在函数签名或调用点里，
想调一个系数要翻五个文件，而且**调完无法对比**——没有评测集，
没有参数快照，"改好了"只是错觉。

现在全部参数集中为 dataclass：
  1) 代码里统一从 rag_config 读，改一处全局生效；
  2) 每个参数都支持环境变量覆盖（RAG_ 前缀），不改代码就能做 A/B；
  3) 评测脚本 (e2e_rag_eval.py) 网格搜索产出的最优参数，直接
     用 config 实例化写回默认值，形成「参数 → 评测 → 回写」闭环。

参数分四组：
  切分组    控制父子块的粒度（检索粒度的地基）
  词法组    BM25 的 TF/DF/IDF 系数（关键词检索）
  向量组    HNSW 索引与检索参数
  融合/重排组 两路权重配比与精排控制
============================================================
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


@dataclass
class RAGConfig:
    # --------------------------------------------------------
    # ① 切分组
    # --------------------------------------------------------
    # 子块 token 上限。检索用子块（粒度细、相关性强），返回父块（上下文完整）。
    CHILD_TOKENS: int = 128

    # 父块 = 相邻子块合并，目标 token 量 = CHILD_TOKENS × PARENT_RATIO。
    # 你给的区间是 1.5~2.0，取 1.8 做初值，靠网格搜索收敛。
    PARENT_RATIO: float = 1.8

    # 符号递归切分的层级：从粗到细依次尝试，直到块 ≤ CHILD_TOKENS
    SPLIT_SEPARATORS: tuple = ("\n\n", "\n", "。", "，")

    # --------------------------------------------------------
    # ② 词法组（chunk 级 TF × DF/IDF）
    # --------------------------------------------------------
    K1: float = 1.5          # BM25 词频饱和系数：越大高频词增益越持久
    B: float = 0.75          # 长度归一强度：0=不归一，1=完全归一
    SUBLINEAR_TF: bool = True  # 用 1+log(tf) 替代 tf，抑制重复刷分
    DELTA_IDF: float = 0.5   # IDF 平滑项（Lucene 口径）
    # DF 超过 DF_MAX_RATIO×N 的词视为"准停用词"直接剔除
    # （自动停用词发现：像"老人"这种每段都有的词不该参与区分）
    DF_MAX_RATIO: float = 0.6

    # --------------------------------------------------------
    # ③ 向量组（Chroma HNSW）
    # --------------------------------------------------------
    HNSW_SPACE: str = "cosine"     # 距离度量（归一化向量下等价点积）
    HNSW_M: int = 16               # 每节点双向连接数：越大召回越高、内存越大
    HNSW_CONSTRUCTION_EF: int = 200  # 建图时候选队列长度
    HNSW_SEARCH_EF: int = 64       # 查询时候选队列长度

    # --------------------------------------------------------
    # ④ 融合 / 重排组
    # --------------------------------------------------------
    # 融合方式：relative（归一化线性加权）| rrf | weighted_rrf
    # relative 的权重直白可比：换更强 embedding 就把 W_VECTOR 往上调
    FUSION_MODE: str = "relative"
    W_LEXICAL: float = 0.4   # 词法路权重
    W_VECTOR: float = 0.6    # 向量路权重（模型越强此值越高）
    RRF_K: int = 60          # rrf/weighted_rrf 的排名平滑常数

    # 送入 CrossEncoder 精排的候选数（从融合结果取前 N）
    RERANK_TOP_N: int = 20
    # 重排低分阈值：低于它的候选直接丢弃，防噪声片段进 prompt
    RERANK_MIN_SCORE: float = 0.0
    # 重排开关（评测网格搜索阶段关闭，只对最终 top-3 组合开启验证）
    ENABLE_RERANK: bool = True

    # --------------------------------------------------------
    # ⑤ 检索行为
    # --------------------------------------------------------
    TOP_K: int = 4            # 最终返回父块数
    VECTOR_TOP_K: int = 20    # 向量路召回子块数
    LEXICAL_TOP_K: int = 20   # 词法路召回子块数

    # --------------------------------------------------------
    # ⑥ 模型
    # --------------------------------------------------------
    EMBEDDING_MODEL: str = "shibing624/text2vec-base-chinese"
    # 【换型】重排模型从 cross-encoder/mmarco-mMiniLMv2-L12-H384-v1（英文系）
    # 换成 BAAI/bge-reranker-base（中文系，已下载到本地，离线可用）。
    # 实测旧模型对中文打分为负：「补钙吃什么好」18 条候选全被低分过滤
    # 丢弃（0 结果）、「血压」被排到「血糖」之后 —— 模型选型问题，非阈值问题。
    _LOCAL_RERANKER = os.path.join(
        os.path.dirname(__file__), "models", "bge-reranker-base"
    )
    RERANKER_MODEL: str = _LOCAL_RERANKER if os.path.isdir(_LOCAL_RERANKER) else \
        "BAAI/bge-reranker-base"

    # --------------------------------------------------------
    # ⑦ 持久化
    # --------------------------------------------------------
    # Chroma 落盘目录（在线路径用 PersistentClient；网格搜索仍走内存版）。
    # 指纹（文档内容+切分参数+embedding 模型）不匹配时自动重建，否则直接复用。
    _PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    PERSIST_DIR: str = _PERSIST_DIR

    # --------------------------------------------------------
    # ⑧ HyDE 第三路（可选）：无 HYDE_API_KEY 时 hyde.py 自动返回空串，
    #    fusion 侧也不把 W_HYDE 计入权重归一，结果与双路完全一致。
    # --------------------------------------------------------
    HYDE_ENABLED: bool = True
    HYDE_TOP_K: int = 20
    W_HYDE: float = 0.3
    HYDE_MAX_TOKENS: int = 256
    HYDE_TIMEOUT: float = 5.0

    def __post_init__(self) -> None:
        # 环境变量覆盖（RAG_ 前缀）：不改代码做 A/B 的口子
        self.CHILD_TOKENS = _env_int("RAG_CHILD_TOKENS", self.CHILD_TOKENS)
        self.PARENT_RATIO = _env_float("RAG_PARENT_RATIO", self.PARENT_RATIO)
        self.K1 = _env_float("RAG_K1", self.K1)
        self.B = _env_float("RAG_B", self.B)
        self.SUBLINEAR_TF = _env_bool("RAG_SUBLINEAR_TF", self.SUBLINEAR_TF)
        self.DELTA_IDF = _env_float("RAG_DELTA_IDF", self.DELTA_IDF)
        self.DF_MAX_RATIO = _env_float("RAG_DF_MAX_RATIO", self.DF_MAX_RATIO)
        self.HNSW_SPACE = os.getenv("RAG_HNSW_SPACE", self.HNSW_SPACE)
        self.HNSW_M = _env_int("RAG_HNSW_M", self.HNSW_M)
        self.HNSW_CONSTRUCTION_EF = _env_int("RAG_HNSW_CONSTRUCTION_EF", self.HNSW_CONSTRUCTION_EF)
        self.HNSW_SEARCH_EF = _env_int("RAG_HNSW_SEARCH_EF", self.HNSW_SEARCH_EF)
        self.FUSION_MODE = os.getenv("RAG_FUSION_MODE", self.FUSION_MODE).lower()
        self.W_LEXICAL = _env_float("RAG_W_LEXICAL", self.W_LEXICAL)
        self.W_VECTOR = _env_float("RAG_W_VECTOR", self.W_VECTOR)
        self.RRF_K = _env_int("RAG_RRF_K", self.RRF_K)
        self.HYDE_ENABLED = _env_bool("HYDE_ENABLED", self.HYDE_ENABLED)
        self.HYDE_TOP_K = _env_int("HYDE_TOP_K", self.HYDE_TOP_K)
        self.W_HYDE = _env_float("W_HYDE", self.W_HYDE)
        self.HYDE_MAX_TOKENS = _env_int("HYDE_MAX_TOKENS", self.HYDE_MAX_TOKENS)
        self.HYDE_TIMEOUT = _env_float("HYDE_TIMEOUT", self.HYDE_TIMEOUT)
        self.RERANK_TOP_N = _env_int("RAG_RERANK_TOP_N", self.RERANK_TOP_N)
        self.RERANK_MIN_SCORE = _env_float("RAG_RERANK_MIN_SCORE", self.RERANK_MIN_SCORE)
        self.ENABLE_RERANK = _env_bool("RAG_ENABLE_RERANK", self.ENABLE_RERANK)
        self.TOP_K = _env_int("RAG_TOP_K", self.TOP_K)
        self.VECTOR_TOP_K = _env_int("RAG_VECTOR_TOP_K", self.VECTOR_TOP_K)
        self.LEXICAL_TOP_K = _env_int("RAG_LEXICAL_TOP_K", self.LEXICAL_TOP_K)
        self.EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", self.EMBEDDING_MODEL)
        self.RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", self.RERANKER_MODEL)
        self.PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", self.PERSIST_DIR)
        # FUSION_MODE 白名单校验
        if self.FUSION_MODE not in ("relative", "rrf", "weighted_rrf"):
            self.FUSION_MODE = "relative"

    def with_overrides(self, **overrides: Any) -> "RAGConfig":
        """生成一个带临时覆盖的新配置（网格搜索用，不动全局）。"""
        data = asdict(self)
        data.pop("SPLIT_SEPARATORS", None)
        cfg = RAGConfig(**{k: v for k, v in data.items() if k != "SPLIT_SEPARATORS"})
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 全局默认实例：检索链路统一从这里取参数
rag_config = RAGConfig()
