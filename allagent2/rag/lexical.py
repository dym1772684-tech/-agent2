"""词法打分器：chunk 级 TF × DF/IDF，系数全暴露。

================================================------------
对照你的需求逐条落地
------------------------------------------------------------
【TF 按 chunk 而非整篇算】
  原实现的 TF/BM25 确实已是 chunk 级（这点原代码就对了），
  本模块保持 chunk 粒度——"文章越大的情况下，相关性在细小chunk里更强"。

【DF/IDF 给 chunk 打分】
  DF 统计 = 一个词出现在多少个 chunk（去重计数）；
  IDF = log((N - df + Δ) / (df + Δ) + 1)，Lucene 平滑口径，Δ 可调。
  稀缺词（只出现在个别 chunk）权重高，正是"最符合逻辑"的区分信号。

【微调系数】
  K1（词频饱和）、B（长度归一）、SUBLINEAR_TF（1+log tf）、
  DELTA_IDF（平滑）、DF_MAX_RATIO（高频剔除）全部来自 rag_config，
  网格搜索直接换 config 实例扫，不重 tokenize（索引可复用）。

【自动停用词发现】
  DF > DF_MAX_RATIO × N 的词（如"老人"每段都有）直接剔除——
  它们不携带区分信息，还会把 IDF 权重搅乱。

改的意义（对照原 BM_25）：
  - 原 k1/b 写死在签名默认值，调用方从不传 → 从未调过；
  - 无 sublinear TF、无 DF 过滤、无平滑控制；
  - 每次检索重新算 DF/IDF/avgdl（全量重扫），现在建索引时一次算好。
============================================================
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

import jieba

from .config import RAGConfig
from .splitter import ChildChunk


class LexicalIndex:
    """建一次、查多次的词法索引（词法路的全部状态都在这里）。"""

    def __init__(self, children: List[ChildChunk], config: RAGConfig):
        self.cfg = config
        self.children = children
        # chunk 粒度分词 + 词频表
        # 【修复】分词器用 cut_for_search（搜索引擎模式）而非 lcut：
        #   lcut 会把用户查询『血压高』切成 1 个词，而正文里是『高血压』，
        #   三种切法互相匹配不上（中文分词错位）。cut_for_search 会同时
        #   输出全词和词典子词（血压高→血压+血压高，高血压→血压+高血压），
        #   两边在『血压』这个子词上桥接。
        # 【修复】索引文本用 index_text（标题末段+正文）：标题『血压管理』
        #   携带最强主题信号，不拼进来等于把最容易区分的词扔掉了。
        self.doc_words: List[List[str]] = []
        self.doc_tf: List[Counter] = []
        for c in children:
            words = [w for w in jieba.cut_for_search(c.index_text) if w.strip()]
            self.doc_words.append(words)
            self.doc_tf.append(Counter(words))

        n = len(children)
        # DF：词 → 出现在多少个 chunk
        df: Counter = Counter()
        for words in self.doc_words:
            df.update(set(words))
        # 自动停用词发现：DF 占比超限的词剔除（不携带区分信息）
        df_cutoff = max(int(n * config.DF_MAX_RATIO), 1)
        self.stopwords = {w for w, v in df.items() if v > df_cutoff and len(w) <= 2}
        self.df = df

        # IDF（Lucene 平滑口径，Δ 可调）
        delta = config.DELTA_IDF
        self.idf: Dict[str, float] = {
            w: math.log((n - v + delta) / (v + delta) + 1.0) for w, v in df.items()
        }

        # 平均块长（BM25 长度归一用）
        lens = [len(w) for w in self.doc_words]
        self.avgdl = (sum(lens) / len(lens)) if lens else 1.0

    # --------------------------------------------------------
    # 打分
    # --------------------------------------------------------
    def score(self, query: str, top_k: int = 0) -> List[Dict[str, Any]]:
        """对全部 chunk 打 BM25 分，返回按分值降序的 [{id, text, score, ...}]。

        top_k=0 返回全量（评测用），>0 截断（在线检索用）。
        """
        q_words = [w for w in jieba.cut_for_search(query or "") if w.strip()]
        k1, b = self.cfg.K1, self.cfg.B
        sublinear = self.cfg.SUBLINEAR_TF

        results: List[Dict[str, Any]] = []
        for i, c in enumerate(self.children):
            tf_map = self.doc_tf[i]
            dl = len(self.doc_words[i])
            norm = k1 * (1 - b + b * dl / (self.avgdl or 1.0))
            score = 0.0
            matched: List[str] = []
            for w in q_words:
                if w in self.stopwords or w not in tf_map:
                    continue
                tf = tf_map[w]
                if sublinear:
                    # 1+log(tf)：重复出现增益递减，抑制刷分
                    tf_component = (1.0 + math.log(tf)) * (k1 + 1.0) / (1.0 + math.log(tf) + norm)
                else:
                    tf_component = tf * (k1 + 1.0) / (tf + norm)
                score += self.idf.get(w, 0.0) * tf_component
                if w not in matched:
                    matched.append(w)
            results.append({
                "id": c.id,
                "text": c.text,
                "score": score,
                "matched_words": matched,   # 可观测：这个词法命中是哪些词贡献的
                "heading_path": c.heading_path,
                "parent_id": c.parent_id,
                "doc_source": c.doc_source,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k] if top_k else results

    # --------------------------------------------------------
    # 评测辅助
    # --------------------------------------------------------
    def build_with(self, config: RAGConfig) -> "LexicalIndex":
        """换系数（k1/b/sublinear/df_ratio）时无需重新分词，直接换配置。

        分词是这里最贵的步骤；DF/IDF 只受 DF_MAX_RATIO 影响。
        网格搜索单独扫词法参数时用这个，快一个数量级。
        """
        if config.DF_MAX_RATIO == self.cfg.DF_MAX_RATIO and config.DELTA_IDF == self.cfg.DELTA_IDF:
            # 完全可复用：只换 k1/b/sublinear（打分期才用到的系数）
            new = LexicalIndex.__new__(LexicalIndex)
            new.cfg = config
            new.children = self.children
            new.doc_words = self.doc_words
            new.doc_tf = self.doc_tf
            new.df = self.df
            new.stopwords = self.stopwords
            new.idf = self.idf
            new.avgdl = self.avgdl
            return new
        return LexicalIndex(self.children, config)
