"""fastap2 统一评测框架(eval)。

分层与存量资产映射:
  L1 检索评测  suite=rag       数据源 allagent2/rag/evalset.jsonl(56条,单一事实源)
  L2 端到端    suite=qa        数据源 allagent2/tests/evalset_100.jsonl(100条)
                              + eval/datasets/qa_reference.jsonl(参考答案,逐步补)
  L3 安全      suite=security  evalset_100 的 security 类目(32条)
                              + eval/datasets/security_extra.jsonl(补充用例)

判定分工:
  规则断言(assertions.py) —— 确定性:关键词/正则/emergency,安全用例唯一裁决方式
  LLM 裁判(judge.py)      —— 语义四维评分,仅 qa suite 且条目带参考答案时启用

运行(由用户执行,框架不自动跑任何 LLM):
  cd fastap2 && python eval/run_eval.py --suite qa --mode offline --workers 4
  python eval/run_eval.py --suite security --gate          # 安全门禁(必须 100%)
  python eval/run_eval.py --suite rag                       # 只跑检索,不碰 LLM
"""
