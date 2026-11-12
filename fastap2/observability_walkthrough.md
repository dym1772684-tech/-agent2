# 养老智能体 · 全链路走查报告

> 生成时间：2026-10-23 20:39:02　｜　线程：`demo-thread-001`　｜　引擎：`react`（手写思考循环）

> 本报告由 `fastap2/e2e_observability.py` 自动生成：走的是真实的图、真实的事件流、真实的 checkpoint 与中断/续跑代码路径，只有 LLM 的文本是脚本化的（零 token、可重复）。


## 0. 装配（这一轮到底加载了什么）

| 项 | 值 |
|---|---|
| 状态结构版本 | `v2`（documents.py，含逐级迁移） |
| 状态文档域 | care.session / care.conversation / care.turn / care.usage / care.trace |
| 可用工具 | `query_health_records`, `search_knowledge_base` |
| checkpointer | `InMemorySaver`（生产用 RedisSaver，语义相同） |
| 轮次上限 | 4（来自配置 `agent.max_rounds`） |
| 线程 | `demo-thread-001` |

## 1. 用户提问

```text
用户：帮我看看 E001 最近的血压情况，饮食上要注意什么？
```

## 2. 智能体的逐步思考（事件流）

每个事件一行，`#` 是全局序号（续跑时从这里接着编，不重头）。`🧠 模型返回` 后面的缩进行是模型的思考内容、输出文本和它决定调用的工具。

| # | 类型 | 节点 | 耗时 | 详情 |
|---|---|---|---|---|
| 1 | ▶ 进入 | `guard` |  | 节点开始执行 |
| 2 | ✔ 结束 | `guard` | 0ms | 下一步：校验通过，进入思考 |
| 3 | ▶ 进入 | `think` |  | 节点开始执行 |
| 4 | 🧠 思考 | `think` | - | 上下文 2 条 / 预估 560 tokens |
| 5 | 🧠 模型返回 | `think` | 0ms | token in=820(cache 520) out=96 cost=¥0.000544 |
| | ↳ 思考内容 | | | 用户问的是个人健康数据，需要先调 query_health_records 拿到近期记录。 |
| | ↳ 决定调用 | | | `query_health_records({'elder_id': 'E001'})` |
| 6 | ✔ 结束 | `think` | 0ms | 下一步：调用工具 query_health_records(elder_id=E001) |
| 7 | ▶ 进入 | `act` 第1轮 |  | 节点开始执行 |
| 8 | 🔧 工具开始 | `query_health_records` | - | 参数 {'elder_id': 'E001'} |
| 9 | 🔧 工具结束 | `query_health_records` | 10ms | 成功：老人 E001 近 7 天血压：150/95、148/92、152/96（收缩压持续偏高） |
| 10 | ✔ 结束 | `act` | 10ms | 下一步：执行 1 个工具、复用 0 个已执行结果，交回模型继续判断 |
| 11 | ▶ 进入 | `observe` 第1轮 |  | 节点开始执行 |
| 12 | 👁 观察 | `observe` | - | 本轮收到 1 条工具结果：query_health_records（成功 1 / 异常 0） |
| 13 | ✔ 结束 | `observe` | 0ms | 下一步：回到思考：判断信息是否足够，不够就继续调工具 |
| 14 | ▶ 进入 | `think` 第1轮 |  | 节点开始执行 |
| 15 | 🧠 思考 | `think` | - | 上下文 4 条 / 预估 620 tokens |
| 16 | 🧠 模型返回 | `think` | 0ms | token in=760(cache 480) out=130 cost=¥0.000588 |
| | ↳ 思考内容 | | | 已拿到近 7 天血压记录，收缩压均超过 140，属于持续偏高。先给出测量与饮食建议，再提示red flag。 |
| | ↳ 输出文本 | | | 老人近 7 天收缩压 148~152，持续偏高。建议：① 每日固定时间测量并记录；② 食盐控制在 5g 以内，少吃腌制食品；③ 若出现头痛、胸闷，立即就医。 |
| 17 | ✔ 结束 | `think` | 0ms | 下一步：信息已足够，进入结构化收尾 |
| 18 | ▶ 进入 | `finalize` 第2轮 |  | 节点开始执行 |
| 19 | 📦 结构化收尾 | `finalize` | - | 成功，答案 38 字 |
| 20 | ✔ 结束 | `finalize` | 0ms | 下一步：结束本轮，返回结构化结果 |

## 3. 最终回答

```text
老人近 7 天收缩压 148~152，持续偏高，建议规律测量并控制食盐摄入。
```

| 结构化字段 | 值 |
|---|---|
| health_advice | 每日固定时间测血压并记录，保证睡眠。 |
| diet_advice | 每日食盐 <5g，少吃腌制食品，增加富钾食物。 |
| exercise_advice | 以散步为主，每次 20~30 分钟，避免清晨剧烈活动。 |
| medication_advice | None |
| risk_level | 中 |
| emergency_flag | False |

**本轮成本（按模型分桶）**

| 模型 | 调用次数 | 输入 | 输出 | 缓存命中 | 成本（元） |
|---|---|---|---|---|---|
| `demo-model-7b` | 2 | 1580 | 226 | 1000 | 0.001132 |

## 4. checkpoint 快照时间线

LangGraph 每跑完一个节点就落一次 checkpoint。下表是**每一步落盘后的状态**：「下一步节点」为空表示这一轮已经跑完。

| 步 | checkpoint | 下一步节点 | 消息数 | 新增 | trace 长度 | 轮次 | 阶段 |
|---|---|---|---|---|---|---|---|
| 1 | `1f1cedeb-a32` | __start__ | 0 | +0 | 0 | 0 | - |
| 2 | `1f1cedeb-a32` | guard | 1 | +1 | 0 | 0 | - |
| 3 | `1f1cedeb-a33` | think | 1 | +0 | 2 | 0 | guard_pass |
| 4 | `1f1cedeb-a33` | act | 2 | +1 | 6 | 1 | think |
| 5 | `1f1cedeb-a35` | observe | 3 | +1 | 10 | 1 | act |
| 6 | `1f1cedeb-a35` | think | 3 | +0 | 13 | 1 | observe |
| 7 | `1f1cedeb-a35` | finalize | 4 | +1 | 17 | 2 | think |
| 8 | `1f1cedeb-a36` | (结束) | 4 | +0 | 20 | 2 | finalize |

## 5. 用户中断

```text
用户（第二轮）：那运动方面呢？他早上喜欢去公园打太极。
```

模型已经完成第 1 次思考并决定调工具，此时用户点了「停止」。中断不是杀线程，而是给会话打标记 —— 图在**下一个节点入口**看到标记就停下。

| 项 | 值 |
|---|---|
| 是否中断 | 是 |
| 中断异常 | `会话 demo-thread-001 已在节点 act 前中断` |
| 停在哪一步 | 已完成 26 条事件，seq 水位 = 26 |
| 快照还在吗 | 在（下一步待执行：`act`） |
| 已保留消息数 | 6 条 |
| 旁路事故记录 | 1 条（error/interrupted 不进 state，落旁路） |

**中断事件（来自旁路，state 里看不到 —— 这正是 P0 修复的点）**

| # | 类型 | 节点 | 耗时 | 详情 |
|---|---|---|---|---|
| 27 | ⏸ **用户中断** | `act` | - | 收到中断标记，链路停在上一快照 |

## 6. 重启对话：断点内容重新注入

续跑时调用 `graph.invoke(None, config)` —— 第一个参数是 `None`，表示「不要新的用户消息，从最后一个 checkpoint 加载状态继续」，所以 `next` 里的节点会接着跑，历史消息、trace、工具黑板全部被重新注入。

| 项 | 中断时 | 续跑后 | 说明 |
|---|---|---|---|
| 消息数 | 6 | 8 | 续跑把工具结果 + AI 回复补进了上下文 |
| trace 长度 | 26 | 40 | 事件继续追加，不清空 |
| seq 水位 | 26 | 41 | 从断点接着编号，不重头 |
| 轮次 | 3 | 4 | 轮次连续 |
| 续跑错误 | - | `无` | - |

**续跑产生的新事件（seq 接在上次水位之后）**

| # | 类型 | 节点 | 耗时 | 详情 |
|---|---|---|---|---|
| 28 | ▶ 进入 | `act` 第3轮 |  | 节点开始执行 |
| 29 | 🔧 工具开始 | `search_knowledge_base` | - | 参数 {'query': '高血压老人 太极 运动强度'} |
| 30 | 🔧 工具结束 | `search_knowledge_base` | 10ms | 成功：高血压饮食建议：每日食盐 <5g；增加富钾食物（香蕉/菠菜）；限制饮酒。 |
| 31 | ✔ 结束 | `act` | 11ms | 下一步：执行 1 个工具、复用 0 个已执行结果，交回模型继续判断 |
| 32 | ▶ 进入 | `observe` 第3轮 |  | 节点开始执行 |
| 33 | 👁 观察 | `observe` | - | 本轮收到 2 条工具结果：query_health_records, search_knowledge_base（成功 2 / 异常 0） |
| 34 | ✔ 结束 | `observe` | 0ms | 下一步：回到思考：判断信息是否足够，不够就继续调工具 |
| 35 | ▶ 进入 | `think` 第3轮 |  | 节点开始执行 |
| 36 | 🧠 思考 | `think` | - | 上下文 8 条 / 预估 743 tokens |
| 37 | 🧠 模型返回 | `think` | 0ms | token in=760(cache 480) out=130 cost=¥0.000588 |
| | ↳ 思考内容 | | | 用户问运动，先查知识库拿有依据的建议，再结合太极这种低强度运动给具体强度。 |
| | ↳ 输出文本 | | | 太极适合老人，但要注意：① 避开清晨血压高峰，改到上午 9 点后；② 每次 20~30 分钟，以微微出汗、能正常说话为准；③ 出现头晕立即停下休息。 |
| 38 | ✔ 结束 | `think` | 1ms | 下一步：信息已足够，进入结构化收尾 |
| 39 | ▶ 进入 | `finalize` 第4轮 |  | 节点开始执行 |
| 40 | 📦 结构化收尾 | `finalize` | - | 成功，答案 38 字 |
| 41 | ✔ 结束 | `finalize` | 0ms | 下一步：结束本轮，返回结构化结果 |

### 6.1 对照：新开一个会话问同样的问题

同一个问题，如果是**新线程**（`demo-thread-fresh`），模型只看得到「这一问 + 这一答」；而续接原线程时，历史消息 + 断点状态全部被重新注入。这就是「断点内容重新注入」的实际差别。

| 场景 | 线程 | 模型可见消息数 | 能否接着上一轮说 |
|---|---|---|---|
| 新会话 | `demo-thread-fresh` | 2 | 不能（从零开始） |
| 续接 | `demo-thread-001` | 8 | 能（历史 + 断点全部注入） |

### 6.2 幂等：同一请求「重放」只生效一次

第 6 幕是用户**有意**的中断后续跑；客户端超时重试、用户连点是**无意**的重放，靠下面四层挡住。这里实测的是 L1 的三态状态机：

演示后端：真 Redis

| 时刻 | 状态 | 会发生什么 |
|---|---|---|
| 第一次请求 | `new` | SET NX 抢到执行权 → 真正跑模型 |
| 同键并发（如用户连点） | `inflight` | 被挡住 → 返回 409，不会双扣 token / 双写库 |
| 执行完再重试（客户端超时重试） | `done` | 回放缓存 `缓存下来的答案…`，零 token、零工具副作用 |

**四层各管一段，互补而非重复**

| 层 | 位置 | 挡什么 | 单测 |
|---|---|---|---|
| L1 HTTP 幂等键 | `core/idempotency.py` + `/chat` `/chat/stream` | 客户端重试、用户连点：`done` 回放 / `inflight` 409 | 13 例 |
| L2 工具台账 | `react_loop.act_node` 的 `tool_ledger`（key=`tool_call_id`） | checkpoint 重放 / 续跑：同一调用不重复执行 | 4 例 |
| L3 写工具 | `_invoke_with_retry(write=True)` + `mcp_servers/idempotency.py` | 写操作不盲目重试；声明了 `idempotency_key` 就由 MCP 端 SETNX 去重 | 9 例 |
| L4 线程登记 | `core/thread_registry.py` | 首次并发登记同一 thread 撞 unique → 复查，而不是 500 | 7 例 |


## 7. 指标增量（/metrics 里会看到的东西）

走查过程中注入的内存指标收集器捕获到的全部数据：

| 指标 | 次数 | 累计值 |
|---|---|---|
| `agent_interrupt_total{node=act}` | 1 | 1.0 |
| `agent_llm_cost_yuan_total` | 5 | 0.0028 |
| `agent_llm_seconds` | 5 | 0.0 |
| `agent_llm_tokens_total{kind=cache_read}` | 5 | 2480.0 |
| `agent_llm_tokens_total{kind=input}` | 5 | 3920.0 |
| `agent_llm_tokens_total{kind=output}` | 5 | 582.0 |
| `agent_node_seconds{node=act}` | 2 | 0.0212 |
| `agent_node_seconds{node=finalize}` | 3 | 0.0003 |
| `agent_node_seconds{node=guard}` | 3 | 0.0 |
| `agent_node_seconds{node=observe}` | 2 | 0.0 |
| `agent_node_seconds{node=think}` | 5 | 0.0023 |
| `agent_node_total{node=act}` | 2 | 2.0 |
| `agent_node_total{node=finalize}` | 3 | 3.0 |
| `agent_node_total{node=guard}` | 3 | 3.0 |
| `agent_node_total{node=observe}` | 2 | 2.0 |
| `agent_node_total{node=think}` | 5 | 5.0 |
| `agent_tool_seconds{tool=query_health_records}` | 1 | 0.0103 |
| `agent_tool_seconds{tool=search_knowledge_base}` | 1 | 0.0105 |
| `agent_tool_total{tool=query_health_records,status=ok}` | 1 | 1.0 |
| `agent_tool_total{tool=search_knowledge_base,status=ok}` | 1 | 1.0 |

## 8. 这套东西对应到线上怎么看

| 想知道的事 | 看哪里 |
|---|---|
| 这次请求都干了什么 | 日志按 `rid=` 过滤（HTTP 头 / SSE 首帧 `meta` 事件都有） |
| 某轮对话的完整步骤 | `GET /api/v1/agent/trace/{thread_id}`（含失败/中断的旁路事件） |
| 哪个模型烧的钱、哪个工具最慢 | `GET /api/v1/agent/metrics` + `GET /metrics` |
| 当前用什么配置、值从哪来 | `GET /api/v1/agent/config`（密钥已脱敏） |
| 中断 / 续跑 | `POST /interrupt` → `POST /resume` → `GET /trace` 核对 seq 接续 |
| 服务是不是持久化模式 | `GET /api/v1/agent/status` 的 `persistent` 字段 |

---

状态结构版本 v2　｜　事件总数 34　｜　中断 1 次　｜　续跑 成功　｜　报告文件 `observability_walkthrough.md`