"""team 拓扑冒烟测试(零 LLM 调用,不产生费用)。

验证:
  [1] bootstrap 以 AGENT_TOPOLOGY=team 接线成功
  [2] 编译出的图节点含 supervisor/health/knowledge/life/aggregate
  [3] run_agent 的急救入口短路在 team 模式下仍生效(零 token)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    import importlib

    # agent_bridge/__init__ 重导出了 bootstrap 函数,会遮住子模块名,须用 importlib 取模块
    bs = importlib.import_module("agent_bridge.bootstrap")

    st = bs.bootstrap()
    status = bs.get_status()
    check("1.bootstrap 接线(team)", status["ready"] and status["topology"] == "team",
          f"status={json.dumps(status, ensure_ascii=False)}")

    graph = st.agent
    node_names = set(getattr(graph, "nodes", {}) or {})
    expect = {"supervisor", "health", "knowledge", "life", "aggregate"}
    check("2.图节点齐全", expect <= node_names, f"nodes={sorted(node_names)}")

    # 急救入口短路:不进图、不调 LLM
    out = bs.run_agent("老人突然胸痛呼吸困难怎么办")
    check("3.急救入口短路", out.get("structured", {}).get("emergency_flag") is True
          and "120" in (out.get("answer") or ""),
          f"answer前60字={out.get('answer', '')[:60]}")

    print(f"\n结果: {sum(results)}/{len(results)} 通过")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
