"""SSE 流式端到端验证（调真实 LLM，team 拓扑）。

用法:
    1. 先启动 MCP 双服务:
       cd fastap2 && E:/python/python.exe -m mcp_servers.elder_server   # 8001
       cd fastap2 && E:/python/python.exe -m mcp_servers.life_server    # 8002
    2. 启动主服务（team 拓扑）:
       cd fastap2 && AGENT_TOPOLOGY=team E:/python/python.exe -m uvicorn main:app --port 8000
    3. cd fastap2 && E:/python/python.exe e2e_stream.py

验证项:
    [1] /agent/chat/stream 返回 200 + text/event-stream
    [2] phase 事件按时间陆续到达（真流式，不是一次性吐完）
    [3] final 事件 answer 非空（验证 max_tokens 修复：aggregate 不再返回 None）
    [4] final.structured 结构完整（ParsedOutput 字段齐全）
    [5] final.usage 有本轮 token 账单
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000/api/v1"

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def post_json(url, payload, headers=None):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp, json.loads(resp.read().decode("utf-8"))


def main():
    # 1. 登录拿 token
    _, body = post_json(f"{BASE}/login", {"username": "testuser01", "password": "test123456"})
    token = body["access_token"]
    print("登录成功")

    # 2. 打流式端点，逐行读 SSE 并记录每个事件到达时间
    req = urllib.request.Request(
        f"{BASE}/agent/chat/stream",
        data=json.dumps({
            "message": "李奶奶今年75岁，最近血压偏高，饮食上要注意什么？",
            "debug": False,
        }).encode("utf-8"),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {token}")

    t_start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        ctype = resp.headers.get("Content-Type", "")
        check("1.HTTP 200 + event-stream", resp.status == 200 and "text/event-stream" in ctype,
              f"status={resp.status} ctype={ctype}")

        events = []          # (elapsed_s, event, data)
        cur_event, cur_data = None, ""
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n\r")
            if line.startswith("event: "):
                cur_event = line[7:].strip()
            elif line.startswith("data: "):
                cur_data = line[6:]
            elif line == "" and cur_event:
                try:
                    data = json.loads(cur_data)
                except Exception:
                    data = cur_data
                events.append((round(time.perf_counter() - t_start, 2), cur_event, data))
                cur_event, cur_data = None, ""

    # 打印事件时序
    for elapsed, ev, data in events:
        brief = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)[:120]
        print(f"  +{elapsed:>6.2f}s  {ev:<8} {brief}")

    phases = [(e, d) for t, e, d in events if e == "phase"]
    finals = [(t, d) for t, e, d in events if e == "final"]
    check("2.收到 phase 事件", len(phases) >= 2, f"phase数={len(phases)}")

    # 真流式判定：phase 事件到达时间跨度 > 0.5s（一次性吐完则是伪流式）
    if len(phases) >= 2:
        span = events[-1][0] - events[0][0] if len(events) >= 2 else 0
        phase_times = [t for t, e, d in events if e == "phase"]
        span = phase_times[-1] - phase_times[0] if len(phase_times) >= 2 else 0
        check("3.phase 随时间陆续到达（真流式）", span > 0.5, f"时序跨度={span:.2f}s")

    check("4.收到 final 事件", len(finals) == 1)
    if not finals:
        sys.exit(1)
    final = finals[0][1]
    answer = final.get("answer") or ""
    check("5.final.answer 非空（aggregate 修复生效）",
          bool(answer.strip()) and "未生成可展示" not in answer,
          f"长度={len(answer)} 前80字={answer[:80]!r}")
    structured = final.get("structured") or {}
    check("6.structured 结构完整", isinstance(structured, dict) and len(structured) >= 3,
          f"键={list(structured.keys())}")
    usage = final.get("usage") or {}
    check("7.usage 账单齐全", isinstance(usage, dict) and usage.get("calls", 0) > 0,
          f"calls={usage.get('calls')} in={usage.get('input_tokens')} out={usage.get('output_tokens')}")

    failed = [r for r in results if not r[1]]
    print(f"\n==== {len(results) - len(failed)}/{len(results)} 通过 ====")
    if failed:
        for name, _, detail in failed:
            print(f"  失败项: {name} -> {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
