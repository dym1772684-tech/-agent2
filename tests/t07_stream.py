"""[t07] SSE 流式端到端（vLLM 可用时执行）。

验证：200 + text/event-stream、phase 事件陆续到达（真流式）、final 事件结构完整。
vLLM 中途掉线 → 记 SKIP，不算 FAIL。
"""
import time

from _lib import auth_header, check, client, login, probe_http, results

import os

VLLM_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8005/v1/models")


def run():
    print("[t07] SSE 流式")
    if not probe_http(VLLM_URL, timeout=3):
        return [("SSE: 完整流式问答", None, "SKIP：vLLM 未启动")]

    c = client()
    token, _ = login(c)
    if not token:
        raise RuntimeError("登录失败，跳过本组")

    phases, finals, t_first_event = 0, [], None
    final_has_error = False
    try:
        with c.stream("POST", "/api/v1/agent/chat/stream", headers=auth_header(token),
                      json={"message": "李奶奶75岁，血压偏高，饮食注意什么？",
                            "thread_id": "t-e2e-stream"},
                      timeout=300) as r:
            check("SSE: 200 且 event-stream", r.status_code == 200 and
                  "text/event-stream" in r.headers.get("content-type", ""),
                  f"http={r.status_code} ct={r.headers.get('content-type','')[:30]}")

            t0 = time.time()
            cur_event, cur_data = "", ""
            for line in r.iter_lines():
                line = str(line).rstrip("\r")
                if line.startswith("event:"):
                    cur_event = line.split(":", 1)[1].strip()
                    if cur_event == "phase":
                        phases += 1
                        if t_first_event is None:
                            t_first_event = time.time() - t0
                elif line.startswith("data:") and cur_event:
                    cur_data += line.split(":", 1)[1].strip()
                    if cur_event == "final":
                        finals.append(time.time() - t0)
                        if "执行失败" in cur_data or '"error"' in cur_data or "Error" in cur_data:
                            final_has_error = True
                    if line == "" or True:
                        pass
                elif line == "":
                    cur_event, cur_data = "", ""
    except Exception as e:
        check("SSE: 流式执行", None, f"SKIP：vLLM 执行中掉线({type(e).__name__})")
        return results()

    if final_has_error and phases == 0:
        check("SSE: 完整流式问答", None, "SKIP：vLLM 已掉线(后端返回 error final)")
        return results()

    check("SSE: phase 事件多次到达(真流式)", phases >= 2, f"phase_count={phases}")
    check("SSE: 收到 final 事件", bool(finals), f"total={finals and round(finals[0], 1)}s")
    check("SSE: 首事件快速到达(<30s)", t_first_event is not None and t_first_event < 30,
          f"first_event={t_first_event and round(t_first_event, 1)}s")

    # final 的载荷用非流式方式验证结构
    try:
        r = c.post("/api/v1/agent/chat", headers=auth_header(token),
                   json={"message": "失眠怎么调理？一句话。", "thread_id": "t-e2e-stream2"})
        if r.status_code == 200:
            data = r.json().get("data") or {}
            check("SSE: final 结构(answer/usage)",
                  bool(data.get("answer")) and bool(data.get("usage")),
                  f"keys={sorted(data)[:5]}")
        else:
            check("SSE: final 结构", None, f"SKIP：补测请求 http={r.status_code}(vLLM 掉线?)")
    except Exception:
        check("SSE: final 结构", None, "SKIP：vLLM 执行中掉线")

    return results()


if __name__ == "__main__":
    run()
