"""Token 估算误差率实测（独立脚本，不改项目代码）。

================================================================
放在哪都能跑
----------------------------------------------------------------
脚本会自动往上找项目根（含 allagent2/ 的那一层），所以：
    python token_bench.py                      # 项目根
    python allagent2/guance/token_bench.py     # guance 目录
两种都能正常导入 allagent2 与 key.env。

================================================================
怎么保证测得准
----------------------------------------------------------------
DeepSeek 的 usage.prompt_tokens 包含 chat template 固定开销（role 标记、
分隔符），而 count_tokens() 只算纯文本。所以先测一条基准文本拿到
template 开销，再逐条减掉：

    overhead = prompt_tokens("好") - tokenize("好")
    actual   = prompt_tokens(text) - overhead

注意不能直接用 prompt_tokens(text) - prompt_tokens("好")：
那样会把基准文本自身的 1 个 token 也减掉，导致每条都少 1（本脚本第一版
就踩了这个坑，看起来"恒差 1 token"其实全是基准误差）。

================================================================
用法
----------------------------------------------------------------
    cd C:\\Users\\17726\\Desktop\\练手agent
    E:\\python\\python.exe token_bench.py

依赖：allagent2/core/tokenizer/tokenizer.json（已随项目放置）
成本：10 条 × 约 300 token × max_tokens=1 ≈ 几分钱
"""
import os
import statistics
import sys
print(sys.path)

def _find_project_root() -> str:
    """往上找含 allagent2/ 的目录；找不到就用当前工作目录。"""
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(cur, "allagent2")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.getcwd()


ROOT = _find_project_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, "allagent2", "core", "key.env"))

from allagent2.guance.token_accounting import (  # noqa: E402
    _get_official_tokenizer,
    count_tokens,
)

CASES = [
    ("中文短句", "我爸最近血压有点高，平时应该注意些什么？"),
    ("中文长段", (
        "老人今年七十八岁，有高血压和糖尿病史，平时服用氨氯地平和二甲双胍。"
        "最近一周自测血压在150到160之间波动，餐后血糖偏高，晚上睡眠不好，"
        "经常起夜两到三次。家属希望了解日常饮食如何调整，是否需要尽快就医。"
        "另外老人牙口不好，硬的蔬菜咬不动，希望推荐一些软烂的做法。"
    )),
    ("纯英文", "The elderly patient has a history of hypertension and type 2 diabetes. "
               "Blood pressure has been fluctuating between 150 and 160 mmHg over the past week."),
    ("中英混合", "建议每天监测 blood pressure 两次，早晚各一次，记录在 health record 里。"),
    ("数字密集", "患者电话13812345678，身份证110101195001011234，血压150/95，血糖7.8，日期2026-10-30。"),
    ("Markdown表格", "| 时间 | 药品 | 剂量 |\n|---|---|---|\n| 早 | 氨氯地平 | 5mg |\n| 晚 | 二甲双胍 | 0.5g |"),
    ("代码片段", "def check_bp(value):\n    if value > 140:\n        return 'high'\n    return 'normal'"),
    ("标点密集", "真的吗？？？！！！那怎么办……他说：“不要紧”，但我不信。"),
    ("长文档", "高血压是老年人最常见的慢性病之一。" * 40),
]

BASELINE_TEXT = "好"
MODES = ("char", "jieba", "official")


def main() -> int:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    if not api_key:
        print(f"未找到 OPENAI_API_KEY（检查 {os.path.join(ROOT, 'allagent2', 'core', 'key.env')}）")
        return 1

    tok = _get_official_tokenizer()
    print(f"项目根: {ROOT}")
    print(f"tokenizer: {'官方（PreTrainedTokenizerFast）' if tok else '未加载，official 档将回退 char'}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    def probe(text: str) -> int:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            max_tokens=1,
            temperature=0,
        )
        return int(resp.usage.prompt_tokens)

    raw_base = probe(BASELINE_TEXT)
    base_tokens = len(tok.encode(BASELINE_TEXT, add_special_tokens=False)) if tok else 1
    overhead = raw_base - base_tokens
    print(f"模型={model}  baseline={raw_base}（其中 template 开销={overhead}）\n")

    header = f"{'用例':<12}{'官方':>6}" + "".join(f"{m:>10}{'误差':>8}" for m in MODES)
    print(header)
    print("-" * 62)

    rows = []
    for name, text in CASES:
        official = probe(text) - overhead
        if official <= 0:
            print(f"{name:<12}  官方差值 <=0，跳过")
            continue
        rec = {"name": name, "_official": official}
        line = f"{name:<12}{official:>6}"
        for mode in MODES:
            est = count_tokens(text, mode)
            err = (est - official) / official * 100
            rec[mode] = (est, err)
            line += f"{est:>10}{err:>7.1f}%"
        rows.append(rec)
        print(line)

    if not rows:
        return 2

    print("\n=== 汇总 ===")
    for mode in MODES:
        errs = [r[mode][1] for r in rows]
        mape = statistics.mean(abs(e) for e in errs)
        bias = statistics.mean(errs)
        worst = max(rows, key=lambda r: abs(r[mode][1]))
        exact = sum(1 for e in errs if abs(e) < 0.01)
        print(
            f"{mode:<9} MAPE={mape:5.1f}%  平均偏差={bias:+5.1f}%  "
            f"最大={abs(worst[mode][1]):5.1f}%（{worst['name']}）  "
            f"精确相等={exact}/{len(rows)}"
        )

    total_off = sum(r["_official"] for r in rows)
    print()
    for mode in MODES:
        total = sum(r[mode][0] for r in rows)
        print(f"总量口径 {mode:<9} 官方 {total_off} vs 估算 {total} → {(total - total_off) / total_off * 100:+.1f}%")

    print("\n判读：MAPE < 5% 可直接用于预算；< 15% 可用于切块；> 20% 需换档。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
