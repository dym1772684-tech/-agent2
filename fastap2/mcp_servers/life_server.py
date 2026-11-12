"""生活工具 MCP Server(端口 8002,streamable-HTTP):天气/计算器/网页访问/健康趋势/BMI。

启动:cd fastap2 && python -m mcp_servers.life_server

无敏感数据,不校验 agent_context;相对原 allagent2 本地实现的加固:
- requests 全部带 timeout(原来没有 timeout,天气接口卡住会挂死调用线程)
- 地理编码/天气查询各重试 1 次
- fetch_webpage 带 SSRF 防护:仅 http/https、拒绝私网/环回地址、禁自动重定向
"""
import ipaddress
import os
import socket
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(Path(__file__).resolve().parent.parent / "core" / "key.env")

mcp = FastMCP("life-tools", host="127.0.0.1", port=8002)

_TIMEOUT = (3, 10)  # (连接超时, 读超时) 秒


@mcp.tool()
def calculator(fuhao: str, number1: float, number2: float) -> str:
    """计算:支持 加法/减法/乘法/除法(也接受 + - * /)"""
    fuhao_map = {"加法": "+", "减法": "-", "乘法": "*", "除法": "/"}
    fuhao = fuhao_map.get(fuhao, fuhao)
    try:
        if fuhao == "+":
            return str(float(number1) + float(number2))
        if fuhao == "-":
            return str(float(number1) - float(number2))
        if fuhao == "*":
            return str(float(number1) * float(number2))
        if fuhao == "/":
            if float(number2) == 0:
                return "除数不能为0"
            return str(float(number1) / float(number2))
        return "错误输入"
    except Exception:
        return "计算失败"


@mcp.tool()
def get_real_weather(city_name: str) -> str:
    """获取指定城市当前天气(中文描述 + 温度)"""
    api_key = os.getenv("WEATHER_API_KEY")
    last_err: Exception | None = None
    # 【加固】重试 1 次;原来无 timeout,外网接口卡住会挂死调用线程
    for _attempt in range(2):
        try:
            geo = requests.get(
                "http://api.openweathermap.org/geo/1.0/direct",
                params={"q": city_name, "appid": api_key, "limit": 1, "lang": "zh_cn"},
                timeout=_TIMEOUT,
            )
            geo.raise_for_status()
            geo_data = geo.json()
            if not geo_data:
                return f"没有找到城市 {city_name}"
            lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]

            weather = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"lat": lat, "lon": lon, "appid": api_key, "lang": "zh_cn", "units": "metric"},
                timeout=_TIMEOUT,
            )
            weather.raise_for_status()
            w = weather.json()
            return f"{w['name']}:{w['weather'][0]['description']},当前温度{w['main']['temp']}℃"
        except Exception as e:
            last_err = e
    return f"获取天气失败:{last_err}"


# ============================================================
# 网页访问工具(供查养老政策/医保新闻等外部信息)
# ============================================================
class _TextExtractor(HTMLParser):
    """HTML → 纯文本:丢弃 script/style/noscript 内容,其余标签只留文本与换行。"""

    _SKIP = {"script", "style", "noscript", "template", "head"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        out = "".join(self.parts)
        # 压缩连续空白,保留段落换行
        lines = [ln.strip() for ln in out.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def _assert_public_http_url(url: str) -> None:
    """SSRF 防护:仅允许 http/https 且目标主机解析到公网地址。

    私网/环回/链路本地地址一律拒绝 —— 防止模型被诱导访问内网服务
    (本机 8001 MCP elder 服务、MySQL、Redis 都在内网段,这是硬红线)。
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"仅支持 http/https 链接,收到:{parsed.scheme or '(空)'}")
    host = parsed.hostname
    if not host:
        raise ValueError("链接缺少主机名")
    try:
        infos = socket.getaddrinfo(host, parsed.port or 80, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"主机名解析失败:{host}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ValueError(f"禁止访问内网地址:{host}({ip})")


@mcp.tool()
def fetch_webpage(url: str, max_chars: int = 3000) -> str:
    """访问指定网页并提取正文文本(用于查询养老政策、医保新闻、保健资讯等公开信息)。
    返回纯文本(按 max_chars 截断);仅支持公网 http/https 链接,内网地址会被拒绝。"""
    try:
        _assert_public_http_url(url)
    except ValueError as e:
        return f"链接被拒绝:{e}"

    headers = {"User-Agent": "Mozilla/5.0 (compatible; ElderCareAgent/1.0)",
               "Accept-Language": "zh-CN,zh;q=0.9"}
    try:
        # 【安全】禁自动重定向:重定向目标可能绕过上面的公网校验
        resp = requests.get(url, headers=headers, timeout=_TIMEOUT,
                            allow_redirects=False, stream=True)
        if resp.status_code in (301, 302, 303, 307, 308):
            return f"网页要求重定向到 {resp.headers.get('Location')},请改用重定向后的链接"
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "html" not in ctype and "text" not in ctype and "json" not in ctype:
            return f"不支持的内容类型:{ctype or '(未知)'},仅支持网页/文本"
        # 只读有限字节,防止超大文件拖垮服务
        raw = b""
        for chunk in resp.iter_content(chunk_size=65536):
            raw += chunk
            if len(raw) > 2 * 1024 * 1024:
                break
        resp.close()

        encoding = resp.encoding or "utf-8"
        text = raw.decode(encoding, errors="replace")
        if "html" in ctype:
            extractor = _TextExtractor()
            extractor.feed(text)
            text = extractor.text()
        text = text.strip()
        if not text:
            return "网页内容为空或全部为脚本/样式"
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n...(已截断,原文共{len(text)}字符)"
        return text
    except requests.Timeout:
        return "网页访问超时,请稍后重试"
    except Exception as e:
        return f"网页访问失败:{e}"


# ============================================================
# 健康趋势分析(纯计算,不带数据访问 —— 数值由 LLM 从 query_health_records 拿来传入)
# ============================================================
# 各指标的老人参考区间(仅提示,不构成诊断;与 prompts 的"不诊断不开药"红线一致)
_METRIC_REF = {
    "血压": {"field": "收缩压", "unit": "mmHg", "high": 140, "low": 90},
    "收缩压": {"field": "收缩压", "unit": "mmHg", "high": 140, "low": 90},
    "舒张压": {"field": "舒张压", "unit": "mmHg", "high": 90, "low": 60},
    "血糖": {"field": "空腹血糖", "unit": "mmol/L", "high": 7.0, "low": 3.9},
    "心率": {"field": "心率", "unit": "次/分", "high": 100, "low": 60},
}


def _parse_values(values: str) -> list[float]:
    out = []
    for tok in values.replace("，", ",").replace("、", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


@mcp.tool()
def analyze_health_trend(metric_type: str, values: str) -> str:
    """分析一组健康监测数值的趋势(适合血压/血糖/心率等按时间顺序的多次测量值)。
    参数 values 为逗号分隔的数值串(按时间从早到晚,如 "135,138,142,145");
    返回均值/最大/最小/最新值/升降趋势,以及超出老人参考范围的次数提示(仅供参考,不构成诊断)。"""
    try:
        nums = _parse_values(values)
    except ValueError:
        return "数值解析失败:请传入逗号分隔的纯数字(如 135,138,142)"
    if len(nums) < 2:
        return "至少需要 2 个数值才能分析趋势"

    ref = _METRIC_REF.get(metric_type.strip())
    n = len(nums)
    avg = sum(nums) / n
    first_half = sum(nums[: n // 2]) / max(n // 2, 1)
    second_half = sum(nums[n - n // 2:]) / max(n // 2, 1)
    drift = second_half - first_half
    # 平稳判定阈值:参考区间宽度的 5%,无参考区间时取均值的 2%
    threshold = (ref["high"] - ref["low"]) * 0.05 if ref else abs(avg) * 0.02
    if abs(drift) < threshold:
        trend = f"基本平稳(后半段均值较前半段{drift:+.1f})"
    else:
        trend = f"{'上升' if drift > 0 else '下降'}(后半段均值较前半段{drift:+.1f},超过阈值{threshold:.1f})"

    lines = [
        f"指标:{metric_type} | 样本数:{n}",
        f"均值:{avg:.1f} 最大:{max(nums):.1f} 最小:{min(nums):.1f} 最新:{nums[-1]:.1f}",
        f"趋势:{trend}",
    ]
    if ref:
        abnormal = [v for v in nums if v > ref["high"] or v < ref["low"]]
        lines.append(
            f"参考范围(老人,仅供参考):{ref['low']}~{ref['high']} {ref['unit']};"
            f"超范围 {len(abnormal)}/{n} 次"
            + (f"(值:{abnormal})" if abnormal else "")
        )
        if len(abnormal) >= max(2, n // 2):
            lines.append("提示:超范围次数较多,建议就医复查并由医生判断,切勿自行调药。")
    else:
        lines.append("(该指标无内置参考范围,仅给出统计值)")
    return "\n".join(lines)


@mcp.tool()
def compute_bmi(height_cm: float, weight_kg: float) -> str:
    """根据身高(cm)与体重(kg)计算 BMI,并给出老年人适宜体重范围建议(仅供参考)。"""
    if not (50 <= height_cm <= 250):
        return "身高数值异常,请确认单位是厘米(如 165)"
    if not (20 <= weight_kg <= 300):
        return "体重数值异常,请确认单位是千克(如 60)"
    h = height_cm / 100
    bmi = weight_kg / (h * h)
    # 老年人适宜 BMI 略高于普通成人(肌少症风险),参考中国老年医学共识:20~26.9
    if bmi < 18.5:
        level = "偏瘦(老人偏瘦与肌少症、骨折风险相关,建议营养科评估)"
    elif bmi < 20:
        level = "偏低(接近老人适宜下限)"
    elif bmi <= 26.9:
        level = "适宜(老年人适宜区间约 20~26.9)"
    else:
        level = "超重(建议咨询医生制定饮食运动方案)"
    lo = 20 * h * h
    hi = 26.9 * h * h
    return (f"BMI = {bmi:.1f},属于:{level}。\n"
            f"按身高 {height_cm}cm,老人适宜体重范围约 {lo:.1f}~{hi:.1f} kg(仅供参考)。")


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
