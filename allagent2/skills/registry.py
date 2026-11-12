"""技能注册表：扫描 builtin 目录、解析 SKILL.md、按 id 取用 + 规则匹配。

================================================================
技能是什么
----------------------------------------------------------------
一个技能 = 一段有边界的指令 + 一个触发条件 + 可选的工具声明。
它不是代码插件，注入的是**文本**，进 system prompt。这样：
  - 安全：技能只能引导模型，不能执行任意代码；
  - 可控：技能声明的工具与实际权限取交集，越权在 act 节点仍会被拦；
  - 便宜：一次注入约 200-400 token，比多调一次模型便宜一个数量级。

================================================================
文件格式（SKILL.md）
----------------------------------------------------------------
    ---
    id: schedule
    name: 日程规划
    description: 帮老人安排一天的作息、用药、活动时间
    when: 安排|计划|日程|作息|时间表|几点|提醒我
    category: 生活
    tools: query_elder_profile,query_health_records
    budget: 300
    ---
    （正文 = 注入到 prompt 的指令，支持 Markdown）

- `when` 是正则，用来做零成本意图匹配（不调 embedding）；
- `tools` 留空表示不需要工具；
- `budget` 是该技能的字符预算（1 token ≈ 2 汉字，取 2 倍容错）。

================================================================
新增技能
----------------------------------------------------------------
在 builtin/ 下建一个目录放 SKILL.md 即可，服务重启后自动扫描到。
不需要改代码、不需要建表（用户启用状态是另一张表，按需生成行）。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("agent")

BUILTIN_DIR = Path(__file__).resolve().parent / "builtin"

# 单轮注入的技能数上限（再多会稀释注意力，也会挤占记忆块预算）
MAX_ACTIVE = 3


@dataclass
class Skill:
    """一个技能的只读视图。"""

    id: str
    name: str
    description: str
    when: str = ""
    category: str = "通用"
    tools: List[str] = field(default_factory=list)
    budget: int = 300
    instructions: str = ""
    version: str = "1.0.0"

    def matches(self, text: str) -> bool:
        """规则匹配：命中 `when` 正则即认为相关。正则写错时按不匹配处理。"""
        if not self.when or not text:
            return False
        try:
            return re.search(self.when, text) is not None
        except re.error as exc:
            logger.warning("[skill] %s 的 when 正则非法（%s），按不匹配处理", self.id, exc)
            return False

    def render(self) -> str:
        """渲染成注入 prompt 的一段。超预算直接截断，不切碎语义。"""
        body = (self.instructions or "").strip()
        limit = max(120, int(self.budget) * 2)
        if len(body) > limit:
            body = body[:limit].rstrip() + "…"
        return f"## {self.name}\n{body}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tools": list(self.tools),
            "budget": self.budget,
            "version": self.version,
        }


# ============================================================
# 解析
# ============================================================
def _split_front_matter(text: str) -> tuple[Dict[str, str], str]:
    """拆 `---` 包裹的头部与正文。不引 yaml 依赖：技能头都是简单 key: value。"""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    head, body = text[3:end], text[end + 4 :]
    meta: Dict[str, str] = {}
    for line in head.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip().lower().replace(" ", "_")] = value.strip()
    return meta, body


def _parse_file(path: Path) -> Optional[Skill]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("[skill] 读取失败 %s：%s", path, exc)
        return None

    meta, body = _split_front_matter(text)
    sid = (meta.get("id") or path.parent.name).strip()
    if not sid:
        logger.warning("[skill] 缺少 id，跳过 %s", path)
        return None
    name = (meta.get("name") or sid).strip()
    raw_tools = meta.get("tools") or ""
    tools = [t.strip() for t in re.split(r"[,，\s]+", raw_tools) if t.strip()]
    try:
        budget = int(meta.get("budget") or 300)
    except ValueError:
        budget = 300

    return Skill(
        id=sid,
        name=name,
        description=(meta.get("description") or "").strip(),
        when=(meta.get("when") or "").strip(),
        category=(meta.get("category") or "通用").strip(),
        tools=tools,
        budget=budget,
        instructions=body.strip(),
        version=(meta.get("version") or "1.0.0").strip(),
    )


# ============================================================
# 加载与缓存
# ============================================================
_CACHE: Dict[str, Skill] = {}
_LOADED = False


def load(force: bool = False) -> Dict[str, Skill]:
    """扫描 builtin/ 下所有 SKILL.md。首次调用后走缓存（技能是静态文件）。"""
    global _LOADED
    if _LOADED and not force:
        return _CACHE

    found: Dict[str, Skill] = {}
    if BUILTIN_DIR.is_dir():
        for path in sorted(BUILTIN_DIR.glob("*/SKILL.md")):
            skill = _parse_file(path)
            if skill is None:
                continue
            if skill.id in found:
                logger.warning("[skill] id 重复，后者覆盖前者：%s", skill.id)
            found[skill.id] = skill
    _CACHE.clear()
    _CACHE.update(found)
    _LOADED = True
    logger.info("[skill] 技能库加载完成：%d 个（%s）", len(_CACHE), ", ".join(_CACHE) or "空")
    return _CACHE


def all_skills() -> List[Skill]:
    return list(load().values())


def get_skill(skill_id: str) -> Optional[Skill]:
    if not skill_id:
        return None
    return load().get(skill_id.strip())


def match_skills(text: str, limit: int = 2) -> List[Skill]:
    """按 `when` 正则匹配（零 token、零向量）。命中多个时按技能顺序取前 N 个。"""
    if not text:
        return []
    hits = [s for s in load().values() if s.matches(text)]
    return hits[: max(1, limit)]


def render_block(skills: List[Skill]) -> str:
    """把选中的技能渲染成一段 prompt。

    措辞上强调"用户已选择"，让模型把它当约束而不是建议；
    同时保留"与安全红线冲突时以红线为准"的兜底，避免技能被写坏时越过医疗边界。
    """
    if not skills:
        return ""
    head = (
        "# 当前启用技能（用户已选择，按以下流程执行）\n"
        "注意：技能只规定流程与输出结构，不覆盖安全红线；"
        "与【安全边界】冲突时以安全边界为准。"
    )
    return head + "\n\n" + "\n\n".join(s.render() for s in skills)
