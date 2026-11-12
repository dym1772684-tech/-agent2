"""上传文件的落盘与读取（按 user_id 隔离）。

设计取舍：
1. **只存解析后的文本**，不存原始二进制 —— 省空间，也少一份"用户原始文件"
   的合规负担（原件应该在对象存储/网盘里）。
2. **目录按 user_id 隔离** + file_id 白名单校验，别人拿到 file_id 也读不到。
3. 文件名一律用 uuid，**绝不用用户提供的名字拼路径**（防路径穿越），
   原始文件名只写进 meta.json 用于展示。
"""
from __future__ import annotations

import json
import os
import re
import uuid

BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads"
)

MAX_CHARS_PER_FILE = 6000      # 单个文件注入 prompt 的上限
MAX_CHARS_TOTAL = 12000        # 多文件合计上限
MAX_FILES = 5                  # 一次最多引用几个文件

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _user_dir(user_id: str) -> str:
    """用户目录（不存在就建）。user_id 做字符清洗，防目录穿越。"""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(user_id or "anon")) or "anon"
    path = os.path.join(BASE_DIR, safe)
    os.makedirs(path, exist_ok=True)
    return path


def uploads_dir(user_id: str) -> str:
    """该用户的上传目录（摄取管线落临时原文用；名字清洗与 save 一致）。"""
    return _user_dir(user_id)


def new_file_id() -> str:
    return f"f-{uuid.uuid4().hex[:12]}"


def save(user_id: str, name: str, text: str) -> dict:
    """保存解析后的文本，返回 {file_id, name, chars}。"""
    fid = new_file_id()
    folder = _user_dir(user_id)
    with open(os.path.join(folder, fid + ".txt"), "w", encoding="utf-8") as f:
        f.write(text)
    meta = {"file_id": fid, "name": name or "未命名", "chars": len(text)}
    with open(os.path.join(folder, fid + ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return meta


def load(user_id: str, file_id: str):
    """读回一个文件；file_id 非法 / 不存在 / 不属于该用户 → None。"""
    if not _SAFE_ID.match(str(file_id or "")):
        return None
    folder = _user_dir(user_id)
    path = os.path.join(folder, file_id + ".txt")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    name = file_id
    meta_path = os.path.join(folder, file_id + ".meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                name = json.load(f).get("name") or file_id
        except Exception:
            pass
    return {"file_id": file_id, "name": name, "text": text}


def read_files(user_id: str, file_ids, max_chars: int = MAX_CHARS_TOTAL) -> str:
    """把多个文件拼成一段带来源标注的文本（总长受限，防 prompt 爆炸）。"""
    blocks = []
    used = 0
    for fid in list(file_ids or [])[:MAX_FILES]:
        got = load(user_id, fid)
        if not got:
            continue
        remain = max_chars - used
        if remain <= 200:
            break
        body = got["text"][: min(MAX_CHARS_PER_FILE, remain)]
        blocks.append(f"【{got['name']}】\n{body}")
        used += len(body)
    return "\n\n".join(blocks)
