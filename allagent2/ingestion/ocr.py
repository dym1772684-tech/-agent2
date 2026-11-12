"""OCR 可插拔插槽:图片/扫描件的文字识别,引擎按可用性探测。

零下载原则:本模块定义接口 + 探测式实现,**不主动安装/下载任何引擎**。
  - RapidOCR(onnxruntime,模型仅数 MB):装了才用,识别质量/速度均衡
  - WinRT OCR:Windows 系统自带(需中文语言包),零额外下载
  - 都不可用 → ocr_available()=False,管线把图片/扫描页标 needs_review,不崩

将来要更强的识别(手写/印章遮挡):MinerU vlm 后端走同一接口接入。
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("agent")

_engine = None          # 已探测到的引擎名
_engine_checked = False


def ocr_available() -> bool:
    return detect_engine() is not None


def detect_engine() -> Optional[str]:
    """探测可用 OCR 引擎(进程内只探测一次)。返回 'rapidocr' | 'winrt' | None。"""
    global _engine, _engine_checked
    if _engine_checked:
        return _engine
    _engine_checked = True
    try:                                   # ① RapidOCR
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401

        _engine = "rapidocr"
        return _engine
    except Exception:
        pass
    try:                                   # ② WinRT(Windows 自带)
        import winrt.windows.media.ocr as _wocr  # noqa: F401

        _engine = "winrt"
        return _engine
    except Exception:
        pass
    logger.info("[ingest] 未检测到可用 OCR 引擎,图片/扫描页将标记 needs_review")
    return None


def ocr_image(image_bytes: bytes) -> Tuple[str, float]:
    """识别一张图片。返回 (文本, 平均置信度 0~1);无引擎/识别失败返回 ("", 0)。

    RapidOCR 路线:按 y 坐标排序文本框,拼接自然阅读顺序。
    WinRT 路线:需异步 API,这里用同步包装(asyncio.run 限定短任务)。
    """
    eng = detect_engine()
    if eng is None:
        return "", 0.0
    try:
        if eng == "rapidocr":
            from rapidocr_onnxruntime import RapidOCR

            import io

            import numpy as np
            from PIL import Image

            img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
            result, _ = RapidOCR()(img)
            if not result:
                return "", 0.0
            # result: [[box, text, score], ...] → 按 y 排序拼接
            items = sorted(result, key=lambda r: (r[0][0][1], r[0][0][0]))
            text = "\n".join(str(r[1]) for r in items)
            conf = sum(float(r[2]) for r in items) / len(items)
            return text, conf
        if eng == "winrt":
            import asyncio

            return asyncio.run(_winrt_ocr(image_bytes))
    except Exception as exc:
        logger.warning("[ingest] OCR 识别失败(按空文本处理):%s", exc)
    return "", 0.0


async def _winrt_ocr(image_bytes: bytes) -> Tuple[str, float]:
    """WinRT OCR 异步包装(Windows 10+ 自带,中文需安装中文语言包)。"""
    import io

    from winrt.windows.graphics.imaging import BitmapDecoder
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import InMemoryRandomAccessStream

    stream = InMemoryRandomAccessStream()
    await stream.write_async(image_bytes) if hasattr(stream, "write_async") else None
    # 完整 WinRT 数据加载较繁琐:走 BitmapDecoder 标准路径
    decoder = await BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()
    engine = OcrEngine.try_create_from_user_profile_languages()
    if engine is None:
        return "", 0.0
    result = await engine.recognize_async(bitmap)
    lines = [l.text for l in result.lines]
    conf = 0.85 if lines else 0.0   # WinRT 不逐行给置信度,固定中高值
    return "\n".join(lines), conf


def ocr_scanned_pages(ir, get_page_bytes) -> int:
    """扫描件补救:对 IR 里 needs_ocr_pages 逐页 OCR,结果回填 IR。

    get_page_bytes(page_no) -> bytes 由调用方提供(pymupdf 渲染页图)。
    返回成功识别的页数;无引擎时返回 0(调用方标 needs_review)。
    """
    if not ir.needs_ocr_pages or not ocr_available():
        return 0
    done = 0
    for pno in list(ir.needs_ocr_pages):
        try:
            raw = get_page_bytes(pno)
        except Exception as exc:
            logger.warning("[ingest] 扫描页渲染失败 page=%s:%s", pno, exc)
            continue
        text, conf = ocr_image(raw)
        if text.strip():
            from .ir import Block

            ir.blocks.append(Block(type="text", text=text, page=pno,
                                   conf=conf, source="ocr"))
            ir.needs_ocr_pages.remove(pno)
            done += 1
    return done
