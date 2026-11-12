"""摄取层配置(独立于 RAGConfig,避免检索层配置牵连摄取层)。"""
import os

DOCS_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs_index")

# 可插拔解析增强(零配置=全原生,不下载任何模型)
MINERU_API_URL = os.getenv("MINERU_API_URL", "")     # 配置后 needs_review 文档可走 sidecar

# needs_review 阈值(启发式置信度低于它 → 用户确认后才进检索库)
REVIEW_CONF_THRESHOLD = 0.55

# OCR 引擎探测顺序(都未装 → 图片/扫描件标 needs_review,管线不崩)
OCR_ENGINES = ("rapidocr", "winrt")
