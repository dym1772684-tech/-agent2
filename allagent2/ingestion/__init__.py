"""ingestion:文档摄取层(多格式解析 → 统一中间表示 IR → 清洗切分入库)。

零模型原则:本包原生实现不依赖任何视觉/深度学习模型——
  - 电子版文档(docx/xlsx/pptx/txt/csv/md/pdf文本层):纯代码启发式解析
  - PDF 版面重建:字体聚类(标题)/坐标双栏检测/跨页重复模式(页眉页脚)/
    pdfplumber 几何表格(矢量线条聚类,非 ML)
  - 扫描件/图片:OcrEngine 可插拔插槽(未装引擎 → needs_review,管线不崩)
  - MinerU sidecar:可选增强(配置 MINERU_API_URL 才启用),仅多一个 IR adapter

模块导航:
    ir.py          统一中间表示(Block/DocumentIR/置信度)
    router.py      魔数识别真实类型 + 解析器级联路由
    parsers/       各格式原生解析器(产出 IR)
    clean.py       清洗(PII脱敏/注入清洗/乱码处理)
    chunker.py     切分(父子token + 符号递归 + overlap + 表格原子块)
    index_store.py Chroma 入库(acl 元数据/文档指纹/增量删改)
    pipeline.py    编排:parse → clean → chunk → embed → 入库(异步)
    ocr.py         OCR 可插拔接口(RapidOCR/WinRT 探测式实现)

【铁律】重依赖(fitz/pdfplumber/chromadb)一律函数内延迟导入,
与 rag.py 的 torch 段错误教训同源。
"""
