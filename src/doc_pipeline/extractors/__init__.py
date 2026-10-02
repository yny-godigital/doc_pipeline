"""
Per-route extractors. Each module exposes `extract(...)` and is named after
the ProcessingRoute it serves.

There is no `e5` module: E5 (low quality / handwritten) is a reclassification
of a page that already came through E3, so it keeps e3's OCR output rather
than being extracted a second time.
"""