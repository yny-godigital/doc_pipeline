"""
Per-route extractors. Each module exposes `extract(...)` and is named after
the ProcessingRoute it serves.

There is no `dc5` module: DC5 (low quality / handwritten) is a
reclassification of a page that already came through DC3, so it keeps dc3's
OCR output rather than being extracted a second time.
"""