"""
E6: Engineering diagram route (P&ID, floor plan, sensor layout).

A CAD-exported PDF already contains real geometry and real text.
This extractor pulls that geometry and text out directly:

  - `labels`: every text span on the page with its position, which is enough
    to recover room/equipment/tag labels (e.g. "DM003 DATA MODULE 03") and
    the title block metadata (drawing number, revision, scale).
  - `symbol_candidates`: a very rough proxy for repeated symbol instances,
    grouped by (width, height) of their bounding path, since the same
    symbol (e.g. a sensor icon) is drawn at the same size repeatedly across
    a layout. This is NOT true symbol recognition -- it's a cheap signal to
    show where a VLM-based symbol classifier should focus, so you don't run
    it over 126,000 raw vector paths.

Full P&ID -> graph conversion (nodes = equipment/instruments, edges = process
lines) needs a proper vectorization + symbol-library matching step (e.g.
a DEXPI-aware parser, or a fine-tuned symbol detector); that is out of scope
for this pass and is flagged in the output as `graph_extraction: "not_run"`
so it's visible in the record rather than silently missing.
"""
from collections import defaultdict


def extract_labels(fitz_page) -> list[dict]:
    labels = []
    raw = fitz_page.get_text("dict")
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                labels.append({
                    "text": text,
                    "bbox": [round(v, 1) for v in span["bbox"]],
                    "font_size": round(span.get("size", 0), 1),
                })
    return labels


def extract_symbol_candidates(fitz_page, min_repeat: int = 5) -> list[dict]:
    drawings = fitz_page.get_drawings()
    buckets = defaultdict(list)
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        w, h = round(rect.width, 1), round(rect.height, 1)
        if w <= 0 or h <= 0:
            continue
        buckets[(w, h)].append([round(rect.x0, 1), round(rect.y0, 1)])

    candidates = [
        {"shape_wh": list(dims), "instance_count": len(positions), "positions_sample": positions[:10]}
        for dims, positions in buckets.items()
        if len(positions) >= min_repeat
    ]
    candidates.sort(key=lambda c: -c["instance_count"])
    return candidates[:50]   # cap so the record doesn't explode on dense sheets


def extract(fitz_page) -> dict:
    labels = extract_labels(fitz_page)
    symbol_candidates = extract_symbol_candidates(fitz_page)
    return {
        "label_count": len(labels),
        "labels": labels,
        "symbol_candidates": symbol_candidates,
        "graph_extraction": "not_run",   # see module docstring
    }

