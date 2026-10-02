"""
E2: tabular data.

Deterministic extraction via pdfplumber's ruling-line table detector, then a
light normalization pass: merged/None cells from rowspans are forward-filled
within a row, and fully-empty rows are dropped. This does NOT attempt schema
mapping (e.g. "which column is the interval code") -- per the design, that
messy-header-to-schema step is where an LLM earns its keep, not raw
extraction. Keep the two steps separate so the deterministic part stays
auditable.
"""


def _forward_fill_row(row: list) -> list:
    filled = []
    last = ""
    for cell in row:
        val = (cell or "").strip()
        if val:
            last = val
            filled.append(val)
        else:
            filled.append(last if filled and filled[-1] == last else "")
    return filled


def extract(pdfplumber_page) -> dict:
    tables = pdfplumber_page.extract_tables()
    normalized_tables = []

    for t_idx, table in enumerate(tables):
        rows = []
        for raw_row in table:
            # Merged cells arrive as None; keep None as-is here (caller decides
            # fill strategy) but drop rows that are entirely empty.
            if all((c is None or str(c).strip() == "") for c in raw_row):
                continue
            rows.append([("" if c is None else str(c).strip()) for c in raw_row])

        normalized_tables.append({
            "table_index": t_idx,
            "row_count": len(rows),
            "col_count": max((len(r) for r in rows), default=0),
            "rows": rows,
        })

    return {
        "table_count": len(normalized_tables),
        "tables": normalized_tables,
    }

