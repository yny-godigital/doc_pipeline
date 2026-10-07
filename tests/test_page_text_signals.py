"""Page text is available to classifiers and kept out of output records.

A classifier deciding whether a page is a drawing or a table needs the page's
words, not only its measurements. Those words are client document content, so
they must reach classifiers without ever being copied into a record.
"""

import pymupdf

from doc_pipeline.classifier import PageSignals
from doc_pipeline.threshold_classifier import ThresholdClassifier, collect_signals


def test_page_signals_carry_page_text():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 144), "CONTROL POINT SCHEDULE")

    signals = collect_signals(page)

    assert "CONTROL POINT SCHEDULE" in signals.text_sample


def test_page_signals_cap_page_text_at_2000_characters():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 144), "X" * 9000)

    signals = collect_signals(page)

    assert len(signals.text_sample) <= 2000


def test_threshold_signals_exclude_page_text():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 144), "CONTROL POINT SCHEDULE")

    signals = collect_signals(page)
    result = ThresholdClassifier().classify(signals)

    # Records are audit data. Two thousand characters of client document text
    # on every record would put confidential document content into output files
    # and into the committed routing snapshot.
    assert "text_sample" not in result.signals
    assert "text_sample" in signals.__dict__


def test_blank_page_yields_empty_page_text():
    document = pymupdf.open()
    page = document.new_page()

    assert collect_signals(page).text_sample == ""


def test_page_signals_constructor_defaults_text_sample_to_empty():
    signals = PageSignals(
        page_number=0,
        width_pt=595.0,
        height_pt=842.0,
        area_pt2=500990.0,
        text_chars=0,
        image_count=0,
        vector_path_count=0,
        has_table_via_pdfplumber=False,
        embedded_font_count=0,
    )

    assert signals.text_sample == ""