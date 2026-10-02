"""
document_type classifiers.

The document_type axis asks "what kind of document is this whole file?",
as opposed to ProcessingRoute, which asks per page. Several methodologies can
answer it; DocumentTypeGuesser is the seam, so process_pdf stays unchanged
when one is swapped in.
"""

from __future__ import annotations
import abc

from .schema import DocumentType


class DocumentTypeGuesser(abc.ABC):
    """
    Strategy interface for the document_type axis.

    Extends this to swap in a different methodology (e.g. a Tier-2 VLM/LLM
    call) without touching process_pdf. Implementations must be
    deterministic given the same inputs and must not raise -- a guesser
    that cannot decide should return DocumentType.UNCLASSIFIED.
    """

    @abc.abstractmethod
    def guess(self, pdf_path: str, first_page_text: str) -> DocumentType:
        """Classify the whole document from its path and first-page text."""


class KeywordGuesser(DocumentTypeGuesser):
    """
    Cheap keyword fallback so every record still carries a type.

    TODO (Chew): this should be more robust -- in production this axis wants Tier-2
    classifier work (small VLM/LLM call per the design doc). Swap in a
    different DocumentTypeGuesser rather than extending this one.
    """

    def guess(self, pdf_path: str, first_page_text: str) -> DocumentType:
        lowered = (pdf_path + " " + first_page_text[:500]).lower()
        if "sensor layout" in lowered or "floor" in lowered or "layout" in lowered:
            return DocumentType.FLOOR_PLAN_OR_LAYOUT_DRAWING
        if "schedule" in lowered and "maintenance" in lowered:
            return DocumentType.MAINTENANCE_SCHEDULE
        if "procedure" in lowered or "risk assessment" in lowered:
            return DocumentType.PROCEDURE
        return DocumentType.UNCLASSIFIED


#: Used when process_pdf is called without an explicit guesser.
DEFAULT_GUESSER = KeywordGuesser()

