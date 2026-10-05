"""
Canonical record schema.

Every processing route (see ProcessingRoute) must emit records in this
shape. This is what makes cross-document reasoning possible later: the
consumer of these records never needs to know which route produced them.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional
import uuid
import datetime


class ProcessingRoute(str, Enum):
    """
    The processing routes a page can be assigned to.

    Declared as a str-mixin enum so records still serialize to plain
    strings in JSON ("DC1", not "ProcessingRoute.DC1") and so consumers
    reading previously-written records keep working.
    """
    DC1 = "DC1"              # text-heavy / semi-structured prose
    DC2 = "DC2"              # tabular
    DC3 = "DC3"              # scanned / flattened image, layout-aware OCR
    DC4 = "DC4"              # native CAD/DXF (not reachable from PDF pages)
    DC5 = "DC5"              # low quality / handwritten (DC3 OCR pass came back weak)
    DC6 = "DC6"              # engineering diagram (P&ID, floor plan, sensor layout)
    UNKNOWN = "unknown"      # page carries no extractable content of its own


class DocumentType(str, Enum):
    """
    The document_type axis: what kind of document the whole file is, as
    opposed to ProcessingRoute, which is per-page.

    Str-mixin for the same JSON reason as ProcessingRoute.
    """
    # Emitted by the keyword heuristic in pipeline.py
    FLOOR_PLAN_OR_LAYOUT_DRAWING = "floor_plan_or_layout_drawing"
    MAINTENANCE_SCHEDULE = "maintenance_schedule"
    PROCEDURE = "procedure"
    UNCLASSIFIED = "unclassified"

    # Declared in Provenance but not yet produced by any guesser; add a
    # producer before relying on these.
    CONTRACT = "contract"
    SCHEDULE = "schedule"
    PID = "pid"
    FLOOR_PLAN = "floor_plan"


def new_id() -> str:
    return str(uuid.uuid4())


def now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


@dataclass
class Provenance:
    document_id: str
    document_type: DocumentType
    revision: Optional[str] = None
    revision_status: str = "unknown"   # controlled | draft | superseded | unknown
    source_system: str = "file_upload"
    page: Optional[int] = None
    bounding_box: Optional[list] = None
    ingested_at: str = field(default_factory=now)


@dataclass
class Classification:
    processing_route: ProcessingRoute
    quality_tier: str              # clean | degraded | handwritten
    sensitivity: str = "internal"  # public | internal | confidential | export_controlled
    classifier_confidence: float = 0.0
    signals: dict = field(default_factory=dict)   # raw signals that drove the decision (for audit)


@dataclass
class Content:
    raw_ref: str
    parsed_ref: Optional[str] = None
    extracted: dict = field(default_factory=dict)
    extraction_confidence: float = 0.0


@dataclass
class Review:
    status: str = "unreviewed"     # unreviewed | human_verified | flagged
    reviewer: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class CanonicalRecord:
    record_id: str
    provenance: Provenance
    classification: Classification
    content: Content
    graph_links: list = field(default_factory=list)
    review: Review = field(default_factory=Review)

    def to_dict(self) -> dict:
        return asdict(self)


def make_record(document_id, document_type, page, processing_route, quality_tier,
                 signals, confidence, raw_ref, extracted, extraction_confidence,
                 parsed_ref=None, sensitivity="internal") -> CanonicalRecord:
    return CanonicalRecord(
        record_id=new_id(),
        provenance=Provenance(
            document_id=document_id,
            document_type=document_type,
            page=page,
        ),
        classification=Classification(
            processing_route=processing_route,
            quality_tier=quality_tier,
            sensitivity=sensitivity,
            classifier_confidence=confidence,
            signals=signals,
        ),
        content=Content(
            raw_ref=raw_ref,
            parsed_ref=parsed_ref,
            extracted=extracted,
            extraction_confidence=extraction_confidence,
        ),
    )

