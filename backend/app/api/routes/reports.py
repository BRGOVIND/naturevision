"""Nature Intelligence Report generation and export."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Query, Response, status
from sqlalchemy import select

from app.api.deps import AnalysisServiceDep, LanguageDep
from app.core.errors import AnalysisStateError, ResourceNotFoundError
from app.core.logging import get_logger
from app.geospatial.render import RenderedLayer
from app.interpretation.deterministic import build_deterministic_summary
from app.interpretation.evidence import EvidencePackage
from app.interpretation.schemas import InterpretationEnvelope
from app.models import AnalysisStatus, Report
from app.reports.builder import build_report
from app.reports.export import pdf_available, render_html, render_pdf, write_html
from app.schemas.analysis import ReportRequest, ReportResponse
from app.services.analysis_service import analysis_dir

logger = get_logger(__name__)
router = APIRouter(tags=["reports"])


@router.post(
    "/ai/report",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate an evidence-grounded Nature Intelligence Report",
)
async def generate_report(
    payload: ReportRequest, service: AnalysisServiceDep, language: LanguageDep
) -> ReportResponse:
    """Build the report from stored evidence and a validated interpretation.

    The interpretation is generated from the persisted evidence package only.
    If the language provider is unconfigured or its output fails grounding
    validation, the report is still produced — with the interpretation section
    explicitly marked unavailable rather than filled with plausible text.
    """
    analysis = await service.get(payload.analysis_id)

    if analysis.status == AnalysisStatus.FAILED:
        raise AnalysisStateError(
            "This analysis failed, so there are no results to report on.",
            details={"error_code": analysis.error_code},
        )
    if not analysis.evidence:
        raise AnalysisStateError(
            "This analysis has not finished processing yet. Wait for it to reach "
            "the report-ready state.",
            details={"status": analysis.status, "progress": analysis.progress},
        )

    existing = next(iter(analysis.reports), None)
    if existing is not None and not payload.regenerate:
        return _report_response(existing)

    evidence_dict = analysis.evidence
    package = EvidencePackage(
        region=evidence_dict.get("region", {}),
        periods=evidence_dict.get("periods", {}),
        data_sources=evidence_dict.get("data_sources", []),
        observed=evidence_dict.get("observed", {}),
        model_predictions=evidence_dict.get("model_predictions", {}),
        methodology=evidence_dict.get("methodology", {}),
        data_quality=evidence_dict.get("data_quality", {}),
        limitations=evidence_dict.get("limitations", []),
    )

    if analysis.include_interpretation:
        try:
            envelope = await language.interpret(package)
            if envelope.available:
                envelope.source = "language_model"
        except Exception as exc:
            # A provider failure must not destroy a completed analysis.
            logger.warning("interpretation_unavailable", error=str(exc)[:300])
            envelope = InterpretationEnvelope(available=False)

        if not envelope.available:
            # No language-model response passed grounding, or none was
            # requested — a completed analysis still has measured evidence,
            # so a deterministic restatement of it stands in. It goes through
            # the same validator; nothing here is exempt from grounding.
            fallback = build_deterministic_summary(package)
            if fallback is not None:
                summary, grounding = fallback
                envelope = InterpretationEnvelope(
                    interpretation=summary,
                    provider=None,
                    model=None,
                    generated_at=dt.datetime.now(dt.UTC).isoformat(),
                    grounding=grounding.to_dict(),
                    available=True,
                    source="measured",
                )
            else:
                envelope.unavailable_reason = (
                    envelope.unavailable_reason
                    or "There is not enough measured evidence in this analysis to "
                    "summarise. All measured results are unaffected."
                )
    else:
        envelope = InterpretationEnvelope(
            available=False,
            unavailable_reason="Interpretation was disabled for this analysis.",
        )

    visual = None
    if payload.include_visual_interpretation and envelope.available:
        visual = await _describe_ndvi_layer(analysis.id, language, package)
        envelope.visual = visual

    region_name = analysis.region.name or package.region.get("name") or "selected region"
    structure = build_report(
        analysis_id=analysis.id,
        evidence=evidence_dict,
        interpretation=envelope,
        region_name=region_name,
    )

    html_path = write_html(structure, analysis.id)

    if existing is not None:
        await service.session.delete(existing)
        await service.session.flush()

    report = Report(
        analysis_id=analysis.id,
        title=structure["title"],
        sections=structure,
        interpretation=(envelope.interpretation.model_dump() if envelope.interpretation else None),
        visual_interpretation=visual.model_dump() if visual else None,
        interpretation_provider=envelope.provider,
        interpretation_model=envelope.model,
        html_path=str(html_path),
        generated_at=dt.datetime.now(dt.UTC),
    )
    service.session.add(report)
    await service.session.commit()
    logger.info(
        "report_generated",
        analysis_id=analysis.id,
        interpretation_available=envelope.available,
    )
    return _report_response(report)


@router.get(
    "/analysis/{analysis_id}/report",
    response_model=ReportResponse,
    summary="Fetch the report for an analysis",
)
async def get_analysis_report(analysis_id: str, service: AnalysisServiceDep) -> ReportResponse:
    """Return the report already generated for an analysis.

    Reopening an analysis should show the interpretation it already has,
    without regenerating it and without a second provider call.
    """
    analysis = await service.get(analysis_id)
    report = next(iter(analysis.reports), None)
    if report is None:
        raise ResourceNotFoundError(
            "No report has been generated for this analysis yet.",
            details={"analysis_id": analysis_id},
        )
    return _report_response(report)


@router.get("/reports/{report_id}", response_model=ReportResponse, summary="Fetch a report")
async def get_report(report_id: str, service: AnalysisServiceDep) -> ReportResponse:
    report = await _load_report(service, report_id)
    return _report_response(report)


@router.get(
    "/reports/{report_id}/export",
    summary="Export a report as HTML or PDF",
    response_class=Response,
)
async def export_report(
    report_id: str,
    service: AnalysisServiceDep,
    format: str = Query(default="html", pattern="^(html|pdf)$"),
) -> Response:
    """Download the rendered report.

    HTML is always available and prints to PDF from any browser. PDF is served
    natively only when the optional rendering toolchain is installed.
    """
    report = await _load_report(service, report_id)
    filename = f"nature-intelligence-report-{report.analysis_id[:8]}"

    if format == "pdf":
        return Response(
            content=render_pdf(report.sections),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}.pdf"'},
        )
    return Response(
        content=render_html(report.sections),
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f'inline; filename="{filename}.html"'},
    )


async def _load_report(service: AnalysisServiceDep, report_id: str) -> Report:
    result = await service.session.execute(select(Report).where(Report.id == report_id))
    report = result.scalar_one_or_none()
    if report is None:
        raise ResourceNotFoundError(
            "No report exists with that identifier.", details={"report_id": report_id}
        )
    return report


async def _describe_ndvi_layer(analysis_id: str, language: LanguageDep, package: EvidencePackage):
    """Run vision interpretation over the rendered NDVI overlay, if present."""
    path = analysis_dir(analysis_id) / "ndvi_a.png"
    if not path.is_file():
        return None
    layer = RenderedLayer(
        png=path.read_bytes(),
        bounds=[],
        legend=[],
        value_min=None,
        value_max=None,
        kind="continuous",
    )
    context = (
        "Greens indicate higher vegetation index values, browns and blues lower "
        "values. Transparent areas were removed by cloud masking or fall outside "
        "the selected region."
    )
    return await language.interpret_image(layer.to_data_url(), "NDVI map", context)


def _report_response(report: Report) -> ReportResponse:
    structure = report.sections or {}
    return ReportResponse(
        id=report.id,
        analysis_id=report.analysis_id,
        title=report.title,
        generated_at=report.generated_at,
        sections=structure.get("sections", []),
        provenance_legend=structure.get("provenance_legend", {}),
        interpretation=report.interpretation,
        visual_interpretation=report.visual_interpretation,
        interpretation_provider=report.interpretation_provider,
        interpretation_model=report.interpretation_model,
        export_urls={
            "html": f"/api/v1/reports/{report.id}/export?format=html",
            **(
                {"pdf": f"/api/v1/reports/{report.id}/export?format=pdf"} if pdf_available() else {}
            ),
        },
    )
