"""Report rendering and export.

HTML is the primary export format: it is self-contained, needs no native
dependencies, prints to PDF from any browser, and preserves tables exactly.
PDF export is offered when WeasyPrint and its system libraries are present, and
is reported as unavailable rather than silently degraded when they are not.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.core.config import settings
from app.core.errors import ReportGenerationError
from app.core.logging import get_logger

logger = get_logger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "report.html.j2"


@lru_cache
def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_html(report: dict[str, Any]) -> str:
    """Render the report structure to a standalone HTML document."""
    try:
        return _environment().get_template(TEMPLATE_NAME).render(report=report)
    except Exception as exc:
        logger.exception("report_render_failed")
        raise ReportGenerationError("The report template could not be rendered.") from exc


def pdf_available() -> bool:
    """Whether PDF export can actually run in this environment."""
    try:
        import weasyprint  # noqa: F401
    except Exception:
        return False
    return True


def render_pdf(report: dict[str, Any]) -> bytes:
    """Render the report to PDF, if the optional toolchain is installed."""
    if not pdf_available():
        raise ReportGenerationError(
            "PDF export is not enabled on this deployment. Export the report as "
            "HTML, which prints to PDF from any browser.",
            details={"install": "pip install 'naturevision-backend[pdf]'"},
        )
    import weasyprint

    try:
        return weasyprint.HTML(string=render_html(report)).write_pdf()
    except Exception as exc:
        logger.exception("pdf_render_failed")
        raise ReportGenerationError("The report could not be converted to PDF.") from exc


def write_html(report: dict[str, Any], analysis_id: str) -> Path:
    """Persist the rendered report and return its path."""
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    path = settings.artifact_dir / f"{analysis_id}.html"
    path.write_text(render_html(report), encoding="utf-8")
    return path
