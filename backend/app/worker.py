"""Durable Vercel Queue consumer for satellite analysis jobs."""

from vercel.queue import subscribe

from app.services.analysis_service import run_analysis_task


@subscribe(topic="naturevision-analysis", max_concurrency=1)
async def process_analysis(message: dict[str, str]) -> None:
    await run_analysis_task(message["analysis_id"])
