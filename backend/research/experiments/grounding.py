"""Evidence-grounding evaluation for the reporting layer.

Two generation modes are compared on the same evidence packages:

* **Mode A, baseline** — the evidence is supplied with a minimal instruction and
  no grounding constraints, and the output is *not* validated.
* **Mode B, grounded** — the production system prompt and the production
  validators, unchanged.

Both outputs are then scored by the same deterministic checker, so the
comparison measures the effect of grounding rather than the effect of scoring.
The production validator is never weakened to make Mode B look better.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.interpretation.evidence import EvidencePackage
from app.interpretation.language_service import LanguageInterpretationService, _extract_json
from app.interpretation.prompts import build_interpretation_prompt
from app.interpretation.schemas import Interpretation
from app.interpretation.validation import UNSUPPORTED_CLAIM_PATTERNS, validate_interpretation
from research import figures
from research.artifacts import ExperimentRecord, write_table
from research.config import MANIFEST_DIR

logger = get_logger(__name__)

CASES_FILE = MANIFEST_DIR / "grounding_cases.json"

#: Mode A. Deliberately minimal: it states the task without the grounding
#: constraints, which is the condition under test.
BASELINE_PROMPT = """You are an environmental analyst writing the interpretation \
section of a report.

You are given a JSON evidence package from a satellite analysis pipeline.

Respond with a single JSON object and no other text:

{
  "summary": "2-4 sentence overview.",
  "observations": [{"statement": "...", "evidence_key": null}],
  "interpretation": "What the observations indicate.",
  "uncertainty": "What is uncertain.",
  "limitations": ["..."],
  "confidence_qualifier": "low" | "moderate" | "high"
}"""

PROMPT_VERSION = "grounding-eval-1.0.0"


def load_cases() -> list[dict[str, Any]]:
    """Evidence packages captured from real completed analyses."""
    if not CASES_FILE.is_file():
        raise FileNotFoundError(
            f"No grounding cases at {CASES_FILE}. Build them with: "
            "python -m research.run --experiment grounding-cases"
        )
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))


def build_cases_from_database() -> list[dict[str, Any]]:
    """Extract evidence packages from analyses already stored by the product.

    Uses real completed runs only. If none exist, it says so rather than
    inventing environmental numbers.
    """
    import sqlite3
    from pathlib import Path

    db_path = Path(settings.sync_database_url.replace("sqlite:///", "").replace("sqlite://", ""))
    if not db_path.is_file():
        raise FileNotFoundError(
            f"No SQLite database at {db_path}. Run analyses in the product first, "
            "or point DATABASE_URL at the database that holds them."
        )

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT a.id, a.evidence, r.name AS region_name "
        "FROM analyses a JOIN regions r ON r.id = a.region_id "
        "WHERE a.evidence IS NOT NULL AND a.status = 'report_ready'"
    ).fetchall()
    connection.close()

    cases: list[dict[str, Any]] = []
    for row in rows:
        try:
            evidence = json.loads(row["evidence"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not evidence.get("observed"):
            continue
        cases.append(
            {
                "case_id": row["id"],
                "region_name": row["region_name"],
                "evidence": evidence,
                "source": "product analysis (real Sentinel-2 run)",
            }
        )

    if not cases:
        raise RuntimeError(
            "No completed analyses with evidence were found. Run at least one "
            "analysis in the product before building grounding cases; evidence "
            "will not be fabricated."
        )

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CASES_FILE.write_text(json.dumps(cases, indent=2, default=str), encoding="utf-8")
    logger.info("grounding_cases_built", cases=len(cases))
    return cases


def _package(evidence: dict[str, Any]) -> EvidencePackage:
    return EvidencePackage(
        region=evidence.get("region", {}),
        periods=evidence.get("periods", {}),
        data_sources=evidence.get("data_sources", []),
        observed=evidence.get("observed", {}),
        model_predictions=evidence.get("model_predictions", {}),
        methodology=evidence.get("methodology", {}),
        data_quality=evidence.get("data_quality", {}),
        limitations=evidence.get("limitations", []),
    )


def score_output(
    interpretation: Interpretation, package: EvidencePackage, evidence: dict[str, Any]
) -> dict[str, Any]:
    """Deterministic evaluation applied identically to both modes."""
    report = validate_interpretation(interpretation, package.numeric_claims())

    checked = report.checked_number_count
    matched = report.matched_number_count
    faithfulness = round(matched / checked, 4) if checked else None

    # Source attribution: does a named scene id actually appear in the evidence?
    known_scenes = {
        str(s.get("source_id")) for s in evidence.get("data_sources", []) if s.get("source_id")
    }
    text = " ".join(
        [
            interpretation.summary,
            interpretation.interpretation,
            interpretation.uncertainty,
            *(o.statement for o in interpretation.observations),
        ]
    )
    cited = {token for token in text.replace(",", " ").split() if token.startswith("S2")}
    invented_sources = sorted(cited - known_scenes)

    # Completeness: are the core measured quantities represented at all?
    observed = evidence.get("observed", {})
    required = [
        ("period_a_mean_ndvi", (observed.get("period_a") or {}).get("mean_ndvi")),
        ("change", (observed.get("change") or {}).get("absolute_ndvi_change")),
    ]
    represented = 0
    considered = 0
    for _key, value in required:
        if value is None:
            continue
        considered += 1
        if f"{abs(float(value)):.3f}"[:5] in text or f"{float(value):.4f}"[:6] in text:
            represented += 1
    completeness = round(represented / considered, 4) if considered else None

    return {
        "numerical_faithfulness": faithfulness,
        "numbers_checked": checked,
        "numbers_matched": matched,
        "unsupported_numerical_claims": len(report.unsupported_numbers),
        "unsupported_values": report.unsupported_numbers[:10],
        "causal_overreach_flags": len(report.flagged_claims),
        "causal_overreach_issues": sorted({c["issue"] for c in report.flagged_claims}),
        "invented_source_ids": invented_sources,
        "source_attribution_correct": not invented_sources,
        "completeness": completeness,
        "grounding_pass": report.passed,
    }


async def _generate(
    service: LanguageInterpretationService, package: EvidencePackage, mode: str
) -> Interpretation | None:
    """Produce one interpretation in the requested mode."""
    if mode == "grounded":
        envelope = await service.interpret(package)
        return envelope.interpretation

    # Baseline: minimal instruction, single attempt, no validation feedback.
    messages = [
        {"role": "system", "content": BASELINE_PROMPT},
        {"role": "user", "content": build_interpretation_prompt(package.to_dict())},
    ]
    raw = await service._complete(messages, settings.language_model)
    return Interpretation.model_validate(_extract_json(raw))


async def _run(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    service = LanguageInterpretationService()
    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            package = _package(case["evidence"])
            for mode in ("baseline", "grounded"):
                entry: dict[str, Any] = {
                    "case_id": case["case_id"],
                    "region": case.get("region_name"),
                    "mode": mode,
                    "provider": "Groq",
                    "model": settings.language_model,
                    "prompt_version": PROMPT_VERSION,
                    "temperature": settings.language_temperature,
                    "generated_at": dt.datetime.now(dt.UTC).isoformat(),
                }
                try:
                    interpretation = await _generate(service, package, mode)
                except Exception as exc:
                    # A rejected generation is a real outcome, recorded as such.
                    entry.update(
                        {
                            "status": "rejected",
                            "error": type(exc).__name__,
                            "detail": str(exc)[:200],
                        }
                    )
                    rows.append(entry)
                    logger.warning("generation_rejected", mode=mode, error=str(exc)[:160])
                    continue

                if interpretation is None:
                    entry.update({"status": "unavailable"})
                    rows.append(entry)
                    continue

                entry.update({"status": "generated"})
                entry.update(score_output(interpretation, package, case["evidence"]))
                entry["response"] = interpretation.model_dump()
                rows.append(entry)
                logger.info(
                    "grounding_scored",
                    mode=mode,
                    case=case["case_id"][:8],
                    faithfulness=entry.get("numerical_faithfulness"),
                    unsupported=entry.get("unsupported_numerical_claims"),
                )
    finally:
        await service.close()
    return rows


def run_grounding_evaluation() -> ExperimentRecord:
    """Compare baseline and grounded generation on real evidence packages."""
    if not settings.language_enabled:
        raise RuntimeError(
            "No language provider is configured. Set GROQ_API_KEY to run the "
            "grounding evaluation; results will not be simulated."
        )

    cases = load_cases()
    rows = asyncio.run(_run(cases))

    def summarise(mode: str) -> dict[str, Any]:
        scored = [r for r in rows if r["mode"] == mode and r["status"] == "generated"]
        rejected = [r for r in rows if r["mode"] == mode and r["status"] == "rejected"]
        if not scored:
            return {
                "n_generated": 0,
                "n_rejected": len(rejected),
                "note": "No output in this mode passed schema validation.",
            }
        faith = [
            r["numerical_faithfulness"] for r in scored if r["numerical_faithfulness"] is not None
        ]
        return {
            "n_generated": len(scored),
            "n_rejected": len(rejected),
            "mean_numerical_faithfulness": (round(sum(faith) / len(faith), 4) if faith else None),
            "total_unsupported_numerical_claims": sum(
                r["unsupported_numerical_claims"] for r in scored
            ),
            "outputs_with_unsupported_numbers": sum(
                1 for r in scored if r["unsupported_numerical_claims"] > 0
            ),
            "total_causal_overreach_flags": sum(r["causal_overreach_flags"] for r in scored),
            "outputs_with_causal_overreach": sum(
                1 for r in scored if r["causal_overreach_flags"] > 0
            ),
            "outputs_with_invented_sources": sum(
                1 for r in scored if not r["source_attribution_correct"]
            ),
            "grounding_pass_rate": round(
                sum(1 for r in scored if r["grounding_pass"]) / len(scored), 4
            ),
        }

    summary = {mode: summarise(mode) for mode in ("baseline", "grounded")}

    table_rows = [
        {k: v for k, v in row.items() if k not in {"response", "unsupported_values"}}
        for row in rows
    ]

    record = ExperimentRecord(
        experiment="llm_grounding",
        seeds=[],
        config={
            "modes": {
                "baseline": "minimal instruction, no grounding constraints, no validation",
                "grounded": "production system prompt and production validators, unchanged",
            },
            "provider": "Groq",
            "model": settings.language_model,
            "temperature": settings.language_temperature,
            "prompt_version": PROMPT_VERSION,
            "n_cases": len(cases),
            "claim_patterns_checked": len(UNSUPPORTED_CLAIM_PATTERNS),
        },
        dataset_manifest={},
        results={"summary": summary, "per_generation": rows},
        notes=[
            "Both modes are scored by the same deterministic checker, so the "
            "comparison isolates the effect of grounding rather than of scoring.",
            "A rejected grounded generation counts as a rejection, not as a pass; "
            "the validator was not relaxed for this experiment.",
        ],
        limitations=[
            "The checker is lexical, not semantic. It matches numbers against the "
            "evidence and screens a fixed list of causal phrasings; it cannot judge "
            "whether an interpretation is scientifically sound.",
            "Numeric matching allows a relative tolerance and the percentage form of "
            "a stored fraction, so a coincidental match is possible.",
            "Language-model output is non-deterministic and the hosted model may "
            "change without notice, so these figures describe one provider at one "
            "point in time.",
            "The case count is small; these are indicative measurements, not a powered comparison.",
        ],
    )

    write_table(table_rows, "table10_llm_grounding", "llm_grounding")

    modes = ["baseline", "grounded"]
    if all(summary[m].get("n_generated") for m in modes):
        figures.grouped_metric_bars(
            ["Numerical faithfulness", "Grounding pass rate"],
            {
                "Baseline": [
                    summary["baseline"]["mean_numerical_faithfulness"] or 0.0,
                    summary["baseline"]["grounding_pass_rate"],
                ],
                "Grounded": [
                    summary["grounded"]["mean_numerical_faithfulness"] or 0.0,
                    summary["grounded"]["grounding_pass_rate"],
                ],
            },
            title="Evidence grounding evaluation",
            ylabel="Score",
            name="fig12_llm_grounding",
            experiment="llm_grounding",
        )
    record.write()
    return record
