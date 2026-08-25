/** Supporting workspace panels. */

import { Button, ProvenanceBadge, Spinner, type Provenance } from '../design/primitives'
import type {
  AnalysisDetail,
  AnalysisSummary,
  LayerReference,
  ReportResponse,
} from '../types/api'
import { STATUS_LABELS } from '../types/api'

// -------------------------------------------------------------------------
// Map layers
// -------------------------------------------------------------------------
interface LayerControlsProps {
  layers: LayerReference[]
  activeKey: string | null
  opacity: number
  onSelect: (key: string | null) => void
  onOpacityChange: (value: number) => void
}

export function LayerControls({
  layers,
  activeKey,
  opacity,
  onSelect,
  onOpacityChange,
}: LayerControlsProps) {
  const active = layers.find((l) => l.key === activeKey) ?? null

  return (
    <div className="layers">
      <div className="layers__row">
        <div className="layers__chips" role="group" aria-label="Map layer">
          <button
            type="button"
            className={activeKey === null ? 'seg seg--active' : 'seg'}
            aria-pressed={activeKey === null}
            onClick={() => onSelect(null)}
          >
            Basemap
          </button>
          {layers.map((layer) => (
            <button
              key={layer.key}
              type="button"
              className={activeKey === layer.key ? 'seg seg--active' : 'seg'}
              aria-pressed={activeKey === layer.key}
              onClick={() => onSelect(layer.key)}
              title={layer.description ?? layer.label}
            >
              {layer.label}
            </button>
          ))}
        </div>

        {active && (
          <label className="layers__opacity">
            <span>Opacity</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => onOpacityChange(Number(e.target.value))}
              aria-label="Layer opacity"
            />
            <span className="mono">{Math.round(opacity * 100)}%</span>
          </label>
        )}
      </div>

      {active && (
        <div className="layers__legend">
          {active.kind === 'continuous' && active.legend.length > 0 ? (
            <>
              <div
                className="legend-ramp"
                style={{
                  background: `linear-gradient(90deg, ${active.legend.map((s) => s.colour).join(', ')})`,
                }}
                role="img"
                aria-label={`Colour scale from ${active.legend[0]?.label} to ${active.legend[active.legend.length - 1]?.label}`}
              />
              <div className="legend-scale mono">
                <span>{active.legend[0]?.label}</span>
                {active.value_min !== null && active.value_max !== null && (
                  <span className="legend-observed">
                    observed {active.value_min.toFixed(2)} … {active.value_max.toFixed(2)}
                  </span>
                )}
                <span>{active.legend[active.legend.length - 1]?.label}</span>
              </div>
            </>
          ) : (
            <ul className="legend-classes">
              {active.legend.map((entry) => (
                <li key={`${entry.value}-${entry.label}`}>
                  <span className="swatch" style={{ background: entry.colour }} aria-hidden="true" />
                  {entry.label}
                </li>
              ))}
            </ul>
          )}
          {active.description && <p className="layers__note">{active.description}</p>}
        </div>
      )}
    </div>
  )
}

// -------------------------------------------------------------------------
// Satellite observations
// -------------------------------------------------------------------------
export function ObservationsPanel({ detail }: { detail: AnalysisDetail }) {
  if (detail.observations.length === 0) {
    return <p className="chart-empty">No observation metadata was recorded.</p>
  }

  return (
    <div className="observations">
      {/* Wide screens: a comparison table. */}
      <table className="observations__table">
        <caption className="visually-hidden">Satellite observations used by this analysis</caption>
        <thead>
          <tr>
            <th scope="col">Period</th>
            <th scope="col">Scene</th>
            <th scope="col">Acquired</th>
            <th scope="col">Cloud</th>
            <th scope="col">Platform</th>
            <th scope="col">Level</th>
            <th scope="col">Res.</th>
            <th scope="col">Bands</th>
          </tr>
        </thead>
        <tbody>
          {detail.observations.map((o) => (
            <tr key={`${o.period}-${o.source_id}`}>
              <th scope="row">
                <span className="period-tag">{o.period}</span>
              </th>
              <td className="mono nowrap">{o.source_id}</td>
              <td className="mono">{o.observation_date}</td>
              <td className="mono">
                {o.cloud_cover_percent !== null ? `${o.cloud_cover_percent.toFixed(2)}%` : '—'}
              </td>
              <td>{o.platform ?? '—'}</td>
              <td>{o.processing_level ?? '—'}</td>
              <td className="mono">{o.resolution_m ? `${o.resolution_m} m` : '—'}</td>
              <td className="observations__bands">{o.bands_used.join(' · ')}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Narrow screens: stacked cards, so nothing overflows horizontally. */}
      <ul className="observations__cards">
        {detail.observations.map((o) => (
          <li key={`card-${o.period}-${o.source_id}`}>
            <div className="observations__card-head">
              <span className="period-tag">{o.period}</span>
              <span className="mono">{o.observation_date}</span>
            </div>
            <p className="observations__scene mono">{o.source_id}</p>
            <dl className="observations__facts">
              <div>
                <dt>Cloud</dt>
                <dd className="mono">
                  {o.cloud_cover_percent !== null ? `${o.cloud_cover_percent.toFixed(2)}%` : '—'}
                </dd>
              </div>
              <div>
                <dt>Platform</dt>
                <dd>{o.platform ?? '—'}</dd>
              </div>
              <div>
                <dt>Level</dt>
                <dd>{o.processing_level ?? '—'}</dd>
              </div>
              <div>
                <dt>Resolution</dt>
                <dd className="mono">{o.resolution_m ? `${o.resolution_m} m` : '—'}</dd>
              </div>
            </dl>
            <p className="observations__bands">{o.bands_used.join(' · ')}</p>
          </li>
        ))}
      </ul>

      <p className="chart-note">
        Source: {detail.observations[0]?.provider} · {detail.observations[0]?.dataset} ·{' '}
        {detail.observations[0]?.license}
      </p>
    </div>
  )
}

// -------------------------------------------------------------------------
// Interpretation
// -------------------------------------------------------------------------
interface InterpretationProps {
  report: ReportResponse | null
  busy: boolean
  available: boolean
  onGenerate: () => void
}

export function InterpretationPanel({ report, busy, available, onGenerate }: InterpretationProps) {
  if (busy) {
    return (
      <div className="interp interp--waiting">
        <Spinner label="Generating interpretation" />
        <p>Sending the measured evidence to the interpretation provider and validating the response.</p>
      </div>
    )
  }

  if (!report) {
    return (
      <div className="interp interp--waiting">
        <p>
          {available
            ? 'Generate the report to produce an evidence-grounded interpretation of these results.'
            : 'No interpretation provider is configured on this deployment. All measured results above are unaffected.'}
        </p>
        {available && (
          <Button variant="primary" onClick={onGenerate}>
            Generate interpretation
          </Button>
        )}
      </div>
    )
  }

  const interpretation = report.interpretation as
    | {
        summary: string
        observations: { statement: string; evidence_key?: string | null }[]
        interpretation: string
        uncertainty: string
        limitations: string[]
        confidence_qualifier: string
      }
    | null

  if (!interpretation) {
    const section = report.sections.find((s) => s.key === 'interpretation')
    return (
      <div className="interp interp--waiting">
        <p>{section?.body ?? 'No interpretation was generated for this analysis.'}</p>
        <p className="chart-note">All measured values shown above are unaffected.</p>
      </div>
    )
  }

  const grounding = (report as unknown as { grounding?: Record<string, number> }).grounding
  // A response with no provider name only ever comes from the deterministic
  // fallback (see reports.py) — the language-model path always sets one.
  const measured = !report.interpretation_provider

  return (
    <div className="interp">
      <div className="interp__meta">
        <ProvenanceBadge kind="interpretation" label={measured ? 'Evidence summary' : undefined} />
        {measured ? (
          <span className="mono">Generated deterministically from measured values</span>
        ) : (
          <span className="mono">
            {report.interpretation_provider} · {report.interpretation_model}
          </span>
        )}
        {!measured && (
          <span className={`confidence confidence--${interpretation.confidence_qualifier}`}>
            {interpretation.confidence_qualifier} confidence
          </span>
        )}
      </div>

      <div className="interp__layout">
        <div className="interp__text">
          <p className="interp__summary">{interpretation.summary}</p>
          <h3>{measured ? 'What the values show' : 'Interpretation'}</h3>
          <p>{interpretation.interpretation}</p>
          <h3>Uncertainty</h3>
          <p>{interpretation.uncertainty}</p>
        </div>
        <div className="interp__field-mark" aria-hidden="true">
          <svg viewBox="0 0 200 240" preserveAspectRatio="none">
            <path d="M -10,40 Q 60,10 100,50 T 210,45" />
            <path d="M -10,95 Q 70,70 110,105 T 210,95" />
            <path d="M -10,150 Q 55,125 105,160 T 210,150" />
            <path d="M -10,205 Q 65,180 100,210 T 210,200" />
          </svg>
        </div>
      </div>

      <ul className="interp__anchors">
        {interpretation.observations.map((o, i) => (
          <li key={i}>
            <span>{o.statement}</span>
            {o.evidence_key && <code className="mono">{o.evidence_key}</code>}
          </li>
        ))}
      </ul>

      <section className="interp__block interp__block--limits">
        <h3>Limitations</h3>
        <ul>
          {interpretation.limitations.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      </section>

      {report.visual_interpretation && (
        <section className="interp__block">
          <h3>Visual interpretation</h3>
          <p>
            {(report.visual_interpretation as { scene_description: string }).scene_description}
          </p>
          <p className="chart-note">
            Produced by a vision model from the rendered NDVI image. It describes appearance
            only and is not the source of any numeric value.
          </p>
        </section>
      )}

      <p className="interp__footnote">
        {measured
          ? 'Generated deterministically from the measured evidence above, not written by a language model — no provider response passed grounding validation for this analysis.'
          : 'Written by a language model from the measured evidence only.'}
        {grounding?.checked_number_count != null &&
          ` Grounding check: ${grounding.matched_number_count} of ${grounding.checked_number_count} numeric statements matched a measured value.`}{' '}
        It contains no independent measurement.
      </p>
    </div>
  )
}

// -------------------------------------------------------------------------
// Report
// -------------------------------------------------------------------------
interface ReportPanelProps {
  report: ReportResponse | null
  detail: AnalysisDetail | null
  busy: boolean
  onGenerate: (includeVisual: boolean, regenerate: boolean) => void
  exportUrl: (id: string, format: 'html' | 'pdf') => string
  visionAvailable: boolean
}

export function ReportPanel({
  report,
  detail,
  busy,
  onGenerate,
  exportUrl,
  visionAvailable,
}: ReportPanelProps) {
  const ready = detail?.status === 'report_ready'

  return (
    <div className="report">
      <div className="report__actions">
        <Button
          variant="primary"
          disabled={!ready || busy}
          onClick={() => onGenerate(false, Boolean(report))}
        >
          {busy
            ? 'Generating…'
            : report
              ? 'Regenerate report'
              : 'Generate Nature Intelligence Report'}
        </Button>
        {visionAvailable && (
          <Button
            disabled={!ready || busy}
            onClick={() => onGenerate(true, true)}
            title="Additionally ask a vision model to describe the rendered NDVI layer"
          >
            Include visual interpretation
          </Button>
        )}
        {report && (
          <Button href={exportUrl(report.id, 'html')} download>
            Export HTML
          </Button>
        )}
        {report?.export_urls?.pdf && (
          <Button href={exportUrl(report.id, 'pdf')} download>
            Export PDF
          </Button>
        )}
      </div>

      {!report && (
        <p className="chart-note">
          The report collects every measured value, the methodology behind it, the model
          provenance and the limitations of this run into one exportable document.
        </p>
      )}

      {report && (
        <>
          <div className="report__meta">
            <div>
              <h3>{report.title}</h3>
              <p className="mono">
                {report.sections.length} sections · generated{' '}
                {new Date(report.generated_at).toLocaleString()}
              </p>
            </div>
            <ul className="report__legend">
              {Object.entries(report.provenance_legend).map(([key, label]) => (
                <li key={key}>
                  <ProvenanceBadge kind={key as Provenance} label={label} />
                </li>
              ))}
            </ul>
          </div>

          <div className="report__sections">
            {report.sections.map((section, index) => (
              <details key={section.key} className="report__section">
                <summary>
                  <span className="report__section-index mono">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  <span className="report__section-title">{section.title}</span>
                  <ProvenanceBadge
                    kind={section.provenance}
                    label={report.provenance_legend[section.provenance]}
                  />
                </summary>
                <div className="report__body">
                  {section.body && <p>{section.body}</p>}
                  {section.items.length > 0 && (
                    <ul>
                      {section.items.map((item, i) => (
                        <li key={i}>{item}</li>
                      ))}
                    </ul>
                  )}
                  {section.table.length > 0 && (
                    <div className="report__table-wrap">
                      <table>
                        <thead>
                          <tr>
                            {Object.keys(section.table[0]!).map((column) => (
                              <th key={column} scope="col">
                                {column}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {section.table.map((row, i) => (
                            <tr key={i}>
                              {Object.values(row).map((value, j) => (
                                <td key={j}>{value ?? '—'}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {section.notes.length > 0 && (
                    <div className="report__notes">
                      {section.notes.map((note, i) => (
                        <p key={i}>{note}</p>
                      ))}
                    </div>
                  )}
                </div>
              </details>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

// -------------------------------------------------------------------------
// History
// -------------------------------------------------------------------------
interface HistoryProps {
  items: AnalysisSummary[]
  activeId: string | null
  onOpen: (id: string) => void
  onDelete: (id: string) => void
}

export function HistoryPanel({ items, activeId, onOpen, onDelete }: HistoryProps) {
  if (items.length === 0) {
    return (
      <p className="history__empty">
        No analyses yet. Runs are saved automatically and appear here.
      </p>
    )
  }

  return (
    <ul className="history">
      {items.map((item) => (
        <li key={item.id} className={item.id === activeId ? 'history__item history__item--active' : 'history__item'}>
          <button type="button" className="history__open" onClick={() => onOpen(item.id)}>
            <span className="history__title">
              {item.region_name ?? `${item.area_km2.toFixed(1)} km² region`}
            </span>
            <span className="history__period mono">
              {item.period_a}
              {item.period_b ? ` → ${item.period_b}` : ''}
            </span>
            <span className="history__stats">
              <span className={`status-dot status-dot--${item.status}`} aria-hidden="true" />
              <span>{STATUS_LABELS[item.status]}</span>
              {item.mean_ndvi_a !== null && (
                <span className="mono">NDVI {item.mean_ndvi_a.toFixed(3)}</span>
              )}
              {item.ndvi_change !== null && (
                <span className="mono">
                  Δ {item.ndvi_change > 0 ? '+' : ''}
                  {item.ndvi_change.toFixed(3)}
                </span>
              )}
              <span className="history__date">
                {new Date(item.created_at).toLocaleDateString()}
              </span>
            </span>
          </button>
          <button
            type="button"
            className="history__delete"
            aria-label={`Delete analysis of ${item.region_name ?? 'region'}`}
            onClick={() => onDelete(item.id)}
          >
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
              <path
                d="M3 4h10M6.5 4V2.8h3V4M5 4l.6 9h4.8L11 4"
                stroke="currentColor"
                strokeWidth="1.3"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </li>
      ))}
    </ul>
  )
}
