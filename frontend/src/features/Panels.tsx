/** Supporting dashboard panels: layers, observations, interpretation, history, report. */

import type {
  AnalysisDetail,
  AnalysisSummary,
  LayerReference,
  ReportResponse,
} from '../types/api'
import { STATUS_LABELS } from '../types/api'

// --- map layer controls -----------------------------------------------------
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
    <div className="layer-controls">
      <div className="layer-buttons">
        <button
          type="button"
          className={activeKey === null ? 'chipbtn active' : 'chipbtn'}
          onClick={() => onSelect(null)}
        >
          Basemap only
        </button>
        {layers.map((layer) => (
          <button
            key={layer.key}
            type="button"
            className={activeKey === layer.key ? 'chipbtn active' : 'chipbtn'}
            onClick={() => onSelect(layer.key)}
            title={layer.description ?? layer.label}
          >
            {layer.label}
          </button>
        ))}
      </div>

      {active && (
        <>
          <label className="opacity-row">
            <span>Opacity</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => onOpacityChange(Number(e.target.value))}
            />
            <span className="mono">{Math.round(opacity * 100)}%</span>
          </label>

          {active.legend.length > 0 && (
            <div className={`legend ${active.kind}`}>
              {active.kind === 'continuous' ? (
                <>
                  <div
                    className="legend-ramp"
                    style={{
                      background: `linear-gradient(to right, ${active.legend
                        .map((s) => s.colour)
                        .join(', ')})`,
                    }}
                  />
                  <div className="legend-scale">
                    <span>{active.legend[0]?.label}</span>
                    <span>{active.legend[active.legend.length - 1]?.label}</span>
                  </div>
                  {active.value_min !== null && active.value_max !== null && (
                    <p className="footnote">
                      Observed range in this layer: {active.value_min.toFixed(3)} to{' '}
                      {active.value_max.toFixed(3)}
                    </p>
                  )}
                </>
              ) : (
                <ul className="legend-classes">
                  {active.legend.map((entry) => (
                    <li key={`${entry.value}-${entry.label}`}>
                      <span className="swatch" style={{ background: entry.colour }} />
                      {entry.label}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
          {active.description && <p className="footnote">{active.description}</p>}
        </>
      )}
    </div>
  )
}

// --- satellite observations --------------------------------------------------
export function ObservationsPanel({ detail }: { detail: AnalysisDetail }) {
  if (detail.observations.length === 0) {
    return <p className="empty-note">No observation metadata was recorded.</p>
  }
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>Period</th>
            <th>Scene</th>
            <th>Acquired</th>
            <th>Cloud</th>
            <th>Platform</th>
            <th>Level</th>
            <th>Res.</th>
            <th>Bands</th>
          </tr>
        </thead>
        <tbody>
          {detail.observations.map((o) => (
            <tr key={`${o.period}-${o.source_id}`}>
              <td>{o.period}</td>
              <td className="mono small">{o.source_id}</td>
              <td>{o.observation_date}</td>
              <td>
                {o.cloud_cover_percent !== null
                  ? `${o.cloud_cover_percent.toFixed(2)}%`
                  : '—'}
              </td>
              <td>{o.platform ?? '—'}</td>
              <td>{o.processing_level ?? '—'}</td>
              <td>{o.resolution_m ? `${o.resolution_m} m` : '—'}</td>
              <td className="small">{o.bands_used.join(', ')}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="footnote">
        Source: {detail.observations[0]?.provider} ·{' '}
        {detail.observations[0]?.dataset} · {detail.observations[0]?.license}
      </p>
    </div>
  )
}

// --- interpretation ------------------------------------------------------------
interface InterpretationProps {
  report: ReportResponse | null
  busy: boolean
}

export function InterpretationPanel({ report, busy }: InterpretationProps) {
  if (busy) return <p className="empty-note">Generating interpretation…</p>
  if (!report) {
    return (
      <p className="empty-note">
        Generate a report to produce an evidence-grounded interpretation of these
        results.
      </p>
    )
  }

  const interpretation = report.interpretation as
    | {
        summary: string
        observations: { statement: string }[]
        interpretation: string
        uncertainty: string
        limitations: string[]
        confidence_qualifier: string
      }
    | null

  if (!interpretation) {
    const section = report.sections.find((s) => s.key === 'interpretation')
    return (
      <div className="interpretation">
        <p className="empty-note">
          {section?.body ?? 'No interpretation was generated for this analysis.'}
        </p>
        <p className="footnote">
          All measured values shown elsewhere on this dashboard are unaffected.
        </p>
      </div>
    )
  }

  return (
    <div className="interpretation">
      <div className="interpretation-head">
        <span className="chip interpretation">Generated interpretation</span>
        <span className="footnote">
          {report.interpretation_provider} · {report.interpretation_model} ·
          confidence: {interpretation.confidence_qualifier}
        </span>
      </div>
      <p className="lede">{interpretation.summary}</p>

      <h4>Observations</h4>
      <ul>
        {interpretation.observations.map((o, i) => (
          <li key={i}>{o.statement}</li>
        ))}
      </ul>

      <h4>Interpretation</h4>
      <p>{interpretation.interpretation}</p>

      <h4>Uncertainty</h4>
      <p>{interpretation.uncertainty}</p>

      <h4>Limitations</h4>
      <ul>
        {interpretation.limitations.map((l, i) => (
          <li key={i}>{l}</li>
        ))}
      </ul>

      {report.visual_interpretation && (
        <>
          <h4>Visual interpretation of the NDVI layer</h4>
          <p>{(report.visual_interpretation as { scene_description: string }).scene_description}</p>
          <p className="footnote">
            Produced by a vision model from the rendered image. It describes
            appearance only and is not a source of any numeric value.
          </p>
        </>
      )}

      <p className="footnote">
        This section is written by a language model from the measured evidence and
        is validated against it. It contains no independent measurement.
      </p>
    </div>
  )
}

// --- report ----------------------------------------------------------------------
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
    <div className="report-panel">
      <div className="report-actions">
        <button
          type="button"
          className="primary"
          disabled={!ready || busy}
          onClick={() => onGenerate(false, Boolean(report))}
        >
          {busy ? 'Generating…' : report ? 'Regenerate report' : 'Generate report'}
        </button>
        {visionAvailable && (
          <button
            type="button"
            disabled={!ready || busy}
            onClick={() => onGenerate(true, true)}
            title="Additionally ask a vision model to describe the rendered NDVI layer"
          >
            With visual interpretation
          </button>
        )}
        {report && (
          <a className="button-link" href={exportUrl(report.id, 'html')} target="_blank" rel="noreferrer">
            Export HTML
          </a>
        )}
        {report?.export_urls?.pdf && (
          <a className="button-link" href={exportUrl(report.id, 'pdf')}>
            Export PDF
          </a>
        )}
      </div>

      {!ready && (
        <p className="footnote">
          The report becomes available once the analysis reaches the complete state.
        </p>
      )}

      {report && (
        <div className="report-sections">
          {report.sections.map((section) => (
            <details key={section.key}>
              <summary>
                {section.title}
                <span className={`chip ${section.provenance}`}>
                  {report.provenance_legend[section.provenance] ?? section.provenance}
                </span>
              </summary>
              {section.body && <p>{section.body}</p>}
              {section.items.length > 0 && (
                <ul>
                  {section.items.map((item, i) => (
                    <li key={i}>{item}</li>
                  ))}
                </ul>
              )}
              {section.table.length > 0 && (
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        {Object.keys(section.table[0]!).map((column) => (
                          <th key={column}>{column}</th>
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
                <div className="notes">
                  {section.notes.map((note, i) => (
                    <p key={i} className="footnote">
                      {note}
                    </p>
                  ))}
                </div>
              )}
            </details>
          ))}
        </div>
      )}
    </div>
  )
}

// --- history -------------------------------------------------------------------------
interface HistoryProps {
  items: AnalysisSummary[]
  activeId: string | null
  onOpen: (id: string) => void
  onDelete: (id: string) => void
}

export function HistoryPanel({ items, activeId, onOpen, onDelete }: HistoryProps) {
  if (items.length === 0) {
    return <p className="empty-note">No analyses yet. Run one to build your history.</p>
  }
  return (
    <ul className="history">
      {items.map((item) => (
        <li key={item.id} className={item.id === activeId ? 'active' : undefined}>
          <button type="button" className="history-open" onClick={() => onOpen(item.id)}>
            <span className="history-title">
              {item.region_name ?? `${item.area_km2.toFixed(1)} km² region`}
            </span>
            <span className="history-meta">
              {item.period_a}
              {item.period_b ? ` → ${item.period_b}` : ''}
            </span>
            <span className="history-meta">
              <span className={`status-dot ${item.status}`} />
              {STATUS_LABELS[item.status]}
              {item.mean_ndvi_a !== null && ` · NDVI ${item.mean_ndvi_a.toFixed(3)}`}
              {item.ndvi_change !== null &&
                ` · Δ ${item.ndvi_change > 0 ? '+' : ''}${item.ndvi_change.toFixed(3)}`}
            </span>
          </button>
          <button
            type="button"
            className="icon-button"
            aria-label="Delete analysis"
            onClick={() => onDelete(item.id)}
          >
            ×
          </button>
        </li>
      ))}
    </ul>
  )
}
