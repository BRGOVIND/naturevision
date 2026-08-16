/**
 * Headline environmental metrics.
 *
 * Every card carries a provenance chip. Measured values and model predictions
 * are never styled identically, because the difference matters to how the
 * number should be read.
 */

import type { AnalysisDetail, Metric } from '../types/api'

interface Props {
  detail: AnalysisDetail
}

function findMetric(metrics: Metric[], key: string, period?: string): Metric | undefined {
  return metrics.find((m) => m.key === key && (period === undefined || m.period === period))
}

function formatValue(metric: Metric | undefined, digits = 3, signed = false): string {
  if (!metric || metric.value === null) return '—'
  const value = metric.value
  const text = signed ? value.toFixed(digits) : Math.abs(value).toFixed(digits)
  const prefix = signed && value > 0 ? '+' : signed && value < 0 ? '−' : ''
  const body = signed ? `${prefix}${Math.abs(value).toFixed(digits)}` : text
  if (metric.unit === 'percent') return `${body}%`
  if (metric.unit === 'km2') return `${body} km²`
  return body
}

export function MetricCards({ detail }: Props) {
  const metrics = detail.metrics
  const meanA = findMetric(metrics, 'mean_ndvi', 'A')
  const meanB = findMetric(metrics, 'mean_ndvi', 'B')
  const change = findMetric(metrics, 'ndvi_change')
  const changedArea = findMetric(metrics, 'changed_area_percent')
  const forest = findMetric(metrics, 'land_cover_forest')
  const prediction = detail.predictions[0]

  const thresholds = change?.details as { moderate?: number } | undefined
  const belowNoiseFloor =
    change?.value !== null &&
    change?.value !== undefined &&
    thresholds?.moderate !== undefined &&
    Math.abs(change.value) < thresholds.moderate

  const cards = [
    {
      label: 'Mean NDVI',
      sub: meanB ? 'Period B' : 'Period A',
      value: formatValue(meanB ?? meanA),
      provenance: 'observed' as const,
      note: meanA && meanB ? `Period A: ${formatValue(meanA)}` : 'Single-date analysis',
    },
    {
      label: 'NDVI change',
      sub: 'Period B − Period A',
      value: change ? formatValue(change, 3, true) : '—',
      provenance: 'observed' as const,
      note: belowNoiseFloor
        ? `Below the ${thresholds?.moderate} detection threshold`
        : change
          ? 'Mean over pixels valid in both periods'
          : 'Requires two periods',
    },
    {
      label: 'Changed area',
      sub: 'Beyond moderate threshold',
      value: formatValue(changedArea, 2),
      provenance: 'observed' as const,
      note: changedArea ? 'Share of comparable area' : 'Requires two periods',
    },
    {
      label: 'Forest cover',
      sub: 'Predicted share',
      value: formatValue(forest, 1),
      provenance: 'model_prediction' as const,
      note: forest ? 'Classifier output, not a measurement' : 'Land cover not run',
    },
    {
      label: 'Model confidence',
      sub: 'Mean max class probability',
      value:
        prediction?.mean_confidence != null
          ? prediction.mean_confidence.toFixed(3)
          : '—',
      provenance: 'model_prediction' as const,
      note:
        prediction?.evaluation_metrics?.overall_accuracy != null
          ? `Held-out accuracy ${prediction.evaluation_metrics.overall_accuracy.toFixed(3)}`
          : 'No held-out evaluation recorded',
    },
  ]

  return (
    <div className="metric-grid">
      {cards.map((card) => (
        <article key={card.label} className={`metric-card ${card.provenance}`}>
          <header>
            <span className="metric-label">{card.label}</span>
            <span className={`chip ${card.provenance}`}>
              {card.provenance === 'observed' ? 'Observed' : 'Model'}
            </span>
          </header>
          <p className="metric-value">{card.value}</p>
          <p className="metric-sub">{card.sub}</p>
          <p className="metric-note">{card.note}</p>
        </article>
      ))}
    </div>
  )
}
