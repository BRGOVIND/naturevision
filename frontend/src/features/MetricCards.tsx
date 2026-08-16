/**
 * Headline environmental metrics.
 *
 * Each card carries a provenance register, because the difference between a
 * measured value and a model prediction changes how the number should be read.
 * Cards vary in internal layout by what they have to say — a change value gets
 * a direction indicator, a distribution gets a bar — rather than being six
 * copies of one template.
 */

import { ProvenanceBadge, type Provenance } from '../design/primitives'
import type { AnalysisDetail, Metric } from '../types/api'

function find(metrics: Metric[], key: string, period?: string): Metric | undefined {
  return metrics.find((m) => m.key === key && (period === undefined || m.period === period))
}

function fmt(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined ? '—' : value.toFixed(digits)
}

interface CardModel {
  key: string
  label: string
  value: string
  unit?: string
  provenance: Provenance
  note: string
  direction?: 'up' | 'down' | 'flat'
  bar?: number
  belowThreshold?: boolean
}

export function MetricCards({ detail }: { detail: AnalysisDetail }) {
  const metrics = detail.metrics
  const meanA = find(metrics, 'mean_ndvi', 'A')
  const meanB = find(metrics, 'mean_ndvi', 'B')
  const change = find(metrics, 'ndvi_change')
  const changedArea = find(metrics, 'changed_area_percent')
  const forest = find(metrics, 'land_cover_forest')
  const prediction = detail.predictions[0]

  const thresholds = change?.details as { moderate?: number } | undefined
  const moderate = thresholds?.moderate
  const belowThreshold =
    change?.value != null && moderate != null && Math.abs(change.value) < moderate

  const current = meanB ?? meanA
  const cards: CardModel[] = [
    {
      key: 'ndvi',
      label: 'Mean NDVI',
      value: fmt(current?.value),
      provenance: 'observed',
      note: meanA && meanB ? `Period A measured ${fmt(meanA.value)}` : 'Single-date analysis',
      bar: current?.value != null ? Math.max(0, Math.min(1, current.value)) : undefined,
    },
    {
      key: 'change',
      label: 'NDVI change',
      value:
        change?.value == null
          ? '—'
          : `${change.value > 0 ? '+' : change.value < 0 ? '−' : ''}${Math.abs(change.value).toFixed(3)}`,
      provenance: 'observed',
      direction:
        change?.value == null ? undefined : change.value > 0 ? 'up' : change.value < 0 ? 'down' : 'flat',
      note: belowThreshold
        ? `Below the ${moderate} detection threshold`
        : change
          ? 'Mean over pixels valid in both periods'
          : 'Requires a second period',
      belowThreshold,
    },
    {
      key: 'changed-area',
      label: 'Changed area',
      value: changedArea?.value == null ? '—' : changedArea.value.toFixed(2),
      unit: '%',
      provenance: 'observed',
      note: changedArea ? 'Share exceeding the moderate threshold' : 'Requires a second period',
      bar: changedArea?.value != null ? changedArea.value / 100 : undefined,
    },
    {
      key: 'forest',
      label: 'Forest cover',
      value: forest?.value == null ? '—' : forest.value.toFixed(1),
      unit: '%',
      provenance: 'model_prediction',
      note: forest ? 'Classifier output, not a measurement' : 'Land cover not run',
      bar: forest?.value != null ? forest.value / 100 : undefined,
    },
    {
      key: 'confidence',
      label: 'Model confidence',
      value: fmt(prediction?.mean_confidence),
      provenance: 'model_prediction',
      note:
        prediction?.evaluation_metrics?.overall_accuracy != null
          ? `Hold-out accuracy ${prediction.evaluation_metrics.overall_accuracy.toFixed(3)}`
          : 'No hold-out evaluation recorded',
      bar: prediction?.mean_confidence ?? undefined,
    },
  ]

  return (
    <ul className="metrics">
      {cards.map((card) => (
        <li key={card.key} className={`metric metric--${card.provenance}`}>
          <div className="metric__top">
            <h3 className="metric__label">{card.label}</h3>
            <ProvenanceBadge kind={card.provenance} />
          </div>

          <p className="metric__value tabular">
            {card.direction && (
              <span className={`metric__arrow metric__arrow--${card.direction}`} aria-hidden="true">
                {card.direction === 'up' ? '↑' : card.direction === 'down' ? '↓' : '→'}
              </span>
            )}
            {card.value}
            {card.unit && <span className="metric__unit">{card.unit}</span>}
          </p>

          {card.bar !== undefined && (
            <div className="metric__bar" aria-hidden="true">
              <span style={{ width: `${Math.round(card.bar * 100)}%` }} />
            </div>
          )}

          <p className={card.belowThreshold ? 'metric__note metric__note--caution' : 'metric__note'}>
            {card.note}
          </p>
        </li>
      ))}
    </ul>
  )
}
