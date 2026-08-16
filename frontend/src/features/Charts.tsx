/**
 * Result visualisations.
 *
 * Colours come from the shared class palette so a class reads identically on
 * the map, in the legend and in every chart. Slice labels are drawn in white
 * with a dark halo and are suppressed entirely below the angle at which they
 * would collide or clip, rather than being shrunk until unreadable.
 */

import { useEffect, useState } from 'react'
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { AnalysisDetail, Metric } from '../types/api'

/** Mirrors the backend class palette exactly. */
const CLASS_COLOURS: Record<string, string> = {
  Forest: '#1b7f3b',
  Agriculture: '#d9a441',
  Water: '#2b6cb0',
  'Urban / built-up': '#a02c2c',
  'Bare land': '#8a8f98',
}

const CHANGE_COLOURS: Record<string, string> = {
  'Significant decrease': '#8f2f22',
  'Moderate decrease': '#c8734f',
  Stable: '#a8a08c',
  'Moderate increase': '#7fa650',
  'Significant increase': '#1d4b33',
}

const AXIS = { stroke: '#7d8a81', fontSize: 11, fontFamily: 'IBM Plex Mono, monospace' }
const GRID = '#ded5c2'

const TOOLTIP_STYLE = {
  background: '#16301f',
  border: '1px solid #24402f',
  borderRadius: 6,
  color: '#eef3ec',
  fontSize: 12,
  padding: '8px 10px',
  boxShadow: '0 8px 24px rgb(8 20 14 / 0.28)',
}

/** Slices thinner than this are labelled in the table only. */
const MIN_LABEL_PERCENT = 7

function useIsNarrow(breakpoint = 620) {
  const [narrow, setNarrow] = useState(
    () => typeof window !== 'undefined' && window.innerWidth < breakpoint,
  )
  useEffect(() => {
    const query = window.matchMedia(`(max-width: ${breakpoint - 1}px)`)
    const onChange = (e: MediaQueryListEvent) => setNarrow(e.matches)
    setNarrow(query.matches)
    query.addEventListener('change', onChange)
    return () => query.removeEventListener('change', onChange)
  }, [breakpoint])
  return narrow
}

function metric(metrics: Metric[], key: string, period?: string): number | null {
  return metrics.find((m) => m.key === key && (period === undefined || m.period === period))?.value ?? null
}

/** In-slice percentage label: white, haloed, and only where it fits.
 *
 * The dominant slice is skipped because the donut centre already states it;
 * labelling it twice adds no information.
 */
function makeSliceLabel(dominantLabel: string | undefined) {
  return function renderSliceLabel(props: any) {
    const { cx, cy, midAngle, innerRadius, outerRadius, percent, name } = props
    const share = percent * 100
    if (share < MIN_LABEL_PERCENT || name === dominantLabel) return null

    const RAD = Math.PI / 180
    const radius = innerRadius + (outerRadius - innerRadius) * 0.55
    const x = cx + radius * Math.cos(-midAngle * RAD)
    const y = cy + radius * Math.sin(-midAngle * RAD)

    return (
      <text
        x={x}
        y={y}
        fill="#ffffff"
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={12}
        fontWeight={600}
        fontFamily="IBM Plex Mono, monospace"
        // A dark halo keeps the label legible over every slice colour,
        // including the lighter ochre and grey ones.
        style={{ paintOrder: 'stroke', stroke: 'rgba(12,26,18,0.6)', strokeWidth: 3 }}
      >
        {share.toFixed(0)}%
      </text>
    )
  }
}

export function LandCoverChart({ detail }: { detail: AnalysisDetail }) {
  const narrow = useIsNarrow()
  const prediction = detail.predictions[0]

  if (!prediction) {
    return (
      <p className="chart-empty">
        Land-cover classification was not part of this analysis, or no trained model was
        installed when it ran.
      </p>
    )
  }

  const data = Object.entries(prediction.class_distribution)
    .map(([label, value]) => ({ label, value: Number(value) }))
    .filter((d) => d.value > 0)
    .sort((a, b) => b.value - a.value)

  const perClass = prediction.evaluation_metrics?.per_class_metrics ?? {}
  const dominant = data[0]

  return (
    <div className="landcover">
      <figure className="landcover__chart">
        <ResponsiveContainer width="100%" height={narrow ? 210 : 250}>
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="label"
              innerRadius={narrow ? 52 : 62}
              outerRadius={narrow ? 88 : 104}
              paddingAngle={1}
              stroke="#f4efe4"
              strokeWidth={2}
              labelLine={false}
              label={makeSliceLabel(dominant?.label)}
              isAnimationActive={false}
            >
              {data.map((entry) => (
                <Cell key={entry.label} fill={CLASS_COLOURS[entry.label] ?? '#6b7c74'} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              itemStyle={{ color: '#eef3ec' }}
              labelStyle={{ color: '#9cb2a4' }}
              formatter={(value: number) => [`${value.toFixed(2)}%`, 'Share']}
            />
          </PieChart>
        </ResponsiveContainer>

        {dominant && (
          <div className="landcover__center" aria-hidden="true">
            <span className="landcover__center-value tabular">{dominant.value.toFixed(1)}%</span>
            <span className="landcover__center-label">{dominant.label}</span>
          </div>
        )}
        <figcaption className="visually-hidden">
          Predicted land-cover distribution:{' '}
          {data.map((d) => `${d.label} ${d.value.toFixed(1)} percent`).join(', ')}.
        </figcaption>
      </figure>

      <div className="landcover__table">
        <table>
          <caption className="visually-hidden">
            Land-cover class shares with hold-out F1 scores
          </caption>
          <thead>
            <tr>
              <th scope="col">Class</th>
              <th scope="col">Share</th>
              <th scope="col">
                <abbr title="F1 score measured on spatially held-out regions">Hold-out F1</abbr>
              </th>
            </tr>
          </thead>
          <tbody>
            {data.map((entry) => (
              <tr key={entry.label}>
                <th scope="row">
                  <span
                    className="swatch"
                    style={{ background: CLASS_COLOURS[entry.label] ?? '#6b7c74' }}
                    aria-hidden="true"
                  />
                  {entry.label}
                </th>
                <td className="tabular">{entry.value.toFixed(2)}%</td>
                <td className="tabular">{perClass[entry.label]?.f1?.toFixed(3) ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="chart-note">
          Shares are model predictions. Hold-out F1 shows how reliable each class is in
          general, measured on regions the model never saw — not how reliable it is in this
          particular scene.
        </p>
      </div>
    </div>
  )
}

export function TemporalChart({ detail }: { detail: AnalysisDetail }) {
  const meanA = metric(detail.metrics, 'mean_ndvi', 'A')
  const meanB = metric(detail.metrics, 'mean_ndvi', 'B')

  if (meanA === null || meanB === null) {
    return (
      <p className="chart-empty">
        This analysis used a single period, so there is no temporal comparison. Add a second
        period to compare dates.
      </p>
    )
  }

  const obsA = detail.observations.find((o) => o.period === 'A')
  const obsB = detail.observations.find((o) => o.period === 'B')
  const data = [
    { name: obsA?.observation_date ?? 'Period A', ndvi: meanA },
    { name: obsB?.observation_date ?? 'Period B', ndvi: meanB },
  ]

  return (
    <figure className="chart-figure">
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 16, right: 8, bottom: 0, left: -18 }}>
          <XAxis dataKey="name" {...AXIS} tickLine={false} axisLine={{ stroke: GRID }} />
          <YAxis domain={[0, 1]} {...AXIS} tickLine={false} axisLine={{ stroke: GRID }} />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            itemStyle={{ color: '#eef3ec' }}
            labelStyle={{ color: '#9cb2a4' }}
            cursor={{ fill: 'rgba(29,75,51,0.08)' }}
            formatter={(value: number) => [value.toFixed(4), 'Mean NDVI']}
          />
          <Bar dataKey="ndvi" radius={[3, 3, 0, 0]} maxBarSize={72} isAnimationActive={false}>
            {data.map((_, i) => (
              <Cell key={i} fill={i === 0 ? '#7fa650' : '#1d4b33'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <figcaption className="chart-note">
        Mean NDVI was {meanA.toFixed(4)} on {data[0]!.name} and {meanB.toFixed(4)} on{' '}
        {data[1]!.name}. Both means are computed over pixels valid in both periods, so the
        bars describe the same ground.
      </figcaption>
    </figure>
  )
}

export function ChangeClassChart({ detail }: { detail: AnalysisDetail }) {
  const classes = detail.evidence?.observed?.change?.change_classes as
    | Record<string, number>
    | undefined
  if (!classes) return null

  const data = Object.entries(classes).map(([label, value]) => ({ label, value }))

  return (
    <figure className="chart-figure chart-figure--tight">
      <h4 className="chart-subtitle">Change distribution</h4>
      <ResponsiveContainer width="100%" height={188}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 28, bottom: 0, left: 4 }}>
          <XAxis
            type="number"
            {...AXIS}
            tickLine={false}
            axisLine={{ stroke: GRID }}
            unit="%"
          />
          <YAxis
            type="category"
            dataKey="label"
            width={128}
            {...AXIS}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            itemStyle={{ color: '#eef3ec' }}
            labelStyle={{ color: '#9cb2a4' }}
            cursor={{ fill: 'rgba(29,75,51,0.08)' }}
            formatter={(value: number) => [`${value.toFixed(2)}%`, 'Share of area']}
          />
          <Bar dataKey="value" radius={[0, 3, 3, 0]} maxBarSize={18} isAnimationActive={false}>
            {data.map((entry) => (
              <Cell key={entry.label} fill={CHANGE_COLOURS[entry.label] ?? '#6b7c74'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <figcaption className="visually-hidden">
        Change class distribution:{' '}
        {data.map((d) => `${d.label} ${d.value.toFixed(1)} percent`).join(', ')}.
      </figcaption>
    </figure>
  )
}
