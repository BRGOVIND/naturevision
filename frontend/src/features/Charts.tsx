/** Land-cover distribution, temporal comparison and change-class charts. */

import {
  Bar,
  BarChart,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { AnalysisDetail, Metric } from '../types/api'

const AXIS = { stroke: '#7d8f86', fontSize: 11 }
const GRID_COLOUR = '#22322c'

const TOOLTIP_STYLE = {
  background: '#12201b',
  border: '1px solid #2c3f37',
  borderRadius: 6,
  color: '#e8f0ec',
  fontSize: 12,
}

/** Class colours are supplied by the backend so map, legend and chart agree. */
const CLASS_COLOURS: Record<string, string> = {
  Forest: '#1b7f3b',
  Agriculture: '#d9a441',
  Water: '#2b6cb0',
  'Urban / built-up': '#a02c2c',
  'Bare land': '#8a8f98',
}

function metric(metrics: Metric[], key: string, period?: string): number | null {
  const found = metrics.find(
    (m) => m.key === key && (period === undefined || m.period === period),
  )
  return found?.value ?? null
}

export function LandCoverChart({ detail }: { detail: AnalysisDetail }) {
  const prediction = detail.predictions[0]
  if (!prediction) {
    return (
      <p className="empty-note">
        Land-cover classification was not part of this analysis, or no trained model
        was installed when it ran.
      </p>
    )
  }

  const data = Object.entries(prediction.class_distribution)
    .map(([label, value]) => ({ label, value: Number(value) }))
    .filter((d) => d.value > 0)
    .sort((a, b) => b.value - a.value)

  const perClass = prediction.evaluation_metrics?.per_class_metrics ?? {}

  return (
    <div className="chart-split">
      <div className="chart-box">
        <ResponsiveContainer width="100%" height={230}>
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="label"
              innerRadius={52}
              outerRadius={88}
              paddingAngle={1.5}
              stroke="#0d1512"
            >
              {data.map((entry) => (
                <Cell key={entry.label} fill={CLASS_COLOURS[entry.label] ?? '#6b7c74'} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={(value: number) => [`${value.toFixed(2)}%`, 'Share']}
            />
            <Legend
              verticalAlign="bottom"
              height={44}
              wrapperStyle={{ fontSize: 11, color: '#b7c6be' }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="class-table">
        <table>
          <thead>
            <tr>
              <th>Class</th>
              <th>Share</th>
              <th>Held-out F1</th>
            </tr>
          </thead>
          <tbody>
            {data.map((entry) => (
              <tr key={entry.label}>
                <td>
                  <span
                    className="swatch"
                    style={{ background: CLASS_COLOURS[entry.label] ?? '#6b7c74' }}
                  />
                  {entry.label}
                </td>
                <td>{entry.value.toFixed(2)}%</td>
                <td>{perClass[entry.label]?.f1?.toFixed(3) ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="footnote">
          Shares are model predictions. Per-class F1 is measured on spatially
          held-out regions and indicates how reliable each class is in general,
          not in this specific scene.
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
      <p className="empty-note">
        This analysis used a single period, so there is no temporal comparison.
        Add a second period to compare dates.
      </p>
    )
  }

  const observationA = detail.observations.find((o) => o.period === 'A')
  const observationB = detail.observations.find((o) => o.period === 'B')

  const data = [
    { name: observationA?.observation_date ?? 'Period A', ndvi: meanA },
    { name: observationB?.observation_date ?? 'Period B', ndvi: meanB },
  ]

  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height={230}>
        <BarChart data={data} margin={{ top: 12, right: 12, bottom: 4, left: -12 }}>
          <XAxis dataKey="name" {...AXIS} tickLine={false} axisLine={{ stroke: GRID_COLOUR }} />
          <YAxis
            domain={[0, 1]}
            {...AXIS}
            tickLine={false}
            axisLine={{ stroke: GRID_COLOUR }}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            cursor={{ fill: '#1a2a24' }}
            formatter={(value: number) => [value.toFixed(4), 'Mean NDVI']}
          />
          <Bar dataKey="ndvi" fill="#4ade80" radius={[4, 4, 0, 0]} maxBarSize={80} />
        </BarChart>
      </ResponsiveContainer>
      <p className="footnote">
        Means are computed over pixels valid in both periods after cloud masking,
        so the two bars describe the same ground.
      </p>
    </div>
  )
}

export function ChangeClassChart({ detail }: { detail: AnalysisDetail }) {
  const classes = detail.evidence?.observed?.change?.change_classes as
    | Record<string, number>
    | undefined
  if (!classes) return null

  const palette: Record<string, string> = {
    'Significant decrease': '#961e20',
    'Moderate decrease': '#e2846a',
    Stable: '#8fa39a',
    'Moderate increase': '#8cc28a',
    'Significant increase': '#206e3a',
  }
  const data = Object.entries(classes).map(([label, value]) => ({ label, value }))

  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height={210}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 8, right: 20, bottom: 4, left: 8 }}
        >
          <XAxis type="number" {...AXIS} tickLine={false} axisLine={{ stroke: GRID_COLOUR }} unit="%" />
          <YAxis
            type="category"
            dataKey="label"
            width={132}
            {...AXIS}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            cursor={{ fill: '#1a2a24' }}
            formatter={(value: number) => [`${value.toFixed(2)}%`, 'Share of area']}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={22}>
            {data.map((entry) => (
              <Cell key={entry.label} fill={palette[entry.label] ?? '#6b7c74'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
