/** Report library: every analysis that has produced results. */

import { useEffect, useState } from 'react'

import { Button, Container, EmptyState, Section, SectionHead } from '../design/primitives'
import { api } from '../services/api'
import type { AnalysisSummary } from '../types/api'
import type { Navigate } from '../app/router'

export function ReportsPage({ navigate }: { navigate: Navigate }) {
  const [items, setItems] = useState<AnalysisSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .listAnalyses(50)
      .then((r) => setItems(r.items))
      .catch((e) => setError(e instanceof Error ? e.message : 'Could not load reports.'))
  }, [])

  const complete = (items ?? []).filter((i) => i.status === 'report_ready')

  return (
    <>
      <Section tone="canopy" className="page-head">
        <Container>
          <SectionHead
            eyebrow="Reports"
            tone="dark"
            title="Nature Intelligence Reports"
            lede="Every completed analysis can produce a report covering its region, sources, methodology, results and limitations."
          />
        </Container>
      </Section>

      <Section tone="cream">
        <Container>
          {error && (
            <div className="alert alert--error" role="alert">
              <p>{error}</p>
            </div>
          )}

          {items === null && !error && <p className="chart-note">Loading…</p>}

          {items !== null && complete.length === 0 && (
            <EmptyState
              title="No completed analyses yet"
              body="Run an analysis and its report becomes available here, with full data provenance and methodology."
              action={
                <Button variant="primary" onClick={() => navigate('/analysis', 'workspace')}>
                  Launch analysis
                </Button>
              }
            />
          )}

          {complete.length > 0 && (
            <ul className="report-list">
              {complete.map((item) => (
                <li key={item.id} className="report-list__item">
                  <div className="report-list__main">
                    <h3>{item.region_name ?? `${item.area_km2.toFixed(1)} km² region`}</h3>
                    <p className="mono report-list__period">
                      {item.period_a}
                      {item.period_b ? ` → ${item.period_b}` : ''}
                    </p>
                  </div>
                  <dl className="report-list__stats">
                    <div>
                      <dt>Area</dt>
                      <dd className="mono">{item.area_km2.toFixed(1)} km²</dd>
                    </div>
                    <div>
                      <dt>Mean NDVI</dt>
                      <dd className="mono">
                        {item.mean_ndvi_a !== null ? item.mean_ndvi_a.toFixed(3) : '—'}
                      </dd>
                    </div>
                    <div>
                      <dt>Change</dt>
                      <dd className="mono">
                        {item.ndvi_change !== null
                          ? `${item.ndvi_change > 0 ? '+' : ''}${item.ndvi_change.toFixed(3)}`
                          : '—'}
                      </dd>
                    </div>
                    <div>
                      <dt>Completed</dt>
                      <dd className="mono">
                        {item.completed_at
                          ? new Date(item.completed_at).toLocaleDateString()
                          : '—'}
                      </dd>
                    </div>
                  </dl>
                  <Button onClick={() => navigate('/analysis', 'workspace')}>
                    Open in workspace
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </Container>
      </Section>
    </>
  )
}
