/** The analysis workspace: region selection, execution, results. */

import { useEffect, useMemo, useState } from 'react'

import { AnalysisControls, type AnalysisFormValues } from '../features/AnalysisControls'
import { ChangeClassChart, LandCoverChart, TemporalChart } from '../features/Charts'
import { MetricCards } from '../features/MetricCards'
import {
  HistoryPanel,
  InterpretationPanel,
  LayerControls,
  ObservationsPanel,
  ReportPanel,
} from '../features/Panels'
import { LifecycleTracker } from '../features/LifecycleTracker'
import { Button, Container, EmptyState, Section, SectionHead } from '../design/primitives'
import { useAnalysis } from '../hooks/useAnalysis'
import { BASEMAP_LABELS, type BasemapId } from '../map/basemaps'
import { MapView, type DrawMode } from '../map/MapView'
import { polygonBounds } from '../map/geometry'
import { api } from '../services/api'
import type { GeoJSONPolygon, HealthResponse } from '../types/api'

const DEFAULT_FORM: AnalysisFormValues = {
  periodAStart: '2021-01-01',
  periodAEnd: '2021-02-28',
  periodBStart: '2024-01-01',
  periodBEnd: '2024-02-29',
  compareTwoPeriods: true,
  maxCloudCover: 25,
  includeLandCover: true,
  includeInterpretation: true,
  regionName: '',
}

export function WorkspacePage({ health }: { health: HealthResponse | null }) {
  const [basemap, setBasemap] = useState<BasemapId>('satellite')
  const [drawMode, setDrawMode] = useState<DrawMode>('none')
  const [selection, setSelection] = useState<GeoJSONPolygon | null>(null)
  const [form, setForm] = useState<AnalysisFormValues>(DEFAULT_FORM)
  const [activeLayer, setActiveLayer] = useState<string | null>(null)
  const [opacity, setOpacity] = useState(0.85)
  const [fitBounds, setFitBounds] = useState<number[] | null>(null)

  const analysis = useAnalysis()
  const { detail, status, report } = analysis
  const { openAnalysis } = analysis

  // An analysis id in the query string opens that run directly, so a result
  // can be linked to and returned to rather than only reached by clicking
  // through history.
  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get('analysis')
    if (requested) void openAnalysis(requested)
  }, [openAnalysis])

  useEffect(() => {
    if (!detail) return
    const preferred = ['change', 'land_cover', 'ndvi_b', 'ndvi_a']
    const chosen = preferred.find((key) => detail.layers.some((l) => l.key === key))
    setActiveLayer(chosen ?? detail.layers[0]?.key ?? null)
    setSelection(detail.region.geometry)
    setFitBounds(detail.region.bbox)
  }, [detail])

  const landCoverAvailable = Boolean(health?.checks?.land_cover_model?.ok)
  const interpretationAvailable = Boolean(health?.checks?.interpretation?.ok)
  const visionAvailable = Boolean(health?.checks?.interpretation?.vision_enabled)

  const layers = detail?.layers ?? []
  const complete = detail?.status === 'report_ready'
  const running = Boolean(status && !['report_ready', 'failed'].includes(status.status))

  const resultsRef = useMemo(() => ({ id: 'findings' }), [])

  function handleRun() {
    if (!selection) return
    void analysis.runAnalysis({
      region: { geometry: selection, name: form.regionName.trim() || null },
      period_a: { start: form.periodAStart, end: form.periodAEnd },
      period_b: form.compareTwoPeriods
        ? { start: form.periodBStart, end: form.periodBEnd }
        : null,
      max_cloud_cover: form.maxCloudCover,
      include_land_cover: form.includeLandCover && landCoverAvailable,
      include_interpretation: form.includeInterpretation && interpretationAvailable,
    })
    requestAnimationFrame(() =>
      document.getElementById('run-status')?.scrollIntoView({ block: 'center' }),
    )
  }

  return (
    <>
      <Section id="workspace" tone="canopy" className="workspace">
        <Container wide>
          <SectionHead
            tone="dark"
            title="Define a region and an observation window"
            lede="The pipeline runs on live Sentinel-2 data. Nothing here is precomputed."
          />

          <div className="workspace__grid">
            {/* --- configuration ------------------------------------- */}
            <div className="workspace__config">
              <AnalysisControls
                selection={selection}
                drawMode={drawMode}
                values={form}
                busy={analysis.busy}
                landCoverAvailable={landCoverAvailable}
                interpretationAvailable={interpretationAvailable}
                onChange={setForm}
                onDrawModeChange={setDrawMode}
                onClearSelection={() => {
                  setSelection(null)
                  setDrawMode('none')
                }}
                onSubmit={handleRun}
              />
            </div>

            {/* --- map ------------------------------------------------ */}
            <div className="workspace__map">
              <div className="map-frame">
                <div className="map-frame__bar">
                  <div className="map-frame__title">
                    <h3>{complete ? 'Environmental map' : 'Region'}</h3>
                    <p className="map-frame__hint">
                      {selection
                        ? complete
                          ? 'Switch layers to inspect the analysed rasters'
                          : 'Region selected — configure periods and run'
                        : 'Draw a box or polygon to begin'}
                    </p>
                  </div>
                  <div className="basemap-toggle" role="group" aria-label="Basemap">
                    {(Object.keys(BASEMAP_LABELS) as BasemapId[]).map((id) => (
                      <button
                        key={id}
                        type="button"
                        className={basemap === id ? 'seg seg--active' : 'seg'}
                        aria-pressed={basemap === id}
                        onClick={() => setBasemap(id)}
                      >
                        {BASEMAP_LABELS[id]}
                      </button>
                    ))}
                  </div>
                </div>

                <MapView
                  basemap={basemap}
                  drawMode={drawMode}
                  selection={selection}
                  onSelectionChange={(polygon) => {
                    setSelection(polygon)
                    if (polygon) setFitBounds(polygonBounds(polygon))
                  }}
                  onDrawModeChange={setDrawMode}
                  layers={layers}
                  activeLayerKey={activeLayer}
                  layerOpacity={opacity}
                  fitBounds={fitBounds}
                />

                {layers.length > 0 && (
                  <LayerControls
                    layers={layers}
                    activeKey={activeLayer}
                    opacity={opacity}
                    onSelect={setActiveLayer}
                    onOpacityChange={setOpacity}
                  />
                )}
              </div>
            </div>
          </div>
        </Container>
      </Section>

      {/* --- lifecycle ------------------------------------------------- */}
      {status && (
        <Section id="run-status" tone="charcoal" className="status-section">
          <Container wide>
            <LifecycleTracker status={status} />
            {analysis.error && (
              <div className="alert alert--error" role="alert">
                <div>
                  <strong>Analysis could not complete</strong>
                  <p>{analysis.error}</p>
                </div>
                <Button variant="quiet" onClick={analysis.clearError}>
                  Dismiss
                </Button>
              </div>
            )}
          </Container>
        </Section>
      )}

      {/* --- results ---------------------------------------------------- */}
      {complete && detail && (
        <>
          <Section id={resultsRef.id} tone="cream">
            <Container wide>
              <SectionHead
                title={detail.region.name ?? 'Analysis results'}
                lede={`${detail.period_a}${detail.period_b ? ` compared with ${detail.period_b}` : ''} · ${detail.region.area_km2.toFixed(1)} km²`}
              />
              <MetricCards detail={detail} />
            </Container>
          </Section>

          <Section id="composition" tone="cream" flush>
            <Container wide>
              <div className="results-grid">
                <article className="panel">
                  <header className="panel__head">
                    <h3>Land-cover composition</h3>
                    <p>Predicted share of classified pixels</p>
                  </header>
                  <LandCoverChart detail={detail} />
                </article>

                <article className="panel">
                  <header className="panel__head">
                    <h3>Temporal comparison</h3>
                    <p>Mean vegetation index by acquisition</p>
                  </header>
                  <TemporalChart detail={detail} />
                  <ChangeClassChart detail={detail} />
                </article>
              </div>
            </Container>
          </Section>

          <Section id="observations" tone="cream" flush>
            <Container wide>
              <article className="panel">
                <header className="panel__head">
                  <h3>Satellite observations</h3>
                  <p>The scenes these results were computed from</p>
                </header>
                <ObservationsPanel detail={detail} />
              </article>
            </Container>
          </Section>

          <Section id="interpretation" tone="charcoal">
            <Container wide>
              <SectionHead
                tone="dark"
                title="Interpretation of the measured evidence"
                lede="Generated from the values above and validated against them before display."
              />
              <InterpretationPanel
                report={report}
                busy={analysis.reportBusy}
                available={interpretationAvailable}
                onGenerate={() => void analysis.generateReport(detail.id, false, false)}
              />
            </Container>
          </Section>

          <Section id="report" tone="cream">
            <Container wide>
              <SectionHead
                title="Nature Intelligence Report"
                lede="Thirteen sections, each tagged with where its content came from."
              />
              <ReportPanel
                report={report}
                detail={detail}
                busy={analysis.reportBusy}
                visionAvailable={visionAvailable}
                onGenerate={(includeVisual, regenerate) =>
                  void analysis.generateReport(detail.id, includeVisual, regenerate)
                }
                exportUrl={api.reportExportUrl}
              />
            </Container>
          </Section>
        </>
      )}

      {/* --- empty state ------------------------------------------------- */}
      {!detail && !running && (
        <Section tone="cream">
          <Container>
            <EmptyState
              title="No analysis yet"
              body="Draw a region on the map above, choose one period or two, and run the analysis. Results appear here with their full provenance."
              action={
                <Button
                  variant="secondary"
                  onClick={() =>
                    document.getElementById('workspace')?.scrollIntoView({ block: 'start' })
                  }
                >
                  Back to region selection
                </Button>
              }
            />
          </Container>
        </Section>
      )}

      {/* --- history ------------------------------------------------------ */}
      <Section id="history" tone="canopy">
        <Container wide>
          <SectionHead
            tone="dark"
            title="Previous analyses"
            lede="Stored server-side with their metrics and observations. Reopen any run to inspect it."
          />
          <HistoryPanel
            items={analysis.history}
            activeId={detail?.id ?? null}
            onOpen={(id) => {
              void analysis.openAnalysis(id)
              requestAnimationFrame(() =>
                document.getElementById('findings')?.scrollIntoView({ block: 'start' }),
              )
            }}
            onDelete={(id) => void analysis.deleteAnalysis(id)}
          />
        </Container>
      </Section>
    </>
  )
}
