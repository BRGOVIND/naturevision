/** NatureVision dashboard. */

import { useEffect, useMemo, useState } from 'react'

import { AnalysisControls, type AnalysisFormValues } from './features/AnalysisControls'
import { ChangeClassChart, LandCoverChart, TemporalChart } from './features/Charts'
import { MetricCards } from './features/MetricCards'
import {
  HistoryPanel,
  InterpretationPanel,
  LayerControls,
  ObservationsPanel,
  ReportPanel,
} from './features/Panels'
import { useAnalysis } from './hooks/useAnalysis'
import { BASEMAP_LABELS, type BasemapId } from './map/basemaps'
import { MapView, type DrawMode } from './map/MapView'
import { polygonBounds } from './map/geometry'
import { api } from './services/api'
import type { GeoJSONPolygon, HealthResponse } from './types/api'
import { STATUS_LABELS, STATUS_SEQUENCE } from './types/api'

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

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [basemap, setBasemap] = useState<BasemapId>('satellite')
  const [drawMode, setDrawMode] = useState<DrawMode>('none')
  const [selection, setSelection] = useState<GeoJSONPolygon | null>(null)
  const [form, setForm] = useState<AnalysisFormValues>(DEFAULT_FORM)
  const [activeLayer, setActiveLayer] = useState<string | null>(null)
  const [opacity, setOpacity] = useState(0.85)
  const [fitBounds, setFitBounds] = useState<number[] | null>(null)

  const analysis = useAnalysis()
  const { detail, status, report } = analysis

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  // When a result arrives, show its most informative layer and frame the region.
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

  const progressIndex = useMemo(() => {
    if (!status) return -1
    return STATUS_SEQUENCE.indexOf(status.status)
  }, [status])

  function handleRun() {
    if (!selection) return
    void analysis.runAnalysis({
      region: {
        geometry: selection,
        name: form.regionName.trim() || null,
      },
      period_a: { start: form.periodAStart, end: form.periodAEnd },
      period_b: form.compareTwoPeriods
        ? { start: form.periodBStart, end: form.periodBEnd }
        : null,
      max_cloud_cover: form.maxCloudCover,
      include_land_cover: form.includeLandCover && landCoverAvailable,
      include_interpretation: form.includeInterpretation && interpretationAvailable,
    })
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <h1>NatureVision</h1>
            <p>Environmental Intelligence</p>
          </div>
        </div>
        <div className="header-status">
          <StatusPill
            label="Imagery"
            value={health?.imagery_provider ?? 'unknown'}
            ok={Boolean(health)}
          />
          <StatusPill
            label="Land-cover model"
            value={landCoverAvailable ? health!.land_cover_model : 'not installed'}
            ok={landCoverAvailable}
          />
          <StatusPill
            label="Interpretation"
            value={interpretationAvailable ? 'configured' : 'not configured'}
            ok={interpretationAvailable}
          />
        </div>
      </header>

      {analysis.error && (
        <div className="banner error" role="alert">
          <span>{analysis.error}</span>
          <button type="button" onClick={analysis.clearError} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}

      <div className="layout">
        <aside className="sidebar">
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

          <section className="control-block">
            <h3>History</h3>
            <HistoryPanel
              items={analysis.history}
              activeId={detail?.id ?? null}
              onOpen={(id) => void analysis.openAnalysis(id)}
              onDelete={(id) => void analysis.deleteAnalysis(id)}
            />
          </section>
        </aside>

        <main className="main">
          {status && (
            <section className="panel progress-panel">
              <div className="progress-head">
                <h2>
                  {STATUS_LABELS[status.status]}
                  {status.status !== 'failed' && status.status !== 'report_ready' && '…'}
                </h2>
                <span className="footnote">{status.status_detail}</span>
              </div>
              <ol className="lifecycle">
                {STATUS_SEQUENCE.map((step, index) => (
                  <li
                    key={step}
                    className={
                      status.status === 'failed'
                        ? index <= progressIndex
                          ? 'failed'
                          : ''
                        : index < progressIndex
                          ? 'done'
                          : index === progressIndex
                            ? 'current'
                            : ''
                    }
                  >
                    <span className="dot" />
                    <span className="lifecycle-label">{STATUS_LABELS[step]}</span>
                  </li>
                ))}
              </ol>
              <div className="progress-bar">
                <div
                  className={`progress-fill ${status.status}`}
                  style={{ width: `${Math.round(status.progress * 100)}%` }}
                />
              </div>
            </section>
          )}

          {detail && detail.status === 'report_ready' && <MetricCards detail={detail} />}

          <section className="panel map-panel">
            <div className="panel-head">
              <h2>Map</h2>
              <div className="basemap-toggle">
                {(Object.keys(BASEMAP_LABELS) as BasemapId[]).map((id) => (
                  <button
                    key={id}
                    type="button"
                    className={basemap === id ? 'chipbtn active' : 'chipbtn'}
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
          </section>

          {detail && detail.status === 'report_ready' && (
            <>
              <div className="panel-row">
                <section className="panel">
                  <h2>Land-cover distribution</h2>
                  <LandCoverChart detail={detail} />
                </section>
                <section className="panel">
                  <h2>Period comparison</h2>
                  <TemporalChart detail={detail} />
                  <ChangeClassChart detail={detail} />
                </section>
              </div>

              <section className="panel">
                <h2>Satellite observations</h2>
                <ObservationsPanel detail={detail} />
              </section>

              <section className="panel">
                <h2>Interpretation</h2>
                <InterpretationPanel report={report} busy={analysis.reportBusy} />
              </section>

              <section className="panel">
                <h2>Nature Intelligence Report</h2>
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
              </section>
            </>
          )}

          {!detail && !status && (
            <section className="panel intro">
              <h2>Get started</h2>
              <ol className="intro-steps">
                <li>Draw a box or polygon over an area of interest.</li>
                <li>Pick one period, or two to compare dates.</li>
                <li>
                  Run the analysis. Sentinel-2 imagery is retrieved, cloud-masked and
                  processed into NDVI, change detection and land cover.
                </li>
                <li>Generate a report and export it.</li>
              </ol>
              <p className="footnote">
                Measured values come from deterministic raster processing. Land-cover
                shares are model predictions and are labelled as such. Written
                interpretation is generated only from measured evidence and is
                validated against it before display.
              </p>
            </section>
          )}
        </main>
      </div>
    </div>
  )
}

function StatusPill({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <div className={`status-pill ${ok ? 'ok' : 'warn'}`}>
      <span className="status-pill-label">{label}</span>
      <span className="status-pill-value">{value}</span>
    </div>
  )
}
