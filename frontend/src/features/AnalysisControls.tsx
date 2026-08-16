/** Region, period and configuration controls. */

import { useState } from 'react'

import { Button, Spinner } from '../design/primitives'
import { api } from '../services/api'
import type { GeoJSONPolygon, Observation } from '../types/api'
import { describeCentre, formatBounds, polygonBounds, validatePolygon } from '../map/geometry'
import type { DrawMode } from '../map/MapView'

export interface AnalysisFormValues {
  periodAStart: string
  periodAEnd: string
  periodBStart: string
  periodBEnd: string
  compareTwoPeriods: boolean
  maxCloudCover: number
  includeLandCover: boolean
  includeInterpretation: boolean
  regionName: string
}

interface Props {
  selection: GeoJSONPolygon | null
  drawMode: DrawMode
  values: AnalysisFormValues
  busy: boolean
  landCoverAvailable: boolean
  interpretationAvailable: boolean
  onChange: (values: AnalysisFormValues) => void
  onDrawModeChange: (mode: DrawMode) => void
  onClearSelection: () => void
  onSubmit: () => void
}

export function AnalysisControls({
  selection,
  drawMode,
  values,
  busy,
  landCoverAvailable,
  interpretationAvailable,
  onChange,
  onDrawModeChange,
  onClearSelection,
  onSubmit,
}: Props) {
  const [observations, setObservations] = useState<Observation[] | null>(null)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)

  const validation = validatePolygon(selection)
  const bounds = selection ? polygonBounds(selection) : null

  const set = <K extends keyof AnalysisFormValues>(key: K, value: AnalysisFormValues[K]) =>
    onChange({ ...values, [key]: value })

  const datesValid =
    values.periodAStart <= values.periodAEnd &&
    (!values.compareTwoPeriods || values.periodBStart <= values.periodBEnd)

  const canRun = validation.valid && datesValid && !busy

  async function previewObservations() {
    if (!selection) return
    setSearching(true)
    setSearchError(null)
    try {
      const response = await api.searchImagery({
        region: { geometry: selection },
        start_date: values.periodAStart,
        end_date: values.compareTwoPeriods ? values.periodBEnd : values.periodAEnd,
        max_cloud_cover: values.maxCloudCover,
        limit: 20,
      })
      setObservations(response.observations)
    } catch (error) {
      setObservations(null)
      setSearchError(error instanceof Error ? error.message : 'The imagery search failed.')
    } finally {
      setSearching(false)
    }
  }

  return (
    <div className="controls">
      {/* --- 1. Region ------------------------------------------------- */}
      <section className="control-card">
        <header className="control-card__head">
          <span className="control-card__step mono">01</span>
          <h3>Region</h3>
        </header>

        <div className="control-card__body">
          <div className="draw-modes" role="group" aria-label="Selection tool">
            <button
              type="button"
              className={drawMode === 'rectangle' ? 'seg seg--active' : 'seg'}
              aria-pressed={drawMode === 'rectangle'}
              onClick={() => onDrawModeChange(drawMode === 'rectangle' ? 'none' : 'rectangle')}
            >
              Box
            </button>
            <button
              type="button"
              className={drawMode === 'polygon' ? 'seg seg--active' : 'seg'}
              aria-pressed={drawMode === 'polygon'}
              onClick={() => onDrawModeChange(drawMode === 'polygon' ? 'none' : 'polygon')}
            >
              Polygon
            </button>
            <button type="button" className="seg" onClick={onClearSelection} disabled={!selection}>
              Clear
            </button>
          </div>

          {selection && bounds ? (
            <dl className="region-facts">
              <div>
                <dt>Centre</dt>
                <dd className="mono">{describeCentre(bounds)}</dd>
              </div>
              <div>
                <dt>Area</dt>
                <dd className="mono">{validation.areaKm2.toFixed(2)} km²</dd>
              </div>
              <div className="region-facts__wide">
                <dt>Bounds</dt>
                <dd className="mono">{formatBounds(bounds)}</dd>
              </div>
            </dl>
          ) : (
            <p className="control-hint">
              Draw a box or polygon on the map. Regions between 0.01 and 2,500 km² can be
              analysed in one run.
            </p>
          )}

          {validation.message && selection && (
            <p className="control-error" role="alert">
              {validation.message}
            </p>
          )}

          <label className="field">
            <span>Region name</span>
            <input
              type="text"
              value={values.regionName}
              maxLength={200}
              placeholder="Optional"
              onChange={(e) => set('regionName', e.target.value)}
            />
          </label>
        </div>
      </section>

      {/* --- 2. Periods ------------------------------------------------- */}
      <section className="control-card">
        <header className="control-card__head">
          <span className="control-card__step mono">02</span>
          <h3>Observation periods</h3>
        </header>

        <div className="control-card__body">
          <fieldset className="period">
            <legend>Period A</legend>
            <div className="field-row">
              <label className="field">
                <span>From</span>
                <input
                  type="date"
                  value={values.periodAStart}
                  onChange={(e) => set('periodAStart', e.target.value)}
                />
              </label>
              <label className="field">
                <span>To</span>
                <input
                  type="date"
                  value={values.periodAEnd}
                  onChange={(e) => set('periodAEnd', e.target.value)}
                />
              </label>
            </div>
          </fieldset>

          <label className="switch">
            <input
              type="checkbox"
              checked={values.compareTwoPeriods}
              onChange={(e) => set('compareTwoPeriods', e.target.checked)}
            />
            <span className="switch__track" aria-hidden="true">
              <span className="switch__thumb" />
            </span>
            <span className="switch__label">Compare against a second period</span>
          </label>

          {values.compareTwoPeriods && (
            <fieldset className="period">
              <legend>Period B</legend>
              <div className="field-row">
                <label className="field">
                  <span>From</span>
                  <input
                    type="date"
                    value={values.periodBStart}
                    onChange={(e) => set('periodBStart', e.target.value)}
                  />
                </label>
                <label className="field">
                  <span>To</span>
                  <input
                    type="date"
                    value={values.periodBEnd}
                    onChange={(e) => set('periodBEnd', e.target.value)}
                  />
                </label>
              </div>
            </fieldset>
          )}

          {!datesValid && (
            <p className="control-error" role="alert">
              Each period must start on or before it ends.
            </p>
          )}
        </div>
      </section>

      {/* --- 3. Configuration --------------------------------------------- */}
      <section className="control-card">
        <header className="control-card__head">
          <span className="control-card__step mono">03</span>
          <h3>Configuration</h3>
        </header>

        <div className="control-card__body">
          <label className="field">
            <span>
              Maximum scene cloud cover
              <strong className="mono field__value">{values.maxCloudCover}%</strong>
            </span>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={values.maxCloudCover}
              onChange={(e) => set('maxCloudCover', Number(e.target.value))}
            />
          </label>
          <p className="control-hint">
            A wider window and a higher limit make a usable scene more likely. Cloudy pixels
            are masked regardless, using the Level-2A scene classification.
          </p>

          <label className="switch">
            <input
              type="checkbox"
              checked={values.includeLandCover && landCoverAvailable}
              disabled={!landCoverAvailable}
              onChange={(e) => set('includeLandCover', e.target.checked)}
            />
            <span className="switch__track" aria-hidden="true">
              <span className="switch__thumb" />
            </span>
            <span className="switch__label">
              Land-cover classification
              {!landCoverAvailable && <em> — no trained model installed</em>}
            </span>
          </label>

          <label className="switch">
            <input
              type="checkbox"
              checked={values.includeInterpretation && interpretationAvailable}
              disabled={!interpretationAvailable}
              onChange={(e) => set('includeInterpretation', e.target.checked)}
            />
            <span className="switch__track" aria-hidden="true">
              <span className="switch__thumb" />
            </span>
            <span className="switch__label">
              Written interpretation
              {!interpretationAvailable && <em> — no provider configured</em>}
            </span>
          </label>
        </div>
      </section>

      {/* --- Run ------------------------------------------------------------ */}
      <div className="control-run">
        <Button variant="primary" size="lg" full onClick={onSubmit} disabled={!canRun}>
          {busy ? 'Running analysis…' : 'Run analysis'}
        </Button>
        <Button variant="quiet" full onClick={previewObservations} disabled={!selection || searching}>
          {searching ? 'Searching…' : 'Preview available imagery'}
        </Button>

        {searchError && (
          <p className="control-error" role="alert">
            {searchError}
          </p>
        )}

        {searching && <Spinner label="Searching the satellite catalogue" />}

        {observations && !searching && (
          <div className="preview">
            <p className="preview__count">
              {observations.length} observation{observations.length === 1 ? '' : 's'} match this
              region and window
            </p>
            <ul className="preview__list">
              {observations.slice(0, 5).map((o) => (
                <li key={o.source_id}>
                  <span className="mono">{o.observation_date}</span>
                  <span className="preview__cloud">
                    {o.cloud_cover_percent !== null
                      ? `${o.cloud_cover_percent.toFixed(1)}% cloud`
                      : 'cloud unknown'}
                  </span>
                  {o.region_coverage !== null && (
                    <span className="preview__coverage mono">
                      {Math.round(o.region_coverage * 100)}%
                    </span>
                  )}
                </li>
              ))}
            </ul>
            <p className="control-hint">
              The pipeline selects by regional coverage first, then cloud cover.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
