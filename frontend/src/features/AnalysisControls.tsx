/** Region selection, period configuration and imagery preview controls. */

import { useState } from 'react'

import { api } from '../services/api'
import type { GeoJSONPolygon, Observation } from '../types/api'
import {
  describeCentre,
  formatBounds,
  polygonBounds,
  validatePolygon,
} from '../map/geometry'
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
      setSearchError(
        error instanceof Error ? error.message : 'The imagery search failed.',
      )
    } finally {
      setSearching(false)
    }
  }

  return (
    <div className="controls">
      <section className="control-block">
        <h3>1. Select a region</h3>
        <div className="button-row">
          <button
            type="button"
            className={drawMode === 'rectangle' ? 'active' : undefined}
            onClick={() => onDrawModeChange(drawMode === 'rectangle' ? 'none' : 'rectangle')}
          >
            Draw box
          </button>
          <button
            type="button"
            className={drawMode === 'polygon' ? 'active' : undefined}
            onClick={() => onDrawModeChange(drawMode === 'polygon' ? 'none' : 'polygon')}
          >
            Draw polygon
          </button>
          <button type="button" onClick={onClearSelection} disabled={!selection}>
            Clear
          </button>
        </div>

        {selection && bounds ? (
          <dl className="region-facts">
            <div>
              <dt>Centre</dt>
              <dd>{describeCentre(bounds)}</dd>
            </div>
            <div>
              <dt>Area</dt>
              <dd>{validation.areaKm2.toFixed(2)} km²</dd>
            </div>
            <div>
              <dt>Bounds</dt>
              <dd className="mono small">{formatBounds(bounds)}</dd>
            </div>
            <div>
              <dt>CRS</dt>
              <dd>EPSG:4326</dd>
            </div>
          </dl>
        ) : (
          <p className="footnote">
            Draw a box or polygon on the map. Regions between 0.01 and 2,500 km² can be
            analysed in one run.
          </p>
        )}

        {validation.message && selection && (
          <p className="inline-error">{validation.message}</p>
        )}

        <label className="field">
          <span>Region name (optional)</span>
          <input
            type="text"
            value={values.regionName}
            maxLength={200}
            placeholder="e.g. Northern catchment"
            onChange={(e) => set('regionName', e.target.value)}
          />
        </label>
      </section>

      <section className="control-block">
        <h3>2. Choose periods</h3>
        <div className="field-row">
          <label className="field">
            <span>Period A start</span>
            <input
              type="date"
              value={values.periodAStart}
              onChange={(e) => set('periodAStart', e.target.value)}
            />
          </label>
          <label className="field">
            <span>Period A end</span>
            <input
              type="date"
              value={values.periodAEnd}
              onChange={(e) => set('periodAEnd', e.target.value)}
            />
          </label>
        </div>

        <label className="checkbox">
          <input
            type="checkbox"
            checked={values.compareTwoPeriods}
            onChange={(e) => set('compareTwoPeriods', e.target.checked)}
          />
          <span>Compare against a second period</span>
        </label>

        {values.compareTwoPeriods && (
          <div className="field-row">
            <label className="field">
              <span>Period B start</span>
              <input
                type="date"
                value={values.periodBStart}
                onChange={(e) => set('periodBStart', e.target.value)}
              />
            </label>
            <label className="field">
              <span>Period B end</span>
              <input
                type="date"
                value={values.periodBEnd}
                onChange={(e) => set('periodBEnd', e.target.value)}
              />
            </label>
          </div>
        )}

        {!datesValid && (
          <p className="inline-error">Each period must start on or before it ends.</p>
        )}

        <label className="field">
          <span>
            Maximum scene cloud cover: <strong>{values.maxCloudCover}%</strong>
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
        <p className="footnote">
          A wider window and a higher cloud limit make it more likely that a usable
          scene exists. Cloudy pixels are removed regardless, using the Level-2A
          scene classification.
        </p>
      </section>

      <section className="control-block">
        <h3>3. Options</h3>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={values.includeLandCover && landCoverAvailable}
            disabled={!landCoverAvailable}
            onChange={(e) => set('includeLandCover', e.target.checked)}
          />
          <span>
            Run land-cover classification
            {!landCoverAvailable && ' (no trained model installed)'}
          </span>
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={values.includeInterpretation && interpretationAvailable}
            disabled={!interpretationAvailable}
            onChange={(e) => set('includeInterpretation', e.target.checked)}
          />
          <span>
            Generate written interpretation
            {!interpretationAvailable && ' (no language provider configured)'}
          </span>
        </label>
      </section>

      <section className="control-block">
        <div className="button-row">
          <button
            type="button"
            onClick={previewObservations}
            disabled={!selection || searching}
          >
            {searching ? 'Searching…' : 'Preview available imagery'}
          </button>
          <button type="button" className="primary run" onClick={onSubmit} disabled={!canRun}>
            {busy ? 'Running analysis…' : 'Run analysis'}
          </button>
        </div>
        {searchError && <p className="inline-error">{searchError}</p>}
        {observations && (
          <div className="observation-preview">
            <p className="footnote">
              {observations.length} observation{observations.length === 1 ? '' : 's'} match
              this region and window. The pipeline picks the best by coverage, then cloud.
            </p>
            <ul>
              {observations.slice(0, 6).map((o) => (
                <li key={o.source_id}>
                  <span className="mono small">{o.observation_date}</span>
                  <span>
                    {o.cloud_cover_percent !== null
                      ? `${o.cloud_cover_percent.toFixed(1)}% cloud`
                      : 'cloud unknown'}
                  </span>
                  <span>
                    {o.region_coverage !== null
                      ? `${Math.round(o.region_coverage * 100)}% coverage`
                      : ''}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>
    </div>
  )
}
