/**
 * Interactive analysis map.
 *
 * Handles genuine geographic interaction: pan, zoom, rectangle and polygon
 * selection, and georeferenced raster overlays produced by the backend. Overlay
 * images are placed with the exact WGS84 corner coordinates the server returned
 * with each layer, so what is drawn lines up with the pixels that were analysed.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import maplibregl, { type Map as MapLibreMap, type MapMouseEvent } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'

import { BASEMAP_STYLES, type BasemapId } from './basemaps'
import { bboxToPolygon, polygonBounds, polygonFromPoints, type LngLat } from './geometry'
import type { GeoJSONPolygon, LayerReference } from '../types/api'

export type DrawMode = 'none' | 'rectangle' | 'polygon'

const SELECTION_SOURCE = 'nv-selection'
const DRAFT_SOURCE = 'nv-draft'
const OVERLAY_PREFIX = 'nv-overlay-'

interface MapViewProps {
  basemap: BasemapId
  drawMode: DrawMode
  selection: GeoJSONPolygon | null
  onSelectionChange: (polygon: GeoJSONPolygon | null) => void
  onDrawModeChange: (mode: DrawMode) => void
  layers: LayerReference[]
  activeLayerKey: string | null
  layerOpacity: number
  fitBounds: number[] | null
}

export function MapView({
  basemap,
  drawMode,
  selection,
  onSelectionChange,
  onDrawModeChange,
  layers,
  activeLayerKey,
  layerOpacity,
  fitBounds,
}: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const [ready, setReady] = useState(false)

  // Drawing state lives in refs so map event handlers registered once do not
  // capture stale values from a previous render.
  const drawModeRef = useRef(drawMode)
  const draftPointsRef = useRef<LngLat[]>([])
  const rectangleStartRef = useRef<LngLat | null>(null)
  const [vertexCount, setVertexCount] = useState(0)

  useEffect(() => {
    drawModeRef.current = drawMode
    if (drawMode === 'none') {
      draftPointsRef.current = []
      rectangleStartRef.current = null
      setVertexCount(0)
      updateDraft(mapRef.current, null)
    }
    const map = mapRef.current
    if (map) {
      map.getCanvas().style.cursor = drawMode === 'none' ? '' : 'crosshair'
      map.dragPan[drawMode === 'rectangle' ? 'disable' : 'enable']()
    }
  }, [drawMode])

  // --- map bootstrap ----------------------------------------------------
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: BASEMAP_STYLES[basemap],
      center: [76.66, 10.16],
      zoom: 10,
      attributionControl: { compact: true },
      // Analysis regions are local, not world-spanning, so extra copies of
      // the map only mean extra basemap tile requests for no visible gain.
      renderWorldCopies: false,
    })
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), 'top-right')
    map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left')
    map.addControl(
      new maplibregl.GeolocateControl({ trackUserLocation: false }),
      'top-right',
    )

    map.on('load', () => {
      ensureVectorLayers(map)
      setReady(true)
    })

    // Basemap raster tiles come from a third-party server outside this
    // app's control (network errors, rate limiting, transient corrupt
    // responses). MapLibre surfaces every failed/undecodable tile as an
    // `error` event; without a listener those become unhandled console
    // errors that read as if the map itself broke. A tile source error
    // just means one tile stays blank — it is not an application error,
    // so it is logged (not hidden) and otherwise left to MapLibre's own
    // retry-on-next-pan behaviour. Errors from other sources (the
    // Sentinel-2 overlay layers, the selection/draft vector sources)
    // still surface exactly as before.
    map.on('error', (event) => {
      // `sourceId` is set by MapLibre on source/tile errors at runtime but
      // is not part of the public ErrorEvent type, hence the narrow cast.
      const sourceId = (event as unknown as { sourceId?: string }).sourceId
      if (sourceId === 'basemap') {
        console.warn('basemap tile failed to load', event.error)
        return
      }
      console.error('map error', event.error)
    })

    mapRef.current = map
    return () => {
      map.remove()
      mapRef.current = null
    }
    // Basemap changes are handled by a dedicated effect; this must run once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // --- drawing interaction ------------------------------------------------
  const handleClick = useCallback(
    (event: MapMouseEvent) => {
      if (drawModeRef.current !== 'polygon') return
      draftPointsRef.current = [...draftPointsRef.current, event.lngLat]
      setVertexCount(draftPointsRef.current.length)
      updateDraft(mapRef.current, draftPointsRef.current)
    },
    [],
  )

  const finishPolygon = useCallback(() => {
    const polygon = polygonFromPoints(draftPointsRef.current)
    draftPointsRef.current = []
    setVertexCount(0)
    updateDraft(mapRef.current, null)
    if (polygon) {
      onSelectionChange(polygon)
      onDrawModeChange('none')
    }
  }, [onSelectionChange, onDrawModeChange])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return

    const onMouseDown = (event: MapMouseEvent) => {
      if (drawModeRef.current !== 'rectangle') return
      rectangleStartRef.current = event.lngLat
    }
    const onMouseMove = (event: MapMouseEvent) => {
      const start = rectangleStartRef.current
      if (drawModeRef.current !== 'rectangle' || !start) return
      updateDraftPolygon(map, rectangleBetween(start, event.lngLat))
    }
    const onMouseUp = (event: MapMouseEvent) => {
      const start = rectangleStartRef.current
      if (drawModeRef.current !== 'rectangle' || !start) return
      rectangleStartRef.current = null
      const polygon = rectangleBetween(start, event.lngLat)
      updateDraft(map, null)
      // A click without a drag is not a rectangle; ignore degenerate boxes.
      const [west, south, east, north] = polygonBounds(polygon)
      if (Math.abs(east! - west!) < 1e-5 || Math.abs(north! - south!) < 1e-5) return
      onSelectionChange(polygon)
      onDrawModeChange('none')
    }
    const onDoubleClick = (event: MapMouseEvent) => {
      if (drawModeRef.current !== 'polygon') return
      event.preventDefault()
      finishPolygon()
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Enter') finishPolygon()
      if (event.key === 'Escape') onDrawModeChange('none')
    }

    map.on('click', handleClick)
    map.on('mousedown', onMouseDown)
    map.on('mousemove', onMouseMove)
    map.on('mouseup', onMouseUp)
    map.on('dblclick', onDoubleClick)
    window.addEventListener('keydown', onKey)

    return () => {
      map.off('click', handleClick)
      map.off('mousedown', onMouseDown)
      map.off('mousemove', onMouseMove)
      map.off('mouseup', onMouseUp)
      map.off('dblclick', onDoubleClick)
      window.removeEventListener('keydown', onKey)
    }
  }, [ready, handleClick, finishPolygon, onSelectionChange, onDrawModeChange])

  // --- basemap switching ---------------------------------------------------
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    map.setStyle(BASEMAP_STYLES[basemap])
    // A style swap discards custom sources, so they are rebuilt once the new
    // style has loaded.
    map.once('styledata', () => {
      ensureVectorLayers(map)
      updateSelection(map, selection)
      syncOverlays(map, layers, activeLayerKey, layerOpacity)
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [basemap, ready])

  // --- selection rendering ---------------------------------------------------
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    updateSelection(map, selection)
  }, [selection, ready])

  // --- overlay rendering ------------------------------------------------------
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    syncOverlays(map, layers, activeLayerKey, layerOpacity)
  }, [layers, activeLayerKey, layerOpacity, ready])

  // --- viewport --------------------------------------------------------------
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready || !fitBounds || fitBounds.length !== 4) return
    const [west, south, east, north] = fitBounds as [number, number, number, number]
    map.fitBounds(
      [
        [west, south],
        [east, north],
      ],
      { padding: 60, duration: 700, maxZoom: 15 },
    )
  }, [fitBounds, ready])

  return (
    <div className="map-shell">
      <div ref={containerRef} className="map-canvas" />
      {drawMode === 'polygon' && (
        <div className="map-hint">
          Click to add corners ({vertexCount} placed). Double-click or press Enter to
          close the shape. Escape cancels.
        </div>
      )}
      {drawMode === 'rectangle' && (
        <div className="map-hint">Drag across the map to draw a box. Escape cancels.</div>
      )}
    </div>
  )
}

// --- map helpers ------------------------------------------------------------
function ensureVectorLayers(map: MapLibreMap) {
  if (!map.getSource(SELECTION_SOURCE)) {
    map.addSource(SELECTION_SOURCE, { type: 'geojson', data: emptyCollection() })
    map.addLayer({
      id: `${SELECTION_SOURCE}-fill`,
      type: 'fill',
      source: SELECTION_SOURCE,
      paint: { 'fill-color': '#4ade80', 'fill-opacity': 0.12 },
    })
    map.addLayer({
      id: `${SELECTION_SOURCE}-line`,
      type: 'line',
      source: SELECTION_SOURCE,
      paint: { 'line-color': '#4ade80', 'line-width': 2.5 },
    })
  }
  if (!map.getSource(DRAFT_SOURCE)) {
    map.addSource(DRAFT_SOURCE, { type: 'geojson', data: emptyCollection() })
    map.addLayer({
      id: `${DRAFT_SOURCE}-line`,
      type: 'line',
      source: DRAFT_SOURCE,
      paint: { 'line-color': '#fbbf24', 'line-width': 2, 'line-dasharray': [2, 1.5] },
    })
    map.addLayer({
      id: `${DRAFT_SOURCE}-point`,
      type: 'circle',
      source: DRAFT_SOURCE,
      filter: ['==', '$type', 'Point'],
      paint: {
        'circle-radius': 4,
        'circle-color': '#fbbf24',
        'circle-stroke-width': 1.5,
        'circle-stroke-color': '#1c2b24',
      },
    })
  }
}

function emptyCollection() {
  return { type: 'FeatureCollection' as const, features: [] }
}

function updateSelection(map: MapLibreMap, polygon: GeoJSONPolygon | null) {
  const source = map.getSource(SELECTION_SOURCE) as maplibregl.GeoJSONSource | undefined
  if (!source) return
  source.setData(
    polygon
      ? { type: 'FeatureCollection', features: [{ type: 'Feature', properties: {}, geometry: polygon }] }
      : emptyCollection(),
  )
}

function updateDraft(map: MapLibreMap | null, points: LngLat[] | null) {
  if (!map) return
  const source = map.getSource(DRAFT_SOURCE) as maplibregl.GeoJSONSource | undefined
  if (!source) return
  if (!points || points.length === 0) {
    source.setData(emptyCollection())
    return
  }
  const coordinates = points.map((p) => [p.lng, p.lat])
  source.setData({
    type: 'FeatureCollection',
    features: [
      ...points.map((p) => ({
        type: 'Feature' as const,
        properties: {},
        geometry: { type: 'Point' as const, coordinates: [p.lng, p.lat] },
      })),
      ...(coordinates.length > 1
        ? [
            {
              type: 'Feature' as const,
              properties: {},
              geometry: { type: 'LineString' as const, coordinates },
            },
          ]
        : []),
    ],
  })
}

function updateDraftPolygon(map: MapLibreMap, polygon: GeoJSONPolygon) {
  const source = map.getSource(DRAFT_SOURCE) as maplibregl.GeoJSONSource | undefined
  if (!source) return
  source.setData({
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        properties: {},
        geometry: { type: 'LineString', coordinates: polygon.coordinates[0]! },
      },
    ],
  })
}

function rectangleBetween(a: LngLat, b: LngLat): GeoJSONPolygon {
  return bboxToPolygon([
    Math.min(a.lng, b.lng),
    Math.min(a.lat, b.lat),
    Math.max(a.lng, b.lng),
    Math.max(a.lat, b.lat),
  ])
}

/** Add, remove and restyle raster overlays to match the requested state. */
function syncOverlays(
  map: MapLibreMap,
  layers: LayerReference[],
  activeKey: string | null,
  opacity: number,
) {
  const wanted = new Set(layers.map((l) => `${OVERLAY_PREFIX}${l.key}`))

  for (const layer of map.getStyle().layers ?? []) {
    if (layer.id.startsWith(OVERLAY_PREFIX) && !wanted.has(layer.id)) {
      map.removeLayer(layer.id)
      if (map.getSource(layer.id)) map.removeSource(layer.id)
    }
  }

  for (const layer of layers) {
    const id = `${OVERLAY_PREFIX}${layer.key}`
    const [west, south, east, north] = layer.bounds as [number, number, number, number]
    if ([west, south, east, north].some((v) => typeof v !== 'number')) continue

    if (!map.getSource(id)) {
      map.addSource(id, {
        type: 'image',
        url: layer.image_url,
        // MapLibre image sources take corners clockwise from the top left.
        coordinates: [
          [west, north],
          [east, north],
          [east, south],
          [west, south],
        ],
      })
      map.addLayer(
        {
          id,
          type: 'raster',
          source: id,
          paint: { 'raster-opacity': 0, 'raster-resampling': 'nearest' },
        },
        map.getLayer(`${SELECTION_SOURCE}-fill`) ? `${SELECTION_SOURCE}-fill` : undefined,
      )
    }
    map.setPaintProperty(
      id,
      'raster-opacity',
      layer.key === activeKey ? opacity : 0,
    )
  }
}
