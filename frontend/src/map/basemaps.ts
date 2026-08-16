/**
 * Basemap style definitions.
 *
 * Both sources are keyless raster tile services, so the application has no
 * mapping-provider dependency and runs offline-of-account. Attribution is
 * mandatory under each service's terms and is rendered by MapLibre from the
 * `attribution` field below.
 */

import type { StyleSpecification } from 'maplibre-gl'

export type BasemapId = 'satellite' | 'street'

export const BASEMAP_LABELS: Record<BasemapId, string> = {
  satellite: 'Satellite',
  street: 'Street',
}

function rasterStyle(
  tiles: string[],
  attribution: string,
  maxzoom = 19,
): StyleSpecification {
  return {
    version: 8,
    sources: {
      basemap: {
        type: 'raster',
        tiles,
        tileSize: 256,
        maxzoom,
        attribution,
      },
    },
    layers: [
      { id: 'background', type: 'background', paint: { 'background-color': '#0d1512' } },
      { id: 'basemap', type: 'raster', source: 'basemap' },
    ],
  }
}

export const BASEMAP_STYLES: Record<BasemapId, StyleSpecification> = {
  satellite: rasterStyle(
    [
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    ],
    'Imagery &copy; Esri, Maxar, Earthstar Geographics',
    18,
  ),
  street: rasterStyle(
    ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
    '&copy; OpenStreetMap contributors',
    19,
  ),
}
