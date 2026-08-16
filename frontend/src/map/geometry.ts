/**
 * Client-side geometry helpers.
 *
 * These give immediate feedback while drawing. They are deliberately not the
 * authority: every geometry is re-validated server-side before any analysis
 * runs, because client checks can be bypassed and floating-point behaviour
 * differs between environments.
 */

import type { GeoJSONPolygon } from '../types/api'

export interface LngLat {
  lng: number
  lat: number
}

export const MIN_AREA_KM2 = 0.01
export const MAX_AREA_KM2 = 2500

/** Ellipsoidal area on the WGS84 spheroid, matching the server's calculation. */
export function geodesicAreaKm2(ring: number[][]): number {
  if (ring.length < 4) return 0
  const R = 6378137
  let total = 0
  for (let i = 0; i < ring.length - 1; i += 1) {
    const p1 = ring[i]!
    const p2 = ring[i + 1]!
    total +=
      toRad(p2[0]! - p1[0]!) *
      (2 + Math.sin(toRad(p1[1]!)) + Math.sin(toRad(p2[1]!)))
  }
  return Math.abs((total * R * R) / 2) / 1_000_000
}

function toRad(deg: number): number {
  return (deg * Math.PI) / 180
}

export function bboxToPolygon(bbox: number[]): GeoJSONPolygon {
  const [west, south, east, north] = bbox as [number, number, number, number]
  return {
    type: 'Polygon',
    coordinates: [
      [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
      ],
    ],
  }
}

export function polygonFromPoints(points: LngLat[]): GeoJSONPolygon | null {
  if (points.length < 3) return null
  const ring = points.map((p) => [p.lng, p.lat])
  const first = ring[0]!
  const last = ring[ring.length - 1]!
  if (first[0] !== last[0] || first[1] !== last[1]) ring.push([first[0]!, first[1]!])
  return { type: 'Polygon', coordinates: [ring] }
}

export function polygonBounds(polygon: GeoJSONPolygon): number[] {
  let west = 180
  let south = 90
  let east = -180
  let north = -90
  for (const ring of polygon.coordinates) {
    for (const [lng, lat] of ring as [number, number][]) {
      west = Math.min(west, lng)
      east = Math.max(east, lng)
      south = Math.min(south, lat)
      north = Math.max(north, lat)
    }
  }
  return [west, south, east, north]
}

export interface GeometryValidation {
  valid: boolean
  areaKm2: number
  message: string | null
}

/** Mirror of the server's extent rules, for pre-submit feedback only. */
export function validatePolygon(polygon: GeoJSONPolygon | null): GeometryValidation {
  if (!polygon) {
    return { valid: false, areaKm2: 0, message: 'Select a region on the map to continue.' }
  }
  const ring = polygon.coordinates[0]
  if (!ring || ring.length < 4) {
    return { valid: false, areaKm2: 0, message: 'A region needs at least three corners.' }
  }
  const areaKm2 = geodesicAreaKm2(ring)
  if (areaKm2 < MIN_AREA_KM2) {
    return {
      valid: false,
      areaKm2,
      message: `Region is too small (${areaKm2.toFixed(4)} km²). Minimum is ${MIN_AREA_KM2} km².`,
    }
  }
  if (areaKm2 > MAX_AREA_KM2) {
    return {
      valid: false,
      areaKm2,
      message: `Region is too large (${areaKm2.toFixed(0)} km²). Maximum is ${MAX_AREA_KM2} km² per analysis.`,
    }
  }
  return { valid: true, areaKm2, message: null }
}

export function formatBounds(bounds: number[]): string {
  return bounds.map((v) => v.toFixed(4)).join(', ')
}

export function describeCentre(bounds: number[]): string {
  const [west, south, east, north] = bounds as [number, number, number, number]
  const lng = (west + east) / 2
  const lat = (south + north) / 2
  return `${Math.abs(lat).toFixed(3)}°${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lng).toFixed(3)}°${
    lng >= 0 ? 'E' : 'W'
  }`
}
