/** Types mirroring the backend API contract. */

export type AnalysisStatus =
  | 'created'
  | 'searching'
  | 'acquiring'
  | 'processing'
  | 'analyzing'
  | 'interpreting'
  | 'report_ready'
  | 'failed'

/** Ordered lifecycle used to render determinate progress. */
export const STATUS_SEQUENCE: AnalysisStatus[] = [
  'created',
  'searching',
  'acquiring',
  'processing',
  'analyzing',
  'interpreting',
  'report_ready',
]

export const STATUS_LABELS: Record<AnalysisStatus, string> = {
  created: 'Queued',
  searching: 'Searching satellite catalogue',
  acquiring: 'Retrieving imagery',
  processing: 'Processing rasters',
  analyzing: 'Analysing vegetation',
  interpreting: 'Rendering layers',
  report_ready: 'Complete',
  failed: 'Failed',
}

export interface ApiError {
  code: string
  message: string
  details?: Record<string, unknown> | null
  request_id?: string | null
}

export interface BoundingBox {
  west: number
  south: number
  east: number
  north: number
}

export interface GeoJSONPolygon {
  type: 'Polygon'
  coordinates: number[][][]
}

export interface RegionInput {
  geometry?: GeoJSONPolygon
  bbox?: number[]
  name?: string | null
  crs?: string | null
}

export interface DateRange {
  start: string
  end: string
}

export interface Observation {
  source_id: string
  provider: string
  dataset: string
  observation_date: string
  acquisition_timestamp: string | null
  cloud_cover_percent: number | null
  bbox: number[]
  processing_level: string | null
  platform: string | null
  instrument: string | null
  crs: string | null
  resolution_m: number | null
  license: string | null
  bands: string[]
  region_coverage: number | null
  properties: Record<string, unknown>
}

export interface ImagerySearchResponse {
  region: { bbox: number[]; area_km2: number; description: string; crs: string }
  query: Record<string, unknown>
  count: number
  observations: Observation[]
}

export interface LayerReference {
  key: string
  label: string
  kind: 'continuous' | 'categorical'
  image_url: string
  bounds: number[]
  value_min: number | null
  value_max: number | null
  legend: { value: number; colour: string; label: string; series?: string }[]
  units: string | null
  description: string | null
}

export interface Metric {
  key: string
  label: string
  value: number | null
  unit: string | null
  period: string | null
  category: string
  provenance: 'observed' | 'model_prediction'
  details?: Record<string, unknown> | null
}

export interface ObservationRecord {
  period: string
  source_id: string
  provider: string
  dataset: string
  observation_date: string
  acquisition_timestamp: string | null
  cloud_cover_percent: number | null
  processing_level: string | null
  platform: string | null
  instrument: string | null
  crs: string | null
  resolution_m: number | null
  license: string | null
  bands_used: string[]
  scene_metadata: Record<string, unknown> | null
}

export interface ModelPrediction {
  model_name: string
  model_version: string
  model_backend: string
  task: string
  predicted_at: string
  class_distribution: Record<string, number>
  mean_confidence: number | null
  low_confidence_fraction: number | null
  evaluation_metrics: {
    overall_accuracy: number | null
    macro_f1: number | null
    per_class_metrics?: Record<string, Record<string, number>>
    evaluation_protocol?: string | null
    evaluation_samples?: number | null
  } | null
  preprocessing_version: string | null
  prediction_metadata: Record<string, unknown> | null
}

export interface AnalysisStatusPayload {
  id: string
  status: AnalysisStatus
  status_detail: string | null
  progress: number
  error_code?: string | null
  error_message?: string | null
  updated_at: string
}

export interface AnalysisSummary {
  id: string
  status: AnalysisStatus
  status_detail: string | null
  progress: number
  region_name: string | null
  region_bbox: number[]
  area_km2: number
  period_a: string
  period_b: string | null
  mean_ndvi_a: number | null
  ndvi_change: number | null
  created_at: string
  completed_at: string | null
}

export interface AnalysisDetail {
  id: string
  status: AnalysisStatus
  status_detail: string | null
  progress: number
  error_code: string | null
  error_message: string | null
  region: {
    id: string
    name: string | null
    geometry: GeoJSONPolygon
    bbox: number[]
    area_km2: number
    crs: string
  }
  period_a: string
  period_b: string | null
  max_cloud_cover: number
  include_land_cover: boolean
  created_at: string
  updated_at: string
  completed_at: string | null
  observations: ObservationRecord[]
  metrics: Metric[]
  predictions: ModelPrediction[]
  evidence: Record<string, any> | null
  methodology: Record<string, any> | null
  layers: LayerReference[]
  has_report: boolean
}

export interface AnalysisListResponse {
  total: number
  limit: number
  offset: number
  items: AnalysisSummary[]
}

export interface ReportSection {
  key: string
  title: string
  provenance: 'observed' | 'model_prediction' | 'interpretation' | 'metadata'
  body: string | null
  items: string[]
  table: Record<string, string | number | null>[]
  notes: string[]
}

export interface ReportResponse {
  id: string
  analysis_id: string
  title: string
  generated_at: string
  sections: ReportSection[]
  provenance_legend: Record<string, string>
  interpretation: Record<string, any> | null
  visual_interpretation: Record<string, any> | null
  interpretation_provider: string | null
  interpretation_model: string | null
  export_urls: Record<string, string>
}

export interface HealthResponse {
  status: string
  application: string
  environment: string
  version: string
  database: string
  imagery_provider: string
  land_cover_model: string
  interpretation: string
  checks: Record<string, any>
}

export interface ModelInfo {
  installed: boolean
  models: Record<string, any>[]
  active_backend: string
  feature_version: string
  classes: { id: number; label: string; description: string; colour: string }[]
}
