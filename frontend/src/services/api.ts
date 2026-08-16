/**
 * Backend API client.
 *
 * Errors from the server arrive in a stable envelope; this module converts
 * them into a typed exception so the UI can show the server's safe message
 * instead of a generic failure string.
 */

import type {
  AnalysisDetail,
  AnalysisListResponse,
  AnalysisStatusPayload,
  ApiError,
  DateRange,
  HealthResponse,
  ImagerySearchResponse,
  ModelInfo,
  RegionInput,
  ReportResponse,
} from '../types/api'

const BASE = import.meta.env.VITE_API_BASE ?? '/api/v1'

export class ApiRequestError extends Error {
  readonly code: string
  readonly details?: Record<string, unknown> | null
  readonly status: number

  constructor(status: number, payload: ApiError) {
    super(payload.message)
    this.name = 'ApiRequestError'
    this.status = status
    this.code = payload.code
    this.details = payload.details
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch {
    throw new ApiRequestError(0, {
      code: 'network_unreachable',
      message: 'The application server could not be reached. Check that it is running.',
    })
  }

  if (response.status === 204) return undefined as T

  const text = await response.text()
  const payload = text ? safeParse(text) : null

  if (!response.ok) {
    throw new ApiRequestError(response.status, {
      code: (payload?.code as string) ?? 'request_failed',
      message:
        (payload?.message as string) ??
        `The request failed with status ${response.status}.`,
      details: (payload?.details as Record<string, unknown>) ?? null,
    })
  }
  return payload as T
}

function safeParse(text: string): Record<string, unknown> | null {
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

export interface AnalysisCreatePayload {
  region: RegionInput
  period_a: DateRange
  period_b?: DateRange | null
  max_cloud_cover: number
  include_land_cover: boolean
  include_interpretation: boolean
  change_moderate_threshold?: number | null
  change_significant_threshold?: number | null
}

export const api = {
  health: () => request<HealthResponse>('/health'),

  models: () => request<ModelInfo>('/models'),

  methodology: () => request<Record<string, unknown>>('/methodology'),

  searchImagery: (body: {
    region: RegionInput
    start_date: string
    end_date: string
    max_cloud_cover: number
    limit?: number
  }) =>
    request<ImagerySearchResponse>('/imagery/search', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  createAnalysis: (body: AnalysisCreatePayload) =>
    request<AnalysisStatusPayload>('/analysis', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  analysisStatus: (id: string) =>
    request<AnalysisStatusPayload>(`/analysis/${id}/status`),

  analysis: (id: string) => request<AnalysisDetail>(`/analysis/${id}`),

  listAnalyses: (limit = 25, offset = 0) =>
    request<AnalysisListResponse>(`/analysis?limit=${limit}&offset=${offset}`),

  deleteAnalysis: (id: string) =>
    request<void>(`/analysis/${id}`, { method: 'DELETE' }),

  generateReport: (analysisId: string, includeVisual = false, regenerate = false) =>
    request<ReportResponse>('/ai/report', {
      method: 'POST',
      body: JSON.stringify({
        analysis_id: analysisId,
        include_visual_interpretation: includeVisual,
        regenerate,
      }),
    }),

  report: (id: string) => request<ReportResponse>(`/reports/${id}`),

  analysisReport: (analysisId: string) =>
    request<ReportResponse>(`/analysis/${analysisId}/report`),

  reportExportUrl: (id: string, format: 'html' | 'pdf' = 'html') =>
    `${BASE}/reports/${id}/export?format=${format}`,

  layerUrl: (path: string) => path,
}
