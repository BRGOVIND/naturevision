/** Analysis lifecycle state: creation, progress polling and result loading. */

import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiRequestError, api, type AnalysisCreatePayload } from '../services/api'
import type {
  AnalysisDetail,
  AnalysisStatusPayload,
  AnalysisSummary,
  ReportResponse,
} from '../types/api'

const POLL_INTERVAL_MS = 2000
const TERMINAL = new Set(['report_ready', 'failed'])

export interface AnalysisState {
  status: AnalysisStatusPayload | null
  detail: AnalysisDetail | null
  report: ReportResponse | null
  history: AnalysisSummary[]
  error: string | null
  busy: boolean
  reportBusy: boolean
}

export function useAnalysis() {
  const [state, setState] = useState<AnalysisState>({
    status: null,
    detail: null,
    report: null,
    history: [],
    error: null,
    busy: false,
    reportBusy: false,
  })
  const pollRef = useRef<number | null>(null)

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  useEffect(() => stopPolling, [stopPolling])

  const refreshHistory = useCallback(async () => {
    try {
      const response = await api.listAnalyses(30)
      setState((s) => ({ ...s, history: response.items }))
    } catch {
      // History is supplementary; a failure here must not blank the dashboard.
    }
  }, [])

  useEffect(() => {
    void refreshHistory()
  }, [refreshHistory])

  const loadDetail = useCallback(
    async (id: string) => {
      const detail = await api.analysis(id)
      setState((s) => ({ ...s, detail, busy: false }))
      // Reflect the open analysis in the URL so the view is linkable and
      // survives a reload, without adding a history entry per poll.
      const url = new URL(window.location.href)
      if (url.searchParams.get('analysis') !== id) {
        url.searchParams.set('analysis', id)
        window.history.replaceState({}, '', url)
      }
      // An analysis that already has a report shows it immediately, rather
      // than offering to generate one that exists.
      if (detail.has_report) {
        try {
          const report = await api.analysisReport(id)
          setState((s) => ({ ...s, report }))
        } catch {
          // A missing or unreadable report must not block the results view.
        }
      }
      void refreshHistory()
      return detail
    },
    [refreshHistory],
  )

  const startPolling = useCallback(
    (id: string) => {
      stopPolling()
      pollRef.current = window.setInterval(async () => {
        try {
          const status = await api.analysisStatus(id)
          setState((s) => ({ ...s, status }))
          if (TERMINAL.has(status.status)) {
            stopPolling()
            if (status.status === 'failed') {
              setState((s) => ({
                ...s,
                busy: false,
                error:
                  status.error_message ??
                  'The analysis failed. See the status detail for more information.',
              }))
              void refreshHistory()
            } else {
              await loadDetail(id)
            }
          }
        } catch (error) {
          stopPolling()
          setState((s) => ({
            ...s,
            busy: false,
            error:
              error instanceof ApiRequestError
                ? error.message
                : 'Lost contact with the server while the analysis was running.',
          }))
        }
      }, POLL_INTERVAL_MS)
    },
    [loadDetail, refreshHistory, stopPolling],
  )

  const runAnalysis = useCallback(
    async (payload: AnalysisCreatePayload) => {
      setState((s) => ({
        ...s,
        busy: true,
        error: null,
        detail: null,
        report: null,
        status: null,
      }))
      try {
        const status = await api.createAnalysis(payload)
        setState((s) => ({ ...s, status }))
        startPolling(status.id)
        void refreshHistory()
      } catch (error) {
        setState((s) => ({
          ...s,
          busy: false,
          error:
            error instanceof ApiRequestError
              ? error.message
              : 'The analysis could not be started.',
        }))
      }
    },
    [refreshHistory, startPolling],
  )

  const openAnalysis = useCallback(
    async (id: string) => {
      stopPolling()
      setState((s) => ({ ...s, busy: true, error: null, report: null }))
      try {
        const detail = await loadDetail(id)
        setState((s) => ({
          ...s,
          status: {
            id: detail.id,
            status: detail.status,
            status_detail: detail.status_detail,
            progress: detail.progress,
            error_code: detail.error_code,
            error_message: detail.error_message,
            updated_at: detail.updated_at,
          },
        }))
        if (!TERMINAL.has(detail.status)) startPolling(id)
      } catch (error) {
        setState((s) => ({
          ...s,
          busy: false,
          error:
            error instanceof ApiRequestError
              ? error.message
              : 'That analysis could not be loaded.',
        }))
      }
    },
    [loadDetail, startPolling, stopPolling],
  )

  const generateReport = useCallback(
    async (id: string, includeVisual = false, regenerate = false) => {
      setState((s) => ({ ...s, reportBusy: true, error: null }))
      try {
        const report = await api.generateReport(id, includeVisual, regenerate)
        setState((s) => ({ ...s, report, reportBusy: false }))
        void refreshHistory()
      } catch (error) {
        setState((s) => ({
          ...s,
          reportBusy: false,
          error:
            error instanceof ApiRequestError
              ? error.message
              : 'The report could not be generated.',
        }))
      }
    },
    [refreshHistory],
  )

  const deleteAnalysis = useCallback(
    async (id: string) => {
      try {
        await api.deleteAnalysis(id)
        setState((s) => ({
          ...s,
          detail: s.detail?.id === id ? null : s.detail,
          status: s.status?.id === id ? null : s.status,
          report: s.report?.analysis_id === id ? null : s.report,
        }))
        void refreshHistory()
      } catch (error) {
        setState((s) => ({
          ...s,
          error: error instanceof ApiRequestError ? error.message : 'Delete failed.',
        }))
      }
    },
    [refreshHistory],
  )

  const clearError = useCallback(() => setState((s) => ({ ...s, error: null })), [])

  return {
    ...state,
    runAnalysis,
    openAnalysis,
    generateReport,
    deleteAnalysis,
    refreshHistory,
    clearError,
  }
}
