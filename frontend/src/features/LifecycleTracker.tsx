/**
 * Analysis lifecycle presentation.
 *
 * Every step shown here corresponds to a real backend stage transition. The
 * progress fraction comes from the server, not from a timer, so a stalled run
 * looks stalled rather than appearing to advance.
 */

import { Spinner } from '../design/primitives'
import {
  STATUS_LABELS,
  STATUS_SEQUENCE,
  type AnalysisStatus,
  type AnalysisStatusPayload,
} from '../types/api'

const STEP_DETAIL: Record<AnalysisStatus, string> = {
  created: 'Analysis queued',
  searching: 'Querying the Sentinel-2 catalogue for usable observations',
  acquiring: 'Reading the required bands over the selected region',
  processing: 'Co-registering bands and computing the vegetation index',
  analyzing: 'Comparing periods and classifying land cover',
  interpreting: 'Rendering map layers and assembling evidence',
  report_ready: 'All results available',
  failed: 'The run stopped before completing',
}

export function LifecycleTracker({ status }: { status: AnalysisStatusPayload }) {
  const currentIndex = STATUS_SEQUENCE.indexOf(status.status)
  const failed = status.status === 'failed'
  const done = status.status === 'report_ready'
  const percent = Math.round(status.progress * 100)

  return (
    <div className={`lifecycle${failed ? ' lifecycle--failed' : ''}`}>
      <div className="lifecycle__head">
        <div>
          <p className="eyebrow eyebrow--dark">Analysis status</p>
          <h2 className="lifecycle__title">
            {failed ? 'Analysis failed' : done ? 'Analysis complete' : STATUS_LABELS[status.status]}
            {!failed && !done && <Spinner label="Analysis in progress" />}
          </h2>
          <p className="lifecycle__detail">{status.status_detail ?? STEP_DETAIL[status.status]}</p>
        </div>
        <p className="lifecycle__percent mono" aria-hidden="true">
          {percent}%
        </p>
      </div>

      <div
        className="lifecycle__bar"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-label="Analysis progress"
      >
        <div
          className={`lifecycle__fill${failed ? ' lifecycle__fill--failed' : ''}`}
          style={{ width: `${percent}%` }}
        />
      </div>

      <ol className="lifecycle__steps">
        {STATUS_SEQUENCE.map((step, index) => {
          const state = failed
            ? index < currentIndex
              ? 'done'
              : index === currentIndex
                ? 'failed'
                : 'pending'
            : index < currentIndex
              ? 'done'
              : index === currentIndex
                ? 'current'
                : 'pending'
          return (
            <li key={step} className={`lifecycle__step lifecycle__step--${state}`}>
              <span className="lifecycle__marker" aria-hidden="true" />
              <span className="lifecycle__step-label">{STATUS_LABELS[step]}</span>
              <span className="lifecycle__step-note">{STEP_DETAIL[step]}</span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
