/**
 * Analysis lifecycle presentation.
 *
 * Every step shown here corresponds to a real backend stage transition. The
 * progress fraction comes from the server, not from a timer, so a stalled run
 * looks stalled rather than appearing to advance. The path below is a fixed
 * hand-drawn contour — seven stages never change count, so it is drawn once
 * rather than computed at runtime — and the lit portion of it is length-bound
 * to that same server-reported fraction via SVG `pathLength`.
 */

import { useEffect, useRef, useState } from 'react'

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

/** Short marks for the seven points along the contour. Abbreviations of the
 * real stage labels above, not a separate invented vocabulary. */
const STEP_MARK: Record<AnalysisStatus, string> = {
  created: 'Region',
  searching: 'Catalogue',
  acquiring: 'Imagery',
  processing: 'Raster',
  analyzing: 'Indices',
  interpreting: 'Evidence',
  report_ready: 'Report',
  failed: 'Stopped',
}

/** Seven fixed points the contour passes through, hand-placed for a gentle
 * rise and fall rather than a flat line — this is a signature visual, drawn
 * once, not generated from the data. */
const POINTS: [number, number][] = [
  [20, 45],
  [133, 25],
  [247, 60],
  [360, 25],
  [473, 60],
  [587, 25],
  [700, 45],
]

const PATH_D =
  'M 20,45 ' +
  'C 76,45 76,25 133,25 ' +
  'C 190,25 190,60 247,60 ' +
  'C 304,60 304,25 360,25 ' +
  'C 416,25 416,60 473,60 ' +
  'C 530,60 530,25 587,25 ' +
  'C 643,25 643,45 700,45'

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false)
  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)')
    setReduced(query.matches)
    const onChange = () => setReduced(query.matches)
    query.addEventListener('change', onChange)
    return () => query.removeEventListener('change', onChange)
  }, [])
  return reduced
}

export function LifecycleTracker({ status }: { status: AnalysisStatusPayload }) {
  const currentIndex = STATUS_SEQUENCE.indexOf(status.status)
  const failed = status.status === 'failed'
  const done = status.status === 'report_ready'
  const percent = Math.round(status.progress * 100)
  const reducedMotion = usePrefersReducedMotion()

  const pathRef = useRef<SVGPathElement>(null)
  const [signal, setSignal] = useState<{ x: number; y: number } | null>(null)

  // The travelling mark sits at the exact point the lit contour currently
  // ends, found by measuring the real SVG geometry rather than approximating
  // it — so it is never visibly out of step with the fill it rides on.
  useEffect(() => {
    if (reducedMotion || failed || done) {
      setSignal(null)
      return
    }
    const path = pathRef.current
    if (!path) return
    const total = path.getTotalLength()
    const point = path.getPointAtLength(total * (percent / 100))
    setSignal({ x: point.x, y: point.y })
  }, [percent, reducedMotion, failed, done])

  return (
    <div className={`lifecycle${failed ? ' lifecycle--failed' : ''}`}>
      <div className="lifecycle__head">
        <div>
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

      <svg
        className="lifecycle__journey"
        viewBox="0 0 720 90"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-label="Analysis progress"
      >
        <path className="lifecycle__contour" d={PATH_D} />
        <path
          ref={pathRef}
          className={`lifecycle__contour-fill${failed ? ' lifecycle__contour-fill--failed' : ''}`}
          d={PATH_D}
          pathLength={100}
          style={{ strokeDasharray: 100, strokeDashoffset: 100 - (failed ? 100 : percent) }}
        />
        {signal && (
          <circle className="lifecycle__signal" cx={signal.x} cy={signal.y} r={4.5} />
        )}
        {POINTS.map(([x, y], index) => {
          const step = STATUS_SEQUENCE[index]
          if (!step) return null
          const state = failed
            ? index < currentIndex
              ? 'done'
              : index === currentIndex
                ? 'failed'
                : 'pending'
            : index < currentIndex || done
              ? 'done'
              : index === currentIndex
                ? 'current'
                : 'pending'
          return (
            <g key={step} className={`lifecycle__mark lifecycle__mark--${state}`}>
              <circle cx={x} cy={y} r={state === 'current' ? 6 : 4.5} />
              <text x={x} y={index % 2 === 0 ? y + 18 : y - 12} textAnchor="middle">
                {STEP_MARK[step]}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
