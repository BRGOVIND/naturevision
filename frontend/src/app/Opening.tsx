/**
 * Cinematic entry.
 *
 * Shows the second forest photograph full-screen with the wordmark, then
 * dismisses itself after ~1.6s. Three rules govern it:
 *
 *  - It is purely additive. The landing page renders underneath from the first
 *    frame, so a slow or failed image can only cost the animation, never the
 *    content. Nothing waits on a preload.
 *  - It runs on first entry per session only, so internal navigation never
 *    replays it.
 *  - Reduced-motion and slow connections skip it entirely.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

const SEEN_KEY = 'nv:opening-seen'
const HOLD_MS = 1600
const FADE_MS = 700

function shouldSkip(): boolean {
  if (typeof window === 'undefined') return true
  // Deep links into a result should land on the result, not on an intro.
  if (window.location.search.includes('analysis=')) return true
  if (window.location.pathname !== '/') return true
  try {
    if (window.sessionStorage.getItem(SEEN_KEY)) return true
  } catch {
    // Private modes can throw on storage access; the intro is not worth failing over.
  }
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return true
  const connection = (navigator as { connection?: { saveData?: boolean; effectiveType?: string } })
    .connection
  if (connection?.saveData) return true
  if (connection?.effectiveType && /2g/.test(connection.effectiveType)) return true
  return false
}

export function Opening() {
  const [phase, setPhase] = useState<'hidden' | 'showing' | 'leaving'>(() =>
    shouldSkip() ? 'hidden' : 'showing',
  )
  const timers = useRef<number[]>([])

  const dismiss = useCallback(() => {
    setPhase((current) => (current === 'showing' ? 'leaving' : current))
    timers.current.push(window.setTimeout(() => setPhase('hidden'), FADE_MS))
  }, [])

  useEffect(() => {
    if (phase === 'hidden') return
    try {
      window.sessionStorage.setItem(SEEN_KEY, '1')
    } catch {
      // Non-fatal; the intro simply repeats next time.
    }

    // The overlay is inert to assistive tech and hidden from the tab order, but
    // it still covers the page, so any deliberate input dismisses it early.
    timers.current.push(window.setTimeout(dismiss, HOLD_MS))
    const onInput = () => dismiss()
    window.addEventListener('keydown', onInput)
    window.addEventListener('pointerdown', onInput)
    window.addEventListener('wheel', onInput, { passive: true })

    const captured = timers.current
    return () => {
      window.removeEventListener('keydown', onInput)
      window.removeEventListener('pointerdown', onInput)
      window.removeEventListener('wheel', onInput)
      captured.forEach(window.clearTimeout)
    }
  }, [phase, dismiss])

  if (phase === 'hidden') return null

  return (
    <div
      className={phase === 'leaving' ? 'opening opening--leaving' : 'opening'}
      aria-hidden="true"
    >
      <picture className="opening__media">
        <source media="(max-width: 640px)" srcSet="/hero/opening-portrait.webp" type="image/webp" />
        <source media="(max-width: 640px)" srcSet="/hero/opening-portrait.jpg" />
        <source media="(max-width: 1100px)" srcSet="/hero/opening-mid.webp" type="image/webp" />
        <source srcSet="/hero/opening-wide.webp" type="image/webp" />
        <img src="/hero/opening-wide.jpg" alt="" fetchPriority="high" decoding="async" />
      </picture>
      <div className="opening__veil" />

      <div className="opening__mark">
        <span className="opening__glyph">
          <svg viewBox="0 0 64 64" width="40" height="40" fill="none" aria-hidden="true">
            <path
              d="M32 12c-11.2 8.3-17.6 17.6-17.6 27.2A17.6 17.6 0 0 0 32 52a17.6 17.6 0 0 0 17.6-23.5C49.6 21.4 43.2 12.1 32 12Z"
              stroke="currentColor"
              strokeWidth="2.4"
              strokeLinejoin="round"
            />
            <path
              d="M32 52V27.7M32 38.9l8.3-8.3M32 47.4l-7.2-7.2"
              stroke="currentColor"
              strokeWidth="2.4"
              strokeLinecap="round"
            />
          </svg>
        </span>
        <h1 className="opening__wordmark">NatureVision</h1>
        <p className="opening__tagline">Environmental Intelligence</p>
      </div>
    </div>
  )
}
