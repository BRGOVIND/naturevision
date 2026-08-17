/**
 * Cinematic entry.
 *
 * The intro is an application state, not an overlay laid over a live page.
 * `App` keeps the shell hidden until `onFinished` fires, so the landing never
 * shows through. Three rules govern it:
 *
 *  - It is opaque from the first painted frame. An overlay that fades *in*
 *    reveals the page underneath for the length of the fade, which is exactly
 *    the flash this replaces.
 *  - It runs once per session, so internal navigation never replays it.
 *  - Reduced motion, deep links and slow connections skip it entirely; the
 *    caller resolves that before mounting this component.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

const SEEN_KEY = 'nv:opening-seen'

/** Visual timing. Total ≈ 3.2s from first paint to the shell being handed over. */
const HOLD_MS = 2500
const FADE_MS = 700

export function shouldPlayOpening(): boolean {
  if (typeof window === 'undefined') return false
  // A deep link into a result should land on the result, not on an intro.
  if (window.location.search.includes('analysis=')) return false
  if (window.location.pathname !== '/') return false
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return false
  try {
    if (window.sessionStorage.getItem(SEEN_KEY)) return false
  } catch {
    // Private modes can throw on storage access; not worth failing over.
  }
  const connection = (navigator as { connection?: { saveData?: boolean; effectiveType?: string } })
    .connection
  if (connection?.saveData) return false
  if (connection?.effectiveType && /(^|-)2g/.test(connection.effectiveType)) return false
  return true
}

export function Opening({
  onExitStart,
  onFinished,
}: {
  /** Fired as the overlay begins fading, so the shell is already visible beneath it. */
  onExitStart: () => void
  /** Fired once the fade completes and the overlay can unmount. */
  onFinished: () => void
}) {
  const [leaving, setLeaving] = useState(false)
  const finished = useRef(false)
  const timers = useRef<number[]>([])

  const finish = useCallback(() => {
    if (finished.current) return
    finished.current = true
    // Reveal the shell first, then dissolve the overlay over it. The shell is
    // never mid-animation, so it cannot be left invisible by a transition that
    // does not run.
    onExitStart()
    setLeaving(true)
    timers.current.push(window.setTimeout(onFinished, FADE_MS))
  }, [onExitStart, onFinished])

  useEffect(() => {
    try {
      window.sessionStorage.setItem(SEEN_KEY, '1')
    } catch {
      // Non-fatal; the intro simply repeats next session.
    }

    timers.current.push(window.setTimeout(finish, HOLD_MS))

    // Any deliberate input skips ahead rather than trapping the visitor.
    const skip = () => finish()
    window.addEventListener('keydown', skip)
    window.addEventListener('pointerdown', skip)
    window.addEventListener('wheel', skip, { passive: true })

    const captured = timers.current
    return () => {
      window.removeEventListener('keydown', skip)
      window.removeEventListener('pointerdown', skip)
      window.removeEventListener('wheel', skip)
      captured.forEach(window.clearTimeout)
    }
  }, [finish])

  return (
    <div className={leaving ? 'opening opening--leaving' : 'opening'} aria-hidden="true">
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
          <svg viewBox="0 0 64 64" width="44" height="44" fill="none" aria-hidden="true">
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
        <p className="opening__wordmark">NatureVision</p>
        <p className="opening__tagline">Environmental Intelligence</p>
      </div>
    </div>
  )
}
