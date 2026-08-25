/**
 * Minimal history router.
 *
 * A dedicated routing library would add a dependency for what this product
 * needs: five flat routes, no nesting, no data loaders. This is the whole
 * implementation, and it keeps scroll restoration under our control.
 */

import { useCallback, useEffect, useState } from 'react'

export const ROUTES = [
  '/',
  '/explore',
  '/analysis',
  '/methodology',
  '/reports',
  '/about',
] as const
export type Route = (typeof ROUTES)[number]

/** `navigate()` only ever targets a real page; `'not-found'` is a render
 * state the router can land in on its own, for a path nothing matches. */
export type RouterState = Route | 'not-found'

function normalise(pathname: string): RouterState {
  const candidate = pathname.replace(/\/+$/, '') || '/'
  return (ROUTES as readonly string[]).includes(candidate) ? (candidate as Route) : 'not-found'
}

export function useRouter() {
  const [route, setRoute] = useState<RouterState>(() => normalise(window.location.pathname))

  useEffect(() => {
    const onPop = () => setRoute(normalise(window.location.pathname))
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const navigate = useCallback(
    (to: Route, hash?: string) => {
      const url = hash ? `${to}#${hash}` : to
      if (window.location.pathname !== to || hash) {
        window.history.pushState({}, '', url)
      }
      setRoute(normalise(to))

      // Defer so the destination has mounted before we scroll to it.
      requestAnimationFrame(() => {
        if (hash) {
          document.getElementById(hash)?.scrollIntoView({ block: 'start' })
        } else {
          window.scrollTo({ top: 0, behavior: 'auto' })
        }
      })
    },
    [],
  )

  return { route, navigate }
}

export type Navigate = ReturnType<typeof useRouter>['navigate']
