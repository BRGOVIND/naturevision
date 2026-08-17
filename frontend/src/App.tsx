/** Application shell: routing, intro state, global capability probing, chrome. */

import { useCallback, useEffect, useState } from 'react'

import { Footer } from './app/Footer'
import { Nav } from './app/Nav'
import { Opening, shouldPlayOpening } from './app/Opening'
import { useRouter } from './app/router'
import { AboutPage } from './pages/AboutPage'
import { ExplorePage } from './pages/ExplorePage'
import { LandingPage } from './pages/LandingPage'
import { MethodologyPage } from './pages/MethodologyPage'
import { ReportsPage } from './pages/ReportsPage'
import { WorkspacePage } from './pages/WorkspacePage'
import { api } from './services/api'
import type { HealthResponse } from './types/api'

const PAGE_TITLES: Record<string, string> = {
  '/': 'NatureVision — Environmental Intelligence',
  '/explore': 'Explore — NatureVision',
  '/analysis': 'Analysis workspace — NatureVision',
  '/methodology': 'Methodology — NatureVision',
  '/reports': 'Reports — NatureVision',
  '/about': 'About — NatureVision',
}

export default function App() {
  const { route, navigate } = useRouter()
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [offline, setOffline] = useState(false)

  // Resolved once, synchronously, before the first paint.
  //   playing   overlay opaque, shell hidden
  //   revealing shell visible, overlay dissolving over it
  //   done      overlay unmounted
  const [intro, setIntro] = useState<'playing' | 'revealing' | 'done'>(() =>
    shouldPlayOpening() ? 'playing' : 'done',
  )
  const introVisible = intro !== 'done'
  const shellHidden = intro === 'playing'

  const revealShell = useCallback(() => setIntro('revealing'), [])
  const finishIntro = useCallback(() => setIntro('done'), [])

  // Scroll is locked for the duration of the intro and released afterwards.
  // The previous inline value is restored rather than being cleared, so this
  // cannot strand the document in a non-scrolling state.
  useEffect(() => {
    if (!introVisible) return
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = previous
    }
  }, [introVisible])

  // Capabilities are probed once. The UI disables controls for features this
  // deployment genuinely cannot perform, rather than letting a user trigger a
  // guaranteed failure.
  useEffect(() => {
    api
      .health()
      .then((h) => {
        setHealth(h)
        setOffline(false)
      })
      .catch(() => {
        setHealth(null)
        setOffline(true)
      })
  }, [])

  useEffect(() => {
    document.title = PAGE_TITLES[route] ?? 'NatureVision'
  }, [route])

  return (
    <>
      {introVisible && <Opening onExitStart={revealShell} onFinished={finishIntro} />}

      {/* The shell is present from the first render so fonts and the hero
          image load during the intro, but it is not visible or focusable
          until the intro hands over. */}
      <div
        className={shellHidden ? 'app-shell app-shell--introing' : 'app-shell'}
        {...(shellHidden ? { inert: '' as unknown as boolean, 'aria-hidden': true } : {})}
      >
        <a className="skip-link" href="#main">
          Skip to content
        </a>

        <Nav route={route} navigate={navigate} />

        {offline && (
          <div className="banner banner--offline" role="status">
            The application server could not be reached, so live analysis is unavailable.
            Start the backend and reload.
          </div>
        )}

        {/* Keying on the route guarantees each page unmounts cleanly — the map
            instance in the workspace releases its WebGL context rather than
            lingering when the visitor returns to the landing page. */}
        <main id="main" key={route}>
          {route === '/' && <LandingPage navigate={navigate} health={health} />}
          {route === '/explore' && <ExplorePage navigate={navigate} />}
          {route === '/analysis' && <WorkspacePage health={health} />}
          {route === '/methodology' && <MethodologyPage />}
          {route === '/reports' && <ReportsPage navigate={navigate} />}
          {route === '/about' && <AboutPage navigate={navigate} />}
        </main>

        <Footer navigate={navigate} />
      </div>
    </>
  )
}
