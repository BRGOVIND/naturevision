/** Application shell: routing, global capability state, chrome. */

import { useEffect, useState } from 'react'

import { Footer } from './app/Footer'
import { Nav } from './app/Nav'
import { Opening } from './app/Opening'
import { useRouter } from './app/router'
import { AboutPage } from './pages/AboutPage'
import { ExplorePage } from './pages/ExplorePage'
import { LandingPage } from './pages/LandingPage'
import { MethodologyPage } from './pages/MethodologyPage'
import { ReportsPage } from './pages/ReportsPage'
import { WorkspacePage } from './pages/WorkspacePage'
import { api } from './services/api'
import type { HealthResponse } from './types/api'

export default function App() {
  const { route, navigate } = useRouter()
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [offline, setOffline] = useState(false)

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
    const titles: Record<string, string> = {
      '/': 'NatureVision — Environmental Intelligence',
      '/explore': 'Explore — NatureVision',
      '/analysis': 'Analysis workspace — NatureVision',
      '/methodology': 'Methodology — NatureVision',
      '/reports': 'Reports — NatureVision',
      '/about': 'About — NatureVision',
    }
    document.title = titles[route] ?? 'NatureVision'
  }, [route])

  return (
    <>
      <Opening />

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

      <main id="main">
        {route === '/' && <LandingPage navigate={navigate} health={health} />}
        {route === '/explore' && <ExplorePage navigate={navigate} />}
        {route === '/analysis' && <WorkspacePage health={health} />}
        {route === '/methodology' && <MethodologyPage />}
        {route === '/reports' && <ReportsPage navigate={navigate} />}
        {route === '/about' && <AboutPage navigate={navigate} />}
      </main>

      <Footer navigate={navigate} />
    </>
  )
}
