/** Site footer. */

import { CalibrationStrip } from '../design/primitives'
import { RECOGNITION, SITE } from './siteContent'
import type { Navigate, Route } from './router'

const COLUMNS: { heading: string; links: { label: string; route: Route; hash?: string }[] }[] = [
  {
    heading: 'Product',
    links: [
      { label: 'Analysis workspace', route: '/analysis', hash: 'workspace' },
      { label: 'What it does', route: '/explore' },
      { label: 'Reports', route: '/reports' },
      { label: 'Methodology', route: '/methodology' },
    ],
  },
  {
    heading: 'Technology',
    links: [
      { label: 'Remote sensing', route: '/methodology', hash: 'acquisition' },
      { label: 'Vegetation analysis', route: '/methodology', hash: 'index' },
      { label: 'Environmental ML', route: '/methodology', hash: 'land-cover' },
    ],
  },
  {
    heading: 'Project',
    links: [
      { label: 'About', route: '/about' },
      { label: 'Scientific scope', route: '/methodology', hash: 'limitations' },
      { label: 'Data sources', route: '/methodology', hash: 'sources' },
    ],
  },
]

export function Footer({ navigate }: { navigate: Navigate }) {
  return (
    <footer className="footer">
      <CalibrationStrip />
      <div className="container container--wide footer__inner">
        <div className="footer__brand">
          <h2 className="footer__wordmark">NatureVision</h2>
          <p className="footer__blurb">
            Environmental intelligence from satellite observations. Measured values come
            from deterministic geospatial processing; model predictions and generated
            interpretation are labelled as such throughout.
          </p>
          <dl className="footer__facts">
            <div>
              <dt>Imagery</dt>
              <dd className="mono">Sentinel-2 L2A</dd>
            </div>
            <div>
              <dt>Resolution</dt>
              <dd className="mono">10 m</dd>
            </div>
            <div>
              <dt>Reference labels</dt>
              <dd className="mono">ESA WorldCover</dd>
            </div>
          </dl>
        </div>

        <nav className="footer__columns" aria-label="Footer">
          {COLUMNS.map((column) => (
            <div key={column.heading} className="footer__column">
              <h3>{column.heading}</h3>
              <ul>
                {column.links.map((link) => (
                  <li key={link.label}>
                    <a
                      href={link.route}
                      onClick={(e) => {
                        e.preventDefault()
                        navigate(link.route, link.hash)
                      }}
                    >
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}

          <div className="footer__column">
            <h3>Contact</h3>
            <ul>
              <li>
                <a href={`mailto:${SITE.author.email}`}>{SITE.author.email}</a>
              </li>
              <li>
                <a href={SITE.author.github} target="_blank" rel="noreferrer">
                  github.com/{SITE.author.name}
                </a>
              </li>
            </ul>
          </div>
        </nav>
      </div>

      {/* Rendered only when a verified entry exists; see siteContent.ts. */}
      {RECOGNITION.length > 0 && (
        <div className="container container--wide footer__recognition">
          <h3>Recognition</h3>
          <ul>
            {RECOGNITION.map((item) => (
              <li key={item.project}>
                <strong>{item.project}</strong> — {item.award}, {item.event} ({item.year})
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="container container--wide footer__base">
        <p className="mono">
          Copernicus Sentinel data · ESA WorldCover CC BY 4.0 · Basemaps © Esri, ©
          OpenStreetMap contributors
        </p>
        <p className="footer__note">
          NDVI is a vegetation indicator, not a measure of biomass or biodiversity.
          Land-cover values are model predictions.
        </p>
      </div>
    </footer>
  )
}
