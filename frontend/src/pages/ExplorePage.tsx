/**
 * Explore: what the system does, in detail.
 *
 * This is where the capability material lives — moved off the landing page so
 * the first screen introduces rather than specifies.
 */

import { Button, CalibrationStrip, Container } from '../design/primitives'
import type { Navigate } from '../app/router'

/** Sentinel-2 bands read per analysis, mirroring the backend band registry. */
const BANDS = [
  { id: 'B02', name: 'Blue', nm: 492.4, res: 10 },
  { id: 'B03', name: 'Green', nm: 559.8, res: 10 },
  { id: 'B04', name: 'Red', nm: 664.6, res: 10 },
  { id: 'B08', name: 'NIR', nm: 832.8, res: 10 },
  { id: 'B11', name: 'SWIR-1', nm: 1613.7, res: 20 },
  { id: 'B12', name: 'SWIR-2', nm: 2202.4, res: 20 },
]

const CAPABILITIES = [
  {
    key: 'remote-sensing',
    title: 'Remote sensing',
    body: 'Windowed reads over Sentinel-2 Level-2A cloud-optimised imagery, so only the bytes covering your region are fetched. The Level-2A scene classification removes cloud, shadow, cirrus and snow before any statistic is computed.',
    detail: '6 spectral bands · 10 m',
    accent: 'var(--lichen)',
  },
  {
    key: 'vegetation',
    title: 'Vegetation intelligence',
    body: 'NDVI computed per pixel with explicit handling of nodata and near-zero denominators, then summarised into density classes and full descriptive statistics over the valid footprint.',
    detail: '(NIR − Red) / (NIR + Red)',
    accent: 'var(--moss-bright)',
    mono: true,
  },
  {
    key: 'land-cover',
    title: 'Land-cover intelligence',
    body: 'A trained classifier assigns forest, agriculture, water, built-up and bare land from eleven per-pixel spectral features, with per-pixel confidence and a published hold-out accuracy.',
    detail: '5 classes · 11 features',
    accent: 'var(--model)',
  },
  {
    key: 'temporal',
    title: 'Temporal change',
    body: 'Two acquisitions co-registered onto a single pixel grid and differenced only where both dates are cloud-free, then classified against documented, configurable thresholds.',
    detail: 'Δ 0.10 / 0.20 thresholds',
    accent: 'var(--clay)',
  },
  {
    key: 'interpretation',
    title: 'Nature intelligence',
    body: 'Measured evidence becomes structured interpretation, then is checked back against that evidence. A number that was never measured is treated as fatal, not tidied up.',
    detail: 'Grounding-validated',
    accent: 'var(--interpreted)',
  },
  {
    key: 'reporting',
    title: 'Reporting',
    body: 'Thirteen sections covering region, sources, methodology, statistics, results and limitations — each tagged with whether its content was observed, predicted or generated.',
    detail: 'HTML export',
    accent: 'var(--sandstone)',
  },
]

export function ExplorePage({ navigate }: { navigate: Navigate }) {
  return (
    <>
      <section className="page-hero">
        <Container>
          <h1 className="page-hero__title">
            A measurement pipeline,
            <br />
            <em>end to end</em>
          </h1>
          <p className="page-hero__lede">
            Each stage is deterministic where it can be, and explicit about its
            uncertainty where it cannot.
          </p>
        </Container>
        <CalibrationStrip />
      </section>

      <section className="capabilities">
        <Container wide>
          <ul className="capability-grid">
            {CAPABILITIES.map((c, index) => (
              <li key={c.key} className="capability" style={{ '--card-accent': c.accent } as React.CSSProperties}>
                <span className="capability__index mono" aria-hidden="true">
                  {String(index + 1).padStart(2, '0')}
                </span>
                <h2>{c.title}</h2>
                <p>{c.body}</p>
                <p className={c.mono ? 'capability__detail mono' : 'capability__detail'}>
                  {c.detail}
                </p>
              </li>
            ))}
          </ul>
        </Container>
      </section>

      <section className="bands">
        <Container>
          <div className="bands__layout">
            <div className="bands__lead">
              <h2 className="bands__title">
                What the satellite
                <br />
                <em>actually sees</em>
              </h2>
              <p>
                Six reflectance bands plus the scene classification layer are read for
                every analysis. Visible light describes what the eye would see; the
                infrared bands are where vegetation condition, moisture and built
                surfaces separate from one another.
              </p>
            </div>

            <div className="bands__panel">
              <ul className="band-list">
                {BANDS.map((band) => (
                  <li key={band.id}>
                    <span className="band-list__id mono">{band.id}</span>
                    <span className="band-list__name">{band.name}</span>
                    <span className="band-list__nm mono">{band.nm} nm</span>
                    <span className="band-list__res mono">{band.res} m</span>
                  </li>
                ))}
              </ul>
              <p className="bands__note">
                Plus the Level-2A scene classification layer, used to mask cloud,
                shadow, cirrus and snow before any statistic is computed.
              </p>
            </div>
          </div>
        </Container>
      </section>

      <section className="invitation">
        <Container>
          <h2 className="invitation__title">
            See it on real
            <br />
            <em>satellite data</em>
          </h2>
          <p className="invitation__body">
            The workspace runs the whole pipeline live. Nothing in it is precomputed.
          </p>
          <div className="invitation__actions">
            <Button variant="primary" size="lg" onClick={() => navigate('/analysis', 'workspace')}>
              Launch analysis
            </Button>
            <Button variant="secondary" size="lg" onClick={() => navigate('/methodology')}>
              Read the methodology
            </Button>
          </div>
        </Container>
      </section>
    </>
  )
}
