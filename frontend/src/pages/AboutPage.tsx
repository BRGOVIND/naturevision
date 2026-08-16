/** About the project. */

import { Button, Container, Section, SectionHead } from '../design/primitives'
import { RECOGNITION, SITE } from '../app/siteContent'
import type { Navigate } from '../app/router'

const STACK = [
  { group: 'Geospatial', items: ['rasterio', 'shapely', 'pyproj', 'GeoPandas', 'PostGIS'] },
  { group: 'Machine learning', items: ['scikit-learn', 'PyTorch', 'NumPy'] },
  { group: 'Backend', items: ['FastAPI', 'SQLAlchemy', 'Alembic', 'PostgreSQL'] },
  { group: 'Frontend', items: ['React', 'TypeScript', 'MapLibre GL', 'Recharts', 'Vite'] },
  { group: 'Delivery', items: ['Docker', 'GitHub Actions', 'pytest'] },
]

export function AboutPage({ navigate }: { navigate: Navigate }) {
  return (
    <>
      <Section tone="canopy" className="page-head">
        <Container>
          <SectionHead
            tone="dark"
            title="Why NatureVision exists"
            lede="Satellite data is abundant and public. Turning it into something a non-specialist can act on, without overstating what it shows, is the hard part."
          />
        </Container>
      </Section>

      <Section tone="cream">
        <Container>
          <div className="prose">
            <p>
              Most environmental dashboards present a number without saying where it came
              from, how it was measured, or how far it can be trusted. NatureVision takes the
              opposite position: every value carries its provenance, and the boundary between
              a measurement, a model prediction and a written interpretation is never blurred.
            </p>
            <p>
              That constraint shapes the whole system. Statistics are computed only over
              pixels that survived cloud masking in every period being compared. Model
              accuracy is measured on regions the model never saw, which produces lower and
              more honest numbers than a random split would. The language layer sees only
              finished evidence, and its output is checked back against that evidence before
              anyone reads it.
            </p>
          </div>
        </Container>
      </Section>

      <Section tone="charcoal">
        <Container>
          <SectionHead tone="dark" title="Technology" />
          <ul className="stack-list">
            {STACK.map((group) => (
              <li key={group.group}>
                <h3>{group.group}</h3>
                <p className="mono">{group.items.join(' · ')}</p>
              </li>
            ))}
          </ul>
        </Container>
      </Section>

      {RECOGNITION.length > 0 && (
        <Section tone="cream">
          <Container>
            <SectionHead title="Awards" />
            <ul className="limit-list">
              {RECOGNITION.map((item) => (
                <li key={item.project}>
                  <strong>{item.project}</strong> — {item.award}, {item.event} ({item.year})
                </li>
              ))}
            </ul>
          </Container>
        </Section>
      )}

      <Section tone="canopy">
        <Container>
          <SectionHead tone="dark" title="Get in touch" />
          <div className="contact">
            <a className="contact__link" href={`mailto:${SITE.author.email}`}>
              {SITE.author.email}
            </a>
            <a
              className="contact__link"
              href={SITE.author.github}
              target="_blank"
              rel="noreferrer"
            >
              github.com/{SITE.author.name}
            </a>
          </div>
          <div className="contact__cta">
            <Button variant="primary" onClick={() => navigate('/analysis', 'workspace')}>
              Launch analysis
            </Button>
          </div>
        </Container>
      </Section>
    </>
  )
}
