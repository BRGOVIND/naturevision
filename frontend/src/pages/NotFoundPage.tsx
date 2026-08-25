/** Rendered when a path resolves to nowhere: the survey doesn't extend here. */

import { Button, CalibrationStrip, Container, Section, SectionHead } from '../design/primitives'
import type { Navigate } from '../app/router'

export function NotFoundPage({ navigate }: { navigate: Navigate }) {
  return (
    <Section tone="canopy" className="not-found">
      <Container>
        <p className="not-found__mark" aria-hidden="true">
          404
        </p>
        <SectionHead
          tone="dark"
          align="center"
          title="This region hasn't been mapped."
          lede="The address doesn't resolve to a page here. Check it for a typo, or head back to charted ground."
        />
        <div className="not-found__strip">
          <CalibrationStrip />
        </div>
        <div className="not-found__actions">
          <Button variant="primary" size="lg" onClick={() => navigate('/')}>
            Return home
          </Button>
          <Button variant="secondary" size="lg" onClick={() => navigate('/analysis')}>
            Open analysis workspace
          </Button>
        </div>
      </Container>
    </Section>
  )
}
