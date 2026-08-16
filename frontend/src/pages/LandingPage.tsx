/**
 * Landing experience.
 *
 * An invitation, not a specification. It establishes identity, says what the
 * system does and what its evidence looks like, and hands the visitor to the
 * workspace. The full capability and methodology material lives on /explore
 * and /methodology, reachable from the navigation.
 */

import { useEffect, useRef, useState } from 'react'

import { Button, CalibrationStrip, Container } from '../design/primitives'
import type { Navigate } from '../app/router'
import type { HealthResponse } from '../types/api'

/**
 * Reveals children as they scroll into view.
 *
 * Content is visible by default and only opts into the hidden start state
 * after mount, once we know an observer is available and the element is
 * genuinely below the fold. A safety timer releases it regardless. An
 * animation must never be able to leave content permanently invisible.
 */
function Reveal({
  children,
  className,
  delay = 0,
}: {
  children: React.ReactNode
  className?: string
  delay?: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [state, setState] = useState<'visible' | 'pending'>('visible')

  useEffect(() => {
    const node = ref.current
    if (!node) return
    if (
      typeof IntersectionObserver === 'undefined' ||
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ) {
      return
    }
    // Already on screen at mount: leave it alone rather than hiding then fading.
    if (node.getBoundingClientRect().top < window.innerHeight * 0.9) return

    setState('pending')
    const reveal = () => setState('visible')
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          reveal()
          observer.disconnect()
        }
      },
      { rootMargin: '0px 0px -10% 0px' },
    )
    observer.observe(node)

    // Backstop: if the observer never fires, the content still appears.
    const safety = window.setTimeout(reveal, 2500)
    return () => {
      observer.disconnect()
      window.clearTimeout(safety)
    }
  }, [])

  return (
    <div
      ref={ref}
      className={[state === 'pending' ? 'reveal reveal--pending' : 'reveal', className ?? '']
        .filter(Boolean)
        .join(' ')}
      style={delay ? { transitionDelay: `${delay}ms` } : undefined}
    >
      {children}
    </div>
  )
}

/** What the system actually produces, stated as evidence rather than features. */
const EVIDENCE = [
  {
    value: '10 m',
    label: 'Ground sample',
    note: 'Every measurement resolves to a hundred square metres of surface.',
  },
  {
    value: '~5 days',
    label: 'Revisit',
    note: 'Sentinel-2 returns to the same ground twice a week.',
  },
  {
    value: '13',
    label: 'Report sections',
    note: 'Region, sources, methodology, results and limitations.',
  },
]

export function LandingPage({
  navigate,
  health,
}: {
  navigate: Navigate
  health: HealthResponse | null
}) {
  return (
    <>
      {/* ---------------------------------------------------------------- */}
      {/* Hero                                                              */}
      {/* ---------------------------------------------------------------- */}
      <section className="hero">
        <picture className="hero__media">
          <source media="(max-width: 640px)" srcSet="/hero/forest-portrait.webp" type="image/webp" />
          <source media="(max-width: 640px)" srcSet="/hero/forest-portrait.jpg" />
          <source media="(max-width: 1200px)" srcSet="/hero/forest-mid.webp" type="image/webp" />
          <source srcSet="/hero/forest-wide.webp" type="image/webp" />
          {/* Visible by default; nothing gates it on a preload. */}
          <img src="/hero/forest-wide.jpg" alt="" fetchPriority="high" decoding="async" />
        </picture>
        <div className="hero__veil" aria-hidden="true" />

        <Container wide className="hero__content">
          <h1 className="hero__title">
            Environmental
            <br />
            intelligence
            <br />
            <em>from Earth’s</em>
            <br />
            changing surface
          </h1>

          <p className="hero__lede">
            Satellite observations, geospatial analysis and machine learning, for
            understanding observable environmental change.
          </p>

          <div className="hero__actions">
            <Button variant="primary" size="lg" onClick={() => navigate('/analysis', 'workspace')}>
              Launch analysis
            </Button>
            <Button variant="ghost" size="lg" onClick={() => navigate('/explore')}>
              Explore NatureVision
            </Button>
          </div>
        </Container>

        <div className="hero__foot">
          <Container wide>
            <CalibrationStrip labelled ariaHidden={false} />
            <p className="hero__scale-caption">
              Vegetation index scale — the ramp every map layer in the product uses
            </p>
          </Container>
        </div>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Statement                                                         */}
      {/* ---------------------------------------------------------------- */}
      <section className="statement">
        <Container>
          <Reveal>
            <p className="statement__body">
              NatureVision turns satellite observations into environmental evidence.
              Pick a region and a period, and it finds usable imagery, removes cloud
              and shadow, measures vegetation, compares dates and classifies land
              cover.
            </p>
          </Reveal>
          <Reveal delay={90}>
            <p className="statement__body statement__body--quiet">
              What comes back is not a score. It is a set of measurements with their
              provenance attached — which scene, acquired when, at what cloud cover,
              over how many valid pixels — alongside model predictions that are
              labelled as predictions, and never confused with them.
            </p>
          </Reveal>
        </Container>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Evidence                                                          */}
      {/* ---------------------------------------------------------------- */}
      <section className="evidence">
        <Container>
          <Reveal>
            <h2 className="evidence__title">
              Measured, <em>not estimated</em>
            </h2>
          </Reveal>
          <div className="evidence__grid">
            {EVIDENCE.map((item, index) => (
              <Reveal key={item.label} delay={index * 90}>
                <div className="evidence__item">
                  <p className="evidence__value">{item.value}</p>
                  <p className="evidence__label">{item.label}</p>
                  <p className="evidence__note">{item.note}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </Container>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Integrity                                                         */}
      {/* ---------------------------------------------------------------- */}
      <section className="integrity">
        <Container>
          <div className="integrity__layout">
            <Reveal className="integrity__lead">
              <h2 className="integrity__title">
                What these results
                <br />
                <em>do not</em> claim
              </h2>
            </Reveal>
            <Reveal className="integrity__body" delay={90}>
              <p>
                A vegetation index responds to chlorophyll and canopy density. It does
                not measure biomass, carbon stock, habitat quality or biodiversity, and
                the product never says it does.
              </p>
              <p>
                Comparing two dates cannot separate land-cover change from phenology,
                harvest, drought or a difference in illumination. Results say where the
                index changed and by how much — never why.
              </p>
              <p>
                Written interpretation is generated from the measured evidence and
                checked back against it. A figure that was never measured is rejected
                before anyone reads it.
              </p>
              <p className="integrity__link">
                <button type="button" onClick={() => navigate('/methodology')}>
                  Read the methodology
                </button>
              </p>
            </Reveal>
          </div>
        </Container>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Invitation                                                        */}
      {/* ---------------------------------------------------------------- */}
      <section className="invitation">
        <Container>
          <Reveal>
            <h2 className="invitation__title">
              Run it on somewhere
              <br />
              <em>you care about</em>
            </h2>
            <p className="invitation__body">
              Draw a region, choose one period or two, and the pipeline runs on live
              Sentinel-2 data.
            </p>
            <div className="invitation__actions">
              <Button variant="primary" size="lg" onClick={() => navigate('/analysis', 'workspace')}>
                Launch analysis
              </Button>
            </div>

            {health && (
              <dl className="invitation__status">
                <div>
                  <dt>Imagery</dt>
                  <dd className="mono">{health.imagery_provider}</dd>
                </div>
                <div>
                  <dt>Land-cover model</dt>
                  <dd className="mono">
                    {health.checks?.land_cover_model?.ok
                      ? health.land_cover_model
                      : 'not installed'}
                  </dd>
                </div>
                <div>
                  <dt>Interpretation</dt>
                  <dd className="mono">
                    {health.checks?.interpretation?.ok ? 'configured' : 'not configured'}
                  </dd>
                </div>
              </dl>
            )}
          </Reveal>
        </Container>
      </section>
    </>
  )
}
