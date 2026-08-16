/** Public landing experience. */

import { useEffect, useRef, useState } from 'react'

import {
  Button,
  CalibrationStrip,
  Card,
  Container,
  Eyebrow,
  Section,
  SectionHead,
} from '../design/primitives'
import type { Navigate } from '../app/router'
import type { HealthResponse } from '../types/api'

/** Real Sentinel-2 band definitions, mirroring the backend's band registry. */
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
    body: 'Windowed reads over Sentinel-2 Level-2A cloud-optimised imagery, with scene-classification masking for cloud, shadow, cirrus and snow.',
    detail: '6 spectral bands · 10 m',
    accent: 'var(--lichen)',
  },
  {
    key: 'vegetation',
    title: 'Vegetation intelligence',
    body: 'NDVI computed per pixel with explicit handling of nodata and near-zero denominators, then summarised into density classes.',
    detail: '(NIR − Red) / (NIR + Red)',
    accent: 'var(--moss-bright)',
    mono: true,
  },
  {
    key: 'land-cover',
    title: 'Land-cover intelligence',
    body: 'A trained classifier assigns forest, agriculture, water, built-up and bare land, with per-pixel confidence and published hold-out accuracy.',
    detail: '5 classes · 11 features',
    accent: 'var(--model)',
  },
  {
    key: 'temporal',
    title: 'Temporal change',
    body: 'Two acquisitions co-registered onto one pixel grid, differenced only where both dates are cloud-free, and classified against documented thresholds.',
    detail: 'Δ 0.10 / 0.20 thresholds',
    accent: 'var(--clay)',
  },
  {
    key: 'interpretation',
    title: 'Nature intelligence',
    body: 'Measured evidence is turned into structured interpretation, then checked back against the evidence. A figure that was never measured is rejected.',
    detail: 'Grounding-validated',
    accent: 'var(--interpreted)',
  },
  {
    key: 'reporting',
    title: 'Reporting',
    body: 'Thirteen sections covering region, sources, methodology, statistics, results and limitations — each tagged with where its content came from.',
    detail: 'HTML export',
    accent: 'var(--sandstone)',
  },
]

const PIPELINE = [
  { stage: 'Satellite observations', note: 'Catalogue search, ranked by coverage then cloud' },
  { stage: 'Geospatial preprocessing', note: 'Windowed reads, calibration, co-registration, masking' },
  { stage: 'Spectral analysis', note: 'Band arithmetic on a single pixel lattice' },
  { stage: 'Vegetation index', note: 'NDVI with nodata and zero-denominator guards' },
  { stage: 'Temporal comparison', note: 'Difference over pixels valid in both periods' },
  { stage: 'Land-cover model', note: 'Per-pixel classification with confidence' },
  { stage: 'Environmental evidence', note: 'Deterministic values, provenance-tagged' },
  { stage: 'Interpretation', note: 'Generated from evidence, validated against it' },
  { stage: 'Report', note: 'Structured, exportable, limitations included' },
]

const TRUST = [
  {
    title: 'NDVI is an indicator, not a census',
    body: 'It responds to chlorophyll and canopy density. It does not measure biomass, carbon stock, habitat quality or biodiversity, and the product never says it does.',
  },
  {
    title: 'Land cover is predicted, not observed',
    body: 'Class shares come from a statistical model with real error. Its hold-out accuracy and per-class F1 are published, including where they are weak.',
  },
  {
    title: 'Interpretation is grounded and checked',
    body: 'The language layer sees only measured evidence. Every number it writes is matched back against that evidence, and unsupported causal claims are flagged.',
  },
  {
    title: 'Change is change, not cause',
    body: 'A two-date optical comparison cannot separate land-cover conversion from phenology, harvest, drought or illumination. Results say where and how much, never why.',
  },
]

export function LandingPage({
  navigate,
  health,
}: {
  navigate: Navigate
  health: HealthResponse | null
}) {
  const [heroReady, setHeroReady] = useState(false)
  const heroImage = useRef<HTMLImageElement>(null)

  // The reveal animation is additive: the image is visible by default, and
  // this flag only adds the fade. A missed load event can slow the animation
  // down but can never leave the hero blank.
  useEffect(() => {
    if (heroImage.current?.complete) setHeroReady(true)
  }, [])

  return (
    <>
      {/* ---------------------------------------------------------------- */}
      {/* Hero                                                              */}
      {/* ---------------------------------------------------------------- */}
      <section className={heroReady ? 'hero hero--ready' : 'hero'}>
        <picture className="hero__media">
          <source
            media="(max-width: 640px)"
            srcSet="/hero/forest-portrait.webp"
            type="image/webp"
          />
          <source media="(max-width: 640px)" srcSet="/hero/forest-portrait.jpg" />
          <source media="(max-width: 1200px)" srcSet="/hero/forest-mid.webp" type="image/webp" />
          <source media="(max-width: 1200px)" srcSet="/hero/forest-mid.jpg" />
          <source srcSet="/hero/forest-wide.webp" type="image/webp" />
          <img
            ref={heroImage}
            src="/hero/forest-wide.jpg"
            alt=""
            fetchPriority="high"
            decoding="async"
            onLoad={() => setHeroReady(true)}
          />
        </picture>
        <div className="hero__veil" aria-hidden="true" />

        <Container wide className="hero__content">
          <Eyebrow tone="dark">Geospatial environmental intelligence</Eyebrow>
          <h1 className="hero__title">
            Environmental intelligence from Earth’s changing surface
          </h1>
          <p className="hero__lede">
            NatureVision analyses Sentinel-2 satellite observations to measure vegetation
            condition, detect change between dates, and classify land cover — then turns
            those measurements into an interpretable environmental report.
          </p>
          <div className="hero__actions">
            <Button variant="primary" size="lg" onClick={() => navigate('/analysis', 'workspace')}>
              Explore NatureVision
            </Button>
            <Button variant="ghost" size="lg" onClick={() => navigate('/methodology')}>
              View methodology
            </Button>
          </div>

          <dl className="hero__stats">
            <div>
              <dt>Imagery</dt>
              <dd>Sentinel-2 L2A</dd>
            </div>
            <div>
              <dt>Ground sample</dt>
              <dd>10 m</dd>
            </div>
            <div>
              <dt>Revisit</dt>
              <dd>~5 days</dd>
            </div>
            <div>
              <dt>Land-cover classes</dt>
              <dd>5</dd>
            </div>
          </dl>
        </Container>

        <div className="hero__scale">
          <CalibrationStrip labelled ariaHidden={false} />
          <p className="hero__scale-caption">
            Vegetation index scale — the same ramp used by every map layer in the product
          </p>
        </div>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Intro                                                             */}
      {/* ---------------------------------------------------------------- */}
      <Section id="intro" tone="cream">
        <Container>
          <div className="intro">
            <div className="intro__lede">
              <SectionHead
                eyebrow="What it does"
                title="Satellite observations, turned into environmental evidence"
              />
              <p>
                Pick a region and a time period. NatureVision finds usable Sentinel-2
                observations, removes cloud and shadow, computes vegetation indices on a
                consistent pixel grid, compares dates, and classifies land cover.
              </p>
              <p>
                What comes back is not a score or a verdict. It is a set of measurements
                with their provenance attached — which scene, acquired when, at what cloud
                cover, over how many valid pixels — alongside model predictions that are
                labelled as predictions.
              </p>
            </div>

            <aside className="intro__spectrum" aria-labelledby="bands-heading">
              <h3 id="bands-heading" className="intro__spectrum-title">
                Bands read per analysis
              </h3>
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
              <p className="intro__spectrum-note">
                Plus the Level-2A scene classification layer, used to mask cloud, shadow,
                cirrus and snow before any statistic is computed.
              </p>
            </aside>
          </div>
        </Container>
      </Section>

      {/* ---------------------------------------------------------------- */}
      {/* Capabilities                                                      */}
      {/* ---------------------------------------------------------------- */}
      <Section id="capabilities" tone="canopy">
        <Container wide>
          <SectionHead
            eyebrow="Capabilities"
            tone="dark"
            title="A measurement pipeline, end to end"
            lede="Each stage is deterministic where it can be, and explicit about its uncertainty where it cannot."
          />
          <ul className="capability-grid">
            {CAPABILITIES.map((c, index) => (
              <Card as="li" key={c.key} accent={c.accent} className="capability">
                <span className="capability__index mono" aria-hidden="true">
                  {String(index + 1).padStart(2, '0')}
                </span>
                <h3>{c.title}</h3>
                <p>{c.body}</p>
                <p className={c.mono ? 'capability__detail mono' : 'capability__detail'}>
                  {c.detail}
                </p>
              </Card>
            ))}
          </ul>
        </Container>
      </Section>

      {/* ---------------------------------------------------------------- */}
      {/* Methodology                                                       */}
      {/* ---------------------------------------------------------------- */}
      <Section id="pipeline" tone="cream">
        <Container>
          <SectionHead
            eyebrow="How it works"
            title="From orbit to report"
            lede="Nine stages. The order matters, because each one constrains what the next is allowed to claim."
          />
          <ol className="pipeline">
            {PIPELINE.map((step, index) => (
              <li key={step.stage} className="pipeline__step">
                <span className="pipeline__rule" aria-hidden="true" />
                <span className="pipeline__num mono">{String(index + 1).padStart(2, '0')}</span>
                <div>
                  <h3>{step.stage}</h3>
                  <p>{step.note}</p>
                </div>
              </li>
            ))}
          </ol>
          <div className="pipeline__cta">
            <Button variant="secondary" onClick={() => navigate('/methodology')}>
              Read the full methodology
            </Button>
          </div>
        </Container>
      </Section>

      {/* ---------------------------------------------------------------- */}
      {/* Trust                                                             */}
      {/* ---------------------------------------------------------------- */}
      <Section id="trust" tone="charcoal">
        <Container>
          <SectionHead
            eyebrow="Scientific scope"
            tone="dark"
            title="What these results do and do not support"
            lede="Stated up front, because the limits of the method are part of the result."
          />
          <div className="trust-grid">
            {TRUST.map((item) => (
              <Card key={item.title} className="trust-card">
                <h3>{item.title}</h3>
                <p>{item.body}</p>
              </Card>
            ))}
          </div>
        </Container>
      </Section>

      {/* ---------------------------------------------------------------- */}
      {/* Launch                                                            */}
      {/* ---------------------------------------------------------------- */}
      <Section id="launch" tone="canopy">
        <Container>
          <div className="launch">
            <div>
              <SectionHead
                eyebrow="Start"
                tone="dark"
                title="Run an analysis on a region you care about"
                lede="Draw an area, choose one period or two, and the pipeline runs on live satellite data."
              />
              <Button variant="primary" size="lg" onClick={() => navigate('/analysis', 'workspace')}>
                Launch analysis
              </Button>
            </div>
            {health && (
              <ul className="launch__status" aria-label="Service status">
                <li>
                  <span>Imagery source</span>
                  <strong className="mono">{health.imagery_provider}</strong>
                </li>
                <li>
                  <span>Land-cover model</span>
                  <strong className="mono">
                    {health.checks?.land_cover_model?.ok ? health.land_cover_model : 'not installed'}
                  </strong>
                </li>
                <li>
                  <span>Interpretation</span>
                  <strong className="mono">
                    {health.checks?.interpretation?.ok ? 'configured' : 'not configured'}
                  </strong>
                </li>
              </ul>
            )}
          </div>
        </Container>
      </Section>
    </>
  )
}
