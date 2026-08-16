/** Methodology, data sources and scientific scope. */

import { useEffect, useState } from 'react'

import { CalibrationStrip, Container, Section, SectionHead } from '../design/primitives'
import { api } from '../services/api'
import type { ModelInfo } from '../types/api'

const SOURCES = [
  {
    name: 'Sentinel-2 Level-2A',
    role: 'All imagery, indices, change detection and classifier inputs',
    detail: 'Element84 Earth Search over AWS Open Data · 10 m · ~5 day revisit',
    licence: 'Copernicus Sentinel data, free and open',
  },
  {
    name: 'ESA WorldCover v200 (2021)',
    role: 'Reference labels for training and evaluating the land-cover model',
    detail: 'Public COG tiles · 10 m global · training only, never read at inference',
    licence: 'CC BY 4.0',
  },
  {
    name: 'Esri World Imagery / OpenStreetMap',
    role: 'Map basemaps',
    detail: 'Keyless raster tiles, attributed in the map',
    licence: 'Provider terms',
  },
]

const STAGES = [
  {
    id: 'acquisition',
    title: 'Acquisition and masking',
    body: 'Observations are ranked by regional coverage first, then cloud cover, because a partially overlapping scene cannot be repaired whereas moderate cloud can be masked. Only the byte ranges covering the region are read from each cloud-optimised GeoTIFF. The Level-2A scene classification then removes no-data, saturated, cloud-shadow, cloud, cirrus and snow pixels.',
  },
  {
    id: 'calibration',
    title: 'Radiometric calibration',
    body: 'Processing baseline 04.00 introduced a −1000 digital-number offset, and the catalogue advertises it on every item — including acquisitions that predate the change. Applying it blindly pushes the NDVI denominator toward zero and destroys the result. The declared offset is therefore tested against a small probe read and rejected when it drives reflectance implausibly negative. The decision is made once per observation and applied to every band.',
  },
  {
    id: 'index',
    title: 'Vegetation index',
    body: 'NDVI = (NIR − Red) / (NIR + Red), from Sentinel-2 B08 and B04. A pixel is discarded when either input is masked, when the denominator is within 1e-6 of zero, or when the result falls outside the achievable −1 to +1 range. Dispersion statistics are withheld below 30 valid pixels rather than reported as if meaningful.',
  },
  {
    id: 'change',
    title: 'Temporal change',
    body: 'Period B is reprojected onto period A’s pixel grid before differencing, so the comparison happens on identical geometry. Only pixels valid in both periods contribute; anything masked in either date is excluded rather than counted as change. Change beyond 0.10 index units is moderate, beyond 0.20 significant — both configurable and reported with every result.',
  },
  {
    id: 'land-cover',
    title: 'Land-cover model',
    body: 'A classifier over 11 per-pixel spectral features assigns five classes with per-pixel confidence. Accuracy is measured on geographically disjoint hold-out regions, never a random pixel split, because adjacent 10 m pixels are strongly autocorrelated and a random split would badly overstate performance.',
  },
  {
    id: 'interpretation',
    title: 'Interpretation and validation',
    body: 'The language layer receives only the finished evidence package and performs no measurement. Its response must match a strict schema, and every number it writes is matched back against the evidence. A figure that was never measured is treated as fatal. Claims the method cannot support are flagged and surfaced in the report.',
  },
]

const LIMITS = [
  'Optical imagery cannot see through cloud. Masked pixels are excluded, not estimated.',
  'A vegetation index measures reflectance-derived greenness, not biomass, carbon stock, habitat quality or biodiversity.',
  'Two dates cannot separate land-cover change from phenology, crop cycles, drought response or differences in illumination and view geometry.',
  'No causal attribution is produced. The analysis identifies where and how much the index changed, not why.',
  'Land-cover classes are per-pixel predictions and carry classification error, particularly on mixed pixels and class boundaries.',
  'Model accuracy is measured against ESA WorldCover, which is itself a model product with its own error. Agreement with it is not ground truth.',
]

export function MethodologyPage() {
  const [models, setModels] = useState<ModelInfo | null>(null)

  useEffect(() => {
    api
      .models()
      .then(setModels)
      .catch(() => setModels(null))
  }, [])

  const installed = models?.models?.[0] as Record<string, any> | undefined

  return (
    <>
      <Section tone="canopy" className="page-head">
        <Container>
          <SectionHead
            eyebrow="Methodology"
            tone="dark"
            title="How a measurement becomes a result"
            lede="Each stage constrains what the next one is allowed to claim. This page documents those constraints."
          />
        </Container>
        <CalibrationStrip />
      </Section>

      <Section tone="cream">
        <Container>
          <div className="stages">
            {STAGES.map((stage) => (
              <article key={stage.id} id={stage.id} className="stage">
                <h3>{stage.title}</h3>
                <p>{stage.body}</p>
              </article>
            ))}
          </div>
        </Container>
      </Section>

      <Section id="sources" tone="charcoal">
        <Container>
          <SectionHead eyebrow="Provenance" tone="dark" title="Data sources" />
          <ul className="source-list">
            {SOURCES.map((source) => (
              <li key={source.name}>
                <h3>{source.name}</h3>
                <p className="source-list__role">{source.role}</p>
                <p className="source-list__detail mono">{source.detail}</p>
                <p className="source-list__licence">{source.licence}</p>
              </li>
            ))}
          </ul>
        </Container>
      </Section>

      {installed && (
        <Section tone="cream">
          <Container>
            <SectionHead
              eyebrow="Model card"
              title="Installed land-cover model"
              lede="Measured on spatially held-out regions. Where performance is weak, it is published as weak."
            />
            <div className="model-card">
              <dl className="model-card__facts">
                <div>
                  <dt>Name</dt>
                  <dd className="mono">{installed.name}</dd>
                </div>
                <div>
                  <dt>Version</dt>
                  <dd className="mono">{installed.version}</dd>
                </div>
                <div>
                  <dt>Backend</dt>
                  <dd className="mono">{installed.backend}</dd>
                </div>
                <div>
                  <dt>Hold-out accuracy</dt>
                  <dd className="mono">
                    {installed.overall_accuracy != null
                      ? installed.overall_accuracy.toFixed(4)
                      : 'not evaluated'}
                  </dd>
                </div>
                <div>
                  <dt>Macro F1</dt>
                  <dd className="mono">
                    {installed.macro_f1 != null ? installed.macro_f1.toFixed(4) : 'not evaluated'}
                  </dd>
                </div>
                <div>
                  <dt>Evaluation samples</dt>
                  <dd className="mono">
                    {installed.evaluation_samples?.toLocaleString() ?? '—'}
                  </dd>
                </div>
              </dl>

              {installed.per_class_metrics && (
                <table className="model-card__table">
                  <caption>Per-class performance on held-out regions</caption>
                  <thead>
                    <tr>
                      <th scope="col">Class</th>
                      <th scope="col">Precision</th>
                      <th scope="col">Recall</th>
                      <th scope="col">F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(
                      installed.per_class_metrics as Record<string, Record<string, number>>,
                    ).map(([label, m]) => (
                      <tr key={label}>
                        <th scope="row">{label}</th>
                        <td className="tabular">{m.precision?.toFixed(3)}</td>
                        <td className="tabular">{m.recall?.toFixed(3)}</td>
                        <td className="tabular">{m.f1?.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}

              {installed.evaluation_protocol && (
                <p className="model-card__protocol">{installed.evaluation_protocol}</p>
              )}
            </div>
          </Container>
        </Section>
      )}

      <Section id="limitations" tone="canopy">
        <Container>
          <SectionHead
            eyebrow="Scientific scope"
            tone="dark"
            title="Limitations"
            lede="Included in every generated report, and specific to each run."
          />
          <ul className="limit-list">
            {LIMITS.map((limit) => (
              <li key={limit}>{limit}</li>
            ))}
          </ul>
        </Container>
      </Section>
    </>
  )
}
