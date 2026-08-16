/**
 * Shared presentation primitives.
 *
 * Every visual element the pages compose from lives here, so the design system
 * is enforced structurally rather than by convention.
 */

import type { CSSProperties, ReactNode } from 'react'

type Tone = 'canopy' | 'cream' | 'charcoal'

interface SectionProps {
  id?: string
  tone?: Tone
  children: ReactNode
  className?: string
  /** Removes the default vertical rhythm, for sections that manage their own. */
  flush?: boolean
}

export function Section({ id, tone = 'cream', children, className, flush }: SectionProps) {
  return (
    <section
      id={id}
      className={[`section section--${tone}`, flush ? 'section--flush' : '', className ?? '']
        .filter(Boolean)
        .join(' ')}
    >
      {children}
    </section>
  )
}

export function Container({
  children,
  wide,
  className,
}: {
  children: ReactNode
  wide?: boolean
  className?: string
}) {
  return (
    <div className={[wide ? 'container container--wide' : 'container', className ?? ''].filter(Boolean).join(' ')}>
      {children}
    </div>
  )
}

/**
 * The signature element: a calibrated NDVI scale.
 *
 * This is the same ramp the map legend and the backend renderer use, drawn as
 * an instrument scale. It separates major sections so the page reads as a
 * measured document rather than a marketing page.
 */
export function CalibrationStrip({
  labelled = false,
  ariaHidden = true,
}: {
  labelled?: boolean
  ariaHidden?: boolean
}) {
  const ticks = ['−1.0', '−0.5', '0', '+0.5', '+1.0']
  return (
    <div
      className={labelled ? 'calibration calibration--labelled' : 'calibration'}
      aria-hidden={ariaHidden || undefined}
      role={ariaHidden ? undefined : 'img'}
      aria-label={ariaHidden ? undefined : 'Vegetation index scale from minus one to plus one'}
    >
      <div className="calibration__ramp" />
      {labelled && (
        <div className="calibration__ticks tabular">
          {ticks.map((t) => (
            <span key={t}>{t}</span>
          ))}
        </div>
      )}
    </div>
  )
}

export function Eyebrow({ children, tone }: { children: ReactNode; tone?: 'dark' | 'light' }) {
  return <p className={tone === 'dark' ? 'eyebrow eyebrow--dark' : 'eyebrow'}>{children}</p>
}

/**
 * A section heading. `eyebrow` exists for the rare case where a small label
 * genuinely aids navigation; it is not decoration and most sections omit it.
 */
export function SectionHead({
  eyebrow,
  title,
  lede,
  tone,
  align = 'left',
}: {
  eyebrow?: string
  title: string
  lede?: string
  tone?: 'dark' | 'light'
  align?: 'left' | 'center'
}) {
  return (
    <header className={`section-head section-head--${align}`}>
      {eyebrow && <Eyebrow tone={tone}>{eyebrow}</Eyebrow>}
      <h2 className="section-title">{title}</h2>
      {lede && <p className="section-lede">{lede}</p>}
    </header>
  )
}

export type Provenance = 'observed' | 'model_prediction' | 'interpretation' | 'metadata'

const PROVENANCE_LABEL: Record<Provenance, string> = {
  observed: 'Observed',
  model_prediction: 'Model',
  interpretation: 'Interpreted',
  metadata: 'Metadata',
}

const PROVENANCE_TITLE: Record<Provenance, string> = {
  observed: 'Measured by deterministic processing of satellite imagery.',
  model_prediction: 'Predicted by a statistical model. Carries classification error.',
  interpretation: 'Written by a language model from the measured evidence.',
  metadata: 'Configuration and provenance.',
}

export function ProvenanceBadge({ kind, label }: { kind: Provenance; label?: string }) {
  return (
    <span className={`badge badge--${kind}`} title={PROVENANCE_TITLE[kind]}>
      <span className="badge__dot" aria-hidden="true" />
      {label ?? PROVENANCE_LABEL[kind]}
    </span>
  )
}

interface ButtonProps {
  children: ReactNode
  onClick?: () => void
  href?: string
  variant?: 'primary' | 'secondary' | 'ghost' | 'quiet'
  size?: 'md' | 'lg'
  disabled?: boolean
  type?: 'button' | 'submit'
  title?: string
  full?: boolean
  download?: boolean
}

export function Button({
  children,
  onClick,
  href,
  variant = 'secondary',
  size = 'md',
  disabled,
  type = 'button',
  title,
  full,
  download,
}: ButtonProps) {
  const className = [
    'btn',
    `btn--${variant}`,
    size === 'lg' ? 'btn--lg' : '',
    full ? 'btn--full' : '',
  ]
    .filter(Boolean)
    .join(' ')

  if (href && !disabled) {
    const external = href.startsWith('http') || download
    return (
      <a
        className={className}
        href={href}
        title={title}
        {...(external ? { target: '_blank', rel: 'noreferrer' } : {})}
      >
        {children}
      </a>
    )
  }
  return (
    <button className={className} onClick={onClick} disabled={disabled} type={type} title={title}>
      {children}
    </button>
  )
}

export function Card({
  children,
  className,
  accent,
  as: Tag = 'article',
  style,
}: {
  children: ReactNode
  className?: string
  accent?: string
  as?: 'article' | 'div' | 'li'
  style?: CSSProperties
}) {
  return (
    <Tag
      className={['card', className ?? ''].filter(Boolean).join(' ')}
      style={accent ? ({ ...style, '--card-accent': accent } as CSSProperties) : style}
    >
      {children}
    </Tag>
  )
}

/** A labelled key/value pair, set in the instrument mono face. */
export function DataRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="data-row">
      <dt>{label}</dt>
      <dd className="mono">{value}</dd>
    </div>
  )
}

export function EmptyState({
  title,
  body,
  action,
}: {
  title: string
  body: string
  action?: ReactNode
}) {
  return (
    <div className="empty-state">
      <div className="empty-state__mark" aria-hidden="true">
        <CalibrationStrip />
      </div>
      <h3>{title}</h3>
      <p>{body}</p>
      {action}
    </div>
  )
}

export function Spinner({ label }: { label: string }) {
  return (
    <span className="spinner" role="status">
      <span className="spinner__ring" aria-hidden="true" />
      <span className="visually-hidden">{label}</span>
    </span>
  )
}
