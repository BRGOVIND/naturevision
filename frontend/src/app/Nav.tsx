/** Primary site navigation. */

import { useEffect, useState } from 'react'

import { Button } from '../design/primitives'
import type { Navigate, Route } from './router'

const LINKS: { label: string; route: Route }[] = [
  { label: 'Explore', route: '/explore' },
  { label: 'Analysis', route: '/analysis' },
  { label: 'Methodology', route: '/methodology' },
  { label: 'Reports', route: '/reports' },
  { label: 'About', route: '/about' },
]

export function Nav({ route, navigate }: { route: Route; navigate: Navigate }) {
  const [open, setOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // Close the mobile menu on route change and on Escape.
  useEffect(() => setOpen(false), [route])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  function go(to: Route) {
    navigate(to)
    setOpen(false)
  }

  return (
    <header className={scrolled ? 'nav nav--scrolled' : 'nav'}>
      <div className="nav__inner">
        <a
          className="nav__brand"
          href="/"
          onClick={(e) => {
            e.preventDefault()
            go('/')
          }}
        >
          <span className="nav__mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none">
              <path
                d="M12 2.5c-4.2 3.1-6.6 6.6-6.6 10.2A6.6 6.6 0 0 0 12 21.5a6.6 6.6 0 0 0 6.6-8.8c0-3.6-2.4-7.1-6.6-10.2Z"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinejoin="round"
              />
              <path d="M12 21.5V8.4M12 12.6l3.1-3.1M12 15.8l-2.7-2.7" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
          </span>
          <span className="nav__wordmark">
            NatureVision
            <span className="nav__sub">Environmental Intelligence</span>
          </span>
        </a>

        <nav className="nav__links" aria-label="Primary">
          {LINKS.map((link) => (
            <a
              key={link.route}
              href={link.route}
              aria-current={route === link.route ? 'page' : undefined}
              className={route === link.route ? 'nav__link nav__link--active' : 'nav__link'}
              onClick={(e) => {
                e.preventDefault()
                go(link.route)
              }}
            >
              {link.label}
            </a>
          ))}
        </nav>

        <div className="nav__action">
          <Button variant="primary" onClick={() => navigate('/analysis', 'workspace')}>
            Launch analysis
          </Button>
        </div>

        <button
          className="nav__toggle"
          aria-expanded={open}
          aria-controls="nav-mobile"
          aria-label={open ? 'Close menu' : 'Open menu'}
          onClick={() => setOpen((v) => !v)}
        >
          <span className={open ? 'nav__burger nav__burger--open' : 'nav__burger'} aria-hidden="true" />
        </button>
      </div>

      <div id="nav-mobile" className={open ? 'nav__mobile nav__mobile--open' : 'nav__mobile'} hidden={!open}>
        {LINKS.map((link) => (
          <a
            key={link.route}
            href={link.route}
            aria-current={route === link.route ? 'page' : undefined}
            onClick={(e) => {
              e.preventDefault()
              go(link.route)
            }}
          >
            {link.label}
          </a>
        ))}
        <Button variant="primary" full onClick={() => navigate('/analysis', 'workspace')}>
          Launch analysis
        </Button>
      </div>
    </header>
  )
}
