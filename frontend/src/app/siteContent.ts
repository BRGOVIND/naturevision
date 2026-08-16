/**
 * Site content that is not derived from the application itself.
 *
 * Anything here is a factual claim about the project or its owner, so it is
 * kept in one reviewable file rather than scattered through components. Values
 * that have not been verified are left as null and the UI omits them, rather
 * than shipping a plausible-looking placeholder.
 */

export const SITE = {
  name: 'NatureVision',
  tagline: 'Environmental intelligence from Earth’s changing surface',
  /** Verified from the repository's git configuration. */
  author: {
    name: 'BRGOVIND',
    email: 'brgovind2005@gmail.com',
    github: 'https://github.com/BRGOVIND',
  },
} as const

/**
 * Recognition to display in the footer and About page.
 *
 * INTENTIONALLY EMPTY. No award or hackathon result could be verified from
 * this repository or its project materials, and inventing one would be a
 * fabricated claim about a real person. To display an entry, add it here with
 * the exact name, awarding event and year; the UI renders the section only
 * when this array is non-empty.
 */
export const RECOGNITION: {
  project: string
  award: string
  event: string
  year: string
  url?: string
}[] = []
