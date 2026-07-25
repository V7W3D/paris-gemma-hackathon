import type { Verdict } from '@/lib/types'

const LABELS: Record<Verdict['label'], string> = {
  true: 'Supported',
  false: 'Refuted',
  mixed: 'Mixed',
  unverified: 'Unverified',
}

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  const percent = Math.round(verdict.confidence * 100)
  return (
    <div className={`verdict verdict--${verdict.label}`}>
      <span className="verdict__label">{LABELS[verdict.label]}</span>
      <span className="verdict__meter" aria-label={`confidence ${percent}%`}>
        <span className="verdict__fill" style={{ width: `${percent}%` }} />
      </span>
      <span className="verdict__confidence">{percent}% confidence</span>
    </div>
  )
}
