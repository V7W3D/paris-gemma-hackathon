import { useState } from 'react'
import { STAGE_LABELS, type Evidence, type Stage, type TraceStep } from '@/lib/types'

interface Props {
  steps: TraceStep[]
  activeStage?: Stage | null
  sources?: Evidence[]
  defaultOpen?: boolean
}

/** The decision trail: what agent 1 decided, and what each search brought back. */
export function TracePanel({ steps, activeStage = null, sources = [], defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen)
  if (steps.length === 0 && !activeStage) return null

  const running = activeStage !== null

  return (
    <div className={`trace${running ? ' trace--running' : ''}`}>
      <button className="trace__toggle" onClick={() => setOpen(!open)}>
        <span className={`chevron${open ? ' chevron--open' : ''}`} />
        <span>
          {running ? STAGE_LABELS[activeStage] : `Reasoning trail · ${steps.length} decision points`}
        </span>
        {running && <span className="pulse" />}
      </button>

      {open && (
        <ol className="trace__list">
          {steps.map((step, index) => (
            <li key={`${step.stage}-${index}`} className="trace__step">
              <div className="trace__head">
                <span className="trace__stage">{STAGE_LABELS[step.stage]}</span>
                {step.retrievals.length > 0 && (
                  <span className="trace__tools">
                    <code>{step.retrievals.length} searches</code>
                  </span>
                )}
              </div>
              {step.summary && <p className="trace__summary">{step.summary}</p>}
              {step.retrievals.map((search, searchIndex) => (
                <div
                  key={searchIndex}
                  className={`trace__call${search.ok ? '' : ' trace__call--failed'}`}
                >
                  <code>search</code>
                  <span className="trace__args">{search.query.slice(0, 120)}</span>
                  <span className="trace__count">
                    {search.ok ? `${search.evidence_count} passages` : search.error.slice(0, 60)}
                  </span>
                </div>
              ))}
            </li>
          ))}
          {running && (
            <li className="trace__step trace__step--pending">
              <span className="trace__stage">{STAGE_LABELS[activeStage]}</span>
              <span className="pulse" />
            </li>
          )}
        </ol>
      )}

      {open && sources.length > 0 && (
        <div className="trace__sources">
          <h3>Evidence</h3>
          <ul>
            {sources.map((source) => (
              <li key={source.id}>
                {source.url ? (
                  <a href={source.url} target="_blank" rel="noreferrer">
                    {source.title || source.url}
                  </a>
                ) : (
                  <span className="trace__source">{source.title || source.source || 'Passage'}</span>
                )}
                <span className={`stance stance--${source.stance}`}>{source.stance}</span>
                {source.snippet && <p>{source.snippet.slice(0, 220)}</p>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
