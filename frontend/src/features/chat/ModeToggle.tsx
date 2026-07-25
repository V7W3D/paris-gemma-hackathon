interface Props {
  enabled: boolean
  disabled: boolean
  onChange: (enabled: boolean) => void
}

const HINT = {
  on: 'Verifier on: claims are decomposed, researched and judged before you see an answer.',
  off: 'Verifier off: one plain model answer, with no sources and no verdict.',
}

/** Switches the turn between the full verification pipeline and a direct answer. */
export function ModeToggle({ enabled, disabled, onChange }: Props) {
  return (
    <div className="mode" title={enabled ? HINT.on : HINT.off}>
      <span className="mode__label">{enabled ? 'Verifier agent' : 'Direct answer'}</span>
      <button
        type="button"
        role="switch"
        aria-checked={enabled}
        aria-label="Verifier agent"
        className={`switch${enabled ? ' switch--on' : ''}`}
        disabled={disabled}
        onClick={() => onChange(!enabled)}
      >
        <span className="switch__knob" />
      </button>
    </div>
  )
}
