import { useEffect, useRef, useState } from 'react'

interface Props {
  busy: boolean
  disabled: boolean
  verifierEnabled: boolean
  onSend: (content: string) => void
  onStop: () => void
}

export function Composer({ busy, disabled, verifierEnabled, onSend, onStop }: Props) {
  const [value, setValue] = useState('')
  const textarea = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const element = textarea.current
    if (!element) return
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 220)}px`
  }, [value])

  const submit = () => {
    const content = value.trim()
    if (!content || busy || disabled) return
    onSend(content)
    setValue('')
  }

  return (
    <div className="composer">
      <div className="composer__box">
        <textarea
          ref={textarea}
          rows={1}
          value={value}
          disabled={disabled}
          placeholder={verifierEnabled ? 'Paste a claim to verify…' : 'Ask the model anything…'}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              submit()
            }
          }}
        />
        {busy ? (
          <button className="send send--stop" onClick={onStop} title="Stop">
            <span className="square" />
          </button>
        ) : (
          <button
            className="send"
            onClick={submit}
            disabled={!value.trim() || disabled}
            title={verifierEnabled ? 'Verify' : 'Send'}
          >
            <span className="arrow" />
          </button>
        )}
      </div>
      <p className="composer__hint">
        {verifierEnabled
          ? 'Two agents, five decision points, sources cited.'
          : 'Verifier off — the model answers unchecked, with no sources.'}{' '}
        Enter to send, Shift + Enter for a new line.
      </p>
    </div>
  )
}
