import { useRef } from 'react'
import { Composer } from '@/features/chat/Composer'
import { MessageList } from '@/features/chat/MessageList'
import { ModeToggle } from '@/features/chat/ModeToggle'
import type { LiveRun } from '@/hooks/useChat'
import type { Message } from '@/lib/types'

const EXAMPLES = [
  'Mitochondria contain their own circular DNA inherited from ancient bacteria.',
  'Antibiotic resistance can spread between bacteria through horizontal gene transfer.',
  'Humans share about 98% of their DNA with chimpanzees.',
]

interface Props {
  title: string
  messages: Message[]
  run: LiveRun
  busy: boolean
  error: string | null
  disabled: boolean
  verifierEnabled: boolean
  onSend: (content: string) => void
  onStop: () => void
  onVerifierChange: (enabled: boolean) => void
}

export function ChatView({
  title,
  messages,
  run,
  busy,
  error,
  disabled,
  verifierEnabled,
  onSend,
  onStop,
  onVerifierChange,
}: Props) {
  const body = useRef<HTMLDivElement>(null)
  const empty = messages.length === 0 && !busy

  return (
    <main className="chat">
      <header className="chat__header">
        <h1>{title}</h1>
        <ModeToggle enabled={verifierEnabled} disabled={busy} onChange={onVerifierChange} />
      </header>

      <div className="chat__body" ref={body}>
        {empty ? (
          <div className="welcome">
            <h2>{verifierEnabled ? 'What should I verify?' : 'What do you want to ask?'}</h2>
            <p>
              {verifierEnabled
                ? 'Every claim is decomposed, researched with live sources, and judged one decision point at a time.'
                : 'The verifier is off, so the model answers on its own — no claims, no sources, no verdict.'}
            </p>
            <div className="examples">
              {EXAMPLES.map((example) => (
                <button key={example} onClick={() => onSend(example)} disabled={disabled}>
                  {example}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <MessageList messages={messages} run={run} busy={busy} scrollRef={body} />
        )}
      </div>

      {error && <div className="banner">{error}</div>}

      <Composer
        busy={busy}
        disabled={disabled}
        verifierEnabled={verifierEnabled}
        onSend={onSend}
        onStop={onStop}
      />
    </main>
  )
}
