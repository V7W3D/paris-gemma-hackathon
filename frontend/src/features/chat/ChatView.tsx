import { useRef } from 'react'
import { Composer } from '@/features/chat/Composer'
import { MessageList } from '@/features/chat/MessageList'
import type { LiveRun } from '@/hooks/useChat'
import type { Message } from '@/lib/types'

const EXAMPLES = [
  'The Eiffel Tower is 330 metres tall including its antennas.',
  'Norway generates more than 95% of its electricity from hydropower.',
  'The 2024 Paris Olympics had more athletes than the 2020 Tokyo games.',
]

interface Props {
  title: string
  messages: Message[]
  run: LiveRun
  busy: boolean
  error: string | null
  disabled: boolean
  onSend: (content: string) => void
  onStop: () => void
}

export function ChatView({ title, messages, run, busy, error, disabled, onSend, onStop }: Props) {
  const body = useRef<HTMLDivElement>(null)
  const empty = messages.length === 0 && !busy

  return (
    <main className="chat">
      <header className="chat__header">
        <h1>{title}</h1>
      </header>

      <div className="chat__body" ref={body}>
        {empty ? (
          <div className="welcome">
            <h2>What should I verify?</h2>
            <p>
              Every claim is decomposed, researched with live sources, and judged one decision point
              at a time.
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

      <Composer busy={busy} disabled={disabled} onSend={onSend} onStop={onStop} />
    </main>
  )
}
