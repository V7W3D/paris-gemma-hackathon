import { useEffect } from 'react'
import type { RefObject } from 'react'
import Markdown from 'react-markdown'
import { MessageBubble } from '@/features/chat/MessageBubble'
import { TracePanel } from '@/features/trace/TracePanel'
import type { LiveRun } from '@/hooks/useChat'
import type { Message } from '@/lib/types'

interface Props {
  messages: Message[]
  run: LiveRun
  busy: boolean
  scrollRef: RefObject<HTMLDivElement | null>
}

const STICK_THRESHOLD = 260

export function MessageList({ messages, run, busy, scrollRef }: Props) {
  useEffect(() => {
    const container = scrollRef.current
    if (!container) return
    // Only follow the conversation while the reader is already at the bottom.
    const distance = container.scrollHeight - container.scrollTop - container.clientHeight
    if (distance > STICK_THRESHOLD) return
    container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' })
  }, [messages.length, run.answer, run.steps.length, scrollRef])

  return (
    <div className="messages">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}

      {busy && (
        <article className="turn turn--assistant">
          <div className="avatar" aria-hidden="true">
            CV
          </div>
          <div className="answer">
            <TracePanel
              steps={run.steps}
              activeStage={run.stage}
              sources={run.evidence}
              defaultOpen
            />
            {run.claims.length > 0 && run.answer === '' && (
              <ul className="claim-preview">
                {run.claims.map((claim) => (
                  <li key={claim.id}>{claim.text}</li>
                ))}
              </ul>
            )}
            {run.answer && (
              <div className="prose">
                <Markdown>{run.answer}</Markdown>
              </div>
            )}
          </div>
        </article>
      )}
    </div>
  )
}
