import Markdown from 'react-markdown'
import { TracePanel } from '@/features/trace/TracePanel'
import { VerdictBadge } from '@/features/chat/VerdictBadge'
import type { Message } from '@/lib/types'

export function MessageBubble({ message }: { message: Message }) {
  if (message.role === 'user') {
    return (
      <article className="turn turn--user">
        <div className="bubble">{message.content}</div>
      </article>
    )
  }

  return (
    <article className="turn turn--assistant">
      <div className="avatar" aria-hidden="true">
        CV
      </div>
      <div className="answer">
        {message.verdict && <VerdictBadge verdict={message.verdict} />}
        <div className="prose">
          <Markdown>{message.content}</Markdown>
        </div>
        <TracePanel steps={message.trace} sources={message.verdict?.sources ?? []} />
      </div>
    </article>
  )
}
