import { useState } from 'react'
import type { ConversationSummary, SystemStatus } from '@/lib/types'

interface Props {
  conversations: ConversationSummary[]
  activeId: string | null
  status: SystemStatus | null
  collapsed: boolean
  onToggle: () => void
  onSelect: (id: string) => void
  onCreate: () => void
  onRename: (id: string, title: string) => void
  onDelete: (id: string) => void
}

type Bucket = 'Today' | 'Previous 7 days' | 'Older'

function bucketOf(iso: string): Bucket {
  const days = (Date.now() - new Date(iso).getTime()) / 86_400_000
  if (days < 1) return 'Today'
  if (days < 7) return 'Previous 7 days'
  return 'Older'
}

function group(conversations: ConversationSummary[]): [Bucket, ConversationSummary[]][] {
  const order: Bucket[] = ['Today', 'Previous 7 days', 'Older']
  return order
    .map((bucket): [Bucket, ConversationSummary[]] => [
      bucket,
      conversations.filter((conversation) => bucketOf(conversation.updated_at) === bucket),
    ])
    .filter(([, items]) => items.length > 0)
}

export function HistorySidebar({
  conversations,
  activeId,
  status,
  collapsed,
  onToggle,
  onSelect,
  onCreate,
  onRename,
  onDelete,
}: Props) {
  const [editing, setEditing] = useState<string | null>(null)
  const [draft, setDraft] = useState('')

  const commit = (id: string) => {
    const title = draft.trim()
    if (title) onRename(id, title)
    setEditing(null)
  }

  return (
    <aside className={`sidebar${collapsed ? ' sidebar--collapsed' : ''}`}>
      <div className="sidebar__top">
        <button className="icon-button" onClick={onToggle} title="Toggle history" aria-label="Toggle history">
          <span className="icon-bars" />
        </button>
        {!collapsed && <span className="wordmark">Claim Verifier</span>}
      </div>

      <button className="new-chat" onClick={onCreate}>
        <span className="plus">+</span>
        {!collapsed && <span>New verification</span>}
      </button>

      {!collapsed && (
        <nav className="history">
          {conversations.length === 0 && <p className="history__empty">No verifications yet.</p>}
          {group(conversations).map(([bucket, items]) => (
            <section key={bucket} className="history__group">
              <h2 className="history__label">{bucket}</h2>
              {items.map((conversation) => (
                <div
                  key={conversation.id}
                  className={`history__item${conversation.id === activeId ? ' history__item--active' : ''}`}
                >
                  {editing === conversation.id ? (
                    <input
                      className="history__input"
                      value={draft}
                      autoFocus
                      onChange={(event) => setDraft(event.target.value)}
                      onBlur={() => commit(conversation.id)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') commit(conversation.id)
                        if (event.key === 'Escape') setEditing(null)
                      }}
                    />
                  ) : (
                    <>
                      <button className="history__title" onClick={() => onSelect(conversation.id)}>
                        {conversation.title}
                      </button>
                      <span className="history__actions">
                        <button
                          className="icon-button icon-button--small"
                          title="Rename"
                          onClick={() => {
                            setEditing(conversation.id)
                            setDraft(conversation.title)
                          }}
                        >
                          ✎
                        </button>
                        <button
                          className="icon-button icon-button--small"
                          title="Delete"
                          onClick={() => onDelete(conversation.id)}
                        >
                          ×
                        </button>
                      </span>
                    </>
                  )}
                </div>
              ))}
            </section>
          ))}
        </nav>
      )}

      {!collapsed && status && (
        <footer className="sidebar__footer">
          <div className="status-row">
            <span className={`dot${status.llm_mocked ? ' dot--hollow' : ''}`} />
            <span>{status.llm_mocked ? 'Mock inference' : status.llm_model}</span>
          </div>
          <div className="status-row">
            <span className={`dot${status.mongo_connected ? '' : ' dot--hollow'}`} />
            <span>{status.mongo_connected ? 'MongoDB' : 'In-memory store'}</span>
          </div>
        </footer>
      )}
    </aside>
  )
}
