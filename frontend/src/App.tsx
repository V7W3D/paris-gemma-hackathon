import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react'
import { ChatView } from '@/features/chat/ChatView'
import { HistorySidebar } from '@/features/history/HistorySidebar'
import { useChat } from '@/hooks/useChat'
import { useConversations } from '@/hooks/useConversations'
import { api } from '@/lib/api'
import type { SystemStatus } from '@/lib/types'
import './App.css'

export default function App() {
  const { conversations, loading, refresh, create, rename, remove } = useConversations()
  const [activeId, setActiveId] = useState<string | null>(null)
  const [collapsed, setCollapsed] = useState(false)
  const [sidebarWidth, setSidebarWidth] = useState(276)
  const [resizing, setResizing] = useState(false)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [verifierEnabled, setVerifierEnabled] = useState(true)
  const creating = useRef(false)

  const { messages, run, busy, error, send, stop } = useChat(activeId, verifierEnabled, refresh)

  useEffect(() => {
    api
      .status()
      .then(setStatus)
      .catch(() => setStatus(null))
  }, [])

  const startChat = useCallback(async () => {
    if (creating.current) return
    creating.current = true
    try {
      setActiveId(await create())
    } finally {
      creating.current = false
    }
  }, [create])

  // Always keep a conversation selected, creating the first one if needed.
  useEffect(() => {
    if (loading || activeId) return
    if (conversations.length > 0) {
      setActiveId(conversations[0].id)
      return
    }
    void startChat()
  }, [activeId, conversations, loading, startChat])

  const handleDelete = useCallback(
    async (id: string) => {
      await remove(id)
      if (id === activeId) setActiveId(null)
    },
    [activeId, remove],
  )

  const active = conversations.find((conversation) => conversation.id === activeId)
  const visible = conversations.filter(
    (conversation) => conversation.message_count > 0 || conversation.id === activeId,
  )

  useEffect(() => {
    if (!resizing) return

    const handlePointerMove = (event: PointerEvent) => {
      setSidebarWidth(Math.min(420, Math.max(220, event.clientX)))
    }
    const stopResizing = () => setResizing(false)

    document.addEventListener('pointermove', handlePointerMove)
    document.addEventListener('pointerup', stopResizing)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    return () => {
      document.removeEventListener('pointermove', handlePointerMove)
      document.removeEventListener('pointerup', stopResizing)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [resizing])

  return (
    <div
      className={`app${collapsed ? ' app--collapsed' : ''}${resizing ? ' app--resizing' : ''}`}
      style={{ '--sidebar-width': `${sidebarWidth}px` } as CSSProperties}
    >
      <HistorySidebar
        conversations={visible}
        activeId={activeId}
        status={status}
        collapsed={collapsed}
        onToggle={() => setCollapsed((value) => !value)}
        onSelect={setActiveId}
        onCreate={() => void startChat()}
        onRename={rename}
        onDelete={(id) => void handleDelete(id)}
      />
      <button
        className="sidebar-resize"
        type="button"
        aria-label="Resize history panel"
        title="Resize history panel"
        aria-valuemin={220}
        aria-valuemax={420}
        aria-valuenow={sidebarWidth}
        onPointerDown={(event) => {
          event.preventDefault()
          setResizing(true)
        }}
        onKeyDown={(event) => {
          if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
          event.preventDefault()
          setSidebarWidth((width) =>
            Math.min(420, Math.max(220, width + (event.key === 'ArrowRight' ? 16 : -16))),
          )
        }}
      />
      <ChatView
        title={active?.title ?? 'New verification'}
        messages={messages}
        run={run}
        busy={busy}
        error={error}
        disabled={!activeId}
        verifierEnabled={verifierEnabled}
        onSend={(content) => void send(content)}
        onStop={stop}
        onVerifierChange={setVerifierEnabled}
      />
    </div>
  )
}
