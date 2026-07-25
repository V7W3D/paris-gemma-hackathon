import { useCallback, useEffect, useState } from 'react'
import { api } from '@/lib/api'
import type { ConversationSummary } from '@/lib/types'

export function useConversations() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setConversations(await api.listChats())
      setError(null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'could not load history')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const create = useCallback(async () => {
    const conversation = await api.createChat()
    await refresh()
    return conversation.id
  }, [refresh])

  const rename = useCallback(
    async (id: string, title: string) => {
      await api.renameChat(id, title)
      await refresh()
    },
    [refresh],
  )

  const remove = useCallback(
    async (id: string) => {
      await api.deleteChat(id)
      await refresh()
    },
    [refresh],
  )

  return { conversations, loading, error, refresh, create, rename, remove }
}
