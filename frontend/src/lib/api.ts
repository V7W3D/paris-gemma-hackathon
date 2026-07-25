import type {
  Conversation,
  ConversationSummary,
  StreamEvent,
  SystemStatus,
} from './types'

/** Empty in local/dev (Vite proxies /api). Set VITE_API_BASE for a split deploy. */
const API_ROOT = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '')
const BASE = `${API_ROOT}/api`

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`${response.status} ${detail.slice(0, 200)}`)
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T)
}

export const api = {
  status: () => request<SystemStatus>('/status'),
  listChats: () => request<ConversationSummary[]>('/chats'),
  createChat: (title?: string) =>
    request<Conversation>('/chats', { method: 'POST', body: JSON.stringify({ title }) }),
  getChat: (id: string) => request<Conversation>(`/chats/${id}`),
  renameChat: (id: string, title: string) =>
    request<Conversation>(`/chats/${id}`, { method: 'PATCH', body: JSON.stringify({ title }) }),
  deleteChat: (id: string) => request<void>(`/chats/${id}`, { method: 'DELETE' }),
}

/** POST a message and yield the decision-point events as they arrive. */
export async function streamMessage(
  chatId: string,
  content: string,
  useVerifier: boolean,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${BASE}/chats/${chatId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ content, use_verifier: useVerifier }),
    signal,
  })
  if (!response.ok || !response.body) {
    throw new Error(`stream failed: ${response.status}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      const payload = frame
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trim())
        .join('')
      if (payload) {
        try {
          onEvent(JSON.parse(payload) as StreamEvent)
        } catch {
          // A partial frame is not worth breaking the stream over.
        }
      }
      boundary = buffer.indexOf('\n\n')
    }
  }
}
