import { useCallback, useEffect, useRef, useState } from 'react'
import { api, streamMessage } from '@/lib/api'
import type {
  Claim,
  Evidence,
  Message,
  Stage,
  StreamEvent,
  TraceRetrieval,
  TraceStep,
} from '@/lib/types'

export interface LiveRun {
  stage: Stage | null
  steps: TraceStep[]
  claims: Claim[]
  evidence: Evidence[]
  answer: string
  /** Searches seen since the stage started, folded into the step when it completes. */
  pending: TraceRetrieval[]
}

const emptyRun: LiveRun = {
  stage: null,
  steps: [],
  claims: [],
  evidence: [],
  answer: '',
  pending: [],
}

/** Loads one conversation and runs verification turns against it. */
export function useChat(conversationId: string | null, onTurnComplete?: () => void) {
  const [messages, setMessages] = useState<Message[]>([])
  const [run, setRun] = useState<LiveRun>(emptyRun)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    abortRef.current?.abort()
    setRun(emptyRun)
    setBusy(false)
    setError(null)
    if (!conversationId) {
      setMessages([])
      return
    }
    let active = true
    api
      .getChat(conversationId)
      .then((conversation) => {
        if (active) setMessages(conversation.messages)
      })
      .catch((cause: unknown) => {
        if (active) setError(cause instanceof Error ? cause.message : 'could not load chat')
      })
    return () => {
      active = false
    }
  }, [conversationId])

  const stop = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setBusy(false)
    setRun(emptyRun)
  }, [])

  const send = useCallback(
    async (content: string) => {
      if (!conversationId || busy) return
      const controller = new AbortController()
      abortRef.current = controller
      setBusy(true)
      setError(null)
      setRun(emptyRun)

      const handle = (event: StreamEvent) => {
        switch (event.type) {
          case 'turn_started':
            setMessages((current) => [...current, event.message])
            break
          case 'claims':
            setRun((current) => ({ ...current, claims: event.claims }))
            break
          case 'stage':
            setRun((current) =>
              event.status === 'started'
                ? { ...current, stage: event.stage, pending: [] }
                : {
                    ...current,
                    stage: null,
                    pending: [],
                    steps: [
                      ...current.steps,
                      {
                        stage: event.stage,
                        summary: event.summary ?? '',
                        retrievals: current.pending,
                        created_at: new Date().toISOString(),
                      },
                    ],
                  },
            )
            break
          case 'retrieval':
            setRun((current) => ({
              ...current,
              evidence: [...current.evidence, ...event.evidence],
              pending: [
                ...current.pending,
                {
                  query: event.query,
                  ok: event.ok,
                  error: event.error,
                  evidence_count: event.evidence.length,
                },
              ],
            }))
            break
          case 'token':
            setRun((current) => ({ ...current, answer: current.answer + event.text }))
            break
          case 'message':
            setMessages((current) => [...current, event.message])
            setRun(emptyRun)
            break
          case 'warning':
            setError(event.message)
            break
          case 'error':
            setError(event.error)
            break
          case 'done':
            setBusy(false)
            onTurnComplete?.()
            break
        }
      }

      try {
        await streamMessage(conversationId, content, handle, controller.signal)
      } catch (cause) {
        if (!controller.signal.aborted) {
          setError(cause instanceof Error ? cause.message : 'verification failed')
        }
      } finally {
        setBusy(false)
        abortRef.current = null
      }
    },
    [busy, conversationId, onTurnComplete],
  )

  return { messages, run, busy, error, send, stop }
}
