export type Stage = 'decompose' | 'plan' | 'gather' | 'assess' | 'verdict'

export type ClaimStatus = 'pending' | 'supported' | 'refuted' | 'insufficient'

export type VerdictLabel = 'true' | 'false' | 'mixed' | 'unverified'

export type Stance = 'supports' | 'refutes' | 'unclear'

export interface Claim {
  id: string
  text: string
  status: ClaimStatus
  rationale: string
  evidence_ids: string[]
}

export interface Evidence {
  id: string
  claim_id: string | null
  title: string
  url: string
  snippet: string
  source: string
  stance: Stance
  credibility: number
  tool: string
  retrieved_at: string
}

export interface Verdict {
  label: VerdictLabel
  confidence: number
  summary: string
  claims: Claim[]
  sources: Evidence[]
}

export interface TraceToolCall {
  tool: string
  arguments: Record<string, unknown>
  ok: boolean
  error: string
  evidence_count: number
}

export interface TraceStep {
  stage: Stage
  summary: string
  curated_tools: string[]
  tool_calls: TraceToolCall[]
  created_at: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
  turn_id: string | null
  verdict: Verdict | null
  trace: TraceStep[]
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  created_at: string
  updated_at: string
}

export interface ConversationSummary {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface SystemStatus {
  mongo_connected: boolean
  llm_mocked: boolean
  llm_model: string
  search_mocked: boolean
  mcp_transport: string
  mcp_tools: string[]
}

export type StreamEvent =
  | { type: 'turn_started'; turn_id: string; conversation_title: string; message: Message }
  | { type: 'stage'; stage: Stage; status: 'started' | 'completed'; curated_tools?: string[]; summary?: string; detail?: Record<string, unknown> }
  | { type: 'claims'; claims: Claim[] }
  | { type: 'tool_call'; stage: Stage; tool: string; arguments: Record<string, unknown>; ok: boolean; error: string; evidence: Evidence[] }
  | { type: 'token'; text: string }
  | { type: 'message'; message: Message }
  | { type: 'warning'; message: string }
  | { type: 'error'; error: string }
  | { type: 'done' }

export const STAGES: Stage[] = ['decompose', 'plan', 'gather', 'assess', 'verdict']

export const STAGE_LABELS: Record<Stage, string> = {
  decompose: 'Decompose claims',
  plan: 'Plan evidence',
  gather: 'Gather sources',
  assess: 'Assess claims',
  verdict: 'Write verdict',
}
