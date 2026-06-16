import type { ModelMessage } from 'ai';
import { estimateMessageTokens, estimateTextTokens, contextWindow } from '../usage/tracker.js';

// ── tool-result output 辅助 ──────────────────────────
// AI SDK 的 tool-result.output 是结构体 { type:'text', value }，不是裸字符串。
// 读写一律走这两个函数，否则把 output 当 string 处理会让整个 messages 过不了 schema 校验。
type TextOutput = { type: 'text'; value: string };

function readOutputText(output: unknown): string {
  if (typeof output === 'string') return output;                       // 兼容历史裸字符串
  if (output && typeof (output as any).value === 'string') return (output as any).value;
  return '';
}

function textOutput(value: string): TextOutput {
  return { type: 'text', value };
}

// ── Layer 1: Token Estimation ────────────────────────

export class TokenTracker {
  private lastPreciseCount = 0;
  private pendingChars = 0;

  updateFromAPI(promptTokens: number): void {
    this.lastPreciseCount = promptTokens;
    this.pendingChars = 0;
  }

  addMessage(content: string): void {
    this.pendingChars += content.length;
  }

  get estimatedTokens(): number {
    return this.lastPreciseCount + Math.ceil(this.pendingChars / 4);
  }

  get status(): { tokens: number; percent: number; needsAction: boolean } {
    const tokens = this.estimatedTokens;
    const percent = Math.round((tokens / contextWindow('mock-model')) * 100);
    return {
      tokens,
      percent,
      needsAction: percent >= 75,
    };
  }
}

// ── Layer 2: Dynamic Tool Result Truncation ──────────

interface TruncationConfig {
  maxSingleResultTokens: number;
  contextBudgetTokens: number;
}

// 截断预算直接按 token（window 本身就是 token）：单条结果 ≤ 50% window，所有工具结果合计 ≤ 75% window
function truncationConfig(window: number): TruncationConfig {
  return {
    maxSingleResultTokens: Math.floor(window * 0.5),
    contextBudgetTokens: Math.floor(window * 0.75),
  };
}

export function truncateToolResults(
  messages: ModelMessage[],
  model: string,
  config: TruncationConfig = truncationConfig(contextWindow('mock-model')),
): { messages: ModelMessage[]; truncated: number; compacted: number } {
  let truncated = 0;
  let compacted = 0;

  // 一条消息的 token 尺寸（工具 output 优先，退而取 text）
  const sizeOf = (msg: ModelMessage): number => {
    if (typeof msg.content === 'string') return estimateTextTokens(msg.content, model);
    if (!Array.isArray(msg.content)) return 0;
    return (msg.content as any[]).reduce(
      (s, p) => s + estimateTextTokens(readOutputText(p.output) || (p.text as string) || '', model), 0,
    );
  };

  // Pass 1: single-result truncation (Head/Tail 60/40)，按 token 量判断与裁剪
  let result = messages.map(msg => {
    if (msg.role !== 'tool' || !Array.isArray(msg.content)) return msg;

    const newContent = msg.content.map((part: any) => {
      const text = readOutputText(part.output);
      const tokens = estimateTextTokens(text, model);
      if (tokens <= config.maxSingleResultTokens) return part;

      truncated++;
      // 用这段文本自己的真实密度把 token 预算换算成字符切点（自校准，无硬编码字符/token 系数）
      const charsPerToken = text.length / Math.max(1, tokens);
      const budgetChars = Math.floor(config.maxSingleResultTokens * charsPerToken);
      const head = text.slice(0, Math.floor(budgetChars * 0.6));
      const tail = text.slice(-Math.floor(budgetChars * 0.4));

      return {
        ...part,
        output: textOutput(`${head}\n\n[truncated: ${tokens} → ~${config.maxSingleResultTokens} tokens]\n\n${tail}`),
      };
    });

    return { ...msg, content: newContent };
  });

  // Pass 2: total budget enforcement — compact oldest tool results first（按 token 总量）
  let totalTokens = result.reduce((sum, msg) => sum + sizeOf(msg), 0);

  if (totalTokens > config.contextBudgetTokens) {
    for (let i = 0; i < result.length && totalTokens > config.contextBudgetTokens; i++) {
      const msg = result[i];
      if (msg.role !== 'tool' || !Array.isArray(msg.content)) continue;
      const toolName = ((msg.content as any[])[0])?.toolName || 'unknown';
      const oldSize = sizeOf(msg);
      result[i] = {
        ...msg,
        content: (msg.content as any[]).map((p: any) => ({
          ...p,
          output: textOutput(`[compacted: ${toolName} output removed to free context]`),
        })),
      };
      totalTokens -= oldSize;
      compacted++;
    }
  }

  return { messages: result, truncated, compacted };
}

// ── Layer 3: TTL Pruning ─────────────────────────────

interface TTLConfig {
  softTTLMs: number;
  hardTTLMs: number;
  keepHeadTail: number;
}

const DEFAULT_TTL: TTLConfig = {
  softTTLMs: 5 * 60 * 1000,    // 5 minutes
  hardTTLMs: 10 * 60 * 1000,   // 10 minutes
  keepHeadTail: 1500,           // chars to keep in soft prune
};

export interface PruneResult {
  messages: ModelMessage[];
  softPruned: number;
  hardPruned: number;
}

export function ttlPrune(
  messages: ModelMessage[],
  timestamps: Map<number, number>,
  config: TTLConfig = DEFAULT_TTL,
): PruneResult {
  const now = Date.now();
  let softPruned = 0;
  let hardPruned = 0;

  const result = messages.map((msg, idx) => {
    // Only prune tool results, never user/assistant messages
    if (msg.role !== 'tool' || !Array.isArray(msg.content)) return msg;

    const ts = timestamps.get(idx);
    if (!ts) return msg;

    const age = now - ts;

    // Preserve error experiences — never prune failed tool results
    const outputText = (msg.content as any[])
      .map((p: any) => readOutputText(p.output))
      .join('');
    const isError = /error|失败|不存在|denied|refused|timeout/i.test(outputText);
    if (isError) return msg;

    // Hard clear: replace entire content with placeholder
    if (age >= config.hardTTLMs) {
      hardPruned++;
      const toolName = (msg.content[0] as any)?.toolName || 'unknown';
      return {
        ...msg,
        content: msg.content.map((part: any) => ({
          ...part,
          output: textOutput(`[tool result expired: ${toolName}]`),
        })),
      };
    }

    // Soft prune: keep head + tail, replace middle
    if (age >= config.softTTLMs) {
      const newContent = msg.content.map((part: any) => {
        const text = readOutputText(part.output);
        if (text.length <= config.keepHeadTail * 2) return part;

        softPruned++;
        const head = text.slice(0, config.keepHeadTail);
        const tail = text.slice(-config.keepHeadTail);
        const removed = text.length - config.keepHeadTail * 2;

        return {
          ...part,
          output: textOutput(`${head}\n\n[soft pruned: ${removed} chars removed, content older than ${Math.round(config.softTTLMs / 60000)}min]\n\n${tail}`),
        };
      });
      return { ...msg, content: newContent };
    }

    return msg;
  });

  return { messages: result, softPruned, hardPruned };
}

// ── Combined Defense ─────────────────────────────────

export interface DefenseResult {
  messages: ModelMessage[];
  tokenEstimate: number;
  truncated: number;
  compacted: number;
  softPruned: number;
  hardPruned: number;
}

export function applyDefense(
  messages: ModelMessage[],
  timestamps: Map<number, number>,
  model: string,
): DefenseResult {
  // Layer 2: truncate oversized tool results（预算按 model 的 context window 派生）
  const trunc = truncateToolResults(messages, model, truncationConfig(contextWindow(model)));
  let result = trunc.messages;

  // Layer 3: TTL prune old tool results
  const prune = ttlPrune(result, timestamps);
  result = prune.messages;

  // Layer 1: estimate final token count
  const tokenEstimate = estimateMessageTokens(result, model);

  return {
    messages: result,
    tokenEstimate,
    truncated: trunc.truncated,
    compacted: trunc.compacted,
    softPruned: prune.softPruned,
    hardPruned: prune.hardPruned,
  };
}
