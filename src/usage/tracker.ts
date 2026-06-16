import { appendFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import type { ModelMessage } from 'ai';

/**
 * 各家模型的 prompt cache 计费规则（单位：$ / 1M tokens，2026-05 数据）。
 *
 * 命中折扣不是行业默认 10x，每家差异不小：
 * - Claude：cache read = 10% input；write 5min = 125%、1h = 200%
 * - OpenAI：自动缓存，命中折扣按模型分档（4o 系列 50%，GPT-5 / 4.1 系列 25%）
 * - Gemini：cache read = 10% input（explicit 模式按存储时长另收费，这里没列）
 * - DeepSeek：cache hit = 10% miss，没有写入费、没有 TTL 概念
 * - Qwen：implicit 20%、explicit 10%（字段跟 Anthropic 一样是 `cache_control: ephemeral`）
 * - Kimi：自动模式 25%
 * - Doubao：显式 cache，命中价 = 40% miss
 *
 * 加新模型直接扩这张表就行。
 */
export interface ModelPricing {
  input: number;       // $ / 1M input tokens (cache miss)
  output: number;      // $ / 1M output tokens
  cacheWrite: number;  // $ / 1M tokens written to cache
  cacheRead: number;   // $ / 1M tokens read from cache (hit)
}

export const PRICE_TABLE: Record<string, ModelPricing> = {
  // Anthropic（最新主力，2026 上半年发布的 4.7 系列）
  'claude-opus-4-7':      { input: 15.00, output: 75.00, cacheWrite: 18.75, cacheRead: 1.50 },
  'claude-sonnet-4-7':    { input: 3.00,  output: 15.00, cacheWrite: 3.75,  cacheRead: 0.30 },
  'claude-haiku-4-5':     { input: 1.00,  output: 5.00,  cacheWrite: 1.25,  cacheRead: 0.10 },
  // OpenAI（GPT-5 系列；GPT-5.5 默认 24h extended cache）
  'gpt-5-5':              { input: 5.00,  output: 20.00, cacheWrite: 5.00,  cacheRead: 0.50 },
  'gpt-5':                { input: 5.00,  output: 15.00, cacheWrite: 5.00,  cacheRead: 1.25 },
  // Google（Gemini 3 系列，最新 preview）
  'gemini-3-pro':         { input: 2.50,  output: 12.00, cacheWrite: 2.50,  cacheRead: 0.625 },
  'gemini-3-flash':       { input: 0.30,  output: 1.20,  cacheWrite: 0.30,  cacheRead: 0.075 },
  // 国产
  'deepseek-v3-2':        { input: 0.27,  output: 1.10,  cacheWrite: 0.27,  cacheRead: 0.027 },
  'deepseek-v4-flash':    { input: 1.00,  output: 2.00,  cacheWrite: 1.00,  cacheRead: 0.10 },
  'qwen3-6-plus':         { input: 0.40,  output: 1.20,  cacheWrite: 0.40,  cacheRead: 0.04 },
  'kimi-k2-6':            { input: 0.60,  output: 2.50,  cacheWrite: 0.60,  cacheRead: 0.15 },
  'doubao-2-0-pro':       { input: 0.30,  output: 0.90,  cacheWrite: 0.30,  cacheRead: 0.12 },
  // 课程内 mock，用 Haiku 4.5 同档价格
  'mock-model':           { input: 1.00,  output: 5.00,  cacheWrite: 1.25,  cacheRead: 0.10 },
};

export interface StepUsage {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
}

export interface StepRecord extends StepUsage {
  ts: number;
  model: string;
  cost: number;
}

export class UsageTracker {
  private steps: StepRecord[] = [];
  private logPath?: string;

  constructor(logPath?: string) {
    this.logPath = logPath;
    if (logPath) mkdirSync(dirname(logPath), { recursive: true });
  }

  record(model: string, usage: StepUsage): StepRecord {
    const cost = computeCost(model, usage);
    const record: StepRecord = { ts: Date.now(), model, cost, ...usage };
    this.steps.push(record);

    if (this.logPath) {
      appendFileSync(this.logPath, JSON.stringify(record) + '\n');
    }
    return record;
  }

  totals() {
    const t = this.steps.reduce(
      (a, s) => ({
        inputTokens: a.inputTokens + s.inputTokens,
        outputTokens: a.outputTokens + s.outputTokens,
        cacheReadTokens: a.cacheReadTokens + s.cacheReadTokens,
        cacheWriteTokens: a.cacheWriteTokens + s.cacheWriteTokens,
        cost: a.cost + s.cost,
      }),
      { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, cost: 0 },
    );
    const totalInputLike = t.inputTokens + t.cacheReadTokens + t.cacheWriteTokens;
    const hitRate = totalInputLike > 0 ? t.cacheReadTokens / totalInputLike : 0;
    // 没有 cache 时的"假想成本"：把所有 input-like token 当成 miss 全付
    const baselineCost = (() => {
      let c = 0;
      for (const s of this.steps) {
        const p = PRICE_TABLE[s.model] || PRICE_TABLE['mock-model'];
        const inputLike = s.inputTokens + s.cacheReadTokens + s.cacheWriteTokens;
        c += (inputLike * p.input) / 1_000_000;
        c += (s.outputTokens * p.output) / 1_000_000;
      }
      return c;
    })();
    return { ...t, hitRate, baselineCost, savedCost: baselineCost - t.cost, steps: this.steps.length };
  }

  recent(n: number): StepRecord[] {
    return this.steps.slice(-n);
  }
}

export function computeCost(model: string, usage: StepUsage): number {
  const p = PRICE_TABLE[model] || PRICE_TABLE['mock-model'];
  return (
    (usage.inputTokens * p.input
      + usage.outputTokens * p.output
      + usage.cacheReadTokens * p.cacheRead
      + usage.cacheWriteTokens * p.cacheWrite)
    / 1_000_000
  );
}

/**
 * 把 AI SDK 返回的 usage 对象规范化成四类 token。
 *
 * AI SDK v5 把 cache read 标准化到顶层 `cachedInputTokens`（OpenAI、DashScope 都映射到这里）。
 * cache write 没有 AI SDK 标准字段，Anthropic provider 元数据用 `cacheCreationInputTokens`。
 * 这里把两个来源都兜一遍，以后接新 provider 就在对应位置补一行。
 */
export function normalizeUsage(usage: any): StepUsage {
  if (!usage) return { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 };

  const cacheRead =
    usage.cachedInputTokens                                        // AI SDK 标准字段
    ?? usage.providerMetadata?.openai?.cachedTokens                // OpenAI 原生
    ?? 0;

  const cacheWrite =
    usage.cacheCreationInputTokens                                 // Anthropic SDK 直接挂顶层
    ?? usage.providerMetadata?.anthropic?.cacheCreationInputTokens // AI SDK 走 provider 元数据
    ?? 0;

  // OpenAI 把 cached tokens 含在 inputTokens 总数里 → 减出来；Anthropic 单列 → 不用减
  let inputTokens = usage.inputTokens ?? 0;
  if (cacheRead && inputTokens >= cacheRead) inputTokens -= cacheRead;

  return {
    inputTokens: Math.max(0, inputTokens),
    outputTokens: usage.outputTokens ?? 0,
    cacheReadTokens: cacheRead,
    cacheWriteTokens: cacheWrite,
  };
}

/**
 * 各家模型的 token 估算口径（token / 字符）。
 *
 * `chars / 4` 是 OpenAI 英文 BPE 的经验值，套到中文 / 别家模型都偏：token 数取决于
 * tokenizer，而 tokenizer 按「家族」走（同一家不同型号词表一致）。所以这里按家族定义
 * 一套权重、再映射到具体 model id —— 和 PRICE_TABLE 同构，按 id 查、缺省回落 mock。
 *
 * 口径来源（粗口径，足够支撑「够没够阈值」的触发判断；要精确就读 API 返回的真实 usage）：
 * - DeepSeek 官方：英文 ~0.3 token/char、中文 ~0.6 token/char
 * - GPT / Claude / Gemini 系 BPE：英文 ~0.27、中文 ~0.6（CJK 多为 1~2 token/字，混合取中值）
 * 常数可用 .usage 的真实 total_tokens 反向校准。加新模型直接扩这张表。
 */
export interface TokenWeights {
  cjk: number;        // 每个中日韩表意字符
  ascii: number;      // 每个 ASCII 字符
  other: number;      // 其他（标点 / emoji / 其它脚本）
  perMessage: number; // 每条消息的固定结构开销（role 标签 / 分隔符）
}

const TW_DEEPSEEK: TokenWeights = { cjk: 0.6, ascii: 0.3,  other: 0.5, perMessage: 3 };
const TW_BPE:      TokenWeights = { cjk: 0.6, ascii: 0.27, other: 0.5, perMessage: 3 };

export const TOKEN_WEIGHTS: Record<string, TokenWeights> = {
  'claude-opus-4-7':   TW_BPE,
  'claude-sonnet-4-7': TW_BPE,
  'claude-haiku-4-5':  TW_BPE,
  'gpt-5-5':           TW_BPE,
  'gpt-5':             TW_BPE,
  'gemini-3-pro':      TW_BPE,
  'gemini-3-flash':    TW_BPE,
  'deepseek-v3-2':     TW_DEEPSEEK,
  'deepseek-v4-flash': TW_DEEPSEEK,
  'qwen3-6-plus':      TW_DEEPSEEK, // 国产中文 BPE，口径接近 DeepSeek
  'kimi-k2-6':         TW_DEEPSEEK,
  'doubao-2-0-pro':    TW_DEEPSEEK,
  'mock-model':        TW_DEEPSEEK,
};

function isCjkCode(c: number): boolean {
  return (c >= 0x4e00 && c <= 0x9fff)     // CJK 基本区
    || (c >= 0x3400 && c <= 0x4dbf)       // 扩展 A
    || (c >= 0x20000 && c <= 0x2a6df)     // 扩展 B（for...of 已合并 surrogate）
    || (c >= 0x3000 && c <= 0x303f)       // CJK 标点
    || (c >= 0xff00 && c <= 0xffef);      // 全角
}

/** 把一条消息的文本内容抽出来（string 内容 / parts 数组里的 text / 工具 output）。 */
function messageText(msg: ModelMessage): string {
  if (typeof msg.content === 'string') return msg.content;
  if (!Array.isArray(msg.content)) return '';
  let s = '';
  for (const part of msg.content as any[]) {
    if ('text' in part && typeof part.text === 'string') s += part.text;
    else if ('output' in part) s += typeof part.output === 'string' ? part.output : JSON.stringify(part.output);
  }
  return s;
}

/** 按字符分桶估算一段文本的 token 数。model 决定权重，未知 model 回落 mock 口径。 */
export function estimateTextTokens(text: string, model: string): number {
  const w = TOKEN_WEIGHTS[model] ?? TOKEN_WEIGHTS['mock-model'];
  let cjk = 0, ascii = 0, other = 0;
  for (const ch of text) {
    const c = ch.codePointAt(0)!;
    if (c <= 0x7f) ascii++;
    else if (isCjkCode(c)) cjk++;
    else other++;
  }
  return Math.ceil(cjk * w.cjk + ascii * w.ascii + other * w.other);
}

/** 估算整个消息列表的 token 数（含每条消息的结构开销）。token 估算的唯一真相入口。 */
export function estimateMessageTokens(messages: ModelMessage[], model: string): number {
  const w = TOKEN_WEIGHTS[model] ?? TOKEN_WEIGHTS['mock-model'];
  let total = 0;
  for (const msg of messages) total += w.perMessage + estimateTextTokens(messageText(msg), model);
  return total;
}

/**
 * 各家模型的上下文窗口（tokens）。压缩阈值一律由「window × 比例」派生，告别魔法数。
 *
 * 课程内可调：把 WINDOW_OVERRIDE 设成数字，会强制所有模型用这个窗口，便于在短会话里
 * 观察压缩触发；设 null 则用下面的真实值。（真实模型 128k+，短会话压缩自然很少触发。）
 */
const WINDOW_OVERRIDE: number | null = 4_000;

export const CONTEXT_WINDOWS: Record<string, number> = {
  'claude-opus-4-7':   200_000,
  'claude-sonnet-4-7': 200_000,
  'claude-haiku-4-5':  200_000,
  'gpt-5-5':           400_000,
  'gpt-5':             272_000,
  'gemini-3-pro':      1_000_000,
  'gemini-3-flash':    1_000_000,
  'deepseek-v3-2':     128_000,
  'deepseek-v4-flash': 128_000,
  'qwen3-6-plus':      262_144,
  'kimi-k2-6':         262_144,
  'doubao-2-0-pro':    262_144,
  'mock-model':        8_000,
};

/** 触发压缩的窗口占比（Claude Code 实际约 92%，这里留些余量）。 */
export const COMPACT_TRIGGER_RATIO = 0.85;
/** 低于窗口这个占比就别压——太小，省得白调一次 LLM。 */
export const COMPACT_FLOOR_RATIO = 0.10;

export function contextWindow(model: string): number {
  return WINDOW_OVERRIDE ?? CONTEXT_WINDOWS[model] ?? CONTEXT_WINDOWS['mock-model'];
}

/** 触发压缩的 token 阈值 = window × COMPACT_TRIGGER_RATIO。 */
export function compactTriggerTokens(model: string): number {
  return Math.floor(contextWindow(model) * COMPACT_TRIGGER_RATIO);
}

/** 不值得压的下限 token 数 = window × COMPACT_FLOOR_RATIO。 */
export function compactFloorTokens(model: string): number {
  return Math.floor(contextWindow(model) * COMPACT_FLOOR_RATIO);
}
