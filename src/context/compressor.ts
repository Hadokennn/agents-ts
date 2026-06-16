import { generateText, type ModelMessage } from 'ai';
import { estimateMessageTokens, compactFloorTokens } from '../usage/tracker.js';

// ── Layer 1: Microcompact ────────────────────────────

// 保留名单：这些工具的结果不自动清（结果短小或后续强依赖，清了得不偿失）。
// 注意是「保留名单」不是「白名单」——默认清理任意工具，避免 web_fetch / MCP / memory
// 等大输出因为不在白名单里而永远清不掉（旧实现的根因）。
const PRESERVE_TOOLS = new Set<string>([]);
const KEEP_RECENT_TOOL_RESULTS = 3;   // 最近 N 个工具结果一律保留，维持就近推理连贯
const MIN_CLEAR_CHARS = 500;          // output 小于此长度不值得清，清了省不下多少还丢上下文

function partOutputText(part: any): string {
  if (!('output' in part)) return '';
  return typeof part.output === 'string'
    ? part.output
    : (part.output?.value ?? JSON.stringify(part.output ?? ''));
}

export function microcompact(messages: ModelMessage[]): {
  messages: ModelMessage[];
  cleared: number;
} {
  let cleared = 0;

  // 收集所有工具结果消息的下标，最近 KEEP_RECENT_TOOL_RESULTS 个不动
  const toolIdx: number[] = [];
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role === 'tool' && Array.isArray(messages[i].content)) toolIdx.push(i);
  }
  const cutoff = Math.max(0, toolIdx.length - KEEP_RECENT_TOOL_RESULTS);
  const toClear = new Set(toolIdx.slice(0, cutoff));

  const result = messages.map((msg, idx) => {
    if (!toClear.has(idx) || msg.role !== 'tool' || !Array.isArray(msg.content)) return msg;

    let changed = false;
    const content = msg.content.map((part: any) => {
      if (!('output' in part)) return part;
      const toolName = part.toolName || 'unknown';
      if (PRESERVE_TOOLS.has(toolName)) return part;        // 保留名单：跳过
      const text = partOutputText(part);
      if (text.length < MIN_CLEAR_CHARS) return part;        // 太小：不值得清
      changed = true;
      // output 必须保持结构体 { type:'text', value }，否则过不了 ModelMessage schema 校验。
      // 留面包屑（工具名 + 原长度）而非全抹，给模型一个「这里曾有什么」的线索。
      return { ...part, output: { type: 'text' as const, value: `[${toolName} 结果已清理，原 ${text.length} 字符]` } };
    });

    if (changed) cleared++;
    return { ...msg, content };
  });

  return { messages: result, cleared };
}

// ── Layer 2: LLM Summarization ───────────────────────

const COMPRESS_PROMPT = `你是一个对话压缩系统。你的任务是把 Agent 和用户之间的对话历史压缩成一份结构化摘要，确保后续对话能够无缝继续。

请严格按照以下模板输出，每个字段都要填写。如果某个字段没有相关内容，写"无"：

## 用户意图
（用户在这次对话中想要完成什么）

## 已完成的操作
（Agent 执行了哪些工具调用、产生了什么结果）

## 关键发现
（读取的文件内容要点、搜索结果、命令输出中的关键信息）

## 当前状态
（对话进行到哪一步了、还有什么没做完）

## 需要保留的细节
（文件路径、变量名、配置值、错误信息等不能丢失的具体内容）

注意事项：
- 用对话中使用的语言（中文或英文）输出
- 文件路径、UUID、版本号等标识符必须原样保留，不要翻译或改写
- 不要写笼统的概述，只保留具体的、可操作的信息
- 总长度控制在 800 字以内`;

const KEEP_RECENT_MESSAGES = 6;

export interface CompactionResult {
  messages: ModelMessage[];
  summary: string;
  compressedCount: number;
}

export async function summarize(
  model: any,
  messages: ModelMessage[],
  existingSummary?: string,
): Promise<CompactionResult> {
  const modelId = model?.modelId ?? 'mock-model';
  const tokenEstimate = estimateMessageTokens(messages, modelId);
  if (tokenEstimate < compactFloorTokens(modelId) || messages.length <= KEEP_RECENT_MESSAGES) {
    return { messages, summary: existingSummary || '', compressedCount: 0 };
  }

  const splitIdx = Math.max(0, messages.length - KEEP_RECENT_MESSAGES);

  // Align to user message boundary
  let alignedIdx = splitIdx;
  while (alignedIdx > 0 && messages[alignedIdx].role !== 'user') {
    alignedIdx--;
  }
  if (alignedIdx === 0) {
    return { messages, summary: existingSummary || '', compressedCount: 0 };
  }

  const toCompress = messages.slice(0, alignedIdx);
  const toKeep = messages.slice(alignedIdx);

  const conversationText = toCompress
    .map(msg => {
      const content = typeof msg.content === 'string'
        ? msg.content
        : Array.isArray(msg.content)
          ? msg.content.map((p: any) => p.text || JSON.stringify(p.output || '')).join('')
          : '';
      return content ? `**${msg.role}**: ${content}` : '';
    })
    .filter(Boolean)
    .join('\n\n');

  if (!conversationText.trim()) {
    return { messages, summary: existingSummary || '', compressedCount: 0 };
  }

  const userPrompt = existingSummary
    ? `## 已有摘要（上一次压缩的结果）\n\n${existingSummary}\n\n## 需要压缩的新对话\n\n${conversationText}`
    : conversationText;

  try {
    const { text: summary } = await generateText({
      model,
      system: COMPRESS_PROMPT,
      prompt: userPrompt,
    });

    const summaryMessage: ModelMessage = {
      role: 'user',
      content: `[以下是之前对话的压缩摘要]\n\n${summary}\n\n[摘要结束，以下是最近的对话]`,
    };

    const newMessages: ModelMessage[] = [summaryMessage, ...toKeep];

    return {
      messages: newMessages,
      summary,
      compressedCount: toCompress.length,
    };
  } catch (err) {
    console.error('[Compaction] LLM 摘要失败:', err);
    return { messages, summary: existingSummary || '', compressedCount: 0 };
  }
}
