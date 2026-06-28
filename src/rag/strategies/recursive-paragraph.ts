import { buildChunk, type ChunkStrategy, type ChunkWithSpan } from './types.js';

export interface RecursiveParagraphOptions {
  targetTokens?: number;
  charsPerToken?: number;
}

/**
 * 递归段落分块（项目基线策略）：段落优先 → 超长段落退化到句子边界 → 缓冲累积到目标大小。
 *
 * 核心切分逻辑与历史 chunkDocument 逐字一致（已用快照 diff 验证），
 * 仅额外计算近似 span 供评测截断率诊断。
 */
export class RecursiveParagraphStrategy implements ChunkStrategy {
  readonly name = 'recursive-paragraph';
  readonly params: Record<string, number>;
  private readonly targetChars: number;
  private readonly charsPerToken: number;

  constructor(opts: RecursiveParagraphOptions = {}) {
    const targetTokens = opts.targetTokens ?? 512;   // 生产常用 512 token，大概 1k 个字符
    this.charsPerToken = opts.charsPerToken ?? 1.6;  // cjk 0.6 token/字符；ASCII 0.3 token/字符
    this.targetChars = targetTokens * this.charsPerToken;
    this.params = { targetTokens, charsPerToken: this.charsPerToken };
  }

  chunk(source: string, text: string): ChunkWithSpan[] {
    const TARGET_CHARS = this.targetChars;
    const charsPerToken = this.charsPerToken;
    const paragraphs = text.split(/\n{2,}/);
    const chunks: ChunkWithSpan[] = [];
    let current = '';
    let idx = 0;
    let cursor = 0;

    // 按文档顺序定位每个 chunk 的近似 span，并落盘
    const push = (body: string) => {
      const { start, end, next } = locate(text, body, cursor);
      cursor = next;
      chunks.push(buildChunk(source, body, idx++, charsPerToken, start, end));
    };

    for (const para of paragraphs) {
      const trimmed = para.trim();
      if (!trimmed) continue;

      // 当前缓冲区 + 新段落超过目标大小，先把缓冲区存下来
      if (current.length + trimmed.length + 2 > TARGET_CHARS && current.length > 0) {
        push(current.trim());
        current = '';
      }

      // 单个段落就超过目标大小，按句子切分
      if (trimmed.length > TARGET_CHARS) {
        // 先把当前缓冲区存下来
        if (current.length > 0) {
          push(current.trim());
          current = '';
        }
        // ... 按句子边界（句号、问号、感叹号）继续切分
        const sentences = trimmed.split(/(?<=[。！？.!?])\s*/);
        let sentBuf = '';
        for (const sent of sentences) {
          // 句子缓冲区 + 新句子超过目标大小，先把缓冲区存下来
          if (sentBuf.length + sent.length + 1 > TARGET_CHARS && sentBuf.length > 0) {
            push(sentBuf.trim());
            sentBuf = '';
          }
          sentBuf += (sentBuf ? ' ' : '') + sent;
        }
        if (sentBuf.trim()) {
          current = sentBuf.trim();
        }
      } else {
        current += (current ? '\n\n' : '') + trimmed;
      }
    }

    if (current.trim()) {
      push(current.trim());
    }

    return chunks;
  }
}

/**
 * 近似定位（D1 方案 A）：用 chunk 首行在原文中顺序查找位置，仅供截断率诊断，
 * 不参与切分决策。对 trim/段落重拼鲁棒；同句重复时可能偏移，靠 cursor 单调前进缓解。
 */
function locate(text: string, body: string, cursor: number): { start: number; end: number; next: number } {
  const firstLine = body.split('\n', 1)[0].slice(0, 50);
  let start = firstLine ? text.indexOf(firstLine, cursor) : -1;
  if (start < 0 && firstLine) start = text.indexOf(firstLine); // 回退：从头找
  if (start < 0) start = Math.min(cursor, text.length);        // 兜底：用游标近似
  const end = Math.min(start + body.length, text.length);
  return { start, end, next: Math.max(cursor, start + 1) };
}
