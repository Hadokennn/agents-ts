import { buildChunk, type ChunkStrategy, type ChunkWithSpan } from './types.js';

export interface FixedOverlapOptions {
  targetTokens?: number;
  overlapTokens?: number;
  charsPerToken?: number;
}

/**
 * 固定大小 + 重叠分块。
 *
 * 与 FixedSizeStrategy 仅差「overlap」一个变量（窗口大小、charsPerToken 全相同），
 * 是一组受控变量实验：截断率的差值即纯粹归因于 overlap，用于回答
 * 「加 overlap 到底值不值」。理论上重叠应显著降低答案截断率——评测要能验证这一点。
 * 详见 docs/rag-eval-spec.md §4.3。
 */
export class FixedOverlapStrategy implements ChunkStrategy {
  readonly name = 'fixed-overlap';
  readonly params: Record<string, number>;
  private readonly windowChars: number;
  private readonly stepChars: number;
  private readonly charsPerToken: number;

  constructor(opts: FixedOverlapOptions = {}) {
    const targetTokens = opts.targetTokens ?? 256;
    const overlapTokens = opts.overlapTokens ?? 32;
    this.charsPerToken = opts.charsPerToken ?? 1.6;
    this.windowChars = Math.max(1, Math.round(targetTokens * this.charsPerToken));
    const overlapChars = Math.max(0, Math.round(overlapTokens * this.charsPerToken));
    this.stepChars = Math.max(1, this.windowChars - overlapChars); // 步进 = 窗口 - 重叠
    this.params = { targetTokens, overlapTokens, charsPerToken: this.charsPerToken };
  }

  chunk(source: string, text: string): ChunkWithSpan[] {
    const chunks: ChunkWithSpan[] = [];
    let idx = 0;
    for (let pos = 0; pos < text.length; pos += this.stepChars) {
      const end = Math.min(pos + this.windowChars, text.length);
      const body = text.slice(pos, end);
      if (body.trim()) {
        chunks.push(buildChunk(source, body, idx++, this.charsPerToken, pos, end));
      }
      if (end >= text.length) break; // 窗口已触底，避免重复尾块
    }
    return chunks;
  }
}
