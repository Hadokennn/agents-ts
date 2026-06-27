import { buildChunk, type ChunkStrategy, type ChunkWithSpan } from './types.js';

export interface FixedSizeOptions {
  targetTokens?: number;
  charsPerToken?: number;
}

/**
 * 固定大小分块（故意的「坏基线」/ 负向对照）。
 *
 * 硬按字符窗口切，完全无视段落与句子边界，必然频繁把答案拦腰斩断。
 * 它存在的意义不是上生产，而是给评测台一个「事先确信很烂」的对照组：
 *   - 真实 embedder 下它必须排到最底 → 证明评测台有分辨力；
 *   - 它若没垫底 → 是评测坏了，不是它意外地好。
 * 同时它给所有指标提供一条「地板线」，让其它策略的提升有可比的参照。
 * 详见 docs/rag-eval-spec.md §4.3。
 */
export class FixedSizeStrategy implements ChunkStrategy {
  readonly name = 'fixed-size';
  readonly params: Record<string, number>;
  private readonly windowChars: number;
  private readonly charsPerToken: number;

  constructor(opts: FixedSizeOptions = {}) {
    const targetTokens = opts.targetTokens ?? 256;
    this.charsPerToken = opts.charsPerToken ?? 1.6;
    this.windowChars = Math.max(1, Math.round(targetTokens * this.charsPerToken));
    this.params = { targetTokens, charsPerToken: this.charsPerToken };
  }

  chunk(source: string, text: string): ChunkWithSpan[] {
    const chunks: ChunkWithSpan[] = [];
    let idx = 0;
    for (let pos = 0; pos < text.length; pos += this.windowChars) {
      const end = Math.min(pos + this.windowChars, text.length);
      const body = text.slice(pos, end);
      if (!body.trim()) continue; // 跳过纯空白窗口，但 offset 仍精确
      chunks.push(buildChunk(source, body, idx++, this.charsPerToken, pos, end));
    }
    return chunks;
  }
}
