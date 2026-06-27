// 分块策略的公共类型与构造工具。
// Chunk 下沉到这里（原在 chunker.ts），由 chunker.ts 再导出，保证现有调用方零感知，
// 同时打破 chunker.ts ←→ strategy 的循环依赖。

export interface Chunk {
  id: string;
  text: string;
  source: string;
  index: number;
  tokenEstimate: number;
}

// 评测需要把命中锚回原文，故每个 chunk 额外携带在原文档中的字符区间。
// 注意：递归策略因 trim/重拼，offset 为「近似」（见 recursive-paragraph.ts: locate）；
// 固定窗口策略的 offset 是精确的。详见 docs/rag-eval-spec.md §4.1 D1。
export interface ChunkWithSpan extends Chunk {
  startOffset: number; // 在原文档中的起始字符下标（含）
  endOffset: number;   // 结束字符下标（不含）
}

export interface ChunkStrategy {
  readonly name: string;                                       // "recursive-paragraph" / "fixed-size" ...
  readonly params: Record<string, number | string | boolean>; // 写进评测报告，保证可复现
  chunk(source: string, text: string): ChunkWithSpan[];
}

// 统一构造：保证 id 格式与 tokenEstimate 公式跨所有策略一致（单一真相源）。
// 字段顺序固定为 id,text,source,index,tokenEstimate,startOffset,endOffset——
// 这样 chunker.ts 剥掉 span 两字段后，剩余对象与历史 makeChunk 逐字一致。
export function buildChunk(
  source: string,
  text: string,
  index: number,
  charsPerToken: number,
  startOffset: number,
  endOffset: number,
): ChunkWithSpan {
  return {
    id: `${source}#${index}`,
    text,
    source,
    index,
    tokenEstimate: Math.ceil(text.length / charsPerToken),
    startOffset,
    endOffset,
  };
}
