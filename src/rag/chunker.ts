// import { RecursiveParagraphStrategy } from './strategies/recursive-paragraph.js';
import { MarkdownCodeAwareStrategy } from './strategies/markdown-code-aware.js';
import type { Chunk, ChunkWithSpan } from './strategies/types.js';

// Chunk 定义已下沉到 strategies/types.ts；在此再导出，保持历史导入路径不变
// （store.ts / sqlite-store.ts 仍 `import { Chunk } from './chunker.js'`）。
export type { Chunk };

// 默认分块策略 = 代码感知的 Markdown 切分（markdown-section 的超集：无代码时输出一致，
// 有代码时把 ``` 围栏块整块保留，不再被空行/点号撕碎）。语料含内嵌代码片段，故用它。
const defaultStrategy = new MarkdownCodeAwareStrategy();

// --- 文档切分（历史入口，签名与行为保持不变）---
// 评测需要对比多种策略时，请直接实例化 strategies/ 下的策略类，而非走这里。
export function chunkDocument(source: string, text: string): Chunk[] {
  return defaultStrategy.chunk(source, text).map(stripSpan);
}

// 剥掉评测专用的 span 两字段，使返回对象与历史输出逐字一致
function stripSpan({ startOffset, endOffset, ...chunk }: ChunkWithSpan): Chunk {
  return chunk;
}
