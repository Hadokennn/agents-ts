import type { StoredChunk } from '../store.js';

// SearchResult 下沉到这里（原在 search.ts），由 search.ts 再导出，
// 打破 search.ts ←→ fusion 的循环依赖。score/vectorScore/keywordScore 为对外展示契约。
export interface SearchResult {
  chunk: StoredChunk;
  score: number;        // 融合后的综合分
  vectorScore: number;  // 向量路对该结果的贡献（语义随融合策略而异）
  keywordScore: number; // 关键词路对该结果的贡献
}

// 单路检索结果：已按原始 score 降序排列的 {chunk, 原始分} 列表
export type RankedList = Array<{ chunk: StoredChunk; score: number }>;

export interface FusionStrategy {
  readonly name: string;
  readonly params: Record<string, number | string | boolean>;
  // 融合「向量路」「关键词路」两个有序列表，返回按融合分降序的结果（MMR 去重在融合之后单独做）
  fuse(vectorResults: RankedList, keywordResults: RankedList): SearchResult[];
}
