import type { SearchResult } from '../fusion/types.js';

// rerank 复用 fusion 的 SearchResult 作为对外契约（精排分写入其 rerankScore 字段）。
export type { SearchResult };

/**
 * 重排策略：在「融合」之后、「MMR」之前，对候选用 cross-encoder 逐个精算相关性。
 *
 * 与 FusionStrategy 是流水线上的前后两道工序，不是并列替代：
 *   召回(bi-encoder + BM25) → fusion 合并 → rerank 精排 → MMR 去重 → topK
 *
 * fusion 只搬运已有的分数/名次（便宜、全量）；rerank 把 query 与文档原文拼在一起
 * 喂进模型读一遍（贵、精度高），所以只作用于融合后的头部候选。
 */
export interface RerankStrategy {
  readonly name: string;
  readonly params: Record<string, number | string | boolean>;
  /** 对融合后的候选重排，返回按相关性降序的结果（写入各项的 rerankScore）。 */
  rerank(query: string, candidates: SearchResult[]): Promise<SearchResult[]>;
}
