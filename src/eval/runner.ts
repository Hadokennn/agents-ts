import { embed, type EmbeddingFn } from '../rag/embedder.js';
import { VectorStore } from '../rag/store.js';
import { hybridSearch } from '../rag/search.js';
import type { ChunkStrategy } from '../rag/strategies/types.js';
import type { FusionStrategy } from '../rag/fusion/types.js';
import { WeightedScoreFusion } from '../rag/fusion/weighted-score.js';
import { loadCorpus, loadGolden, type GoldenTag } from './dataset.js';
import { isHit, isAnswerSplit, HIT_THRESHOLD } from './matching.js';
import {
  computeQueryMetrics, macroAverage, mean, contextEfficiency, type QueryMetrics,
} from './metrics.js';

export interface EvalConfig {
  corpusDir: string;
  datasetPath: string;
  strategies: ChunkStrategy[];     // 切分轴
  embedFn: EmbeddingFn;
  embedder: 'mock' | 'dashscope';  // §8 护栏：mock 仅供冒烟，不可下结论
  embedderModel: string;
  topK?: number;
  tau?: number;
  fusions?: FusionStrategy[];      // 融合轴（可选）；不传则默认仅加权分数一种
}

// 矩阵单元 = 一对（切分策略 × 融合策略）的评测结果
export interface MatrixCell {
  chunking: string;
  chunkingParams: Record<string, number | string | boolean>;
  fusion: string;
  fusionParams: Record<string, number | string | boolean>;
  metrics: QueryMetrics;  // 全部 query 的宏平均
  byTag: Partial<Record<GoldenTag, QueryMetrics>>; // 按 tag 分桶的宏平均（分层诊断）
  splitRate: number;      // 答案截断率（仅取决于切分）
  ctxEfficiency: number;  // 上下文效率，有限值均值
  chunkCount: number;     // 仅取决于切分
  avgChunkChars: number;  // 仅取决于切分
  queryCount: number;
}

export interface EvalReport {
  embedder: 'mock' | 'dashscope';
  embedderModel: string;
  topK: number;
  tau: number;
  corpusDocs: number;
  chunkingCount: number;
  fusionCount: number;
  tagCounts: Record<string, number>; // 各 tag 的 query 数量
  results: MatrixCell[]; // 切分 × 融合，按主指标 NDCG 降序
}

export async function runEval(config: EvalConfig): Promise<EvalReport> {
  const topK = config.topK ?? 5;
  const tau = config.tau ?? HIT_THRESHOLD;
  const fusions = config.fusions ?? [new WeightedScoreFusion()];
  const corpus = loadCorpus(config.corpusDir);
  const golden = loadGolden(config.datasetPath);

  const results: MatrixCell[] = [];

  // golden 里出现过的 tag 及其样本数（用于分层诊断）
  const tagsInGolden = [...new Set(golden.flatMap(g => g.tags))];
  const tagCounts: Record<string, number> = {};
  for (const tag of tagsInGolden) tagCounts[tag] = golden.filter(g => g.tags.includes(tag)).length;

  for (const strategy of config.strategies) {
    // 切分 + 向量化 + 建索引：每个切分策略只做一次（embedding 最贵，融合复用同一索引）
    const allChunks = [...corpus.entries()].flatMap(([source, text]) => strategy.chunk(source, text));
    const store = new VectorStore();
    const embeddings = await embed(config.embedFn, allChunks.map(c => c.text));
    store.addBatch(allChunks.map((c, i) => ({ chunk: c, embedding: embeddings[i] })));

    const bySource = new Map<string, string[]>();
    for (const c of allChunks) {
      const arr = bySource.get(c.source) ?? [];
      arr.push(c.text);
      bySource.set(c.source, arr);
    }
    const allTexts = allChunks.map(c => c.text);

    // 仅取决于切分、与融合无关的量，只算一次
    const splitRate = mean(golden.map(g => (isAnswerSplit(g.goldenAnswer, bySource.get(g.source) ?? [], tau) ? 1 : 0)));
    const numRelevantByQuery = golden.map(g => allTexts.reduce((n, t) => n + (isHit(g.goldenAnswer, t, tau) ? 1 : 0), 0));
    const chunkCount = allChunks.length;
    const avgChunkChars = Math.round(mean(allChunks.map(c => c.text.length)));

    // 融合是便宜的内层轴：复用同一 store，逐融合跑查询
    for (const fusion of fusions) {
      const perQuery: QueryMetrics[] = [];
      const ctxEffs: number[] = [];
      for (let qi = 0; qi < golden.length; qi++) {
        const g = golden[qi];
        const hits = await hybridSearch(store, config.embedFn, g.query, topK, fusion);
        const relevant = hits.map(h => isHit(g.goldenAnswer, h.chunk.text, tau));
        perQuery.push(computeQueryMetrics(relevant, numRelevantByQuery[qi]));
        const ce = contextEfficiency(hits.map(h => h.chunk.tokenEstimate), relevant);
        if (Number.isFinite(ce)) ctxEffs.push(ce);
      }
      // 按 tag 分桶宏平均（perQuery 与 golden 同序）
      const byTag: Partial<Record<GoldenTag, QueryMetrics>> = {};
      for (const tag of tagsInGolden) {
        const subset = perQuery.filter((_, qi) => golden[qi].tags.includes(tag));
        if (subset.length) byTag[tag] = macroAverage(subset);
      }
      results.push({
        chunking: strategy.name,
        chunkingParams: strategy.params,
        fusion: fusion.name,
        fusionParams: fusion.params,
        metrics: macroAverage(perQuery),
        byTag,
        splitRate,
        ctxEfficiency: ctxEffs.length ? mean(ctxEffs) : Infinity,
        chunkCount,
        avgChunkChars,
        queryCount: golden.length,
      });
    }
  }

  // 主排序键：NDCG 降序（排序质量综合指标）
  results.sort((a, b) => b.metrics.ndcg - a.metrics.ndcg);

  return {
    embedder: config.embedder,
    embedderModel: config.embedderModel,
    topK,
    tau,
    corpusDocs: corpus.size,
    chunkingCount: config.strategies.length,
    fusionCount: fusions.length,
    tagCounts,
    results,
  };
}
