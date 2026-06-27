import type { FusionStrategy, RankedList, SearchResult } from './types.js';

export interface WeightedScoreOptions {
  vectorWeight?: number;
  keywordWeight?: number;
}

/**
 * 加权分数融合（项目原有、默认策略）。
 *
 * 两路各自归一化到 [0,1] 后加权求和：score = vecNorm·wv + kwNorm·wk。
 * 权重调的是两路「分数大小」的话语权。注意：向量用 min-max、关键词用 sigmoid，
 * 两种归一化口径不同，这也是 weight 难以干净解释的原因（对照 RrfFusion）。
 *
 * 逻辑与历史 hybridSearch 内联融合段逐字一致（已用快照 diff 验证）。
 */
export class WeightedScoreFusion implements FusionStrategy {
  readonly name = 'weighted-score';
  readonly params: Record<string, number>;
  private readonly vw: number;
  private readonly kw: number;

  constructor(opts: WeightedScoreOptions = {}) {
    this.vw = opts.vectorWeight ?? 0.7;
    this.kw = opts.keywordWeight ?? 0.3;
    this.params = { vectorWeight: this.vw, keywordWeight: this.kw };
  }

  fuse(vectorResults: RankedList, keywordResults: RankedList): SearchResult[] {
    // Normalize scores to [0, 1]
    const vecNorm = normalizeMinMax(vectorResults.map(r => r.score));
    const kwNorm = normalizeViaSigmoid(keywordResults.map(r => r.score));

    // Merge into unified candidate set
    const candidates = new Map<string, SearchResult>();

    for (let i = 0; i < vectorResults.length; i++) {
      const id = vectorResults[i].chunk.id;
      candidates.set(id, {
        chunk: vectorResults[i].chunk,
        score: vecNorm[i] * this.vw,
        vectorScore: vecNorm[i],
        keywordScore: 0,
      });
    }

    for (let i = 0; i < keywordResults.length; i++) {
      const id = keywordResults[i].chunk.id;
      const existing = candidates.get(id);
      if (existing) {
        existing.keywordScore = kwNorm[i];
        existing.score += kwNorm[i] * this.kw;
      } else {
        candidates.set(id, {
          chunk: keywordResults[i].chunk,
          score: kwNorm[i] * this.kw,
          vectorScore: 0,
          keywordScore: kwNorm[i],
        });
      }
    }

    return [...candidates.values()].sort((a, b) => b.score - a.score);
  }
}

// ── Normalization（自 search.ts 原样迁移）──────────────────────────

function normalizeMinMax(scores: number[]): number[] {
  if (scores.length === 0) return [];
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const range = max - min || 1;
  return scores.map(s => (s - min) / range);
}

function normalizeViaSigmoid(scores: number[]): number[] {
  return scores.map(s => 1 / (1 + Math.exp(-s)));
}
