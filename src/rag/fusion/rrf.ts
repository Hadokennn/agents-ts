import type { FusionStrategy, RankedList, SearchResult } from './types.js';

export interface RrfOptions {
  k?: number;            // 平滑常数，常取 60；越大越淡化名次差距
  vectorWeight?: number;
  keywordWeight?: number;
}


/**
 * 核心就一行：contrib = weight / (k + rank)，rank 从 1 起，每路累加。

- 思想：只用名次，不用原始分数。某文档在某路排第 1 → 贡献 1/(60+1)，第 2 → 1/(60+2)…… 把它在各路的贡献加起来就是综合分。
- 为什么这么干：向量路的 cosine（~0~1）和 BM25（无上界）分数尺度根本不可比，直接加权求和必须先归一化、还很难调（这正是你 weighted-score.ts 里 min-max 和 sigmoid 两套口径"难干净解释"的痛点）。RRF 用名次（无量纲）彻底绕开归一化。
- k 的作用：平滑常数。k 越大，名次之间的差距越被抹平（第 1 和第 10 没差那么多）；k=60 是原论文经典值。
- 代价：丢掉了"好多少"的幅度信息——一个断崖式领先的文档，也只体现为"名次高一位"。RRF 偏爱"两路都还行"的共识结果，weighted-score 则保留幅度但要处理归一化。二者是你 FusionStrategy 下真正并列可换的两个实现。
 */

/**
 * 倒数排名融合 RRF（Reciprocal Rank Fusion），作为加权分数融合的对照。
 *
 * 只用「名次」不用「分数」：每路贡献 = w / (k + rank)，rank 从 1 起，累加得综合分。
 * 因此天生无需归一化（名次无量纲），对两路分数尺度不一致鲁棒，
 * 并偏爱「两路都还行」的共识结果。代价是丢弃了「差多少」的分数大小信息。
 * 经典 RRF 两路等权（wv=wk=1）；权重可调即为 weighted-RRF。
 */
export class RrfFusion implements FusionStrategy {
  readonly name = 'rrf';
  readonly params: Record<string, number>;
  private readonly k: number;
  private readonly vw: number;
  private readonly kw: number;

  constructor(opts: RrfOptions = {}) {
    this.k = opts.k ?? 60;
    this.vw = opts.vectorWeight ?? 1;
    this.kw = opts.keywordWeight ?? 1;
    this.params = { k: this.k, vectorWeight: this.vw, keywordWeight: this.kw };
  }

  fuse(vectorResults: RankedList, keywordResults: RankedList): SearchResult[] {
    const candidates = new Map<string, SearchResult>();

    const accumulate = (list: RankedList, weight: number, path: 'v' | 'k') => {
      for (let i = 0; i < list.length; i++) {
        const contrib = weight / (this.k + (i + 1)); // i+1 = 该路名次
        const id = list[i].chunk.id;
        const existing = candidates.get(id);
        if (existing) {
          existing.score += contrib;
          if (path === 'v') existing.vectorScore = contrib;
          else existing.keywordScore = contrib;
        } else {
          candidates.set(id, {
            chunk: list[i].chunk,
            score: contrib,
            vectorScore: path === 'v' ? contrib : 0,
            keywordScore: path === 'k' ? contrib : 0,
          });
        }
      }
    };

    accumulate(vectorResults, this.vw, 'v');
    accumulate(keywordResults, this.kw, 'k');

    return [...candidates.values()].sort((a, b) => b.score - a.score);
  }
}
