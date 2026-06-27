// 检索 IR 指标（docs/rag-eval-spec.md §6），全部为纯函数，便于单测。
//
// 约定：单条 query 检索回 top-k 结果，relevant[i] 表示第 i 名结果是否相关（已由 matching 判定），
// 按排名降序排列；numRelevant = 整库中该 query 的相关 chunk 总数（|G|，recall 的分母）。

export interface QueryMetrics {
  hitRate: number;
  recall: number;
  precision: number;
  mrr: number;
  ndcg: number;
}

const ZERO: QueryMetrics = { hitRate: 0, recall: 0, precision: 0, mrr: 0, ndcg: 0 };

// 至少捞回一个相关结果即为 1（最宽松）
export function hitRate(relevant: boolean[]): number {
  return relevant.some(Boolean) ? 1 : 0;
}

// 该捞的捞回了多少：命中数 / 相关总数
export function recallAtK(relevant: boolean[], numRelevant: number): number {
  if (numRelevant <= 0) return 0;
  return relevant.filter(Boolean).length / numRelevant;
}

// 捞回的有多少是对的：命中数 / 返回数（噪声反指标）
export function precisionAtK(relevant: boolean[]): number {
  if (relevant.length === 0) return 0;
  return relevant.filter(Boolean).length / relevant.length;
}

// 第一个相关结果排多前：第一名命中=1，第二名=1/2，无命中=0
export function mrrAtK(relevant: boolean[]): number {
  const idx = relevant.findIndex(Boolean);
  return idx < 0 ? 0 : 1 / (idx + 1);
}

// 排序质量（位置加权），二值相关性：DCG = Σ rel_i / log2(i+2)，IDCG 为理想排序的 DCG
export function ndcgAtK(relevant: boolean[], numRelevant: number): number {
  const dcg = relevant.reduce((s, r, i) => s + (r ? 1 / Math.log2(i + 2) : 0), 0);
  const ideal = Math.min(Math.max(numRelevant, 0), relevant.length); // 理想情况下相关结果全部排在最前
  let idcg = 0;
  for (let i = 0; i < ideal; i++) idcg += 1 / Math.log2(i + 2);
  return idcg === 0 ? 0 : dcg / idcg;
}

export function computeQueryMetrics(relevant: boolean[], numRelevant: number): QueryMetrics {
  return {
    hitRate: hitRate(relevant),
    recall: recallAtK(relevant, numRelevant),
    precision: precisionAtK(relevant),
    mrr: mrrAtK(relevant),
    ndcg: ndcgAtK(relevant, numRelevant),
  };
}

// 宏平均：对每条 query 的指标取算术平均（每条 query 等权）
export function macroAverage(all: QueryMetrics[]): QueryMetrics {
  if (all.length === 0) return { ...ZERO };
  const acc = { ...ZERO };
  for (const m of all) {
    acc.hitRate += m.hitRate;
    acc.recall += m.recall;
    acc.precision += m.precision;
    acc.mrr += m.mrr;
    acc.ndcg += m.ndcg;
  }
  const n = all.length;
  return {
    hitRate: acc.hitRate / n,
    recall: acc.recall / n,
    precision: acc.precision / n,
    mrr: acc.mrr / n,
    ndcg: acc.ndcg / n,
  };
}

export function mean(xs: number[]): number {
  return xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length;
}

/**
 * 上下文效率（§6）：召回的总 token / 其中相关 chunk 的 token。越低越好（噪声越少）。
 * 若本次召回没有任何相关结果，返回 Infinity（最差），聚合时应过滤掉非有限值。
 */
export function contextEfficiency(retrievedTokens: number[], relevant: boolean[]): number {
  const total = retrievedTokens.reduce((a, b) => a + b, 0);
  const relevantTokens = retrievedTokens.reduce((a, t, i) => a + (relevant[i] ? t : 0), 0);
  return relevantTokens === 0 ? Infinity : total / relevantTokens;
}
