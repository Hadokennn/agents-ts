import { cosineSimilarity } from './embedder.js';
import type { StoredChunk } from './store.js';
import type { EmbeddingFn } from './embedder.js';
import { embed } from './embedder.js';
import type { FusionStrategy, SearchResult } from './fusion/types.js';
import { WeightedScoreFusion } from './fusion/weighted-score.js';
import type { RerankStrategy } from './rerank/types.js';

// hybridSearch 只依赖 getAll()，结构化类型让内存 VectorStore 与 SqliteVectorStore 都能传入
// （评测用内存索引复用同一套检索逻辑，不污染 knowledge.db）。
export interface RetrievalStore {
  getAll(): StoredChunk[];
}

// SearchResult 已下沉到 fusion/types.ts；在此再导出，保持历史导入路径不变。
export type { SearchResult };

const CANDIDATE_MULTIPLIER = 4;
const MMR_LAMBDA = 0.7;  // 70% 看相关性，30% 看多样性

// 默认融合策略 = 加权分数（参数为历史常量 0.7/0.3），行为零变更。
const defaultFusion = new WeightedScoreFusion();

export async function hybridSearch(
  store: RetrievalStore,
  embedFn: EmbeddingFn,
  query: string,
  topK: number = 5,
  fusion: FusionStrategy = defaultFusion,
  rerank?: RerankStrategy,  // 可选：传入则在融合后、MMR 前做 cross-encoder 精排
): Promise<SearchResult[]> {
  const all = store.getAll();
  if (all.length === 0) return [];

  const candidateCount = Math.min(topK * CANDIDATE_MULTIPLIER, all.length);

  // Path 1: Vector search
  // 将query转换为向量
  const [queryVec] = await embed(embedFn, [query]);
  const vectorResults = all
    .map(chunk => ({ chunk, score: cosineSimilarity(queryVec, chunk.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, candidateCount);

  // Path 2: Keyword search (BM25-like TF-IDF scoring)
  const queryTerms = tokenize(query);
  const docCount = all.length;
  const keywordResults = all
    .map(chunk => ({ chunk, score: bm25Score(queryTerms, chunk.text, docCount, all) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, candidateCount);

  // 融合两路（默认加权分数；可传入 RRF 等对照策略）
  let fused = fusion.fuse(vectorResults, keywordResults);

  // 可选：cross-encoder 精排。读 query+原文重算相关性，比召回/融合更准但贵，
  // 故置于融合之后、只作用于头部候选；未传入则行为零变更。
  if (rerank) {
    fused = await rerank.rerank(query, fused);
  }

  // 再做 MMR 去重，兼顾相关性与多样性
  return mmrSelect(fused, topK);
}

// ── BM25 scoring ──────────────────────────

/**
 * 对文本进行分词，按空格分割成数组，移除非字母字符和单字符
 * @param text 待分词的文本
 * @returns 分词后的字符串数组
 */
export function tokenize(text: string): string[] {
  return text.toLowerCase()
    .replace(/[^\w一-鿿]+/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 1);
}

/**
 * 计算BM25分数
 * 某个词在文档中出现越多 → TF 越高 → 分数越高
 * 某个词在所有文档中越稀有 → IDF 越高 → 分数越高
 * 文档越长，会被适当惩罚（防止长文档占优势）
 * TF (Term Frequency)，词频，该词在文档中出现多少次
 * DF (Document Frequency)，文档频率，有多少文档包含该词
 * IDF(Inverse Document Frequency)，逆文档频率，越稀有的词权重越高
 * @param queryTerms 查询词数组
 * @param docText 当前文档文本
 * @param N 总文档数
 * @param allDocs 全部文档数组
 * @returns BM25分数值
 */
function bm25Score(queryTerms: string[], docText: string, N: number, allDocs: StoredChunk[]): number {
  const k1 = 1.2; // 控制词频饱和程度（标准值）
  const b = 0.75; // 控制文档长度归一化程度（标准值）
  const docTokens = tokenize(docText); // 当前文档的 token 数组
  const avgDl = allDocs.reduce((s, d) => s + tokenize(d.text).length, 0) / (N || 1); // 平均文档长度
  const dl = docTokens.length; // 当前文档长度
  let score = 0;

  for (const term of queryTerms) {
    const tf = docTokens.filter(t => t === term).length;  // 词频：该词在文档中出现多少次
    const df = allDocs.filter(d => tokenize(d.text).includes(term)).length;  // 文档频率：有多少文档包含该词
    const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);  // 逆文档频率：越稀有分数越高
    const tfNorm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgDl)));  // 词频归一化：控制词频对结果的影响
    score += idf * tfNorm;
  }

  return score;
}

// ── MMR deduplication MMR（Maximal Marginal Relevance）在选结果时兼顾相关性和多样性 ──────────────────────

//  mmrSelect，核心在：mmr = λ·relevance − (1−λ)·maxSim，λ=0.7。

// - 思想：在"相关"和"不重复"之间权衡。纯按相关性取 topK，很容易返回 5 个几乎重复的 chunk（同一段话的不同切片）。MMR 贪心地每次挑"既跟 query 相关、又跟已选结果不雷同"的那个。
// - 怎么算：从最相关的开一个头，之后每一步对剩余候选算 λ·自身相关性 − (1−λ)·与已选集合的最大相似度（你用 jaccard 词重叠衡量相似），挑分最高的。
// - λ 旋钮：λ=1 → 纯相关性、不管多样性；λ=0 → 纯多样性。0.7 = 七分相关三分多样。
// - 和 rerank 的协同：精排后 .score 变成了 cross-encoder 概率，所以现在 MMR 是在"cross-encoder 相关性 vs 多样性"之间权衡，比之前用融合分更准。

/**
 * 对搜索结果数组进行MMR去重，返回 topK 个结果
 * @param results 待去重的搜索结果数组
 * @param topK 返回的结果数量
 * @returns 去重后的搜索结果数组
 */
function mmrSelect(results: SearchResult[], topK: number): SearchResult[] {
  if (results.length <= topK) return results;

  const selected: SearchResult[] = [results[0]];
  const remaining = results.slice(1);

  while (selected.length < topK && remaining.length > 0) {
    let bestIdx = 0;
    let bestMmr = -Infinity;

    for (let i = 0; i < remaining.length; i++) {
      const relevance = remaining[i].score; // 相关性分数
      const maxSim = Math.max(...selected.map(s => jaccardSimilarity(s.chunk.text, remaining[i].chunk.text)));  // 最大相似度分数
      const mmr = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * maxSim;  // MMR分数
      if (mmr > bestMmr) {
        bestMmr = mmr;
        bestIdx = i;
      }
    }

    selected.push(remaining[bestIdx]);
    remaining.splice(bestIdx, 1);
  }

  return selected;
}

function jaccardSimilarity(a: string, b: string): number {
  const setA = new Set(tokenize(a));
  const setB = new Set(tokenize(b));
  const intersection = [...setA].filter(t => setB.has(t)).length;
  const union = new Set([...setA, ...setB]).size;
  return union === 0 ? 0 : intersection / union;
}
