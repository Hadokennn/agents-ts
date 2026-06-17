import { cosineSimilarity } from './embedder.js';
import type { StoredChunk, VectorStore } from './store.js';
import type { EmbeddingFn } from './embedder.js';
import { embed } from './embedder.js';

export interface SearchResult {
  chunk: StoredChunk;
  score: number;
  vectorScore: number;
  keywordScore: number;
}

const VECTOR_WEIGHT = 0.7;
const KEYWORD_WEIGHT = 0.3;
const CANDIDATE_MULTIPLIER = 4;
const MMR_LAMBDA = 0.7;  // 70% 看相关性，30% 看多样性

export async function hybridSearch(
  store: VectorStore,
  embedFn: EmbeddingFn,
  query: string,
  topK: number = 5,
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

  // Normalize scores to [0, 1]
  const vecNorm = normalizeMinMax(vectorResults.map(r => r.score));
  const kwNorm = normalizeViaSigmoid(keywordResults.map(r => r.score));

  // Merge into unified candidate set
  const candidates = new Map<string, SearchResult>();

  for (let i = 0; i < vectorResults.length; i++) {
    const id = vectorResults[i].chunk.id;
    candidates.set(id, {
      chunk: vectorResults[i].chunk,
      score: vecNorm[i] * VECTOR_WEIGHT,
      vectorScore: vecNorm[i],
      keywordScore: 0,
    });
  }

  for (let i = 0; i < keywordResults.length; i++) {
    const id = keywordResults[i].chunk.id;
    const existing = candidates.get(id);
    if (existing) {
      existing.keywordScore = kwNorm[i];
      existing.score += kwNorm[i] * KEYWORD_WEIGHT;
    } else {
      candidates.set(id, {
        chunk: keywordResults[i].chunk,
        score: kwNorm[i] * KEYWORD_WEIGHT,
        vectorScore: 0,
        keywordScore: kwNorm[i],
      });
    }
  }

  // Sort by combined score
  const sorted = [...candidates.values()].sort((a, b) => b.score - a.score);

  // MMR deduplication
  return mmrSelect(sorted, topK);
}

// ── BM25 scoring ──────────────────────────

/**
 * 对文本进行分词，按空格分割成数组，移除非字母字符和单字符
 * @param text 待分词的文本
 * @returns 分词后的字符串数组
 */
function tokenize(text: string): string[] {
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

// ── Normalization ──────────────────────────

/**
 * 对分数数组进行归一化，将所有分数映射到 [0, 1] 范围内
 * @param scores 待归一化的分数数组
 * @returns 归一化后的分数数组
 * 原始分数: [10, 20, 30]
 * min=10, max=30, range=20
 * 归一化后: [(10-10)/20=0, (20-10)/20=0.5, (30-10)/20=1] → [0, 0.5, 1]
 */
function normalizeMinMax(scores: number[]): number[] {
  if (scores.length === 0) return [];
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const range = max - min || 1;
  return scores.map(s => (s - min) / range);
}

/**
 * 对分数数组进行归一化，将所有分数映射到 [0, 1] 范围内
 * @param scores 待归一化的分数数组
 * @returns 归一化后的分数数组
 * 待归一化的分数: [10, 20, 30]
 * 归一化后: [0.2689414213, 0.7310585787, 0.9999999999] → [0.269, 0.731, 0.999]
 * 归一化后的分数数组，每个分数都介于 0 和 1 之间，且总和为 1
 */
function normalizeViaSigmoid(scores: number[]): number[] {
  return scores.map(s => 1 / (1 + Math.exp(-s)));
}

// ── MMR deduplication MMR（Maximal Marginal Relevance）在选结果时兼顾相关性和多样性 ──────────────────────

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
