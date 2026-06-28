import type { RerankStrategy, SearchResult } from './types.js';

export interface QwenRerankerOptions {
  /** GGUF cross-encoder 模型：HF URI（hf:repo/file）或本地 .gguf 路径。
   *  默认 Qwen3-Reranker-0.6B（与 embedder 的 Qwen3-Embedding 同源，中文强）。 */
  model?: string;
  /** 模型缓存目录，首次自动下载到此处（离线、免 HF token）。与 embedder 共用 ./models。 */
  cacheDir?: string;
  /** 只精排融合后的前 N 个候选——cross-encoder 每对都要现算前向，贵，故设上限。 */
  topN?: number;
  /** ranking context 的上下文窗口（容纳 query + 单篇文档）。 */
  contextSize?: number;
  /** 位置加权混合：true 时按名次混合「检索分」与「重排分」(头部更信检索、尾部更信
   *  reranker，权重沿用 qmd 经验值)；false(默认) 为纯精排，直接用 cross-encoder 概率排序。 */
  blend?: boolean;
}

/**
 * 基于 Qwen3-Reranker-0.6B 的 cross-encoder 精排。
 *
 * 与 embedder.ts 的 bi-encoder 本质不同：bi-encoder 把 query、doc 分开各编码成向量再比
 * 余弦（可预存、快、精度有上限）；cross-encoder 把 (query, doc) 拼在一起喂模型一次前向，
 * 注意力跨两者交叉 → 精度高得多，但无法预计算，只能对头部候选现算。
 *
 * node-llama-cpp 的 rankAll 直接返回 0~1 的「相关概率」，与输入顺序对齐。
 * 默认行为是「纯精排」：用 rerank 概率覆盖 score 重新排序；原融合分保留在 vector/keywordScore，
 * cross-encoder 分单独存入 rerankScore，便于调试与对照。
 *
 * 用前先确保已安装：pnpm add node-llama-cpp（项目已有）。
 */
export class QwenReranker implements RerankStrategy {
  readonly name = 'qwen3-reranker-0.6b';
  readonly params: Record<string, number | string | boolean>;

  private readonly modelUri: string;
  private readonly cacheDir: string;
  private readonly topN: number;
  private readonly contextSize: number;
  private readonly blend: boolean;

  // 懒加载：首次 rerank 时才下载/加载模型，之后复用同一 ranking context。
  private ctxPromise: Promise<any> | null = null;

  constructor(opts: QwenRerankerOptions = {}) {
    // 用 || 而非 ??：空字符串也回落默认，与 embedder 口径一致。
    this.modelUri =
      opts.model ||
      'hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf';
    this.cacheDir = opts.cacheDir || './models';
    this.topN = opts.topN ?? 20;
    this.contextSize = opts.contextSize ?? 4096;
    this.blend = opts.blend ?? false;
    this.params = { topN: this.topN, contextSize: this.contextSize, blend: this.blend };
  }

  private getContext(): Promise<any> {
    if (!this.ctxPromise) {
      this.ctxPromise = (async () => {
        const { getLlama, resolveModelFile, LlamaLogLevel } = await import('node-llama-cpp');
        const llama = await getLlama({ logLevel: LlamaLogLevel.error });                       // Mac 自动走 Metal
        const modelPath = await resolveModelFile(this.modelUri, this.cacheDir);
        const model = await llama.loadModel({ modelPath });
        return model.createRankingContext({ contextSize: this.contextSize });
      })();
    }
    return this.ctxPromise;
  }

  async rerank(query: string, candidates: SearchResult[]): Promise<SearchResult[]> {
    if (candidates.length === 0) return candidates;

    // 只对头部 topN 跑 cross-encoder；尾部保持原融合序接在后面，不丢召回。
    const head = candidates.slice(0, this.topN);
    const tail = candidates.slice(this.topN);

    const ctx = await this.getContext();
    const rerankScores: number[] = await ctx.rankAll(query, head.map(c => c.chunk.text));

    // blend 模式需把融合分归一化到 [0,1]，才能与 0~1 的 rerank 概率同尺度加权
    // （不同 fusion 量纲差异大：weighted≈[0,1]，RRF≈0.0x，不归一化检索项会被淹没）。
    const retrievalNorm = this.blend ? normalizeMinMax(head.map(c => c.score)) : [];

    const reranked = head
      .map((c, i) => {
        const rerankScore = rerankScores[i];
        // 纯精排：直接用概率排序；blend：按名次混合检索分与重排分。
        const score = this.blend
          ? blendByPosition(i, retrievalNorm[i], rerankScore)
          : rerankScore;
        return { ...c, rerankScore, score };
      })
      .sort((a, b) => b.score - a.score);

    return [...reranked, ...tail];
  }
}

// 位置加权：i 为候选在融合列表中的名次（0 起）。头部是多路共识、可信度高，少让
// reranker 扰动（更信检索）；越往后检索越不可靠，越放手让 reranker 捞遗珠。档位沿用 qmd。
function blendByPosition(i: number, retrieval: number, rerank: number): number {
  const rank = i + 1;
  const [wRetrieval, wRerank] =
    rank <= 3 ? [0.75, 0.25] :
    rank <= 10 ? [0.6, 0.4] :
    [0.4, 0.6];
  return wRetrieval * retrieval + wRerank * rerank;
}

// 融合分归一化到 [0,1]（min-max），仅用于 blend 同尺度加权。
function normalizeMinMax(scores: number[]): number[] {
  if (scores.length === 0) return [];
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const range = max - min || 1;
  return scores.map(s => (s - min) / range);
}
