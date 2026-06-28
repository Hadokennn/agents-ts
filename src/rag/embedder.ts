const DIMS = 1024;

export type EmbeddingFn = (texts: string[]) => Promise<number[][]>;

/** 可选的 embedder 实现：本地 GGUF / 云 API / 冒烟用字符哈希 */
export type EmbedderKind = 'mock' | 'dashscope' | 'local';

export function createMockEmbedder(): EmbeddingFn {
  return async (texts: string[]) => texts.map(mockEmbed);
}

export function createDashScopeEmbedder(apiKey: string): EmbeddingFn {
  return async (texts: string[]) => {
    const resp = await fetch(
      'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings',
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'text-embedding-v3',
          input: texts,
          dimensions: DIMS,
        }),
      },
    );
    if (!resp.ok) {
      throw new Error(`Embedding API error: ${resp.status} ${await resp.text()}`);
    }
    const data = await resp.json() as any;
    return data.data.map((d: any) => d.embedding as number[]);
  };
}

export interface LocalEmbedderOptions {
  /** GGUF 模型：HF URI（hf:repo/file）或本地 .gguf 路径。
   *  默认 Qwen3-Embedding-0.6B（通义血统、中文强，text-embedding-v3 的本地等价物）。 */
  model?: string;
  /** 模型缓存目录，首次自动下载到此处（离线、免 HF token）。 */
  cacheDir?: string;
}

/**
 * 本地 embedding：用 node-llama-cpp 加载 GGUF 模型，离线、免 API key、数据不出本机。
 * 输出经 MRL 截断 + L2 归一化到 DIMS 维，与现有向量库（DashScope 同维）兼容，可直接互换。
 * 注意：不同模型向量空间不通用——切换 embedder 后需对整个语料重新 embed 建库。
 *
 * 用前先安装依赖：pnpm add node-llama-cpp
 */
export function createLocalEmbedder(options: LocalEmbedderOptions = {}): EmbeddingFn {
  // 用 || 而非 ??：空字符串（如 .env 里 LOCAL_EMBED_MODEL=）也回落到默认，避免被当本地路径。
  const modelUri =
    options.model ||
    'hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf';
  const cacheDir = options.cacheDir || './models';

  // 懒加载：首次调用时才下载/加载模型，避免启动开销与顶层 await；之后复用同一 context。
  let ctxPromise: Promise<any> | null = null;
  const getContext = () => {
    if (!ctxPromise) {
      ctxPromise = (async () => {
        const { getLlama, resolveModelFile, LlamaLogLevel } = await import('node-llama-cpp');
        const llama = await getLlama({ logLevel: LlamaLogLevel.error });                      // Mac 自动走 Metal
        const modelPath = await resolveModelFile(modelUri, cacheDir);
        const model = await llama.loadModel({ modelPath });
        return model.createEmbeddingContext();
      })();
    }
    return ctxPromise;
  };

  return async (texts: string[]) => {
    const ctx = await getContext();
    const out: number[][] = [];
    for (const text of texts) {
      const { vector } = await ctx.getEmbeddingFor(text);
      out.push(truncateAndNormalize(Array.from(vector as Float32Array), DIMS));
    }
    return out;
  };
}

/** MRL 截断到 dims 维并 L2 归一化（Qwen3-Embedding / EmbeddingGemma 均支持 Matryoshka 截断）。 */
function truncateAndNormalize(vec: number[], dims: number): number[] {
  const v = vec.slice(0, dims);
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map(x => x / norm);
}

const embedCache = new Map<string, number[]>();

export async function embed(fn: EmbeddingFn, texts: string[]): Promise<number[][]> {
  const results: number[][] = new Array(texts.length);
  const uncached: { idx: number; text: string }[] = [];

  for (let i = 0; i < texts.length; i++) {
    const cached = embedCache.get(texts[i]);
    if (cached) {
      results[i] = cached;
    } else {
      uncached.push({ idx: i, text: texts[i] });
    }
  }

  if (uncached.length > 0) {
    const BATCH_SIZE = 10; // DashScope API 限制最多 10 个
    for (let i = 0; i < uncached.length; i += BATCH_SIZE) {
      const batch = uncached.slice(i, i + BATCH_SIZE);
      const vectors = await fn(batch.map(u => u.text));
      for (let j = 0; j < batch.length; j++) {
        results[batch[j].idx] = vectors[j];
        embedCache.set(batch[j].text, vectors[j]);
      }
    }
  }

  return results;
}

function mockEmbed(text: string): number[] {
  const vec = new Array(DIMS).fill(0);
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    vec[i % DIMS] += code;
    vec[(i * 7 + 13) % DIMS] += code * 0.3;
  }
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
  return vec.map(v => v / norm);
}

export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

export { DIMS };
