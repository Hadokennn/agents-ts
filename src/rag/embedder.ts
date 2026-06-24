const DIMS = 128;

export type EmbeddingFn = (texts: string[]) => Promise<number[][]>;

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
