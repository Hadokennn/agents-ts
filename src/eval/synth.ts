import { generateText } from 'ai';
import type { GoldenItem } from './dataset.js';

// 从 generateText 的入参反推 model 类型，避免硬编码 SDK 版本相关的类型名
type Model = Parameters<typeof generateText>[0]['model'];

const SYNTH_SYSTEM = `你是 RAG 评测集构造助手。给你一段知识库文本，你要生成 1~2 个「仅凭这段文字就能明确回答」的中文问题。
要求：
1. 问题必须能仅凭这段文字回答，不依赖外部常识。
2. 问题要具体，避免「这段讲了什么」这类泛问。
3. answerSpan 必须是这段文字中能回答该问题的原句，逐字摘录，不要改写。
4. 只输出 JSON 数组，每个元素形如 {"query": "...", "answerSpan": "..."}，不要任何解释或代码块标记。`;

export interface SynthOptions {
  maxPerPassage?: number;
}

/**
 * 用 LLM 为每段语料合成候选 golden 问答对。
 *
 * 产出 origin 标记为 'synth'，必须再经 validateGolden 锚定校验 + 人工抽检（≥20%）
 * 才能入库；合成集与手写集在报告里应分桶看指标，避免合成噪声掩盖真实差异。
 * 详见 docs/rag-eval-spec.md §9.2。
 */
export async function synthesizeGolden(
  model: Model,
  passages: Array<{ source: string; text: string }>,
  opts: SynthOptions = {},
): Promise<GoldenItem[]> {
  const maxPer = opts.maxPerPassage ?? 2;
  const out: GoldenItem[] = [];
  let n = 0;

  for (const p of passages) {
    let parsed: Array<{ query?: string; answerSpan?: string }> = [];
    try {
      const { text } = await generateText({ model, system: SYNTH_SYSTEM, prompt: p.text });
      parsed = extractJsonArray(text);
    } catch {
      continue; // 单段失败不影响整体，跳过
    }
    for (const item of parsed.slice(0, maxPer)) {
      if (!item?.query?.trim() || !item?.answerSpan?.trim()) continue;
      out.push({
        id: `syn-${String(++n).padStart(3, '0')}`,
        query: item.query.trim(),
        source: p.source,
        goldenAnswer: item.answerSpan.trim(),
        tags: ['factoid'],
        origin: 'synth',
      });
    }
  }
  return out;
}

// 容错提取响应里的 JSON 数组（模型可能裹一层代码块或前后缀文字）
function extractJsonArray(text: string): Array<{ query?: string; answerSpan?: string }> {
  const start = text.indexOf('[');
  const end = text.lastIndexOf(']');
  if (start < 0 || end <= start) return [];
  try {
    const arr = JSON.parse(text.slice(start, end + 1));
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}
