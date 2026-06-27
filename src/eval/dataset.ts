import fs from 'node:fs';
import path from 'node:path';
import { containment } from './matching.js';

// 黄金集条目。ground truth 锚在「答案文本」而非 chunk id，
// 这样对任何切分策略都不变（详见 docs/rag-eval-spec.md §3、§9）。
export type GoldenTag = 'factoid' | 'multi-hop' | 'long-answer';
export type GoldenOrigin = 'handwritten' | 'synth';

export interface GoldenItem {
  id: string;
  query: string;
  source: string;        // 相对 corpusDir 的路径，如 "chunking.md"
  goldenAnswer: string;  // 标准答案文本，用于 §5 的 token 重叠命中判定
  tags: GoldenTag[];     // factoid / multi-hop / long-answer，分层看指标
  origin: GoldenOrigin;  // synth 的需经锚定校验 + 人工抽检
}

export interface ValidationIssue {
  id: string;
  level: 'error' | 'warn';
  message: string;
}

const ALLOWED_TAGS: GoldenTag[] = ['factoid', 'multi-hop', 'long-answer'];
// 锚定有效性阈值：goldenAnswer 至少 60% 的 token 出现在 source 中，否则视为与语料漂移。
const GROUNDING_THRESHOLD = 0.6;

// ── 加载 ──────────────────────────────

export function loadGolden(datasetPath: string): GoldenItem[] {
  const raw = fs.readFileSync(datasetPath, 'utf-8');
  const items: GoldenItem[] = [];
  raw.split('\n').forEach((line, i) => {
    const t = line.trim();
    if (!t || t.startsWith('//')) return; // 跳过空行与注释行
    try {
      items.push(JSON.parse(t) as GoldenItem);
    } catch (e: any) {
      throw new Error(`golden 第 ${i + 1} 行 JSON 解析失败: ${e.message}`);
    }
  });
  return items;
}

// 加载整个语料目录为「相对路径 → 文本」映射
export function loadCorpus(corpusDir: string): Map<string, string> {
  const map = new Map<string, string>();
  for (const file of walk(corpusDir)) {
    map.set(path.relative(corpusDir, file), fs.readFileSync(file, 'utf-8'));
  }
  return map;
}

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(full));
    else if (/\.(md|txt)$/i.test(e.name)) out.push(full);
  }
  return out;
}

// ── 校验 ──────────────────────────────

// 锚定有效性 = goldenAnswer 对整篇 source 的 token 命中率。
// 复用 matching.containment（与检索命中判定同一基元），口径单一真相源。
export function grounding(answer: string, source: string): number {
  return containment(answer, source);
}

/**
 * 校验黄金集：字段完整性 + id 唯一 + tag/origin 合法 + source 存在 + 答案锚定有效。
 * 返回所有问题；error 级别非空即视为校验不通过。
 */
export function validateGolden(items: GoldenItem[], corpus: Map<string, string>): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const seen = new Set<string>();

  for (const it of items) {
    const id = it.id || '(no-id)';
    const err = (message: string) => issues.push({ id, level: 'error', message });

    if (!it.id) err('缺少 id');
    else if (seen.has(it.id)) err(`id 重复: ${it.id}`);
    else seen.add(it.id);

    if (!it.query?.trim()) err('query 为空');
    if (!it.goldenAnswer?.trim()) err('goldenAnswer 为空');
    if (!Array.isArray(it.tags) || it.tags.length === 0) err('tags 为空');
    else for (const tag of it.tags) if (!ALLOWED_TAGS.includes(tag)) err(`非法 tag: ${tag}`);
    if (it.origin !== 'handwritten' && it.origin !== 'synth') err(`非法 origin: ${it.origin}`);

    const src = corpus.get(it.source);
    if (src === undefined) {
      err(`source 不存在于语料: ${it.source}`);
      continue;
    }
    if (it.goldenAnswer?.trim()) {
      const g = grounding(it.goldenAnswer, src);
      if (g < GROUNDING_THRESHOLD) {
        err(`goldenAnswer 锚定无效（token 命中率 ${(g * 100).toFixed(0)}% < ${GROUNDING_THRESHOLD * 100}%），疑与语料漂移`);
      }
    }
  }
  return issues;
}
