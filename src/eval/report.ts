import fs from 'node:fs';
import path from 'node:path';
import type { EvalReport, MatrixCell } from './runner.js';
import type { GoldenTag } from './dataset.js';

const NAME_W = 40;
const pct = (x: number) => (x * 100).toFixed(1).padStart(5);
const eff = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : '  ∞ ');

// 单元标签 = 切分 × 融合（两根轴），参数留在 JSON 里
function label(r: MatrixCell): string {
  return `${r.chunking} × ${r.fusion}`;
}

// 终端排行榜。mock embedder 下打出醒目护栏，提示结果仅供冒烟。
export function renderLeaderboard(report: EvalReport): string {
  const lines: string[] = [];
  lines.push('');
  if (report.embedder === 'mock') {
    lines.push('⚠️  MOCK EMBEDDER — 结果仅供冒烟，不可用于策略决策（字符哈希无语义）');
  }
  lines.push(
    `检索评测排行榜  embedder=${report.embedder}/${report.embedderModel}  ` +
    `矩阵=${report.chunkingCount}切分×${report.fusionCount}融合  ` +
    `topK=${report.topK}  τ=${report.tau}  语料=${report.corpusDocs}篇  查询=${report.results[0]?.queryCount ?? 0}条`,
  );
  lines.push('（NDCG/Recall/MRR/P@k/Hit 越高越好；Split截断率/CtxEff上下文效率 越低越好）');
  lines.push('');

  const head = ['#', '切分 × 融合', 'NDCG', 'Recall', 'MRR', 'P@k', 'Hit', 'Split', 'CtxEff', 'chunks', 'avgCh'];
  const rows = report.results.map((r, i) => [
    String(i + 1),
    label(r).padEnd(NAME_W),
    pct(r.metrics.ndcg),
    pct(r.metrics.recall),
    pct(r.metrics.mrr),
    pct(r.metrics.precision),
    pct(r.metrics.hitRate),
    pct(r.splitRate),
    eff(r.ctxEfficiency).padStart(6),
    String(r.chunkCount).padStart(6),
    String(r.avgChunkChars).padStart(5),
  ]);

  lines.push([head[0].padEnd(2), head[1].padEnd(NAME_W), ...head.slice(2).map(h => h.padStart(6))].join(' '));
  lines.push('─'.repeat(NAME_W + 62));
  for (const row of rows) {
    lines.push([row[0].padEnd(2), row[1], ...row.slice(2).map(c => c.padStart(6))].join(' '));
  }
  lines.push('');
  lines.push(`榜首：${label(report.results[0])}（NDCG ${pct(report.results[0].metrics.ndcg)}%）`);
  lines.push(...renderTagBreakdown(report));
  return lines.join('\n');
}

// 按 tag 分层的 Recall@k 分解：定位每个策略在哪类查询上掉链子（只有一类时不必分层）。
function renderTagBreakdown(report: EvalReport): string[] {
  const tags = Object.keys(report.tagCounts) as GoldenTag[];
  if (tags.length <= 1) return [];
  const lines: string[] = ['', `按 tag 分层 — Recall@k（${tags.map(t => `${t}×${report.tagCounts[t]}`).join('  ')}）`];
  lines.push(['切分 × 融合'.padEnd(NAME_W), ...tags.map(t => t.padStart(12))].join(' '));
  lines.push('─'.repeat(NAME_W + tags.length * 13));
  for (const r of report.results) {
    const cells = tags.map(t => {
      const m = r.byTag[t];
      return (m ? (m.recall * 100).toFixed(1) : '-').padStart(12);
    });
    lines.push([label(r).padEnd(NAME_W), ...cells].join(' '));
  }
  return lines;
}

/**
 * 落盘 JSON。§8 护栏：mock 只写 *-smoke.json，绝不写 *-latest.json，
 * 杜绝拿无语义结果当可信基线 diff。返回写入路径。
 */
export function writeResults(report: EvalReport, dir: string): string {
  fs.mkdirSync(dir, { recursive: true });
  const file = report.embedder === 'mock' ? 'mock-smoke.json' : `${report.embedder}-latest.json`;
  const full = path.join(dir, file);
  fs.writeFileSync(full, JSON.stringify(report, null, 2));
  return full;
}
