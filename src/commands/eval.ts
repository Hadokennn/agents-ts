import type { CommandHandler } from './index.js';
import type { EmbeddingFn } from '../rag/embedder.js';
import { runEval } from '../eval/runner.js';
import { renderLeaderboard, writeResults } from '../eval/report.js';
import { RecursiveParagraphStrategy } from '../rag/strategies/recursive-paragraph.js';
import { FixedSizeStrategy } from '../rag/strategies/fixed-size.js';
import { FixedOverlapStrategy } from '../rag/strategies/fixed-overlap.js';
import { MarkdownSectionStrategy } from '../rag/strategies/markdown-section.js';
import { WeightedScoreFusion } from '../rag/fusion/weighted-score.js';
import { RrfFusion } from '../rag/fusion/rrf.js';

const CORPUS_DIR = 'eval-data/corpus';
const DATASET = 'eval-data/golden.jsonl';
const RESULTS_DIR = 'eval-data/results';

// 切分轴：基线 vs 负向对照 vs 受控变量组（详见 docs/rag-eval-spec.md §4.3）。
function defaultStrategies() {
  return [
    new RecursiveParagraphStrategy({ targetTokens: 256 }),
    new MarkdownSectionStrategy({ targetTokens: 256 }),
    new FixedSizeStrategy({ targetTokens: 256 }),
    new FixedOverlapStrategy({ targetTokens: 256, overlapTokens: 32 }),
  ];
}

export function createEvalCommands(
  embedFn: EmbeddingFn,
  embedder: 'mock' | 'dashscope',
  embedderModel: string,
): CommandHandler[] {
  return [
    (cmd, ctx) => {
      const m = cmd.match(/^\/?eval(\s+fusion)?$/);
      if (!m) return false;
      // 默认只跑加权融合一种；`/eval fusion` 才加上 RRF，把融合也作为一根评测轴
      const withFusionAxis = !!m[1];
      const fusions = withFusionAxis
        ? [new WeightedScoreFusion(), new RrfFusion()]
        : [new WeightedScoreFusion()];
      const strategies = defaultStrategies();
      console.log(`\n[评测] 正在跑矩阵（${strategies.length}切分 × ${fusions.length}融合）...`);
      runEval({
        corpusDir: CORPUS_DIR,
        datasetPath: DATASET,
        strategies,
        fusions,
        embedFn,
        embedder,
        embedderModel,
        topK: 5,
      })
        .then(report => {
          console.log(renderLeaderboard(report));
          const out = writeResults(report, RESULTS_DIR);
          console.log(`\n结果已写入 ${out}\n`);
          ctx.ask();
        })
        .catch((e: any) => {
          console.error(`[评测] 失败: ${e.message}`);
          ctx.ask();
        });
      return 'async';
    },
  ];
}
