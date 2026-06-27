import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  containment, isHit, judgeRelevance, countRelevant, isAnswerSplit, HIT_THRESHOLD, ngramTokens,
} from '../src/eval/matching.js';

const close = (a: number, b: number, eps = 1e-9) =>
  assert.ok(Math.abs(a - b) < eps, `expected ${a} ≈ ${b}`);

// 用空格分隔的 ASCII token，规避 CJK 无空格分词的歧义，保证断言确定性
const GOLD = 'alpha beta gamma delta';

test('containment: golden token 被覆盖的比例', () => {
  assert.equal(containment(GOLD, 'alpha beta gamma delta extra'), 1); // 全覆盖
  assert.equal(containment(GOLD, 'alpha beta'), 0.5);                  // 2/4
  assert.equal(containment(GOLD, 'nothing here'), 0);
  assert.equal(containment('', 'whatever'), 0);                        // 空 golden 兜底 0
});

test('isHit: 以 τ 为界', () => {
  // containment=0.75 (3/4)
  assert.equal(isHit(GOLD, 'alpha beta gamma'), true);  // 0.75 ≥0.7
  assert.equal(isHit(GOLD, 'alpha beta'), false);       // 0.5 <0.7
  assert.equal(isHit(GOLD, 'alpha beta', 0.5), true);   // 自定义 τ
  assert.equal(HIT_THRESHOLD, 0.7);
});

test('judgeRelevance: 与输入等长的布尔数组', () => {
  assert.deepEqual(
    judgeRelevance(GOLD, ['alpha beta gamma delta', 'alpha beta', 'zzz']),
    [true, false, false],
  );
});

test('countRelevant: 整库命中计数', () => {
  assert.equal(countRelevant(GOLD, ['alpha beta gamma delta', 'alpha beta', 'alpha beta gamma']), 2);
});

test('isAnswerSplit: 有单块覆盖→未切断；跨相邻块→切断', () => {
  // 单块完整覆盖 → false
  assert.equal(isAnswerSplit(GOLD, ['alpha beta gamma delta extra']), false);
  // 切成两块（直接拼接还原原文）：单块均 0.5<0.6，拼接=1 ≥0.6 → 被切断
  assert.equal(isAnswerSplit(GOLD, ['alpha beta g', 'amma delta']), true);
  // 仅一块无从切断 → false
  assert.equal(isAnswerSplit(GOLD, ['alpha']), false);
});

test('containment 对真实黄金句应高命中（与 dataset 锚定口径一致）', () => {
  const ans = '余弦相似度衡量两个向量方向上的接近程度，取值范围在负一到正一之间。';
  const src = '前面一些铺垫。' + ans + '后面还有别的句子。';
  close(containment(ans, src), 1); // 逐字包含 → 全覆盖
});

test('ngramTokens: 英文整词保留，中文拆字符 bigram', () => {
  const toks = ngramTokens('向量 NDCG 排序质量');
  assert.ok(toks.has('ndcg'), '英文术语应作整词保留');
  assert.ok(toks.has('向量'), '中文应拆 bigram');
  assert.ok(!toks.has('nd'), '不应把英文拆成碎片 bigram');
});

test('containment: 中文改写不再被误杀（bigram 平滑降级）', () => {
  const golden = '余弦相似度衡量两个向量方向上的接近程度';
  const paraphrase = '余弦相似度用来衡量两个向量在方向上的接近程度'; // 同义、加字、换词
  const c = containment(golden, paraphrase);
  assert.ok(c > 0.6, `改写应保留高重叠（整句级会是 0），实得 ${c.toFixed(3)}`);
});

test('containment: 完全无关仍为 0（无假阳）', () => {
  assert.equal(containment('余弦相似度衡量向量', '今天天气很好适合出门散步'), 0);
});
