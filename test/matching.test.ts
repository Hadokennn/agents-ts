import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  containment, isHit, judgeRelevance, countRelevant, isAnswerSplit, HIT_THRESHOLD,
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
  assert.equal(isHit(GOLD, 'alpha beta gamma'), true);  // ≥0.6
  assert.equal(isHit(GOLD, 'alpha beta'), false);       // 0.5 <0.6
  assert.equal(isHit(GOLD, 'alpha beta', 0.5), true);   // 自定义 τ
  assert.equal(HIT_THRESHOLD, 0.6);
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
