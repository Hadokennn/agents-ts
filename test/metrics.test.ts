import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  hitRate, recallAtK, precisionAtK, mrrAtK, ndcgAtK,
  computeQueryMetrics, macroAverage, mean, contextEfficiency,
} from '../src/eval/metrics.js';

const close = (a: number, b: number, eps = 1e-9) =>
  assert.ok(Math.abs(a - b) < eps, `expected ${a} ≈ ${b}`);

test('hitRate: 有命中=1，无命中=0', () => {
  assert.equal(hitRate([false, true, false]), 1);
  assert.equal(hitRate([false, false]), 0);
  assert.equal(hitRate([]), 0);
});

test('recallAtK: 命中数/相关总数，分母为 0 兜底 0', () => {
  assert.equal(recallAtK([true, false, true], 4), 0.5); // 2/4
  assert.equal(recallAtK([true, true], 2), 1);
  assert.equal(recallAtK([true], 0), 0);
});

test('precisionAtK: 命中数/返回数，空集兜底 0', () => {
  close(precisionAtK([true, false, false]), 1 / 3);
  assert.equal(precisionAtK([true, true]), 1);
  assert.equal(precisionAtK([]), 0);
});

test('mrrAtK: 第一名=1，第三名=1/3，无命中=0', () => {
  assert.equal(mrrAtK([true, false, false]), 1);
  close(mrrAtK([false, false, true]), 1 / 3);
  assert.equal(mrrAtK([false, false]), 0);
});

test('ndcgAtK: 相关在首位=1；相关在次位<1', () => {
  assert.equal(ndcgAtK([true, false], 1), 1); // DCG=1/log2(2)=1, IDCG=1
  close(ndcgAtK([false, true], 1), 1 / Math.log2(3)); // DCG=1/log2(3), IDCG=1
  assert.equal(ndcgAtK([false, false], 1), 0);
  assert.equal(ndcgAtK([true], 0), 0); // 无相关→IDCG=0→兜底0
});

test('computeQueryMetrics: 组合一致', () => {
  const m = computeQueryMetrics([true, false, false], 2);
  assert.equal(m.hitRate, 1);
  assert.equal(m.recall, 0.5);
  close(m.precision, 1 / 3);
  assert.equal(m.mrr, 1);
});

test('macroAverage: 每条 query 等权平均', () => {
  const avg = macroAverage([
    { hitRate: 1, recall: 1, precision: 1, mrr: 1, ndcg: 1 },
    { hitRate: 0, recall: 0, precision: 0, mrr: 0, ndcg: 0 },
  ]);
  assert.equal(avg.hitRate, 0.5);
  assert.equal(avg.recall, 0.5);
  assert.deepEqual(macroAverage([]), { hitRate: 0, recall: 0, precision: 0, mrr: 0, ndcg: 0 });
});

test('mean: 空数组=0', () => {
  assert.equal(mean([1, 2, 3]), 2);
  assert.equal(mean([]), 0);
});

test('contextEfficiency: 总token/相关token，无相关=Infinity', () => {
  // 召回三块 token=[10,20,30]，相关为第1、3块 → total=60, relevant=40 → 1.5
  close(contextEfficiency([10, 20, 30], [true, false, true]), 60 / 40);
  assert.equal(contextEfficiency([10, 20], [false, false]), Infinity);
});
