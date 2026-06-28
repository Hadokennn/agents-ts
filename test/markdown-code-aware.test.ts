import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MarkdownCodeAwareStrategy } from '../src/rag/strategies/markdown-code-aware.js';
import { MarkdownSectionStrategy } from '../src/rag/strategies/markdown-section.js';

// 失败模式①：代码块内部含空行 → 旧策略按 \n{2,} 撕开，开闭 fence 分家。
test('含空行的代码块整块保留在同一个 chunk（开闭 fence 不分家）', () => {
  const doc = [
    '# 标题', '',
    '说明文字。', '',
    '```ts',
    'function foo() {',
    '  const x = 1;',
    '',                 // ← 代码块内部空行，旧策略的撕点
    '  return x;',
    '}',
    '```', '',
    '后续文字。',
  ].join('\n');

  const chunks = new MarkdownCodeAwareStrategy({ includeBreadcrumb: false }).chunk('d', doc);
  const codeChunk = chunks.find(c => c.text.includes('function foo'));
  assert.ok(codeChunk, '应有包含代码的 chunk');
  // 同一个 chunk 里既有开 fence、又有闭 fence、还有空行后的 return —— 没被撕开
  assert.ok(codeChunk!.text.includes('```ts'), '开 fence 在内');
  assert.ok(codeChunk!.text.trimEnd().endsWith('```'), '闭 fence 在内');
  assert.ok(codeChunk!.text.includes('\n  return x;'), '空行后的代码仍在同一块、且缩进保留');

  // 对照（size-无关的真实损坏）：markdown-section 对每个段落做 .trim()（recursive-paragraph.ts:62），
  // 空行把代码切成段落后，「作为段落首行」的 `  return x;` 缩进被吃掉。本策略整块保留，不会。
  const old = new MarkdownSectionStrategy({ includeBreadcrumb: false }).chunk('d', doc);
  assert.ok(
    old.every(c => !c.text.includes('\n  return x;')),
    '旧策略会丢掉 return 行的缩进（回归基准）',
  );
});

// 失败模式②：单个代码块超过 target → 旧策略按句末标点(.)切，在 arr.map()/版本号处断开。
test('超长代码块整块保留，不在点号处断开', () => {
  const lines = ['```ts'];
  for (let i = 0; i < 40; i++) lines.push(`const v${i} = arr.map(x => x.foo()).filter(Boolean); // 3.18.1`);
  lines.push('```');
  const doc = `# 大代码\n\n${lines.join('\n')}\n`;

  const chunks = new MarkdownCodeAwareStrategy({ includeBreadcrumb: false }).chunk('d', doc);
  const codeChunks = chunks.filter(c => c.text.includes('arr.map'));
  assert.equal(codeChunks.length, 1, `超长代码块应仍是 1 块，实得 ${codeChunks.length}`);
  assert.ok(codeChunks[0].text.includes('v0 =') && codeChunks[0].text.includes('v39 ='), '首尾行都在同一块');
});

// 代码块带上面包屑（标题路径），保证纯代码块仍可被 NL 查询命中。
test('代码块前置面包屑（NL 上下文）', () => {
  const doc = '# 安装\n\n## 步骤\n\n```sh\npnpm add node-llama-cpp\n```';
  const chunks = new MarkdownCodeAwareStrategy().chunk('d', doc);
  const code = chunks.find(c => c.text.includes('pnpm add'));
  assert.ok(code, '应有代码 chunk');
  assert.ok(code!.text.startsWith('安装 > 步骤\n\n'), `面包屑不对: ${code!.text.slice(0, 20)}`);
});

// 等价保证：无代码的文档，本策略与 markdown-section 输出逐字一致（散文路径未改动）。
test('无代码文档与 markdown-section 输出逐字一致', () => {
  const doc = [
    '# 顶层', '', '引言文字。', '',
    '## 分块', '', '这一节讲分块。第一段。', '',
    '### 递归', '', '递归段落细节。', '',
    '## 检索', '', '检索内容。',
  ].join('\n');
  const a = new MarkdownCodeAwareStrategy().chunk('d', doc);
  const b = new MarkdownSectionStrategy().chunk('d', doc);
  assert.deepEqual(a.map(c => c.text), b.map(c => c.text));
});

// id / index 全局唯一连续（跨节、跨 prose/code 段不冲突）。
test('全局 id / index 唯一且连续', () => {
  const doc = '# A\n\n文字一。\n\n```ts\nconst x = 1;\n```\n\n文字二。\n\n## B\n\n```py\ny = 2\n```';
  const chunks = new MarkdownCodeAwareStrategy().chunk('d', doc);
  const ids = chunks.map(c => c.id);
  assert.equal(new Set(ids).size, ids.length, '出现重复 id');
  chunks.forEach((c, i) => assert.equal(c.index, i));
});
