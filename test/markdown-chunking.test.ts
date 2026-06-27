import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MarkdownSectionStrategy } from '../src/rag/strategies/markdown-section.js';
import { RecursiveParagraphStrategy } from '../src/rag/strategies/recursive-paragraph.js';

const doc = `# 顶层标题

顶层下的一段引言文字，放在所有子标题之前。

## 分块策略

这一节讲分块。第一段内容。

### 递归段落

递归段落的细节段落。

## 检索

检索这一节的内容。`;

test('按标题切节，并为每个块前置完整面包屑路径', () => {
  const chunks = new MarkdownSectionStrategy().chunk('doc', doc);
  assert.ok(chunks.length >= 4, `期望 ≥4 块，实得 ${chunks.length}`);
  const recur = chunks.find(c => c.text.includes('递归段落的细节'));
  assert.ok(recur, '应有包含「递归段落的细节」的块');
  assert.ok(recur!.text.startsWith('顶层标题 > 分块策略 > 递归段落'), `面包屑不对: ${recur!.text.slice(0, 30)}`);
});

test('全局 id / index 唯一，不跨节冲突', () => {
  const chunks = new MarkdownSectionStrategy().chunk('doc', doc);
  const ids = chunks.map(c => c.id);
  assert.equal(new Set(ids).size, ids.length, '出现重复 id');
  chunks.forEach((c, i) => assert.equal(c.index, i)); // index 连续
});

test('preamble（首个标题前的正文）面包屑只含顶层', () => {
  const chunks = new MarkdownSectionStrategy().chunk('doc', doc);
  const pre = chunks.find(c => c.text.includes('引言文字'));
  assert.ok(pre);
  assert.ok(pre!.text.startsWith('顶层标题\n\n'), `preamble 面包屑应只含顶层: ${pre!.text.slice(0, 20)}`);
});

test('可关闭面包屑', () => {
  const chunks = new MarkdownSectionStrategy({ includeBreadcrumb: false }).chunk('doc', doc);
  assert.ok(!chunks.some(c => c.text.includes(' > ')), '关闭后不应出现面包屑分隔符');
});

test('无标题文档退化为纯递归段落（文本逐字一致）', () => {
  const plain = '第一段内容。\n\n第二段内容。\n\n第三段内容。';
  const md = new MarkdownSectionStrategy({ includeBreadcrumb: false }).chunk('p', plain);
  const rp = new RecursiveParagraphStrategy().chunk('p', plain);
  assert.deepEqual(md.map(c => c.text), rp.map(c => c.text));
});

test('代码块内的 # 不被当作标题', () => {
  const withCode = `# 标题\n\n\`\`\`python\n# 这是注释不是标题\nx = 1\n\`\`\`\n\n正文。`;
  const chunks = new MarkdownSectionStrategy({ includeBreadcrumb: false }).chunk('c', withCode);
  // 只有一个真标题 → 只有一个面包屑上下文，不应因代码注释多切出一节
  assert.ok(chunks.some(c => c.text.includes('# 这是注释不是标题')), '代码注释应保留在正文块里');
});
