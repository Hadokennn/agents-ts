import { buildChunk, type ChunkStrategy, type ChunkWithSpan } from './types.js';
import { RecursiveParagraphStrategy } from './recursive-paragraph.js';

export interface MarkdownSectionOptions {
  targetTokens?: number;
  charsPerToken?: number;
  includeBreadcrumb?: boolean; // 每个子块前置标题路径，使其脱离上下文也自包含（默认 true）
}

interface Section {
  breadcrumb: string;    // 从根到本节的标题路径，如 "RAG > 分块策略 > 递归段落"
  content: string;       // 本节正文（不含标题行）
  contentOffset: number; // content 在原文中的起始字符下标（近似，仅供诊断）
}

/**
 * Markdown 标题层级分块。
 *
 * 两段式：先按 Markdown 标题（# ~ ######）切出层级化的「节」，每节带从根到本节的
 * 标题路径作面包屑；再把每节正文**委托**给 RecursiveParagraphStrategy 做段落递归切分
 * （组合复用，不重复实现切分逻辑）。可选地为每个子块前置面包屑，让块独立可检索。
 *
 * 无标题文档自动退化为纯递归段落分块（文本与 recursive-paragraph 等价）。
 * 仅识别 ATX 标题（# 开头）；``` 代码块内的 # 不当标题。
 */
export class MarkdownSectionStrategy implements ChunkStrategy {
  readonly name = 'markdown-section';
  readonly params: Record<string, number | boolean>;
  private readonly para: RecursiveParagraphStrategy;
  private readonly charsPerToken: number;
  private readonly includeBreadcrumb: boolean;

  constructor(opts: MarkdownSectionOptions = {}) {
    const targetTokens = opts.targetTokens ?? 256;
    this.charsPerToken = opts.charsPerToken ?? 1.6;
    this.includeBreadcrumb = opts.includeBreadcrumb ?? true;
    this.para = new RecursiveParagraphStrategy({ targetTokens, charsPerToken: this.charsPerToken });
    this.params = { targetTokens, charsPerToken: this.charsPerToken, includeBreadcrumb: this.includeBreadcrumb };
  }

  chunk(source: string, text: string): ChunkWithSpan[] {
    const out: ChunkWithSpan[] = [];
    let idx = 0;

    for (const sec of splitByHeadings(text)) {
      // 委托：节内正文交给递归段落切分（返回的是节内局部 id/offset，下面重建为全局）
      const subs = this.para.chunk(source, sec.content);
      const prefix = this.includeBreadcrumb && sec.breadcrumb ? `${sec.breadcrumb}\n\n` : '';
      for (const sub of subs) {
        out.push(buildChunk(
          source,
          prefix + sub.text,
          idx++,                               // 全局重新编号，避免跨节 id 冲突
          this.charsPerToken,
          sec.contentOffset + sub.startOffset, // 节内偏移 + 节偏移 = 文档级偏移（近似）
          sec.contentOffset + sub.endOffset,
        ));
      }
    }
    return out;
  }
}

// 按 ATX 标题切节，维护标题栈得到面包屑路径；跳过 ``` 代码块内的伪标题。
function splitByHeadings(text: string): Section[] {
  const sections: Section[] = [];
  const stack: Array<{ level: number; title: string }> = [];
  let buf: string[] = [];
  let offset = 0;    // 当前行起始的运行偏移
  let bufOffset = 0; // 当前缓冲区正文的起始偏移
  let inFence = false;

  const flush = () => {
    const content = buf.join('\n');
    if (content.trim()) {
      sections.push({
        breadcrumb: stack.map(h => h.title).join(' > '),
        content,
        contentOffset: bufOffset,
      });
    }
    buf = [];
  };

  for (const line of text.split('\n')) {
    if (/^\s*```/.test(line)) inFence = !inFence;
    const m = inFence ? null : /^(#{1,6})\s+(.*\S)\s*$/.exec(line);
    if (m) {
      flush();
      const level = m[1].length;
      while (stack.length && stack[stack.length - 1].level >= level) stack.pop();
      stack.push({ level, title: m[2].trim() });
      offset += line.length + 1;
      bufOffset = offset; // 正文从标题行之后开始
    } else {
      buf.push(line);
      offset += line.length + 1;
    }
  }
  flush();
  return sections;
}
