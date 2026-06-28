import { buildChunk, type ChunkStrategy, type ChunkWithSpan } from './types.js';
import { RecursiveParagraphStrategy } from './recursive-paragraph.js';
import { splitByHeadings } from './markdown-section.js';

export interface MarkdownCodeAwareOptions {
  targetTokens?: number;
  charsPerToken?: number;
  includeBreadcrumb?: boolean; // 每个子块前置标题路径，使其脱离上下文也自包含（默认 true）
}

/**
 * 代码感知的 Markdown 分块。
 *
 * 与 markdown-section 的唯一区别：把每节正文里的 ``` 围栏代码块当作**原子单元**，打包时
 * 绝不从 fence 内部下刀。散文部分仍委托 RecursiveParagraphStrategy，行为与 markdown-section
 * 对齐——所以无代码的文档，本策略与 markdown-section 输出逐字一致。
 *
 * 为什么需要：递归段落切分按空行(\n{2,})和句末标点(。.!?)切，二者在代码里到处都是
 * （见 recursive-paragraph.ts:30,61），于是任何含空行或超长的代码块都会被撕成碎片、
 * 开闭 fence 分家——代码与 markdown 双双破损，且基本无法被检索命中。
 *
 * 超长代码块策略：整块保留（即便超过 target）。一个略大但完整的代码块，远胜一堆在
 * `arr.map()`、`3.18.1` 这种点号处断开的碎片。
 *
 * 自然语言上下文：沿用面包屑前缀（标题路径）。embedder 对裸代码向量化很差、而 query 多是
 * 自然语言，靠面包屑让纯代码块仍可被 NL 查询命中。
 *
 * 仅识别三反引号围栏（与 markdown-section 的 fence 判定一致）；行内单反引号不受影响。
 */
export class MarkdownCodeAwareStrategy implements ChunkStrategy {
  readonly name = 'markdown-code-aware';
  readonly params: Record<string, number | boolean>;
  private readonly para: RecursiveParagraphStrategy;
  private readonly charsPerToken: number;
  private readonly includeBreadcrumb: boolean;

  constructor(opts: MarkdownCodeAwareOptions = {}) {
    const targetTokens = opts.targetTokens ?? 512;
    this.charsPerToken = opts.charsPerToken ?? 1.6;
    this.includeBreadcrumb = opts.includeBreadcrumb ?? true;
    this.para = new RecursiveParagraphStrategy({ targetTokens, charsPerToken: this.charsPerToken });
    this.params = { targetTokens, charsPerToken: this.charsPerToken, includeBreadcrumb: this.includeBreadcrumb };
  }

  chunk(source: string, text: string): ChunkWithSpan[] {
    const out: ChunkWithSpan[] = [];
    let idx = 0;

    for (const sec of splitByHeadings(text)) {
      const prefix = this.includeBreadcrumb && sec.breadcrumb ? `${sec.breadcrumb}\n\n` : '';
      for (const seg of splitOutCodeBlocks(sec.content)) {
        if (seg.kind === 'code') {
          // 代码块：整块一个 chunk，绝不内部切分。span 为文档级近似（节偏移 + 段内偏移）。
          const start = sec.contentOffset + seg.offset;
          out.push(buildChunk(
            source, prefix + seg.text, idx++, this.charsPerToken,
            start, start + seg.text.length,
          ));
        } else {
          // 散文：复用基线递归段落切分，行为与 markdown-section 一致。
          for (const sub of this.para.chunk(source, seg.text)) {
            out.push(buildChunk(
              source, prefix + sub.text, idx++, this.charsPerToken,
              sec.contentOffset + seg.offset + sub.startOffset,
              sec.contentOffset + seg.offset + sub.endOffset,
            ));
          }
        }
      }
    }
    return out;
  }
}

interface Segment {
  kind: 'prose' | 'code';
  text: string;
  offset: number; // 该段在节正文 content 中的起始字符下标（近似，供重建文档级 span）
}

/**
 * 把一节正文按 ``` 围栏拆成有序的 prose / code 段。
 * code 段含开闭 fence、整块不可分；相邻散文聚成一个 prose 段交给递归切分。
 * 文档若以未闭合的 fence 结尾，剩余部分按 code 收尾（宁可整块，不破码）。
 */
function splitOutCodeBlocks(content: string): Segment[] {
  const segs: Segment[] = [];
  let buf: string[] = [];
  let bufKind: 'prose' | 'code' = 'prose';
  let bufStart = 0; // 当前 buf 的起始字符偏移
  let pos = 0;      // 运行字符偏移（当前行行首）
  let inFence = false;

  // 冲出当前 buf（非空才入列），并切到下一段的 kind / 起始偏移。
  const flush = (nextKind: 'prose' | 'code', nextStart: number) => {
    const text = buf.join('\n');
    if (text.trim()) segs.push({ kind: bufKind, text, offset: bufStart });
    buf = [];
    bufKind = nextKind;
    bufStart = nextStart;
  };

  for (const line of content.split('\n')) {
    const isFence = /^\s*```/.test(line);
    if (isFence && !inFence) {
      flush('code', pos);       // 先冲掉前面的散文，再开始攒代码块
      inFence = true;
      buf.push(line);           // 开 fence 收进代码段
    } else if (isFence && inFence) {
      buf.push(line);           // 闭 fence 收进代码段
      inFence = false;
      flush('prose', pos + line.length + 1); // 整块代码冲出，回到散文
    } else {
      buf.push(line);
    }
    pos += line.length + 1;
  }
  flush('prose', pos); // 收尾（用的是当前 bufKind，未闭合则仍为 code）
  return segs;
}
