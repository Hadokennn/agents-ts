// τ：命中阈值（docs/rag-eval-spec.md §5.1 决策点 D2）。
// goldenAnswer 的 n-gram 有 ≥τ 比例出现在某 chunk 中，即认定该 chunk 命中此答案。
// 由真实 golden 分布标定：逐字答案块=1.0、改写真阳≈0.77、边界截断的半句块≈0.6、同主题邻块≤0.5。
// 取 0.7：admits 改写、但拒绝半句截断块（保住截断率这一最稳的分块判别信号）、远离邻块。
export const HIT_THRESHOLD = 0.7;

/**
 * 混合 n-gram 分词：英文/数字术语保留整词，中文连续段拆字符 bigram。
 *
 * 中文无空格，若按标点切只能得到「整句级」token，has 匹配退化成整句精确匹配——
 * 改写或差一字就整句作废，误差极大。降到字符 bigram 让重叠平滑降级；
 * 英文术语（NDCG/BM25/cosine）仍按整词，避免 nd/dc/cg 这类碎片误匹配。
 */
export function ngramTokens(text: string): Set<string> {
  const lower = text.toLowerCase();
  const tokens = new Set<string>();
  // 英文/数字整词（长度 >1）
  for (const m of lower.matchAll(/[a-z0-9]+/g)) {
    if (m[0].length > 1) tokens.add(m[0]);
  }
  // 中文连续段 → 字符 bigram（单字段不成 bigram，自然丢弃）
  for (const m of lower.matchAll(/[一-鿿]+/g)) {
    const run = m[0];
    for (let i = 0; i + 2 <= run.length; i++) tokens.add(run.slice(i, i + 2));
  }
  return tokens;
}

/**
 * 命中判定基元（§5.1）：goldenAnswer 的 n-gram 有多少被给定文本覆盖。
 * 是 ground-truth 锚文本不锚 chunk 的核心——对任何切分策略都不变。
 */
export function containment(goldenText: string, text: string): number {
  const g = ngramTokens(goldenText);
  if (g.size === 0) return 0;
  const c = ngramTokens(text);
  let hit = 0;
  for (const t of g) if (c.has(t)) hit++;
  return hit / g.size;
}

export function isHit(goldenText: string, text: string, tau: number = HIT_THRESHOLD): boolean {
  return containment(goldenText, text) >= tau;
}

// 对一组（按排名有序的）检索结果，逐个判定是否命中该 golden，返回与输入等长的布尔数组。
export function judgeRelevance(goldenText: string, chunkTexts: string[], tau: number = HIT_THRESHOLD): boolean[] {
  return chunkTexts.map(t => isHit(goldenText, t, tau));
}

// 在整库 chunk 上统计有多少命中该 golden（= 相关集 |G| 的大小，recall 的分母来源）。
export function countRelevant(goldenText: string, allChunkTexts: string[], tau: number = HIT_THRESHOLD): number {
  return allChunkTexts.reduce((n, t) => n + (isHit(goldenText, t, tau) ? 1 : 0), 0);
}

/**
 * 答案截断判定（§5.2）：没有任何单个 chunk 覆盖到 τ，但相邻两 chunk 拼接后可以。
 * allChunkTexts 必须是「同一文档」按 index 升序排列的全部 chunk。
 * 直接拼接（无分隔符）对固定窗口策略可精确还原原文；递归策略因 trim 为近似。
 */
export function isAnswerSplit(goldenText: string, allChunkTexts: string[], tau: number = HIT_THRESHOLD): boolean {
  if (allChunkTexts.length < 2) return false;
  const maxSingle = Math.max(...allChunkTexts.map(t => containment(goldenText, t)));
  if (maxSingle >= tau) return false; // 有单块完整覆盖 → 未被切断
  for (let i = 0; i < allChunkTexts.length - 1; i++) {
    if (containment(goldenText, allChunkTexts[i] + allChunkTexts[i + 1]) >= tau) return true;
  }
  return false;
}
