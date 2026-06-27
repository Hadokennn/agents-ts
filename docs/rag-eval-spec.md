# RAG 检索评测能力 — 详细规格（Spec v1）

> 目标读者：本项目维护者 / 实现该模块的 Claude。
> 状态：**待 review**。review 通过后按「§10 增量路径」实现。
> 一句话：给 RAG 管线装一个**离线实验台**，把「改一个参数 → 检索质量变化多少」量化成可对比、可回归、可进 CI 的数字。

---

## 1. 背景与问题定义

当前检索链路四段式，每段都有写死的魔法数字，且**没有任何东西在验证它们**：

| 环节 | 文件 | 写死的参数 |
|------|------|-----------|
| 切分 | `src/rag/chunker.ts:9-11` | `TARGET_TOKENS=256`、`CHARS_PER_TOKEN=1.6`、无 overlap |
| 召回融合 | `src/rag/search.ts:14-17` | `VECTOR=0.7 / KEYWORD=0.3`、`CANDIDATE_MULTIPLIER=4`、`MMR_LAMBDA=0.7` |
| 向量化 | `src/rag/embedder.ts:1` | `DIMS=128` |

**直接诉求**：验证「递归段落分块」是不是好策略，对比其它切法谁更优。
**根因**：缺少「ground truth + 指标 + 矩阵跑批」三件套构成的反馈闭环，导致任何参数都无法被证伪。

---

## 2. 范围

### In scope（本 spec 覆盖）
- **Layer 1 检索评测**（component-level，确定性、便宜、不需要 LLM）。
- 可插拔的分块策略接口 + 至少 3 种策略实现。
- span 锚定的黄金集（手写 + LLM 合成 bootstrap）。
- IR 指标纯函数 + 分块专属诊断指标。
- 矩阵 runner + 排行榜报告 + `/eval` 命令。
- 测试基建（项目目前 `npm test` 是占位 echo，借此立起来）。

### Out of scope（本期不做，留接口）
- **Layer 2 生成评测**（LLM-as-judge 忠实度/相关性，RAGAS 系）——P4 可选，单列。
- 在线 A/B、生产 query 日志挖掘。
- embedding 模型本身的微调/对比（只把 embedder 当黑盒记录）。

### 非目标（明确不追求）
- 不追求与 RAGAS/TruLens 的指标数值可比，只追求**项目内部纵向可比**（同一黄金集、同一 embedder 下，策略 A vs 策略 B）。

---

## 3. 架构总览

```
src/rag/
  chunker.ts              重构：保留 chunkDocument 兼容入口，内部委托给策略
  strategies/
    types.ts              ChunkStrategy 接口 + ChunkWithSpan 类型
    recursive-paragraph.ts 现有逻辑迁移（默认策略，行为不变）
    fixed-size.ts         固定大小切分
    fixed-overlap.ts      固定大小 + overlap
    (semantic.ts)         语义分块，P3+ 可选
src/eval/
  dataset.ts              黄金集类型 + 加载/校验
  synth.ts                LLM 合成问答对（bootstrap 数据集）
  matching.ts             span ↔ chunk 命中判定（核心算法，单独成文件）
  metrics.ts              纯函数：recall/precision/mrr/ndcg/hitrate/截断率
  runner.ts               矩阵跑批：策略×参数×query → 指标
  report.ts               排行榜渲染（表格 + JSON，可 diff）
  types.ts                EvalConfig / EvalResult / RunRecord
src/commands/eval.ts      /eval 命令，复用 CommandHandler 机制
eval-data/
  golden.jsonl            黄金集（手写 + 合成，git 跟踪）
  corpus/                 评测用文档（小而固定）
test/
  metrics.test.ts         指标纯函数单测
  matching.test.ts        命中判定单测
```

**数据流**（单次 eval run）：

```
corpus 文档 ──┬─► [ChunkStrategy A] ─► chunks ─► embed ─► VectorStore(内存) ─┐
              │                                                              ├─► 对每条 golden query:
              └─► [ChunkStrategy B] ─► chunks ─► embed ─► VectorStore(内存) ─┘    hybridSearch(topK)
                                                                                  → matching 判定命中
                                                                                  → metrics 累计
                                                                              ─► report 排行榜
```

关键：**每种策略起一个独立的内存 `VectorStore`**（复用 `src/rag/store.ts:8` 现成类），互不污染，也不碰 `knowledge.db`。

---

## 4. 核心接口：ChunkStrategy

### 4.1 类型（`src/rag/strategies/types.ts`）

```ts
// 现有 Chunk 增加原文定位信息，命中判定要用
export interface ChunkWithSpan extends Chunk {
  startOffset: number;  // 在「原文档」中的起始字符下标（含）
  endOffset: number;    // 结束字符下标（不含）
}

export interface ChunkStrategy {
  readonly name: string;                 // "recursive-paragraph" / "fixed-256-o32" ...
  readonly params: Record<string, number | string | boolean>;  // 写进报告，保证可复现
  chunk(source: string, text: string): ChunkWithSpan[];
}
```

**决策点 D1 — offset 怎么算（重要）**：
当前 `chunkDocument` 对段落做了 `trim()` 和 `'\n\n'` 重拼（`chunker.ts:52`），精确还原原文 offset 很脆。两个方案：

- **方案 A（推荐先用）— 文本包含匹配，不依赖精确 offset**：命中判定基于「黄金答案文本 与 chunk 文本」的 token 重叠率，`startOffset/endOffset` 用 `text.indexOf(firstSentence)` 做近似即可，仅用于「截断率」诊断。优点：对 trim/重拼鲁棒，落地快。缺点：同一句话在文档里重复出现时定位可能偏。
- **方案 B（后续精度升级）— 严格 offset**：切分时全程不丢字符位置，chunk 携带精确 `[start,end)`，命中判定用区间覆盖率。优点：严格。缺点：要把现有 chunker 的 trim/rejoin 逻辑改成「记录位置而非修改字符串」，工作量大。

> **本 spec 默认走方案 A**；§5 的命中判定以 token 重叠为主、offset 为辅。方案 B 列为 P3+ 精度升级。请 review 时确认这个取舍。

### 4.2 重构约束

- `chunkDocument(source, text)` **保持现有签名与行为**（`rag-tools.ts:24`、`index.ts:320` 在用），内部改为 `new RecursiveParagraphStrategy(defaults).chunk(...)` 并 `map` 掉 span 字段返回，保证**零行为变更**。
- `RecursiveParagraphStrategy` 的默认参数 = 现有常量（`TARGET_TOKENS=256` 等），迁移即基线。

### 4.3 首批策略实现

| 策略 | 参数 | 说明 |
|------|------|------|
| `recursive-paragraph` | `targetTokens` | 现有逻辑，基线 |
| `fixed-size` | `targetTokens` | 纯按字符窗口硬切，不看段落 — 故意做的「坏基线」，验证评测能否把它排到下面 |
| `fixed-overlap` | `targetTokens`, `overlapTokens` | 固定窗口 + 重叠，对照「overlap 能不能救回被切断的答案」 |

> 三种策略 + 参数扫描已足够回答用户原始问题。`semantic` 留到 P3+。

---

## 5. 命中判定（`src/eval/matching.ts`）— 整个评测的地基

**原则（§3 已述）**：ground truth 锚在「答案文本」，对任何切分策略不变。

### 5.1 黄金答案 → chunk 命中

```ts
// 复用 search.ts 里的 tokenize，保持一致
function containment(goldenText: string, chunkText: string): number {
  const g = new Set(tokenize(goldenText));
  const c = new Set(tokenize(chunkText));
  if (g.size === 0) return 0;
  let hit = 0; for (const t of g) if (c.has(t)) hit++;
  return hit / g.size;          // 黄金答案的 token 有多少被这个 chunk 覆盖
}
// chunk 命中该 golden ⟺ containment ≥ τ
const HIT_THRESHOLD = 0.6;      // τ，决策点 D2，可调
```

> **决策点 D2 — τ 阈值**：0.6 是起点。τ 太低 → 蹭到几个词就算命中，虚高；太高 → 被切断的答案永远算 miss。建议实现后用黄金集人工抽查校准一次再定。

### 5.2 截断率（answer-splitting rate）— chunking 的杀手指标

一个 golden 答案被「切断」⟺ **没有任何单个 chunk 能覆盖到 τ，但相邻 chunk 拼起来可以**。

```
被切断 = (max_c containment(golden, c) < τ)
         AND (containment(golden, concat(相邻chunk对)) ≥ τ)
截断率 = 被切断的 golden 数 / 总 golden 数
```

这个指标直接量化「坏切分把答案拦腰斩断」的失败模式，比 Recall 更早暴露问题。`fixed-overlap` 理论上应显著降低它——评测要能验证这一点。

---

## 6. 指标定义（`src/eval/metrics.ts`，全是纯函数）

记号：单条 query 检索回 top-k 结果 `R = [r₁..r_k]`（按分数降序），该 query 的相关 chunk 集合 `G`（由 §5.1 判定，`relevant(rᵢ) ∈ {0,1}`）。整个黄金集 `Q` 条 query 取宏平均（macro-average）。

| 指标 | 公式 | 含义 |
|------|------|------|
| **HitRate@k** | `1` 若 `∃ rᵢ relevant 否则 0`，对 Q 平均 | 至少捞回一个相关的比例（最宽松） |
| **Recall@k** | `\|{relevant rᵢ}\| / \|G\|` | 该捞的捞回了多少 |
| **Precision@k** | `\|{relevant rᵢ}\| / k` | 捞回的有多少是对的（噪声反指标） |
| **MRR@k** | `1 / rank(第一个 relevant)`，无则 0 | 第一个正确结果排多前 |
| **NDCG@k** | `DCG@k / IDCG@k`，`DCG=Σ relⱼ/log₂(j+1)` | 排序质量（位置加权），rel∈{0,1} |
| **截断率** | §5.2 | chunking 专属，越低越好 |
| **上下文效率** | `Σ 召回 token / Σ 相关 token` | 捞回一份有用信息要塞多少噪声，越低越好 |

辅助统计（每种策略输出一次，非 per-query）：chunk 总数、平均/方差 chunk token 数、embed 调用次数。

> 全部纯函数，输入 `(R, G)` 输出 number，**必须配 `test/metrics.test.ts` 单测**：构造已知 `R/G` 手算期望值断言（例：MRR 第一个就命中=1.0、第三个命中=0.333）。

---

## 7. Runner 与矩阵（`src/eval/runner.ts`）

### 7.1 配置

```ts
export interface EvalConfig {
  corpusDir: string;                 // eval-data/corpus
  datasetPath: string;               // eval-data/golden.jsonl
  strategies: ChunkStrategy[];       // 要对比的策略（含不同参数的同名策略）
  topK: number;                      // 默认 5，与线上 rag_search 默认一致
  embedder: 'mock' | 'dashscope';    // 见 §8 护栏
  embedderModel: string;             // 记录进报告
  seed?: number;                     // 预留，当前链路无随机
}
```

### 7.2 执行语义
1. 对每个 `strategy`：切分整个 corpus → `embed`（复用 `embedder.ts` 的 cache）→ 灌入新建内存 `VectorStore`。
2. 对每条 golden query：`hybridSearch(store, embedFn, query, topK)`（直接复用 `src/rag/search.ts`，**评测即测真实检索路径**，不另写一套）。
3. 用 §5/§6 判定 + 累计指标。
4. 汇总成 `RunRecord[]`（每个 strategy 一行）。

### 7.3 生产级属性（必须满足）
- **确定性可复现**：固定黄金集、固定 corpus、embed 走 cache；同输入两次跑结果完全一致。
- **便宜可 CI**：mock embedder 下应秒级完成（仅作冒烟，见 §8）。
- **可 diff**：报告同时输出 JSON，落到 `eval-data/results/<embedder>-latest.json`，改了某个魔法数字后 `git diff` 能看到每个指标涨跌。

> 注意现有 `hybridSearch` 的 BM25 在 JS 里对 `getAll()` 全量重算（`search.ts:114-118`），是 O(n²)。评测语料保持小（建议 ≤ 几百 chunk）即可，不在本期优化。

---

## 8. Mock embedder 护栏（⚠️ 不可省）

`mockEmbed`（`embedder.ts:64`）是字符哈希，**无语义**，分不清「狗/犬」。用它跑分块评测排名是垃圾。

强制规则：
- `embedder: 'mock'` → 报告头部打 **`⚠️ MOCK EMBEDDER — 结果仅供冒烟，不可用于策略决策`**，且 runner 拒绝写入 `*-latest.json`（只写 `*-smoke.json`）。
- `embedder: 'dashscope'` → 需 `DASHSCOPE_API_KEY`，结果才可用于真实结论。
- 报告里**永远记录 embedder + model**，杜绝「拿 mock 结果下结论」。

（对应 `~/.claude/rules/observe-before-implement.md`：策略决策必须基于真实 embedding，不基于「向量应该能区分语义」的假设。）

---

## 9. 黄金集（`eval-data/golden.jsonl`）

### 9.1 格式（每行一条）

```jsonc
{
  "id": "q001",
  "query": "递归段落分块的目标 token 数是多少？",
  "source": "corpus/rag-design.md",   // 答案所在文档
  "goldenAnswer": "目标 256 token，约等于 410 个字符",  // 用于 §5 token 重叠判定
  "tags": ["factoid"],               // factoid / multi-hop / long-answer，分层看指标
  "origin": "handwritten"            // handwritten | synth，合成的要标，便于抽检
}
```

> 用 `goldenAnswer` 文本而非 chunk id，正是 §3 的「锚文本不锚 chunk」设计。

### 9.2 LLM 合成 bootstrap（`src/eval/synth.ts`）
- 流程：遍历 corpus chunk → 喂给模型「针对这段文字出 1~2 个该段能回答的问题」→ 该 chunk 文本即 `goldenAnswer` → 标 `origin: "synth"`。
- 这是 RAGAS 等的通行 bootstrap 法，便宜可扩。
- **质量门**：合成集要人工抽检 ≥ 20%，剔除「问题脱离该段也能答」的泛问。合成 query 与手写 query 在报告里**分桶看指标**，避免合成噪声掩盖真实差异。

---

## 10. 增量路径与验收标准

| 阶段 | 产出 | 验收标准（Done 的定义） |
|------|------|------------------------|
| **P0** | ChunkStrategy 接口 + 现有逻辑迁移为 `recursive-paragraph` | `chunkDocument` 行为零变更（对若干样例文档，迁移前后输出逐字相等）；`fixed-size`/`fixed-overlap` 可实例化 |
| **P1** | 黄金集：≥10 条手写 + synth 扩充 + `dataset.ts` 校验 | `golden.jsonl` 加载校验通过；手写覆盖 factoid/long-answer 两类 |
| **P2** | `matching.ts` + `metrics.ts` + 单测 | `npm test` 真正跑起来；指标纯函数单测全绿（含手算断言）；命中判定单测覆盖「完整命中/截断/miss」 |
| **P3** | `runner.ts` + `report.ts` + `/eval` 命令 | 一条命令跑出 ≥3 策略排行榜；真实 embedder 下 `fixed-size`（坏基线）指标明显垫底、`fixed-overlap` 截断率显著低于 `fixed-size` —— **评测能正确分辨好坏即闭环成立** |
| **P4**（可选） | Layer 2：LLM-judge 忠实度/答案相关性 | 单列，本期不阻塞 |

---

## 11. 待 review 的决策点（请逐条确认）

- **D1**：命中判定走「方案 A 文本 token 重叠」（推荐，落地快、对 trim 鲁棒），offset 严格匹配留 P3+。是否同意？
- **D2**：命中阈值 τ 起点 `0.6`，实现后人工校准。是否接受先用 0.6？
- **D3**：首批策略 = `recursive-paragraph` / `fixed-size` / `fixed-overlap`，暂不做 `semantic`。够不够？
- **D4**：黄金集体量 P1 先做「≥10 手写 + synth 扩充」，是否够起步？corpus 用什么文档（建议直接用本项目 `docs/` + `src/rag/*` 的说明文字，自产自销）？
- **D5**：`/eval` 输出形态——终端表格 + `eval-data/results/*.json`。是否要再加 markdown 报告？

---

## 附：与「生产级 RAG 评测」的对应关系

| 生产实践 | 本 spec 对应 |
|----------|-------------|
| 检索 vs 生成两层分离 | §2 范围划分，本期只做 Layer 1 |
| Golden set / ground truth | §9，span/文本锚定 |
| LLM 合成评测集 | §9.2 synth |
| IR 指标（Recall/NDCG/MRR） | §6 |
| 离线实验台 / 参数扫描 | §7 矩阵 runner |
| 可复现 + 可回归 + 进 CI | §7.3 |
| 评测集与被测配置解耦 | §7.1 EvalConfig 只换 strategy，黄金集固定 |
