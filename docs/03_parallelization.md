# Ch.3 Parallelization 并行化

## 3.1 一句话定义

**Parallelization 是把多个 LLM call 并发执行——要么对独立子任务做 fan-out（Sectioning），要么对同一任务做多次采样取共识（Voting）——然后聚合结果。** [A1, P1, A2]

它有两条独立的设计理由：(1) **降延迟**（顺序变并发），(2) **提质量**（让每个 LLM call 专注单一关注点）。A2 还加了第三条，2024 年开始变成主流：(3) **隔离上下文**（每个 sub-agent 用干净的 context 探索深，只返回浓缩摘要）。

## 3.2 动机：为什么不能用更简单的方式

朴素做法是 "一个 prompt 干完所有事"。下面是个典型反例：

```text
请审查这段 code diff。同时检查：(1) 安全漏洞、(2) 代码风格、(3) 性能问题。
然后给出综合评分和修改建议。
```

这条 prompt 看起来合理，实际 A1 给出过明确判断：

> "LLMs generally perform better when each consideration is handled by a separate LLM call, allowing focused attention on each specific aspect." [A1]

具体表现：

- **Attention 稀释**：模型必须同时关注三个 dimension，每个 dimension 的深度都打折；
- **顺序偏置**：模型倾向于先回答 prompt 里提到的第一个关注点，后面的关注点输出质量逐降；
- **互相妥协**：模型可能为了"综合评分"调和不同 dimension 的发现，让单个 dimension 的红旗被掩盖；
- **延迟串行**：即使你不在乎质量只想要"列出三类问题"，单 prompt 也比并发慢——你等的是 max(t1+t2+t3) 而不是 max(t1, t2, t3)。

Parallelization 同时解决这四个问题。

A2 的第三条理由更深，它把 parallelization 从"workflow 模式"升级到"context engineering 武器"：

> "Each subagent might explore extensively, using tens of thousands of tokens or more, but returns only a condensed, distilled summary of its work (often 1,000-2,000 tokens)." [A2]

这意味着 main agent 的 context 永远只装 N 个 1-2k 的浓缩摘要，而每个 sub-agent 可以独立烧 10k+ 的 token 做深度探索。**主上下文保持干净，深度探索发生在隔离环境里。** 这是为什么 Anthropic 多 agent research system 在复杂任务上能"substantial improvement over single-agent systems"——不是模型变强了，是 context 管理变好了。

## 3.3 核心结构

Parallelization 有两个根本不同的范式，外加一个 A2 提出的进化版：

```
=== Sectioning（分而治之） ===
                                +--------+
                            +-->| Worker1| (security)
                            |   +--------+
   input  ---> dispatch ----+-->| Worker2| (style)     ---> aggregate ---> output
                            |   +--------+
                            +-->| Worker3| (perf)
                                +--------+

=== Voting（共识/多样性） ===
                            +-->| Judge1 |
                            |   +--------+
   input  ---> fan-out  ----+-->| Judge2 |  ---> majority/avg ---> output
                            |   +--------+
                            +-->| Judge3 |
                                +--------+

=== Sub-agent（A2 的 context 隔离）===
                                +--------+   10k tokens explored
                            +-->|SubAgt1 |   返回 1.5k summary
                            |   +--------+
   main   ---> spawn    ----+-->|SubAgt2 |   ---> main reads summaries ---> output
   agent                    |   +--------+
                            +-->|SubAgt3 |
                                +--------+
```

三者的差异：

| | Sectioning | Voting | Sub-agent |
|---|---|---|---|
| 输入分配 | 不同 worker 看任务的不同切面 | 所有 worker 看完全相同的输入 | 不同 sub-agent 拿到不同子目标 |
| 目的 | 分而治之 + 关注点分离 | 共识 / 减少 hallucination / 安全双校 | Context isolation + 深度探索 |
| Worker 数量 | 通常固定（按 dimension 数） | 通常 3-7 个奇数（便于多数票） | 动态（main agent 决定） |
| 聚合策略 | 合并不同 dimension | majority vote / averaging | main agent reads summaries |
| Workflow / Agent | Workflow | Workflow | 偏 Agent |

工程上三者经常组合。例如 Anthropic 的 multi-agent research 同时用到 sub-agent（每个子主题一个 agent）和 voting（同一子主题跑 3 次取一致）。

## 3.4 最小示例

```python
import asyncio
from collections import Counter

# === Sectioning ===
async def review_code(diff: str) -> dict:
    sec, sty, perf = await asyncio.gather(
        llm_async(f"Review for security vulnerabilities only:\n{diff}"),
        llm_async(f"Review for code style only:\n{diff}"),
        llm_async(f"Review for performance issues only:\n{diff}"),
    )
    return {"security": sec, "style": sty, "performance": perf}

# === Voting (Self-Consistency style) ===
async def safety_check(content: str, votes: int = 5) -> bool:
    judgments = await asyncio.gather(*[
        llm_async(f"Is this content safe? Reply yes or no only.\n{content}",
                  model="claude-haiku-4")
        for _ in range(votes)
    ])
    counter = Counter(j.strip().lower() for j in judgments)
    return counter.most_common(1)[0][0] == "yes"

# === Sub-agent fan-out (simplified) ===
async def research_topic(topic: str) -> str:
    subtopics = decompose(topic)                # main agent plans
    summaries = await asyncio.gather(*[
        sub_agent_run(t, max_tokens=20_000)     # 每个 sub-agent 独立 context
        for t in subtopics
    ])                                          # 每个返回 ~1-2k summary
    return synthesize(summaries)                # main 只读 summary 合成
```

注意几个共同要点：

- **`asyncio.gather`** 是 Python 异步并发的标准做法；同步代码可以用 `concurrent.futures.ThreadPoolExecutor`。
- **每个 worker 的 prompt 都聚焦单一关注点**——不要写"check security AND style AND perf"，那等于把 sectioning 又退化成单 prompt。
- **Voting 必须输出受限集合**（yes/no、A/B/C）才方便统计。让 voter 输出自由文本会让 aggregation 巨复杂。
- **Sub-agent 必须有 token cap**——不然某个失控 sub-agent 烧穿预算，并发优势变成并发烧钱。

## 3.5 框架对比

四大框架对 Parallelization 的支持差异极大（D digest Pattern 3）：

| 框架 | Primitive | 怎么搭 | Idiomatic 片段 |
|---|---|---|---|
| **LangGraph** | `Send` API + fan-in via `Annotated[list, operator.add]` | conditional edge 返回 `Send("worker_node", payload)` 列表，runtime 并发 spawn workers；每个 worker 写到 list-typed state key 实现聚合 | `return [Send("llm_call", {"section": s}) for s in state["sections"]]` + worker 返回 `{"completed_sections": [section.content]}` |
| **AutoGen v0.4+** | Pub-sub fan-out (Mixture of Agents) | Orchestrator 把同一任务 publish 到 N 个 worker topic；worker 并发处理；orchestrator 收齐结果再 dispatch 下一层 | "Dispatch to workers at layer 0" → N 个 `Worker` 同时跑 → 回到 orchestrator |
| **CrewAI** | **没有 first-class primitive** | workaround: 用 `asyncio` 跑多个 `crew.kickoff_async()`；或把并行工作建模成 `Process.sequential` 里的独立 task（executor 可能并行 tool 调用但 task 仍按 list 顺序跑） | `Process` enum 只定义 `sequential` 和 `hierarchical` |
| **Google ADK** | `ParallelAgent` | 用 `ParallelAgent` 包住 N 个 sub-agent；它们在独立线程跑但共享 `session.state`；每个必须写到不同 `output_key` 避免竞态；下游用 `SequentialAgent` 接 aggregator | `parallel_reviews = ParallelAgent(name="CodeReviewSwarm", sub_agents=[security_scanner, style_checker, complexity_analyzer])` |

四种心智模型差异：

- **LangGraph 的 `Send`** 是**动态 fan-out**——worker 数量在 planner 跑完才知道。这是写"未知数量子任务"场景的 sweet spot。
- **AutoGen** 通过 async runtime 隐式并行——你只是 publish 多次，runtime 自然并发。最像 actor model。
- **CrewAI** 在 v0.x 没有 process-level 并行原语——并行处理需要用户写 Python 自己做。这是 CrewAI 在 long-running 复杂场景下的 weakness。
- **ADK 的 `ParallelAgent`** 是**静态声明**——sub_agents 必须 up front 列出。简单但不能动态扩展。

工程经验：

- **未知数量并发** → LangGraph `Send`；
- **已知数量且想要 typed container** → ADK `ParallelAgent`；
- **分布式部署 / 跨机** → AutoGen pub-sub；
- **如果你已经在用 CrewAI** → 用 `asyncio.gather(*[crew.kickoff_async(...) for ...])` 在 crew 之外包一层并发。

### 3.5.1 三种范式的选用决策

下面是经验性的判断顺序，从用户需求倒推合适的 parallelization 范式：

**问题 1：你的并发动机是什么？**

| 动机 | 范式 | 备注 |
|---|---|---|
| 降低 wall-clock 延迟 | Sectioning（独立 dimension）| 必须真独立，否则同步开销大于收益 |
| 减少单个判断的随机性 | Voting | 必须用 3-5 个独立采样，odd number 便于多数票 |
| 同时探索多个候选方案 | Best-of-N | 需要独立 scorer，否则退化为随机 |
| 让 main agent context 不爆 | Sub-agent | 每个 sub-agent 独立 token 预算 |
| 跨多个数据源汇集信息 | Sub-agent + Sectioning 组合 | 每个 sub-agent 看一个 source，main agent 合成 |

**问题 2：你的子任务是否完全独立？**

| 独立度 | 推荐 |
|---|---|
| 完全独立 | 纯 Sectioning |
| 同输入不同视角 | Sectioning 但用 MECE 设计避免重叠 |
| 弱依赖（共享上下文但不互相消费） | 先 chain 出共享 outline，再 fan-out |
| 强依赖 | 不要 parallelize，老老实实 chain 或上 Planning |

**问题 3：你的聚合策略是什么？**

| 聚合 | 适用 |
|---|---|
| 字段合并（每个 worker 写不同 field） | Sectioning + structured output |
| 多数票 | Voting，且 worker 输出必须是受限集合 |
| 加权平均 | Voting，且每个 worker 的权重有业务依据 |
| LLM 合成 | 任何范式，但成本+1 次 LLM call |
| 保留分歧呈现给下游 | 任何范式，把决策推迟到下游 |

把这三个问题答完，范式选型基本就清楚了。

## 3.6 何时不该用

- **任务真的有顺序依赖。** worker B 需要 worker A 的输出才能开始——这是 Chaining 不是 Parallelization。强行 parallel 只会浪费一次 A 的调用。
- **Worker 数量太少（1-2 个）。** 并发的固定开销（线程调度、聚合代码、错误处理）有时大于 2 次 LLM call 串行的延迟。经验阈值：≥ 3 个 worker 才值得并发。
- **聚合逻辑比 LLM 调用本身还贵。** 如果你要聚合 5 个 worker 输出，且聚合需要再调一次 LLM 做合成——总成本 = 5 + 1 = 6 次 call。如果可以用更简单的方式（比如 chain + summarize）做到 3 次 call 同样效果，未必划算。
- **Worker 之间需要协调。** "worker A 看到 worker B 的中间结果再决定怎么做"——这种情况你需要的是 Multi-Agent Collaboration（Ch.7），不是 Parallelization。
- **Rate limit 严苛。** 如果你的 API 配额一秒只能 5 个并发请求，spawn 20 个 worker 会被 throttle，反而总耗时更长。详见坑 2。

## 3.7 常见坑（按危害排序）

### 坑 1：任务其实不独立 → worker 之间需要协调

经典反模式：把"写一篇文章"拆成"写第一段、写第二段、写第三段"并发执行。各段写出来风格不连贯、内容互相重复或冲突，因为各 worker 不知道彼此在写什么。

**判断标准：** 如果 worker B 的合理输出依赖 worker A 的具体内容（不只是"独立做某事"），那就不是 sectioning。重新规划：

- 真独立 → parallel 没问题；
- 弱依赖（B 需要 A 的方向但不需要 A 的具体输出）→ 先 chain 一个 "planning" step 出共享 outline，再 parallel 写各段；
- 强依赖 → 老老实实 chain。

### 坑 2：没考虑 rate limit → 并发 50 个直接 throttle

第一次跑成功，扩大并发到 50，全部 429 Too Many Requests。

**对策：**

- 用 `asyncio.Semaphore` 限制最大并发数：
  ```python
  sem = asyncio.Semaphore(8)   # 同时最多 8 个 in-flight
  async def bounded(coro):
      async with sem:
          return await coro
  results = await asyncio.gather(*[bounded(llm_async(...)) for ...])
  ```
- 配置 client 端的 retry-with-backoff（Anthropic / OpenAI SDK 都内置 `max_retries`）；
- 知道你 plan 的 RPM 和 TPM 配额，提前算并发上限；
- 长期方案：用 batch API（OpenAI Batch、Anthropic Message Batches）跑非实时 workload，单价便宜一半。

### 坑 3：Aggregator 设计糟糕 → 冲突时不知道用哪个

3 个 worker 对同一段 code 给出 3 个截然不同的评分（80 / 60 / 95）。Aggregator 简单求平均得 78——但 60 那个可能是因为发现了真 bug，平均掉了关键信号。

**对策（按复杂度递增）：**

- **保守聚合**：min（对安全/质量类问题取最严格判断）；
- **加权聚合**：给不同 worker 不同权重（如 security worker 的判断比 style worker 优先）；
- **LLM aggregator**：把 N 个 worker 输出喂给另一个 LLM 让它综合判断——成本最高但能解释 disagree；
- **保留分歧**：直接把 N 个判断输出给用户/下游，不强行合一致。Generative Agents [R9] 就是这么做的。

绝对不要无脑取平均或第一个——这两种都隐含"所有 worker 等价"的假设，通常错。

### 坑 4：没用 structured output → 聚合代码巨复杂

如果每个 worker 输出自由文本，aggregator 要先 parse、再 normalize、再合并——parse 失败率高，aggregator 逻辑膨胀到几百行。

**对策：** 每个 worker 用 schema 强约束输出。例如 code review worker 强制返回：

```python
class ReviewItem(BaseModel):
    severity: Literal["low", "medium", "high", "critical"]
    category: Literal["security", "style", "performance", "correctness"]
    line_range: tuple[int, int]
    message: str
    suggested_fix: Optional[str]
```

Aggregator 就只是 `sorted(all_items, key=lambda x: SEVERITY_ORDER[x.severity], reverse=True)`，10 行搞定。

### 坑 5：把 Parallelization 用成 "反复采样取最好" → 不是 Voting 而是 Best-of-N

容易混淆的两个范式：

- **Voting (Self-Consistency)**：N 个采样取**多数票**——目的是减少 hallucination 提升一致性；
- **Best-of-N**：N 个采样取**最好那个**——需要一个 verifier / scorer 来评分；

后者需要一个独立的 evaluator（往往是另一个 LLM 或外部 verifier 如 unit test），本质更接近 Ch.4 Reflection 的退化情形。如果你想要的是"采几次取最好"，请确认你有可信的 verifier；如果没有，就退回 Voting 用多数票。

### 坑 6：Sub-agent 的 summary 设计不当 → 信息瓶颈

A2 的 sub-agent 模式假设每个 sub-agent 返回 1-2k token 的 summary。如果 summary 太短，main agent 拿不到决策需要的细节；如果太长，context 隔离的好处就消失了。

**对策：**

- 给每个 sub-agent 明确的 "return format" 约束（schema），不是自由摘要；
- summary 必须包含 "confidence / sources / open questions" 三个字段——让 main agent 知道这个 sub-agent 有多确定、依据哪里、还有什么没解决；
- 如果 main agent 经常需要回头问 sub-agent 细节 → 你的 summary 不够，重新设计 schema；如果经常忽视 sub-agent 输出 → 你的 sub-agent 设计错了，main agent 自己做更便宜。

### 坑 7：并发错误处理粗暴 → 一个 worker 失败拖垮整个聚合

`asyncio.gather` 默认 fail-fast：任何一个抛异常，整个 gather 抛异常。如果你的 aggregator 是"5 选 3"那种容错型，losing 1 worker 不应该让整个任务失败。

**对策：** 用 `return_exceptions=True`：

```python
results = await asyncio.gather(*coros, return_exceptions=True)
ok = [r for r in results if not isinstance(r, Exception)]
errors = [r for r in results if isinstance(r, Exception)]
if len(ok) < min_quorum:
    raise InsufficientWorkersError(errors)
return aggregate(ok)
```

明确定义 quorum——至少几个 worker 成功才算整体成功——并在 telemetry 里记录失败的 worker 类型，便于后期优化。

### 坑 8：忘了对每个 worker 设 timeout → 慢 worker 拖死整批

`asyncio.gather` 会等所有 worker 完成（或第一个失败）。如果某个 worker 因为 API 抖动 hang 30 秒——整批响应延迟跟着 hang 30 秒。

**对策：** 给每个 worker 包一层 `asyncio.wait_for`：

```python
async def bounded_worker(coro, timeout=10):
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return TimeoutSentinel()

results = await asyncio.gather(*[bounded_worker(w) for w in workers],
                               return_exceptions=True)
```

聚合阶段把 `TimeoutSentinel` 当作 missing data 处理（quorum 判断、降级响应等）。

### 坑 9：Voting 时让 LLM 在同一次对话里给多个判断 → 不是真 voting

反模式：

```python
# bad: 一次 prompt 让模型给 5 个判断
out = llm("Judge this safety 5 times independently: ...")
```

模型在同一 context 里给"5 个独立判断"实际上是高度相关的——它会倾向于第一个判断一致，后面 4 个跟风。Self-Consistency 的统计学基础崩了。

**对策：** 真的开 5 个独立 API call，最好用不同 temperature（0.7-1.0）让采样多样化。如果你只能开 1 个 call，请改用 chain-of-verification 而不是叫它 voting。

### 坑 10：Sectioning 的关注点之间有重叠 → 重复工作 + 计算冲突

把 code review 切成 "security / style / performance / correctness" 四个维度——但 "correctness" 经常和 "security" 重叠（一个 buffer overflow 既是 correctness bug 也是 security 漏洞）。两个 worker 都报告同一个 issue，aggregator 必须 dedupe，否则用户看到双重报告。

**对策：**

- 设计 dimension 时使用 MECE（Mutually Exclusive, Collectively Exhaustive）原则；
- 不能 MECE 时显式定义重叠裁决规则（"重叠 issue 归 security worker"）；
- Aggregator 必须 dedupe（按 line_range + message similarity）；
- 用 schema 字段（如 `primary_category`）让 worker 自己声明归属。

## 3.8 与其他模式的关系

- **Parallelization 是 Multi-Agent 的核心机制**：ADK 的 `ParallelAgent`、AutoGen 的 Mixture of Agents、Anthropic 的 sub-agent research system 都是 Parallelization 在 agent 层的具体化。多 agent 拓扑里的 "fan-out" 阶段本质就是 Parallelization。
- **Parallelization 是 Reflection 的反面**：Reflection（Ch.4）是 generator-critic 顺序迭代，Parallelization 是同时并发；前者深、后者宽。两者经常组合：先 parallel 出 N 个候选，再 reflect 出最好的（Best-of-N + critique）。
- **Parallelization 与 Sub-agent context isolation**（A2 的关键贡献）：让 Parallelization 与 Memory Management（Ch.8）、Resource Optimization（Ch.16）紧密相连——并发不只是降延迟，更是 context 管理武器。
- **Voting 是 Reasoning Techniques 的 Self-Consistency**（Ch.17 的特例）：本质是同一 prompt 跑多次取多数票。
- **Parallelization 受益于 structured output**（坑 4）：每个 worker 用 schema 输出让 aggregation 变 trivial。
- **Parallelization 是 Routing 的下游进化**：Multi-label routing（一个输入对应多个意图）的自然延续就是 parallel 分发到多个 handler。
- **Sectioning 与 Planning 的边界**：Sectioning 的子任务是固定的（你预先知道要并发哪些），Planning 的子任务是动态的（LLM 跑时才决定）。当你的 sectioning"经常需要根据输入动态决定分多少 worker"时，升级到 Planning（Ch.6）。

## 3.8.5 一个完整的 multi-source 研究系统例子

这个例子直接借鉴 Anthropic [A2] 的 multi-agent research system。业务需求：用户提一个开放问题（"对比 LangGraph 和 ADK 哪个更适合做 multi-agent system"），系统给出综合答案。

朴素做法是单 agent + web search tool 自己跑——但这种问题需要同时查多个子主题、读大量 doc，单 agent 的 context 会爆。

并行版本：

```python
async def multi_agent_research(question: str) -> str:
    # 1. Main agent 做 planning：把大问题切成子问题
    subq = await llm_async(
        PLAN_PROMPT.format(q=question),
        schema=SubQuestions,
        model="claude-sonnet-4",
    )
    # subq.questions = ["LangGraph 的架构是什么", "ADK 的架构是什么",
    #                    "两者在 multi-agent 上的 primitives 差异", ...]

    # 2. Sub-agent fan-out：每个子问题一个 sub-agent，独立 context
    sem = asyncio.Semaphore(4)   # 控制并发不超过 4，避免 rate limit

    async def run_sub(sq: str) -> Summary:
        async with sem:
            try:
                return await asyncio.wait_for(
                    sub_agent_research(sq, max_tokens_budget=30_000),
                    timeout=120,
                )
            except (asyncio.TimeoutError, Exception) as e:
                return Summary(question=sq, error=str(e), confidence=0)

    summaries = await asyncio.gather(*[run_sub(q) for q in subq.questions])

    # 3. Main agent 合成：只读 summaries（每个 ~1.5k token），不读 sub-agent 的完整探索
    valid = [s for s in summaries if s.confidence > 0.3]
    if len(valid) < len(summaries) * 0.5:
        return "信息不足，请缩小问题范围"
    answer = await llm_async(
        SYNTHESIZE_PROMPT.format(q=question, summaries=valid),
        model="claude-sonnet-4",
    )
    return answer
```

每个 sub-agent 内部又是一个完整的 tool-using agent（Ch.5）——可能调 web search、读 doc、跑 ReAct loop。但 main agent 只看到每个 sub-agent 最后吐出的 1-2k token summary。

数字感觉：

- 不用 parallelization：单 agent 跑这种问题 60-120 秒，context 经常爆 200k；
- 用 parallelization + sub-agent：总 wall clock 30-50 秒（受 max sub-agent 时间限制），main context 永远 < 30k，token 总消耗增加 2-3 倍但延迟降一半且准确率明显提升。

A2 报告这种架构在复杂研究任务上"substantial improvement over single-agent systems"——上面这套就是把那句话翻译成代码的样子。

## 3.9 一句话总结

**Parallelization 有三个独立的设计回报：降延迟、提质量、隔离 context。前两个是 Workflow 层的 mechanical optimization，第三个是 Agent 层的 context engineering——后者在 2024 年开始变成 multi-agent 系统的核心架构理由。**

当你的系统单 prompt 越写越长、各关注点互相打架时，先想 Sectioning；当你的系统对单次回答的稳定性敏感时，先想 Voting；当你的 main agent context 经常爆掉时，先想 Sub-agent fan-out。
