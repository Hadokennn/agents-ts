# Ch.1 Prompt Chaining 提示链

## 1.1 一句话定义

**Prompt Chaining 是把一个复杂任务拆成固定的、线性的多个 LLM call，前一个 call 的输出喂给后一个 call 的输入，可选在中间插入 programmatic gate 做校验。** [A1, P1]

控制流完全由代码决定，LLM 只是被 orchestrate 的纯函数组件——这是它属于 Part I "工作流模式" 的根本原因。

## 1.2 动机：为什么不能用更简单的方式

如果你只把它理解成"调几次 LLM 串起来"，会觉得这章没什么内容。它的真正价值藏在一个反直觉的 tradeoff 里：

> "The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task." [A1]

直觉上，把多个 step 塞进一个 prompt（"先写大纲，再写正文，最后翻译"）应该更省 token、更省 latency。实际上：

- **单 prompt 多任务**会让模型在不同子任务上互相抢 attention，每个子任务都做得平庸；
- **单 prompt 多任务**很难做中间结果的校验——一旦大纲跑偏，整篇文章一起跑偏；
- **单 prompt 多任务**让 debug 变成黑盒——你拿到的只有最终错误的输出，不知道是哪一步开始崩的。

而 chaining 给了你三个东西：

1. **每步 prompt 都更短更聚焦**，单步准确率提升；
2. **每两步之间可以插 gate**——code-level 校验、schema 校验、甚至再调一次 LLM 做 QA；
3. **失败可以精确定位到具体某步**，重试也只重试那一步，不需要重跑整条链。

代价是：latency 增加（多次 round trip）、token 增加（每步都要带前一步的输出）、累积误差（见 1.7）。这是个明确的 trade。

P1 的判断与 A1 一致：

> "The output of one LLM call sequentially feeds into the input of the next LLM call." [P1]

Schmid 还强调一个 A1 没有展开的点：每步之间的 I/O contract 应该是**结构化的**（JSON schema），而不是自然语言。理由见 1.7 第 3 条坑。

### 1.2.1 一个反直觉的现象：链可以让"弱模型 + 多步"打败"强模型 + 单步"

A1 没有把这点写成公式，但工程上反复观察到：在某些任务上，用 GPT-4o-mini 跑 3 步 chain 的端到端质量超过 GPT-4o 跑 1 步——而总成本反而更低。原因在于：

- 强模型在长 prompt 上**也有 attention 稀释问题**——只是程度比弱模型轻；
- 强模型在"一次性输出长结构化内容"时容易**漏字段、跳步骤**；
- chain 把任务切小后，**模型能力的边际收益快速递减**——每步都简单到 mini 也能做对。

这是为什么"先试 chain 再考虑升级模型"在生产中常常比反过来更经济。每次想加预算时，先问：能不能多切一步？

## 1.3 核心结构

输入：原始任务（user query / 文档 / topic）。
输出：经过 N 步处理后的最终结果。
控制流：固定线性 + 可选 gate。

```
                  +----+        +----+        +----+        +----+
   input  ---->   | L1 |  ----> | L2 |  ----> | L3 |  ----> | L4 |  ----> output
                  +----+        +----+        +----+        +----+
                                  |
                                  v
                               +------+   fail   +-------+
                               | gate | -------> | retry |
                               +------+          +-------+
                                  |
                                pass
                                  v
                             (continue)
```

四个关键元素：

- **L1...Ln**：每个是一次 LLM call，使用独立的 system prompt 和 schema。
- **gate**：programmatic check（不是 LLM），判断中间结果是否满足约束（如长度、schema、关键字段非空、与前一步一致性）。
- **retry**：失败时的重试策略（常见：重新调当前 step 一次、用 fallback prompt 重调、降级到人工）。
- **shared state**：在 Python 里通常是一个 dict 或 Pydantic model，每步读上一步写的字段、写自己产出的字段。

输入/输出契约用 JSON schema 是 P1 的强建议。每步定义 input schema 和 output schema，使用 OpenAI / Anthropic 的 structured output 功能强约束 LLM 输出格式。

## 1.4 最小示例

下面这段 Python 伪代码展示一个常见的"主题 → 大纲 → 正文 → 翻译"链，带一个 gate。框架无关，10-25 行：

```python
from pydantic import BaseModel

class Outline(BaseModel):
    title: str
    sections: list[str]

class Article(BaseModel):
    title: str
    body: str

def outline_valid(outline: Outline) -> bool:
    return 3 <= len(outline.sections) <= 8 and len(outline.title) < 80

def write_localized_article(topic: str, target_lang: str) -> str:
    # Step 1: 生成大纲
    outline = llm(f"Write an outline for: {topic}", schema=Outline)

    # Gate: 大纲必须 3-8 节、标题不超过 80 字
    if not outline_valid(outline):
        outline = llm(f"Fix this outline (need 3-8 sections, title<80 chars): {outline}",
                      schema=Outline)

    # Step 2: 根据大纲写正文
    draft = llm(f"Write article from outline:\n{outline.json()}", schema=Article)

    # Step 3: 翻译到目标语言
    translated = llm(f"Translate to {target_lang}, preserve markdown:\n{draft.body}")

    return translated
```

注意几个细节：

- **每步都有明确 schema**，不依赖 LLM 自由发挥；
- **gate 是 Python 代码不是 LLM**——LLM 不擅长 self-check 数量约束（见 Ch.4 self-evaluation blindness）；
- **gate 失败只重做这一步**，不是重做整条链；
- **prompt 只包含当前步需要的上下文**，不把 topic 一路带到最后一步——避免 context 漂移。

## 1.5 框架对比

四大框架对 Prompt Chaining 的实现 idiom 差异极大，下表从 D digest [F1-F4] 改写：

| 框架 | 主要 primitive | 怎么搭 | Idiomatic 片段 |
|---|---|---|---|
| **LangGraph** | `StateGraph` + `add_edge` | 把 nodes 用固定边连成序列；每个 node mutate 共享 state；后一个 node 读前一个写的字段 | `builder.add_edge(START, "step1"); builder.add_edge("step1", "step2"); builder.add_edge("step2", END)` |
| **AutoGen v0.4+** | Pub-sub topic chain | 每个 agent 用 `@type_subscription(topic_type=X)` 订阅自己的 topic；做完后向下一个 agent 的 topic 发 `publish_message` | `await self.publish_message(Message(response), topic_id=TopicId(writer_topic_type, source=self.id.key))` |
| **CrewAI** | `Process.sequential` | 定义 task list；crew 按 list 顺序执行；前一个 task 的输出通过 `task.context` 喂给下一个 | `crew = Crew(agents=my_agents, tasks=my_tasks, process=Process.sequential)` |
| **Google ADK** | `SequentialAgent` + `output_key` | 用 `SequentialAgent` 容器包住一串 sub-agent；每个 agent 用 `output_key` 把输出写到 `session.state` 的具名 slot；下游 agent 在 `instruction` 里用 `{slot_name}` 引用 | `pipeline = SequentialAgent(name="PDFPipeline", sub_agents=[parser, extractor, summarizer])` |

四种心智模型一句话点出区别：

- **LangGraph**：链是 graph 里的 edge，state 是显式的载体——适合 infra 工程师；
- **AutoGen**：没有 graph，agent 之间通过 async pub-sub 自然串起来——适合 actor model 思维；
- **CrewAI**：最 prescriptive，几乎不需要写代码，list 顺序就是执行顺序——适合非工程师参与；
- **ADK**：链生活在 typed container 里，state 是共享 whiteboard——适合喜欢"容器组合"风格的团队。

工程经验：如果你的链 < 5 步、不需要可视化、不需要持久化中间 state，**不用 framework，直接写 Python function 调用最干净**。A1 反复强调这一点：framework 引入的额外抽象层会遮蔽底层 prompt，调试反而困难。

### 1.5.1 不用 framework 的极简实现

最小可用的 chaining 不需要任何 framework，纯 `httpx` 或 SDK 调用即可：

```python
from anthropic import Anthropic
client = Anthropic()

def llm(prompt: str, model="claude-haiku-4", **kw) -> str:
    resp = client.messages.create(
        model=model, max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        **kw
    )
    return resp.content[0].text

def chain(topic: str) -> str:
    a = llm(f"Write a 5-bullet outline for: {topic}")
    b = llm(f"Expand each bullet into a paragraph:\n{a}")
    c = llm(f"Add a title and conclusion:\n{b}")
    return c
```

15 行代码，完整可跑。Anthropic [A1] 反复强调的"don't reach for framework first"在 chaining 这里最适用——chain 结构本身已经是最简单的 control flow，再加抽象只会增加 debug 成本。

什么时候才有理由上 framework？

- **多人协作 / 长期维护** → graph 可视化和 typed state 有沟通价值；
- **需要 checkpoint / resume**（用户离线后续接） → LangGraph 的 checkpointer 帮你省工；
- **chain 与 routing/parallelization 复杂组合** → graph 能给你 birds-eye 视图；
- **跨语言部署** → 框架往往提供 server / SDK / observability 一站式集成。

否则 Python function 调用就够了。

## 1.6 何时不该用

- **任务真的就一步搞定。** 例如简单的"判断这段文本是否友善"——不要硬拆成 "理解 → 评估 → 输出"。
- **需要动态分支。** 如果 step 2 要不要执行依赖 step 1 的内容，应该用 **Routing**（Ch.2）；如果分支结构本身是模型决定的，应该用 **Planning**（Ch.6）。
- **步骤之间需要双向依赖。** Chaining 是单向的：L2 用 L1 的输出，但 L1 看不到 L2 的输出。如果你需要 L1 根据 L2 的结果回头修改自己——那是 **Reflection** loop（Ch.4），不是 chain。
- **延迟极敏感的实时场景。** 每多一步 LLM call 增加 0.5-3 秒。如果是 voice assistant 这种秒级延迟敏感的场景，宁可用 1 个 long prompt 换 1 次 round trip。
- **你只是想做并发，不需要顺序。** 用 **Parallelization**（Ch.3）。

## 1.7 常见坑（按危害排序）

### 坑 1：链太长 → 累积误差

每步 LLM call 都有失败率。即使每步 90% 准确：

| 链长度 | 端到端准确率 |
|---|---|
| 1 步 | 90% |
| 2 步 | 81% |
| 3 步 | 72.9% |
| 5 步 | **59.0%** |
| 8 步 | 43.0% |
| 10 步 | 34.9% |

5 步以下还能用，5 步以上必须配套以下机制之一：

- **每步 gate 校验**（程序化 / schema / LLM-as-judge），把单步准确率推到 95%+；
- **关键步骤 self-consistency**（同一 prompt 跑 3 次取多数票，见 Ch.3 Voting）；
- **退化为 Planning**（让 LLM 在中间根据情况决定是否继续）。

如果你发现自己在写一个 8 步以上的固定 chain，先暂停问自己：**这些步骤真的应该固定吗？还是其实应该让 LLM 来决定？** 如果是后者，你需要的是 Ch.6 而不是 Ch.1。

### 坑 2：错把分叉逻辑塞进 chaining → 应该用 Routing

新手最常见的反模式：

```python
# bad: chaining 假装做 routing
def handle(user_msg):
    is_billing = llm(f"Is this about billing? {user_msg}")
    if is_billing == "yes":
        ans = llm(f"Billing handler: {user_msg}")
    else:
        is_tech = llm(f"Is this technical? {user_msg}")
        if is_tech == "yes":
            ans = llm(f"Tech handler: {user_msg}")
        else:
            ans = llm(f"General handler: {user_msg}")
    return ans
```

这是 cascade 形式的 hard-coded routing，N 个意图就要 N-1 次 LLM call 做分类。正确做法是 Ch.2 Routing：一次 LLM call 输出 intent，dispatch 到对应 handler。

### 坑 3：中间步骤不做 schema validation → 一步错全错

如果 step 2 期望 step 1 输出一个 JSON list，但 step 1 偶尔输出散文——step 2 的 prompt 模板会注入失败、解析失败、或更糟，看似成功但生成胡言乱语。

**对策：** 每步用 `response_format`（OpenAI）或 `tool_use` 强约束（Anthropic）或 Pydantic schema 校验输出。失败时立刻重试当前 step，不要让坏数据流入下游。

### 坑 4：把 LLM 用作纯转换器 → 其实应该用确定性代码

经典案例：用 LLM 做 "把日期 '2025-05-24' 转换成 'May 24, 2025'"、"把驼峰命名转成下划线"、"把 markdown 表格转成 JSON"。

这些任务有确定性的程序解。用 LLM 会：
- 偶尔出错（"May 24th, 2025" vs "May 24, 2025"）；
- 慢 100-1000 倍；
- 贵 1000-10000 倍。

**对策：** 链里的每一步先问自己："这步必须用 LLM 吗？还是 `datetime.strftime` / `re.sub` / `json.loads` 就够？" 把能确定性化的步骤替换成 Python 函数——这不算 chaining 步骤，但能把 chaining 的有效深度大幅压短。

### 坑 5：上下文穿透 → 后面步骤被前面步骤的废话污染

直觉做法是把前一步的完整输出作为下一步 prompt 的输入。但如果前一步输出 2000 字而下一步只需要其中一个字段——你在烧 token 也在分散 attention。

**对策：**
- 每步明确声明它消费哪些字段、产出哪些字段；
- 用 schema 把 "我需要的字段" 显式 select 出来再传给下一步；
- 把"完整中间状态" vs "下一步实际需要"分离——前者持久化到 store 做 debug，后者才进 prompt。

A2 的 attention budget 论点在这里直接适用：每个 token 都要为它的存在付租金。

### 坑 6：把异步任务做成同步 chain → 浪费可并发的机会

不是所有 chain 步骤都必须串行。如果 step 3 和 step 4 都消费 step 2 的输出但**彼此独立**——它们应该并发，不该 chain。

```python
# bad: 串行
out2 = llm(prompt2(out1))
out3 = llm(prompt3(out2))   # 等 out3 完成
out4 = llm(prompt4(out2))   # 才开始 out4

# good: 并发
out2 = llm(prompt2(out1))
out3, out4 = await asyncio.gather(
    llm_async(prompt3(out2)),
    llm_async(prompt4(out2)),
)
```

判断标准：画一张 step 依赖图（DAG），任何同层节点都应该并发。详见 Ch.3。

### 坑 7：失败重试不做退避 → 偶发 API 错把 retry storm 放大

外层的 retry 逻辑没考虑指数退避，一旦某个 step 因为 rate limit 失败，整条 chain 的 retry 同时打回 API，进一步触发更严重的 rate limit。

**对策：** retry 用指数退避（如 1s, 2s, 4s, 8s），且总尝试次数 ≤ 3；超过就 fail-fast 让上层处理。

## 1.7.5 调试 Prompt Chaining 的实用清单

链跑出错误结果时，按下面顺序排查最有效（按命中率从高到低排序）：

**1. 看每一步的中间输出。** 90% 的 chain bug 在打印中间步骤后立刻暴露。如果你没有可观测性把中间输出落盘——立刻加上，这比任何 framework 抽象都重要。最简单做法：

```python
def llm_traced(prompt, step_name, **kw):
    out = llm(prompt, **kw)
    trace_log.write({"step": step_name, "in": prompt[-500:], "out": out[:500],
                     "ts": time.time()})
    return out
```

**2. 找累积误差的源头。** 如果端到端准确率明显低于各步准确率乘积——某一步的输出格式可能在下一步被错误解析（看起来成功实际数据已损坏）。逐步对比 schema validation 的输出。

**3. 检查每一步的 token usage。** 如果某一步 token 突然飙升，意味着前面步骤把不必要的 context 透过来了。压缩或显式 select 字段。

**4. 把每步独立用真实数据 replay。** 不要在 chain 跑完后才看错——把每步抽出来用一组 fixture data 独立跑（这就是 unit test）。chain 是固定结构，每步都可以独立 eval。

**5. A/B test 单步改进。** 当怀疑某步的 prompt 有问题，固定其他步骤，只换该步 prompt，用 100 条历史数据跑两次对比 metric。如果只能整 chain 重跑——你没法做 attribution。

**6. 检查 LLM 是否在某步默默拒绝。** 偶尔模型会输出 "I cannot help with that"——如果你的 schema 解析没特别处理这种情况，会得到一个空字段然后 downstream 莫名其妙。在每步都加 refusal detection。

**7. 看 latency 的分布而不是平均值。** chain 的 p50 latency 可能很好，p99 因为某一步偶发抖动炸到 30 秒。每步独立监控 p50 / p95 / p99，找出"经常最慢的那一步"是优化的第一目标。

## 1.8 与其他模式的关系

- **Prompt Chaining 是 Planning 的退化情形**——Planning（Ch.6）让 LLM 在运行时决定 step 序列，Chaining 是 step 序列预先固定的特例。当你的 chain 经常需要"根据上一步结果跳过某些步骤"时，就该升级到 Planning。
- **Prompt Chaining 是 Reflection 的 backbone**——Reflection（Ch.4）本质是 "generate → critique → regenerate" 这条三步 chain 加上一个 loop。理解了 chaining 再去看 reflection，会发现后者只是多了一个 "until pass" 条件。
- **Prompt Chaining 与 Parallelization 互补**——Chaining 是顺序（A 完成才能 B），Parallelization 是并发（A 和 B 同时）。复杂系统通常两者交替：先 parallel fan-out N 个分析，再 chain 出最终报告。
- **Prompt Chaining 是 Routing 的下游**——Routing（Ch.2）只负责"分到哪个 handler"，handler 内部往往就是一条 chain。
- **Prompt Chaining 受益于 structured output 和 Guardrails**——schema validation（Ch.18）就是 chain 上 gate 的一种特例。

## 1.8.5 一个完整的生产级例子

下面用一个 "文档审核 → 改写 → 合规检查 → 多语言发布" 的真实场景，把这一章学到的东西串起来。

业务需求：营销团队提交一段产品文案 → 系统自动改写到品牌口吻 → 检查是否合规（无夸大、无禁词）→ 翻译成 5 种语言 → 写入 CMS。

朴素版本（反模式）：

```python
def publish(text):
    out = llm(f"""你是品牌助手。请按以下步骤处理 '{text}':
1) 改写到品牌口吻
2) 检查合规
3) 翻译成 en/zh/ja/ko/es
4) 输出 JSON 包含改写后文案、合规结论、5 个翻译""")
    return json.loads(out)
```

这条 prompt 几乎注定失败：

- 4 个任务挤一个 prompt，每个都做得平庸；
- 输出是 JSON 但 LLM 经常漏掉某个字段；
- 任意一步出错（如翻译只翻 3 种）整个失败；
- 合规检查与改写在同一次 call 里——LLM 倾向于"既然我改写了那肯定合规"，自己审自己绝不可信。

Chaining 版本：

```python
def publish(text: str, langs: list[str]) -> dict:
    # Step 1: 品牌口吻改写
    rewritten = llm(
        BRAND_VOICE_PROMPT.format(text=text),
        schema=Rewritten, model="claude-sonnet-4"
    )

    # Gate 1: 长度约束（品牌规范要求 100-300 字）
    if not (100 <= len(rewritten.body) <= 300):
        rewritten = llm(
            f"Rewrite to 100-300 chars: {rewritten.body}",
            schema=Rewritten, model="claude-sonnet-4"
        )

    # Step 2: 合规检查（独立 LLM call，不让改写者自审）
    compliance = llm(
        COMPLIANCE_PROMPT.format(text=rewritten.body),
        schema=Compliance, model="claude-sonnet-4"
    )

    # Gate 2: 合规失败直接 fail-fast，转人工
    if compliance.verdict != "PASS":
        return {"status": "rejected", "reason": compliance.reason}

    # Step 3: 并发翻译（这里跨入 Ch.3 Parallelization）
    translations = parallel_translate(rewritten.body, langs)

    # Step 4: 写入 CMS（纯代码，不用 LLM）
    cms_id = cms_client.create(
        body=rewritten.body,
        translations=translations,
        compliance_log=compliance.dict(),
    )
    return {"status": "published", "id": cms_id}
```

这个版本展示了 chaining 的所有要点：

- **每步独立 schema** → 错位立刻可见；
- **gate 是 Python 代码**（长度检查 + 合规 verdict 分流）；
- **不让 LLM 自审**（改写和合规是两个独立的 call）；
- **不必要的步骤用纯代码**（CMS 写入）；
- **chain 在 step 3 自然分叉到 parallelization** → 两个模式组合用。

Step 数量是 4，远低于"5 步以上累积误差陷阱"。每步又用 Sonnet 这种强模型，单步准确率应当 95%+，端到端约 0.95^3 ≈ 86%——剩余 14% 由合规 gate 兜住转人工。这是一个可以直接上线的设计。

## 1.9 一句话总结

**Prompt Chaining 是 agent 工程的"汇编语言"——简单、可预测、便宜，绝大多数所谓 agent 应用，本质上就是一条带 gate 的 prompt chain。** 在你考虑任何更复杂的模式之前，先问自己：能不能用 chain 解决？大多数情况答案是能。
