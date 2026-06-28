# Ch.2 Routing 路由

## 2.1 一句话定义

**Routing 是用一个 classifier（LLM 或传统模型）检查输入，把它 dispatch 到 N 个专门 handler 中的一个；每个 handler 有自己的 prompt、model、tool 配置。** [A1, P1, O1]

形式上极简，但藏着两个工程意图：**关注点分离**（每个 handler 只优化自己那类输入）和 **成本分层**（简单输入用便宜模型，复杂输入才用贵模型）。

## 2.2 动机：为什么不能用更简单的方式

最朴素的反例是"用一个万能 prompt 处理所有意图"：

```text
你是一个客服助手。如果用户问账单问题，按 X 处理；如果问技术问题，按 Y 处理；
如果问退款，按 Z 处理；如果问营业时间，按 W 处理；如果……
```

这种 prompt 在前 3 个意图覆盖时还能工作，意图加到 8 个以上必然崩坏，原因 A1 一针见血：

> "Without this workflow, optimizing for one kind of input can hurt performance on other inputs." [A1]

具体表现：

- **Prompt bloat**：每加一个新意图，全局 prompt 增长，所有意图的 attention 都被稀释；
- **Optimization conflict**：调 prompt 让账单更准，技术答疑变差；调回去技术好了，账单又烂；
- **Cost waste**：所有请求不论复杂度都跑同一个模型——简单意图用 Sonnet 是浪费，复杂意图用 Haiku 是赌运气；
- **Tool overload**：所有 handler 的 tool 都暴露给 LLM，相似 tool 互相争抢调用。

Routing 把这些问题一刀切开。每个 handler 是独立的子系统：

- 独立 prompt（可以针对该意图深度优化）；
- 独立模型（cost-tiering：A1 明确把 "Haiku for easy queries, Sonnet for hard" 列为 Routing 的二号动机）；
- 独立 tool 集（不暴露不相关 tool）；
- 独立评估（每个 handler 有自己的 eval set）。

P1 补充了一个 routing 与 chaining 的关键差异：

> "An initial LLM acts as a router, classifying the user's input and directing it to the most appropriate specialized task or LLM... the selected agent 'takes over' the request." [P1]

注意 "takes over"——routing **不做** 合成回写。Router 决定去哪，然后 handler 自己负责整个响应；不像 orchestrator-workers（Ch.6）那样还要把 worker 结果汇总。这是 routing 与 planning 的根本分野。

## 2.3 核心结构

输入：原始任务。
输出：选中 handler 的输出。
控制流：单次分类 → dispatch → handler 独立运行。

```
                       +-----------+
                       |  Router   |  (cheap LLM / embedding NN / rules)
   input  ------->     | classifier|
                       +-----------+
                          |   |   \
                  billing |   | tech  general
                          v   v       v
                   +--------+ +------+ +---------+
                   |Billing | |Tech  | |General  |
                   |Handler | |Hand. | |Handler  |
                   +--------+ +------+ +---------+
                          \   |       /
                           v  v      v
                       (handler output goes directly to user)

                          [fallback]  <-- 置信度低 / 未匹配
                              |
                              v
                       +---------+
                       | Default |
                       +---------+
```

三种 router 实现，从便宜到准确：

1. **规则 router**：关键字、正则、URL pattern。延迟近 0，但只覆盖能枚举的情况。
2. **Embedding NN router**：把 query embed 后与每个 intent 的 prototype embedding 做 cosine，取 top-1。延迟 ~50ms，便宜，对训练数据敏感。
3. **LLM classifier router**：让 LLM 输出 intent 标签。最贵但最灵活，处理 paraphrase 和长尾意图最好。生产中常用 Haiku / GPT-4o-mini / Gemini Flash 这一档。

实际系统经常**三种叠用**：规则先过明显 case，embedding 拦截高频意图，LLM 兜底剩余的。

## 2.4 最小示例

```python
ROUTES = {
    "billing": (billing_agent, "claude-haiku-4"),     # 简单意图用便宜模型
    "tech":    (tech_agent,    "claude-sonnet-4"),    # 复杂意图用强模型
    "general": (general_agent, "claude-haiku-4"),
}

CLASSIFIER_PROMPT = """Classify the user message into exactly one of:
billing, tech, general. If unsure, output 'general'.
User: {msg}
Answer (one word):"""

def route(user_msg: str) -> str:
    intent = llm(
        CLASSIFIER_PROMPT.format(msg=user_msg),
        model="claude-haiku-4",     # router 必须便宜
        max_tokens=5,
    ).strip().lower()

    handler, model = ROUTES.get(intent, ROUTES["general"])  # 兜底 fallback
    return handler(user_msg, model=model)
```

10 行代码涵盖 routing 的全部精髓。注意：

- **Router 用 Haiku（cheap fast）**，handler 才用 Sonnet。如果 router 本身用 Sonnet，你只是付出 routing 的延迟没有获得 routing 的成本优化。
- **fallback 是 explicit 的** `ROUTES.get(intent, ROUTES["general"])`——见坑 1。
- **classifier prompt 限定输出集**（"exactly one of: billing, tech, general"）+ `max_tokens=5` 强约束。否则 LLM 容易输出 "I think this is about billing because..."。
- Handler 之间**没有 cross talk**——routing 决策后控制权完全移交。

## 2.5 框架对比

四大框架对 Routing 的实现 idiom（D digest Pattern 2）：

| 框架 | Primitive | 怎么搭 | Idiomatic 片段 |
|---|---|---|---|
| **LangGraph** | `add_conditional_edges` + Router 多 agent 模式 | router node 做分类（常用 structured output），conditional edge function 根据分类结果挑下游 node | `builder.add_conditional_edges("router", route_fn, {"billing": "billing_node", "support": "support_node"})` |
| **AutoGen v0.4+** | Triage agent + delegate tools | 给 triage `AIAgent` 配 `delegate_tools=[transfer_to_X, transfer_to_Y]`；LLM 选某个 delegate tool 时，agent 把任务 publish 到对应 topic | `delegate_tools=[transfer_to_issues_and_repairs_tool, transfer_to_sales_agent_tool, escalate_to_human_tool]` |
| **CrewAI** | `Process.hierarchical` + manager | `manager_llm` 或 `manager_agent` 读每个 incoming task，分给最合适的成员；没有静态 if/else | `crew = Crew(agents=my_agents, tasks=my_tasks, process=Process.hierarchical, manager_llm="gpt-4o")` |
| **Google ADK** | Coordinator（LlmAgent 带 sub_agents） | 中心 `LlmAgent` 列出 specialist 作为 `sub_agents`；ADK 的 AutoFlow 用每个 child 的 `description` 字段让 LLM 自己 delegate | `coordinator = LlmAgent(name="CoordinatorAgent", instruction="Route billing issues to BillingSpecialist...", sub_agents=[billing_specialist, tech_support])` |

四种心智模型差异：

- **LangGraph** 的 routing 是**显式代码**：你自己写 dispatch function，graph 看得见全部分支；
- **AutoGen** 和 **ADK** 都让 LLM 来 delegate，但 AutoGen 通过 publish 到 topic 实现 routing，ADK 是 in-place 移交执行——前者更适合分布式，后者更适合单进程；
- **CrewAI** 的 hierarchical 把 manager 自己变成一个 agent，**每一回合都重新规划**——比静态 routing 灵活，但成本和延迟都高一档。

工程经验：

- **如果你的意图集稳定且数量少（< 20）**，用 LangGraph 风格的 explicit routing——可观测性最好；
- **如果意图集会动态扩展**（新增 specialist 不应该改 router 代码），用 ADK / AutoGen 风格的 LLM delegation——可扩展性最好；
- **不要在 prototype 阶段用 CrewAI hierarchical 做 routing**——manager 每回合 LLM 调用太贵，等你确定确实需要"动态再规划"再升级。

### 2.5.1 选 router 类型的决策树

工程经验性的选型路径：

```
你的意图集稳定吗？
├── 是，且 < 20 个 → LangGraph explicit routing（条件边 + dict）
├── 否，会动态增加 → Capability Registry + LLM classifier
│                    （ADK sub_agents 或自己实现）
└── 边界模糊 → 改用 Planning（Ch.6）

你的 QPS 高吗？
├── < 100 QPS → LLM classifier 足够
├── 100-1000 QPS → embedding router + LLM 兜底
└── > 1000 QPS → 规则 + embedding 缓存为主，LLM 只兜底未匹配

你需要 cost-tiering 吗？
├── 是 → 强制 router 用便宜模型
└── 否 → 任何 router 实现都行

你的意图边界有 prompt injection 风险吗？
├── 是 → router 前置 guardrail 过滤
└── 否 → 普通 routing
```

实际经验：90% 的中等规模 SaaS（QPS 10-200，意图数 5-15）最适合**规则 + LLM classifier 混合**——规则覆盖 80% 高频 case，LLM 兜底剩余。embedding router 只在 QPS 上千且分类需要语义相似性时才划算。

## 2.6 何时不该用

- **所有路径处理方式相同。** Routing 的前提是 "different handlers have different behavior"。如果只是不同 prompt template，用 Prompt Chaining 的 if-else 即可。
- **下游 specialist 少于 2 个。** 一个 specialist 不需要 router。
- **意图边界模糊到无法清晰分类。** 例如"用户问的可能同时涉及账单和技术"——这种情况要么用 Multi-label routing（输出意图集合），要么直接上 Planning（Ch.6）让 LLM 自己规划。
- **延迟极敏感且分类成本无法摊销。** Router 增加一次 LLM call 的延迟。如果 handler 本身只需要 1 次 call，多加 1 次 router call 让总延迟翻倍——这种情况要么用 embedding router 把分类降到 <50ms，要么取消 routing 让 handler 自己识别意图。
- **意图与执行严重耦合。** 如果识别意图本身需要先做一些操作（如先查数据库才能知道用户是哪类客户），不能用 routing——你需要的是 Agent 模式让 LLM 边做边定。

## 2.7 常见坑（按危害排序）

### 坑 1：没有 fallback → 置信度低时随机分到错地方

最常见的 bug：classifier 在边界 case 上输出 "I think this might be billing or maybe support"，被解析成空字符串或乱码，dict lookup 失败抛 KeyError，或更糟，被默认匹配到字典里的第一个 handler。

**对策：**

```python
intent = classifier_output.strip().lower()
if intent not in ROUTES:
    log_unknown_intent(user_msg, intent)
    intent = "general"   # 显式 fallback，绝不 silent crash
handler, model = ROUTES[intent]
```

更稳的做法：让 classifier 同时输出 confidence，低于阈值（如 0.7）就走 fallback 或 HITL（Ch.13）。

### 坑 2：Router 用了过强的模型 → routing 的成本优势消失

如果 router 用 Sonnet 而 handler 也用 Sonnet，你只是为每个请求多付一次 Sonnet 钱，没省任何成本。Router 的设计目标就是**便宜 + 快 + 在窄范围分类任务上够准**——通常 Haiku / GPT-4o-mini / Gemini Flash 就够。

**实测经验：** 让 Haiku 做 5 类意图分类，准确率通常 95%+；让 GPT-4o-mini 做 10 类意图分类，准确率通常 90%+。如果你的分类准确率低于 90%，先优化 prompt（加 few-shot 示例）而不是换更贵的模型。

### 坑 3：把"分类"和"执行"耦合在一起做 → 违反关注点分离

反模式：

```python
# bad: 一次 LLM call 既分类又生成回答
response = llm(f"""You're a customer service bot.
If the user is asking about billing, answer about billing.
If about tech, answer about tech.
User: {user_msg}""", model="claude-sonnet-4")
```

这等价于"用一个万能 prompt"——前面 2.2 列举的所有问题都会出现。

**对策：** 严格分两步：先用便宜模型分类，再 dispatch 到专门 handler。两步分离也让你可以**独立替换**——升级模型时分别考虑 router 和 handler 是不是要升级。

### 坑 4：新增 specialist 需要改 router 代码 → 不可扩展

如果每加一个 specialist 都要：
1. 改 router 的 prompt（加新 intent label）；
2. 改 ROUTES dict（加新映射）；
3. 改 fallback 逻辑（确保旧 client 不会卡死）；
4. 重新跑 eval（确保不影响旧 intent 准确率）；

那 routing 就成了 ops 瓶颈。

**对策（capability registry 模式）：**

```python
class Specialist:
    name: str
    description: str         # 这是 LLM 用来路由的依据
    handler: Callable

REGISTRY: list[Specialist] = []

def register(spec: Specialist):
    REGISTRY.append(spec)

def route(user_msg: str) -> str:
    catalog = "\n".join(f"- {s.name}: {s.description}" for s in REGISTRY)
    intent = llm(f"Available specialists:\n{catalog}\nUser: {user_msg}\nWhich one?",
                 model="claude-haiku-4").strip()
    spec = next((s for s in REGISTRY if s.name == intent), None)
    return (spec or REGISTRY[-1]).handler(user_msg)   # 最后一个是 fallback
```

新增 specialist 只需 `register(...)` 一次，router 完全不用改。

### 坑 5：Multi-intent 输入被强行单选 → 答非所问

用户说："我账单上多收了 50 块，而且 app 一直崩。"——账单 + 技术两个意图。如果 router 强行选 "billing"，technical 部分就被吞了。

**对策：**

- 单 intent 模式：classifier prompt 显式告诉 LLM "if multiple, pick the most urgent"，并在响应里告知用户 "Your tech issue will be handled separately"；
- Multi-label 模式：classifier 输出 list，由 orchestrator 决定串行或并行处理（这时其实你需要 Ch.6 Planning 而非纯 Routing）。

### 坑 6：Router 和 handler 用同一个 model context → 副作用泄漏

如果 router 和 handler 共用一个 conversation context（同一个 message list），handler 会看到 router 内部的思考（"This looks like billing..."），这可能：

- 引导 handler 也用 router 的风格说话；
- 浪费 token；
- 暴露内部决策给 user（如果 handler 不小心引用 history）。

**对策：** Router 用独立的、短的 context（只看 user 当前 message）；handler 用自己专属的、完整的 conversation context。

### 坑 7：Embedding router 的 prototype 过时 → 静默劣化

Embedding router 依赖每个 intent 的 prototype embedding。如果产品迭代加了新功能，但 prototype 没更新——新意图会被错误分到最相近的旧意图，且**没有任何报错**。

**对策：** 定期（如每月）拿新数据 sample 跑一遍 embedding router，对照 LLM router 的判断做 diff；diff 增大就触发 prototype 重计算。

### 坑 8：Classifier 的 prompt 写成开放问句 → 输出不受限

反模式：

```python
intent = llm(f"What is this user asking about? {user_msg}")
```

LLM 会回答 "They're asking about a billing issue regarding their recent invoice and..."——一段散文。你后面无法 dict lookup。

正确写法：

```python
intent = llm(
    f"""Classify into exactly ONE label. Allowed labels:
- billing
- tech
- general

Output the label and nothing else. No explanation.

User: {user_msg}
Label:""",
    max_tokens=5,
    stop_sequences=["\n"],
)
```

要素：(1) 显式列出允许的 label；(2) "output the label and nothing else"；(3) `max_tokens` 限到 5；(4) `stop_sequences` 截断多余内容。这四条加完后 LLM 几乎不可能输出散文。

更进一步可以用 structured output：

```python
class Intent(BaseModel):
    label: Literal["billing", "tech", "general"]
    confidence: float

intent: Intent = llm_structured(prompt, response_model=Intent)
```

`Literal` 类型让 schema validation 直接拒绝乱来的输出，比字符串解析稳得多。

### 坑 9：Routing 决策没有 audit trail → 出了 bug 不知道为什么分错

线上某个 user 投诉"我问退款怎么被分到了技术支持"——你打开日志只看到最终响应，没有 router 的判断依据。无法 debug，无法回归测试。

**对策：** 每次 routing 决策都记录到结构化日志：

```python
log_routing(
    request_id=req_id,
    user_msg_hash=hash(user_msg),
    intent_predicted=intent,
    confidence=confidence,
    fallback_triggered=(intent not in ROUTES),
    handler_chosen=handler.__name__,
    classifier_model=classifier_model,
    classifier_latency_ms=lat,
)
```

定期 sample 这些日志做 offline eval：拿过去一周的 routing 决策，找一个更强的 LLM 重新判一遍 ground truth，统计 disagreement rate。如果 disagreement > 5%，考虑改 prompt 或换 router 模型。

### 坑 10：Cost-tiering 时 router 自己变成单点瓶颈

如果你为了省钱把 80% 流量分到 cheap handler、20% 分到 expensive handler——但 router 本身每次都要 1 次 LLM call——router 的总成本可能反而成了大头（特别是当你的 cheap handler 也只需要 1-2 次 call 时）。

**对策：**

- **Embedding router** 替代 LLM router：~50ms / ~$0.0001 per query；
- **Rule + LLM 混合**：80% 流量被规则一击命中（0 token），剩余 20% 才走 LLM classifier；
- **Caching router decisions**：相同 user_msg 的 hash → intent 缓存 24h，大幅降低 router 成本。

实际经验：一个高流量客服系统加了 router cache 后，router 部分的 LLM cost 降了 70%——大量用户问的是高频重复问题。

## 2.8 与其他模式的关系

- **Routing 与 Handoff 的关系**：当 receiver 是另一个 agent 而不是固定 handler 时，routing 就成了 **Handoff**（Ch.15）。OpenAI [O1] 明确把 handoff 定义为 "routing where the receiver is itself an agent"。两者结构一致，但 handoff 还会**transfer conversation state**——routing 通常只 transfer the request。
- **Routing 是 Multi-Agent 的入口**：在 Manager pattern（Ch.7）中，manager agent 内部往往就是一个 router——决定把任务交给哪个 specialist agent。
- **Routing vs Planning**：Routing 处理"一个输入分到一个 handler"，Planning 处理"一个输入拆成多步、可能涉及多个 handler"。简单的二分法：如果输入 → 1 个 specialist 能搞定，用 Routing；如果输入 → 需要 ≥ 2 个 specialist 配合，用 Planning。
- **Routing 与 Parallelization**：Multi-label routing 的下游可以是 Parallelization（Ch.3）——把多个 intent 同时分给对应 handler 并发执行，最后聚合。
- **Routing 与 Guardrails**：Guardrails（Ch.18）经常作为 routing 的特殊 case 实现——router 把"违规输入"路由到 refusal handler。
- **Routing 与 Cost Optimization**：Routing 是 Resource-Aware Optimization（Ch.16）最直接的工具。A1 把 Haiku/Sonnet tiering 作为 Routing 的二号动机不是偶然。

## 2.8.5 一个完整的客服 routing 实例

业务背景：电商客服系统，每天 50000 条入站消息，意图分布大致：

| 意图 | 占比 | 复杂度 | 建议 model |
|---|---|---|---|
| 订单查询（查物流、查订单） | 35% | 简单 | Haiku |
| 退换货申请 | 20% | 中等 + 需要 tool 操作 | Sonnet |
| 商品咨询（库存、规格） | 15% | 中等 | Haiku |
| 售后问题（破损、错发） | 10% | 复杂 + HITL | Sonnet + HITL |
| 投诉 / 高情绪 | 5% | 复杂 + HITL | Sonnet + HITL |
| 寒暄 / 闲聊 | 10% | 极简 | Haiku |
| 其他 / 不清楚 | 5% | 兜底 | 转人工 |

设计：

```python
SPECIALISTS = [
    Specialist("order_query",     "查询订单/物流",     handler=order_handler,    model="haiku"),
    Specialist("rma",             "退货/换货",         handler=rma_handler,      model="sonnet"),
    Specialist("product_info",    "商品规格/库存",     handler=product_handler,  model="haiku"),
    Specialist("after_sales",     "商品质量问题/错发", handler=after_handler,    model="sonnet", hitl=True),
    Specialist("complaint",       "投诉/不满",         handler=complaint_handler,model="sonnet", hitl=True),
    Specialist("chitchat",        "寒暄/客套",         handler=chitchat_handler, model="haiku"),
    Specialist("human_takeover",  "兜底/转人工",       handler=human_handler,    model=None),
]

ROUTER_PROMPT = """你是电商客服分诊。把用户消息分到下列之一：
{catalog}

规则：
- 不确定 → human_takeover
- 用户情绪激动（出现"垃圾/差评/投诉/曝光"等）→ complaint
- 涉及钱（退款/赔偿）且情绪温和 → rma
- 同时多意图 → 选最紧急的

只输出 label，不要解释。
用户消息：{msg}
Label:"""

def route(user_msg: str, ctx: SessionCtx) -> Response:
    # 1. 规则 fast-path
    if is_obvious_chitchat(user_msg):
        return chitchat_handler(user_msg)

    # 2. Cache lookup
    cache_key = hashlib.md5(user_msg.encode()).hexdigest()
    if cached := routing_cache.get(cache_key):
        spec = SPECIALISTS_BY_NAME[cached]
    else:
        # 3. LLM classifier
        catalog = "\n".join(f"- {s.name}: {s.desc}" for s in SPECIALISTS)
        label = llm(ROUTER_PROMPT.format(catalog=catalog, msg=user_msg),
                    model="haiku", max_tokens=10).strip()
        if label not in SPECIALISTS_BY_NAME:
            label = "human_takeover"
        routing_cache.set(cache_key, label, ttl=86400)
        spec = SPECIALISTS_BY_NAME[label]

    # 4. HITL pre-check
    if spec.hitl and not ctx.agent_available:
        return queue_for_human(user_msg, reason=spec.name)

    # 5. Dispatch + audit log
    log_routing(req_id=ctx.req_id, label=spec.name, cached=bool(cached))
    return spec.handler(user_msg, model=spec.model, ctx=ctx)
```

这个例子包含了本章几乎所有要点：规则 fast-path、cache、capability registry、显式 fallback、HITL 升级、audit log、cost-tiering。整套 routing 层在生产中跑下来：

- p95 routing 延迟 < 80ms；
- 端到端意图准确率 91-93%；
- 总 router LLM cost < 全系统 LLM cost 的 8%；
- 兜底 human_takeover 触发率 < 6%。

如果你的 routing 系统跑出来数字与上面差距很大，回到坑 1-10 一条条 check。

## 2.9 一句话总结

**Routing 是 cheapest classifier + simplest dispatcher 的组合艺术。它的工程价值不在分类有多准，而在让每个 handler 都能独立优化、独立替换、独立计费。** 当你的系统开始出现"一个 prompt 越改越长越改越累"的症状时，先想 Routing，再想其他更复杂的模式。
