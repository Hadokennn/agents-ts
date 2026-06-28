# Ch.5 Tool Use 工具调用

> "Tools are a new kind of software which reflects a contract between deterministic systems and non-deterministic agents." — Anthropic [A3]

## 5.1 定义

**Tool Use** 是指 LLM 通过结构化输出（典型形式是 JSON 函数调用）调用外部函数、API、代码执行环境或子 agent，从而**感知**或**改变**训练数据之外的世界。一次完整的工具调用包含四件事：

1. 模型决定**调用哪个工具**（tool selection）
2. 模型决定**调用参数**（argument synthesis）
3. 运行时**执行**该工具（execution）
4. 把**结果反馈**回模型上下文（observation injection）

如果 4 之后还能回到 1，就构成了"agent loop"——R1 ReAct 论文给出了这个 loop 的正式形式 [B/R1]。所有现代 agent 栈（Claude tool use、OpenAI function calling、LangGraph ReAct、AutoGen）都是对这个 loop 的不同包装。

工具调用是 Andrew Ng 在 2024 年提出的"4 大 agentic design patterns"之一 [A/N1]，也是 OpenAI Practical Guide 列出的"data / action / orchestration"三种工具类型的基石 [A/O1]。但本章的真正立场来自 Anthropic Engineering 2025 年的文章 *Writing Effective Tools for AI Agents* [A/A3]——**工具是一种新型软件**，它的设计哲学不能照搬给人用的 API，也不能照搬给软件用的 SDK。这个论断是本章的脊梁。

**几个常见的别名（必须先消除歧义）：**

- **Function Calling**（OpenAI 系）≡ Tool Use 的协议级别名 [A/P1]
- **MRKL**（Modular Reasoning, Knowledge and Language，Karpas et al. 2022）— Tool Use 的早期框架化，强调 LLM 作为"router to expert modules" [A/L1]
- **Tool-augmented LLM**（Anthropic 系）— 与"agent"对齐时的一个 building block 名字 [A/A1]
- **ACI**（Agent-Computer Interface）— 不是 Tool Use 的别名，而是它的**设计哲学**[A/A1, A/A3]

## 5.2 动机：为什么 Tool Use 不可绕开

一个孤立的 LLM 可以写诗、做翻译、解释概念，但它做不到三件事：

1. **拿到训练截止之后的数据**（实时股价、今天的会议纪要、用户的当前购物车）
2. **执行确定性计算**（精确算术、SAT solver、SQL aggregation）
3. **改变世界**（发邮件、下订单、提交 PR）

这三件事正是 production agent 99% 的实际需求。Tool Use 是 LLM 跨过这条边界的唯一通路。

Lilian Weng 在 *LLM Powered Autonomous Agents* 中把 Tool Use 列为 agent 的三大支柱之一（Planning, Memory, Tool Use），并引用了 Karpas et al. 的关键判断：

> "When [...] external symbolic tools can work reliably, knowing when to use and how to use the tools are crucial." [A/L1]

注意"knowing **when** to use"——这比"how to use"更难。一个有 100 个工具的 agent，最大的失败模式不是参数填错，而是**该用 calculator 时去硬算，该用 search 时去编造**。后面 5.6 节会回到这个失败模式。

Toolformer [B/R5] 提供了"why this matters"的硬数据：把 6.7B 的 GPT-J 用 self-supervised 方式教会调用 5 个简单工具（QA / Wikipedia / calculator / calendar / translator），在 LAMA QA 上从 17.8 跳到 33.8，**超过了 25× 体量的 GPT-3 175B（26.8）**。数学任务（ASDiv/SVAMP/MAWPS）平均从 8 跳到 38，calculator 在数学样本里被调用率高达 97.9%。一个小模型 + 工具 > 大模型 raw——这是 Tool Use 的"复利"效应。

## 5.3 结构：从 ReAct loop 到 ACI

### 5.3.1 ReAct 是所有 Tool-using Agent 的祖宗

Yao et al. 2022 [B/R1] 提出的 ReAct（Reasoning + Acting）把 LLM 的 action space 扩展为 `A ∪ L`：A 是工具集，L 是自由文本（"Thought"）。每一轮 LLM 输出要么是 Thought（不影响环境，只追加到上下文），要么是 Action（被执行并返回 Observation）。三者交替出现在同一个自回归 trajectory 里：

```text
Thought 1: I need to find the elevation range of the area where the eastern
sector of Colorado orogeny extends into.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building...
Thought 2: It does not mention the eastern sector. So I need to look up
eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1/1) The eastern sector extends into the High Plains
and is called the Central Plains orogeny.
Thought 3: ...
Action 3: Search[High Plains (United States)]
Observation 3: ...elevation from around 1,800 to 7,000 ft (550 to 2,130 m).
Action 4: Finish[1,800 to 7,000 ft]
```

ReAct 在 ALFWorld 上把成功率从 Act-only 的 45% 拉到 71%，在 WebShop 上从 IL+RL 的 28.7% 拉到 40.0% [B/R1]。更关键的是定性结果：CoT 的 56% 失败来自幻觉（编造事实），ReAct 把幻觉降到 0%——因为每个 Action 都被环境 ground 了。代价是出现了 47% "reasoning errors"（往往是 repetitive loops，下文 5.6 会讲）。

ReAct 的伪代码非常短，本章读者必须把它印在脑子里：

```python
def react_loop(task, tools, max_turns=20):
    context = task_prompt + few_shot_examples + f"Question: {task}\n"
    for i in range(1, max_turns + 1):
        completion = llm(context, stop=["\nObservation"])
        # 解析模型输出
        thought, action_name, action_args = parse(completion)
        if action_name == "Finish":
            return action_args
        # 执行工具
        try:
            obs = tools[action_name].run(**action_args)
        except Exception as e:
            obs = f"Error: {format_for_agent(e)}"  # ACI care
        context += completion + f"\nObservation {i}: {obs}\n"
    return "ABORT: max_turns reached"
```

注意三个细节：

- `stop=["\nObservation"]` ——必须在模型尝试自己编造 Observation 之前截断，否则模型会幻觉环境的回复（这是 ReAct 复现时最常踩的坑）。
- `format_for_agent(e)` ——错误信息不是给开发者看的，是给模型看的（5.4.5 详述）。
- `max_turns` ——保险丝。生产 agent 没有这个会跑死循环。

### 5.3.2 现代 tool schema：Anthropic / OpenAI 格式

ReAct 论文用的是纯文本解析。生产环境则全部走结构化 schema。Anthropic 的 tool spec 是当今最干净的格式之一：

```python
TOOLS = [
    {
        "name": "search_orders",
        "description": (
            "Find a customer's recent orders by their full name. Returns up "
            "to 10 most recent orders, sorted by date descending. Returns an "
            "empty list if the customer has no orders. Use this tool when the "
            "user asks about their order history, order status, or wants to "
            "reference a specific past purchase."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": (
                        "Full customer name as it appears on the account, "
                        "case-insensitive. Example: 'Alice Chen'."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "maximum": 50,
                    "description": "How many orders to return. Default 10.",
                },
            },
            "required": ["customer_name"],
        },
    },
    # ...
]

def agent_loop(task, tools=TOOLS, max_turns=20):
    messages = [{"role": "user", "content": task}]
    for _ in range(max_turns):
        resp = client.messages.create(
            model="claude-sonnet-4",
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            return resp.content

        # 收集所有 tool_use blocks，并行执行
        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                result = execute(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": format_for_agent(result),
                })
        messages.append({"role": "user", "content": tool_results})
    return "ABORT: max_turns reached"
```

读这段代码时请注意：模型可以在**一个 turn 里返回多个 tool_use blocks**（parallel tool use），运行时需要全部执行后一次性塞回去。如果你的 agent loop 没处理这一点，Claude 4 / GPT-4 的并行能力会被你白白浪费掉。

## 5.4 ACI 设计五原则（本章的核心）

Anthropic 的 *Writing effective tools for AI agents* [A/A3] 是迄今为止最系统的 ACI manifesto。文章的总论点是：

> "More tools don't always lead to better outcomes. If a human engineer can't definitively say which tool should be used in a given situation, an AI agent can't be expected to do better." [A/A3]

我们把它精炼为五条原则，每条都附"反例 / 正例"。

### 原则 1：Build for affordances, not 1-to-1 API wrapping

**坏例子（来自 A3）：**
当客户支持 agent 需要回答"Alice Chen 的上次订单出什么问题了"时，naive MCP wrapper 会强迫 agent 串联三个 tool：

```text
get_customer_by_id(name="Alice Chen")  →  customer_id
list_transactions(customer_id)         →  recent orders
list_notes(customer_id)                →  support history
```

这是把人类工程师 REST API 设计直接搬给 agent。三次 tool call、三次 LLM 决策、三次往返延迟、三次结果膨胀上下文。

**好例子：**

```python
{
    "name": "get_customer_context",
    "description": (
        "Get a customer's profile, last 10 orders, and last 20 support "
        "notes in one call. Use this as the first tool whenever the user "
        "is asking about a specific customer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"customer_name": {"type": "string"}},
        "required": ["customer_name"],
    },
}
```

一个 consolidated tool 直接对齐 agent 的**实际工作流**（"先查上下文再回答"），而不是 backend API 的**资源切分**。Anthropic 在 SWE-bench 上的内部报告说："we spent more time on tool design than on prompt design"，这是为什么 [A/A3]。

**判断原则：** 如果一个 task 总是触发同一组 tool calls 的固定组合，就把这个组合升级成一个 tool。

### 原则 2：Token-efficient response

每个 tool result 都要进上下文，每次后续 LLM call 都要把它再读一遍。Anthropic 给出了具体手段：

- **Pagination**：明确支持 `offset / limit`，缺省返回前 N 条。
- **Range selection**：让 agent 指定行号范围而不是整文件返回（典型如 file reader）。
- **Filtering**：在 tool 内部完成过滤，agent 只拿过滤后的结果。
- **Truncation defaults**：Claude Code 把所有 tool response 上限设为 **25k tokens**，超出截断并提示 [A/A3]。
- **`response_format` enum**：让 agent 自己选 `concise` 还是 `detailed`。A3 的 Slack tool 例子显示，concise 模式可以省 **3×** token。

```python
{
    "name": "list_slack_messages",
    "input_schema": {
        "type": "object",
        "properties": {
            "channel": {"type": "string"},
            "limit": {"type": "integer", "default": 20, "maximum": 100},
            "response_format": {
                "type": "string",
                "enum": ["concise", "detailed"],
                "default": "concise",
                "description": (
                    "concise = sender + text only. detailed = full message "
                    "object with reactions, threads, edits. Default concise "
                    "to save tokens; switch to detailed only when needed."
                ),
            },
        },
        "required": ["channel"],
    },
}
```

A2 把这件事拔高到了"context as finite resource with diminishing marginal returns"的高度 [A/A2]——n² attention pairs 让上下文长度的 recall 不是线性下降而是 super-linear 下降。每一个 token 都要为自己被保留下来辩护。

### 原则 3：Namespacing

当 agent 同时挂了 5 个 MCP server，每个 server 都有 `send_message`、`list_files`、`get_status` 这种通用名字时，工具选择准确率会断崖式下降。解法是强制 namespace：

```text
slack__send_message    # 而不是 send_message
gmail__send_message
notion__send_message
```

A3 实测："namespacing materially affects tool selection accuracy" [A/A3]。Anthropic 在 MCP 客户端里把这层做成了自动前缀。如果你自己 hosting 多个 server，必须自己做一层 namespace adapter。

### 原则 4：Semantic identifiers over UUIDs

返回值里的标识符要给**人能读懂的名字**而不是 opaque UUID：

```text
// 差
{"order_id": "ord_01H8XYZ2KQRWVN9F4ABCDEFGHJ"}

// 好
{"order_id": "ord_2024_alice_chen_001",
 "summary": "Alice Chen's order from 2024-03-15, 3 items, $147.20"}
```

A3 报告："returning semantic identifiers significantly reduces hallucination" [A/A3]——因为 agent 在 follow-up 调用里需要复述这个 ID 时，semantic 形式可以被它的 reasoning 直接 ground，opaque UUID 则必须靠精确拷贝，错一个字符就 404。

如果 backend schema 不允许改 ID 格式，那就在 response 里同时返回 `id` 和 `human_label`，并在 description 里告诉 agent："always reference items by `human_label` in your reasoning; pass `id` to other tools."

### 原则 5：Helpful error messages

> "Prompt-engineer your error responses to clearly communicate specific and actionable improvements, rather than opaque error codes or tracebacks." [A/A3]

错误信息是 ACI 的一部分。一个差的错误响应（直接吐 Python traceback）会让 agent 反复重试同一个错。一个好的错误响应告诉 agent 哪里错了、怎么改：

```text
// 差
Error: ValueError at line 42 in tool_handler.py:
  Traceback (most recent call last):
    File "tool_handler.py", line 42, in run
      return self.execute(args["customer_name"])
  KeyError: 'customer_name'

// 好
{
  "error": "missing_required_field",
  "message": (
    "The `customer_name` field is required. You passed: "
    "{'name': 'Alice Chen'}. Try again with: "
    "{'customer_name': 'Alice Chen'}."
  )
}
```

后者直接告诉 agent 它把 `customer_name` 拼成了 `name`，并给出修正示例。Reflexion [B/R2] 论文也观察到：当错误信息明确时，agent 在下一次 trial 就能恢复；当错误信息模糊时，agent 会反复犯同样的错。

## 5.5 范式转移：Code Execution with MCP

> "150,000 tokens → 2,000 tokens — Code execution mode dramatically improves agent efficiency." [C/A4]

2025-11 Anthropic 发布的 *Code execution with MCP* 是 Tool Use 范式的一次重要转移。它解决的问题是：当一个 agent 挂了几百上千个 MCP tools 时，传统 tool-calling 模式会爆炸：

1. **Context bloat**：所有 tool definitions（每个动辄几百 token）开机就塞进 context。
2. **Result bloat**：每次 tool call 的完整结果都流过 LLM context。
3. **Round-trip bloat**：每个简单的 control flow（如"循环检查 5 次"）都要走 5 个 LLM turn。

A4 给出的例子是典型的"Google Drive → Salesforce"工作流：把 Google Doc 的会议纪要拷到 Salesforce 的 Meeting 记录里。传统模式：

```text
TOOL CALL: gdrive.getDocument(documentId="abc123")
  → returns "Discussed Q4 goals...[full 50k-token transcript]"
       (loaded into model context)
TOOL CALL: salesforce.updateRecord(
  objectType="SalesMeeting",
  recordId="00Q5f...",
  data={"Notes": "Discussed Q4 goals...[full transcript written out]"}
)
       (model has to retype the transcript into context)
```

50k token 的 transcript 进了 context 一次（gdrive 返回），又被复述了一次（写入 salesforce 的参数），加上其他 tool defs，整个工作流烧 **150,000 tokens**。

Code execution 模式则把 MCP servers 暴露为**文件系统里的 TypeScript 模块**：

```text
./servers/
├── google-drive/
│   ├── getDocument.ts
│   └── index.ts
└── salesforce/
    ├── updateRecord.ts
    └── index.ts
```

每个 `.ts` 文件长这样：

```typescript
// ./servers/google-drive/getDocument.ts
import { callMCPTool } from "../../../client.js";
interface GetDocumentInput  { documentId: string; }
interface GetDocumentResponse { content: string; }
export async function getDocument(
  input: GetDocumentInput,
): Promise<GetDocumentResponse> {
  return callMCPTool<GetDocumentResponse>("google_drive__get_document", input);
}
```

LLM 不再 tool-call，而是**写代码**：

```typescript
import * as gdrive from "./servers/google-drive";
import * as salesforce from "./servers/salesforce";

const transcript = (await gdrive.getDocument({ documentId: "abc123" })).content;
await salesforce.updateRecord({
  objectType: "SalesMeeting",
  recordId: "00Q5f000001abcXYZ",
  data: { Notes: transcript },
});
```

50k 的 transcript **从未进入 model context**——它在 sandbox 里从 gdrive 直接流到 salesforce。整个工作流降到 **2,000 tokens（98.7% 削减）** [C/A4]。Cloudflare 独立报告了同方向的数据，并把这种模式命名为 "Code Mode"。

### 5.5.1 Code Execution 的五大收益

A4 系统枚举了五条：

**(1) Progressive disclosure of tools.** Tools live as files; the model `ls` them and reads only what's needed. 可以选配一个 `search_tools(detail_level=name|name+desc|full)` 让 agent 决定按什么粒度拉 schema。100 个 tool 的 schema 不再开机就塞进上下文。

**(2) Context-efficient tool results.** Filter / transform in the sandbox before returning anything to the model:

```typescript
const allRows = await gdrive.getSheet({ sheetId: "abc123" });
const pendingOrders = allRows.filter((row) => row["Status"] === "pending");
console.log(`Found ${pendingOrders.length} pending orders`);
console.log(pendingOrders.slice(0, 5));  // 5 rows to model, not 10,000
```

模型只看到 `console.log` 输出。

**(3) More powerful control flow.** Loops / conditionals / retries 全部在 sandbox 内完成，不走 LLM round-trip：

```typescript
let found = false;
while (!found) {
  const msgs = await slack.getChannelHistory({ channel: "C123456" });
  found = msgs.some((m) => m.text.includes("deployment complete"));
  if (!found) await new Promise((r) => setTimeout(r, 5000));
}
```

也省了 "time to first token" 的延迟——`if` 判断在 sandbox 里直接做。

**(4) Privacy-preserving operations.** Intermediate results stay in the sandbox by default. 敏感字段可以在 MCP client 边界被 tokenize（如 `{email: "[EMAIL_1]", phone: "[PHONE_1]"}`），传到对端再 untokenize——PII 从 Sheets 流到 Salesforce 而**从未进过 model context**。这一点对合规场景（医疗、金融、政企）的意义巨大。

**(5) State persistence and skills.** Agent 可以写中间文件（`./workspace/leads.csv`）下次恢复，也可以把一段反复用的代码 promote 成 `./skills/save-sheet-as-csv.ts`，加上 `SKILL.md` 就变成下一节要讲的 Agent Skill。

### 5.5.2 Code Mode 的代价

A4 自己也坦白了 trade-off：

- 需要**安全 sandbox**——资源限制、网络隔离、调用监控。直接 tool calling 不需要这层基础设施。
- 模型生成的代码是非确定性的，sandbox 内的 bug 可能静默吞掉错误。
- 文章原话："the benefits vs costs trade-off is real and context-dependent — Anthropic does not claim code-mode is universally better."

实际工程判断：

- Tools 数量 < 20 且每次都用得上 → **直接 tool calling 更简单**。
- Tools 数量 > 100 或涉及大数据 transformation / privacy → **Code mode 不可替代**。
- 中间情形 → 看 result size。如果每个 tool result < 1k token，tool calling 没问题；如果会拿到长 transcript / large CSV / binary blob，转 code mode。

## 5.6 Agent Skills：把 Tool Use 推向 lifelong learning

Anthropic 2025-12 把 Skills 升级为开源标准（agentskills.io）[C/A5]。一个 **Skill** 是一个目录：

```text
my-skill/
├── SKILL.md                    # required, YAML frontmatter + body
├── reference.md                # optional, lazy-loaded
├── forms.md                    # optional, lazy-loaded
└── scripts/
    └── extract_form_fields.py  # optional, executed not read
```

`SKILL.md` 最简形式：

```markdown
---
name: pdf
description: |
  Use when the user needs to manipulate PDFs — extract form fields,
  fill out forms, edit text, merge or split documents. Not for
  read-only PDF question-answering, which Claude can already do.
---

# PDF skill

Core workflow:
1. To inspect a PDF's structure, run `python scripts/extract_form_fields.py <path>`.
2. To fill a form, see forms.md.
3. For other PDF manipulations, see reference.md.
```

**三层 progressive disclosure：**

- **Level 1**：只有 `name + description`（每个 skill ~100 tokens）开机进 system prompt。
- **Level 2**：当模型判断 skill 相关时，`cat SKILL.md` 拉取 body。
- **Level 3**：SKILL.md 提到 "see forms.md" 时再 `cat forms.md`。
- **Level 4**：bundled scripts 当 tool 运行，输出 piped 进上下文，脚本内容**从不**进 context。

A5 原文："the amount of context that can be bundled into a skill is effectively unbounded." 这是 lifelong learning 的新答案：**capability lives on the filesystem, not in the weights, not in the system prompt.**

Skills 与 Code Mode（A4）天然耦合——脚本就是 sandbox 里跑的代码，工作目录就是 state persistence。两者合在一起，构成 Anthropic 2025 的 agent stack：MCP（tool 协议）+ Code Execution（执行底座）+ Skills（能力包装）+ filesystem（state）。

## 5.7 框架对比

| 框架 | 一等原语 | 怎么写 | 关键差异 |
|---|---|---|---|
| **LangGraph** | `bind_tools` + `ToolNode` | `@tool` 装饰函数 → `llm.bind_tools([...])` → 在 graph 里加 `ToolNode([...])` 节点 | 有专门的 `ToolNode`，自动处理并行调用、错误、state 注入 |
| **AutoGen (v0.4+)** | `FunctionTool` | `tools=[FunctionTool(fn)]`，agent 内部 `await self._tools[name].run_json(args, ctx.cancellation_token)` | 区分常规 tool 和 `delegate_tools`（后者触发 handoff） |
| **CrewAI** | `Agent(tools=[...])` | tools 挂在 agent 或 task 上 | 最少 ceremony，但缺乏并行 tool call 控制 |
| **Google ADK** | `tools=[...]` on `LlmAgent` + `AgentTool` | 普通 tool 直接给函数；`AgentTool(child_agent)` 把整个子 agent 当 tool | **独有：把 agent 包装成 tool**，让父 agent 通过单次 tool call 触发整个子工作流 |

[源：D Pattern 5]

四家在 tool spec 层面收敛——都是 `function name + JSON schema + description`，本质上是 OpenAI / Anthropic 的 function calling 协议的不同绑定。差异在于：

- **执行控制粒度**：LangGraph 的 `ToolNode` 是显式的图节点，可以在节点前后插自定义逻辑；其他三家都是 agent 内部黑盒。
- **handoff 集成**：AutoGen 的 `delegate_tools` 把"调 tool"和"调另一个 agent"统一成同一种语法。
- **agent-as-tool**：ADK 的 `AgentTool` 是把多 agent 编排折叠为单个 tool call 的唯一一等原语，对 hierarchical 设计很顺手。

### 5.7.1 何时手写、何时上框架

A1 的判断 [A/A1]：

> "Many patterns can be implemented in a few lines of code. Frameworks often create extra layers of abstraction that can obscure the underlying prompts and responses."

对 Tool Use 这个具体 pattern，作者建议：

- **学习 / 原型阶段**：手写 ReAct loop（5.3.1 的 30 行代码就够），看清每次 LLM 输入输出。
- **生产 single-agent**：用 SDK 原生 client（Anthropic Messages API / OpenAI Chat Completions）的 `tools=` 参数 + 自己管 message history。
- **生产 multi-agent / 复杂 routing**：选一个框架。LangGraph 适合显式状态机；AutoGen 适合 actor 模型 + 分布式；CrewAI 适合"团队 + 任务列表"心智；ADK 适合 typed container 组合。

## 5.8 坑（按踩中频率排序）

### 坑 1：给 agent 100+ tools 且描述都很相似 → 选错率飙升

**症状：** Agent 该用 `search_orders` 时调了 `list_orders`，该用 `update_user_profile` 时调了 `create_user`。

**根因：** Embedding 空间里这些工具名 + 描述距离太近，LLM 在 next-token 选择时区分不开。

**对策（综合 A3 + 实战）：**

- 砍掉冗余 tool：如果两个 tool 80% 场景可互换，合并成一个 + enum 参数。
- 在描述里加 **"Use when..." 和 "Don't use when..."**（这是 A5 Skills 推荐 description 写法的来源）。
- 上 Code Mode：让 agent `ls` 然后按需 `cat` schema，而不是开机灌全部 100 个。
- 离线 eval：用 A3 的 prompt-response pair 做 tool-selection accuracy benchmark，每次改 description 都跑一遍。

### 坑 2：Tool 描述写给开发者看，不是写给模型看

**症状：** 描述里全是 "Returns a JSON object conforming to OrderResponse schema with fields id, status, items, totalAmount where items is a list of LineItem with sku, quantity, unitPrice..."

**根因：** Schema 已经在 `input_schema` / `output_schema` 里了，description 应该讲**意图和场景**。

**正确的描述结构：**

```text
[一句话用途]
[Use when ... 触发场景]
[Don't use when ... 反例]
[关键参数语义]
[返回值的语义解读，不是 schema 复述]
[ failure mode 提示]
```

### 坑 3：返回 raw binary / 长 transcript / 一堆 UUID → 上下文爆炸 + 幻觉

详见原则 2（token efficiency）和原则 4（semantic IDs）。修复方案就是 5.5 的 Code Mode，或者在 tool 内部加 `response_format` enum + `summary` 字段。

### 坑 4：没 namespace，多个 MCP server 同名 tool 冲突

详见原则 3。MCP client 没自动加前缀就自己加。

### 坑 5：Side-effecting tool 没 sandbox / confirm

**症状：** Agent 因为推理失误调了 `delete_user`、`send_email_to_all_customers`、`drop_table`。

**对策：**

- **Risk rating（O1 [A/O1] 提的）：** 给每个 tool 打 low / medium / high，high 走 human-in-the-loop。详见 Ch.13。
- **MCP annotations（A3 + Pattern 10 [A/A3]）：** 在 tool spec 里标 `destructive: true` 或 `open_world: true`，让运行时拦截。
- **Idempotency key**：所有 side-effecting tool 接受 `idempotency_key`，重复 retry 不会重复执行。

### 坑 6：Repetitive loops（ReAct 的经典失败）

**症状：** Agent 反复输出同样的 `Thought → Action` 对，never recovers [B/R1]。Reflexion [B/R2] 把这个失败模式正式定义为 "consecutive identical actions that lead to the same observation"。

**对策：**

- **Detector：** 检测最近 N 个 action 的 hash 是否重复，重复就强制 inject 一条 system message：`"You've been repeating the same action. Try a different approach."`
- **温度提升：** 单纯 greedy decoding 容易卡住，可以在 detected loop 时把 temperature 从 0 拉到 0.7。
- **Reflexion 风格 reset：** Trial 内卡住就抛弃当前 trajectory，把"我刚才卡住了，下次别走同一条路"写入 episodic memory，开新 trial。

### 坑 7：ReAct 的 stop sequence 没设对

**症状：** 模型自己幻觉出 Observation，跳过真正的 tool 调用。

**对策：** 必须设 `stop=["\nObservation"]` 或同类 sentinel。结构化 tool spec（Anthropic / OpenAI 格式）原生支持，因为 tool_use block 是协议级 stop signal，但用纯文本 ReAct 时要手动配。

### 坑 8：Tool error 反馈被运行时吞掉

**症状：** Tool 抛异常时返回了 "Internal Error" 给模型，模型根本不知道改什么。

**对策：** 详见原则 5。所有 tool 抛错都 catch + format 成"specific + actionable"的 error message 再回模型。

## 5.9 何时不该用

- **任务可以用纯 prompt 完成。** 比如 "翻译这段中文"、"总结这篇文章" —— 别给它挂 tools。Tools 永远是 latency + cost + 错误源。
- **Tool 调用的代价比直接让 LLM 算还贵。** 比如算 "12 × 34"，calculator tool round-trip 比 LLM 直接算更慢、且不一定更准（GPT-4 算这个 100% 对）。当然 "12345.67 × 89012.34" 就该上 calculator。判断阈值：模型在 eval 上能稳定算对 → 不用 tool；偶尔出错 → 用 tool。
- **任务本身就是 deterministic pipeline。** 比如 ETL、定时报表——直接写代码，agent 是 over-engineering。
- **离线 batch processing，无需对话。** Tool Use 的价值在交互式动态决策。批处理直接写 Python。

## 5.10 与其他模式的关系

- **Tool Use ⊂ Agent loop（Ch.5 + Ch.4 + Ch.6 + Ch.11）：** ReAct loop = Tool Use + Reasoning Techniques + Goal Setting 的最小组合。
- **Tool Use → MCP（Ch.10）：** MCP 是 Tool Use 的协议标准化，让 tools 跨厂商可移植。
- **Tool Use → Skills（Ch.9）：** Skills = "tool with full instruction bundle"，是 Tool Use 在 lifelong learning 方向的延伸。
- **Tool Use → Multi-Agent（Ch.7）：** Agent-as-Tool（ADK 的 `AgentTool`、OpenAI 的 handoff）把"调子 agent"用 tool 协议统一起来。
- **Tool Use ↔ Guardrails（Ch.18）：** 高危 tool 必须配 risk rating + HITL escalation。
- **Tool Use ↔ Evaluation（Ch.19）：** A3 的核心建议是"评估驱动 tool 设计"——离线 eval 跑 tool selection accuracy、tool call count、error rate，这些数字直接指导你改 tool 描述、改 schema、改 namespace。

## 5.11 小结：Tool Use 的进化方向

把整章压缩为一条 timeline：

1. **2022 ReAct [B/R1]：** Tool Use 找到了 agent loop 这个最小架构。
2. **2022-2023 Toolformer [B/R5]：** 证明 Tool Use 可以 self-supervised 训进权重。
3. **2023 ChatGPT plugins / OpenAI function calling：** Tool Use 协议化，进入产品。
4. **2024 MCP [A/A3, C/A4]：** Tool 跨厂商化，client-server 标准协议。
5. **2025-11 Code Execution with MCP [C/A4]：** 范式从"sequential tool calls"转向"agent writes code"，token 效率 98.7% 提升。
6. **2025-12 Agent Skills [C/A5]：** Filesystem-as-capability，lifelong learning 不再需要 fine-tune。

接下来五年的核心赌注是：**capability lives on the filesystem (Skills), composed via code (Code Mode), exposed via protocol (MCP), grounded by ACI design (A3).** 整本手册的 Tool Use 章节如果只能留一段，就是这一段。

> "Put yourself in the model's shoes. Is it obvious how to use this tool, based on the description and parameters? Or would it require careful thought?" — Anthropic [A/A1]

这一句话是 Tool Use 章节的 north star。每次写新 tool 时把它默念一遍，比读完整本 A3 还有用。
