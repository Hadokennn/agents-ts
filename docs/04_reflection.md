# Ch.4 Reflection 反思

> "What if you automate the step of delivering critical feedback, so the model automatically criticizes its own output and improves its response? This is the crux of Reflection." — Andrew Ng [A/N1]

## 4.1 定义

**Reflection** 让 agent（或一对 agent）批评自己的输出并迭代改进。这是 2024 年 Andrew Ng 提出"4 大 agentic design patterns"中最早被产品化的一个 [A/N1]，也是历史上被多次重命名的同一个模式：

| 名字 | 出处 | 强调点 |
|---|---|---|
| Reflection | Andrew Ng [A/N1] / Schmid [A/P1] | 普及术语 |
| Evaluator-Optimizer | Anthropic [A/A1] | 工作流视角，两个 LLM 角色 |
| Self-Refine | Madaan et al. NeurIPS 2023 [B/R6] | 单 LLM 三角色 |
| Reflexion | Shinn et al. 2023 [B/R2] | 加 episodic memory 的"verbal RL" |
| CRITIC | Gou et al. 2024 | 加外部 verifier 的版本 |

本章把它们统一在 Reflection 这把伞下，但**显式区分三种典型实现**——它们的成本、强度、失败模式都不同。

**核心结构**永远是同一个 loop：

```text
generate → critique → (PASS?)
              │
              └─ no → refine → critique → ...
```

退出条件是 `PASS` 信号、达到 max iterations、或改进幅度低于阈值。

## 4.2 动机：人类写作的迭代过程

A1 的描述是：

> "This is analogous to the iterative writing process a human writer might go through when producing a polished document." [A/A1]

没有人一次写出最终版。第一稿大概率有错字、有结构问题、有论证 gap。人类的工作流是 draft → 自己读一遍 → 改 → 给同事看 → 再改。Reflection 是把这条 loop 编码进 agent。

Ng 的原话更直接：

> "I've been delighted by how much it improved my applications' results." [A/N1]

注意他用了 "delighted"。Reflection 是"低门槛、高收益"模式的代表——加 30 行代码，准确率经常能跳 10-50pp。下面三个数字是本章的核心证据：

- **Self-Refine [B/R6] 在 GPT-4 Dialogue Response 上：25.4% → 74.6%（+49.2pp）** 偏好度。
- **Reflexion [B/R2] 在 HumanEval pass@1：91.0%**，对比 GPT-4 baseline 80.1%，对比之前 SOTA（Codex+CodeT GPT-3.5）65.8%。+11pp。
- **Self-Refine 在数学推理上：+0.2pp**——几乎无效。这条反例和上面两条一样重要。

最后这一条是本章最需要警示的事情。Reflection 不是万能。当模型**识别不出自己错的地方**时，整个 loop 退化为 no-op 或更糟。Madaan et al. 报告：ChatGPT 在 GSM8K 上 94% 的 self-feedback 是 "everything looks good" [B/R6]。这就是 **self-evaluation blindness**，4.6 节会展开。

## 4.3 三种实现

Reflection 的三种典型实现按"角色分离度"递增排列。

### 4.3.1 实现 A：单 LLM 三角色（Self-Refine）

Madaan et al. NeurIPS 2023 [B/R6] 的 Self-Refine 是最朴素的版本：同一个 frozen LLM 通过三个 few-shot prompt 扮演三个角色——

- **Generator** `p_gen`：给定任务 x，产出初稿 y₀。
- **Feedback provider** `p_fb`：批评 y_t，产出 specific + actionable 的 critique fb_t。
- **Refiner** `p_refine`：消费完整历史 `x ∥ y₀ ∥ fb₀ ∥ ... ∥ y_t ∥ fb_t`，写出 y_{t+1}。

```python
def self_refine(x, max_iter=4):
    y = llm(p_gen + x)                            # 初稿
    history = [(y, None)]
    for t in range(max_iter):
        fb = llm(p_fb + x + y)                    # 评审
        if "APPROVED" in fb or stop_signal(fb):
            break
        prefix = "".join(f"{yi}\n{fbi}\n" for yi, fbi in history)
        y = llm(p_refine + x + prefix + fb)       # 修订
        history.append((y, fb))
    return y
```

无需训练、无需多模型、无需 RL，只需要三个 prompt 模板。

**关键设计：feedback prompt 必须给出 specific + actionable 的 critique。** 这是 Self-Refine 的 load-bearing component。Madaan et al. 的 ablation：

- 用 generic feedback（"this could be better"）→ 大部分增益消失。
- 完全去掉 feedback，只重新采样 → 增益归零（Sentiment Reversal 任务 43.2 → 31.2 → 0）。

具体的 feedback prompt 例子（code optimization 任务）：

```text
# p_fb 的 few-shot block
Input code:
def sum(n):
    res = 0
    for i in range(n+1):
        res += i
    return res
Feedback:
This code is slow as it uses a for loop which is brute force. A better
approach is to use the formula sum = n*(n+1)/2 which runs in O(1).
```

注意 feedback 不是 "this is slow"，而是 "use the formula `n*(n+1)/2`"——具体到能让 refiner 直接照做。

**Self-Refine 的硬数据 [B/R6]（GPT-4，平均 7 个任务）：**

- Dialogue Response: 25.4 → 74.6（+49.2pp）
- Sentiment Reversal: 3.8 → 36.2（+32.4pp）
- Code Optimization: 27.3 → 36.0（+8.7pp）
- Constrained Generation（30 concept keywords）: 15 → 45（+30pp）
- **Math GSM8K: 92.9 → 93.1（+0.2pp）** ← 这是本章的警示数字

迭代次数 vs 增益：单调上升但 diminishing returns。Constrained Generation 任务 y₀=29.0 → y₁=40.3（+11.3）→ y₂=46.1（+5.8）→ y₃=49.7（+3.6）。第一轮提升最大，往后 marginal。

### 4.3.2 实现 B：Generator + Critic 两 agent（Evaluator-Optimizer）

A1 的 *Evaluator-Optimizer* workflow [A/A1] 把生成者和评审者**拆成两个 LLM**——通常用不同 prompt，可以选不同模型（critic 可以贵一些、generator 可以便宜一些，或反过来）。

```python
def evaluator_optimizer(task, max_iter=4):
    draft = generator_llm(generator_prompt + task)
    for _ in range(max_iter):
        verdict = critic_llm(critic_prompt + task + draft)
        if verdict.approved:
            return draft
        draft = generator_llm(
            generator_prompt + task +
            f"\nPrevious draft:\n{draft}\n" +
            f"\nFeedback:\n{verdict.feedback}\n" +
            "Rewrite the draft addressing the feedback."
        )
    return draft  # 兜底返回最后一稿
```

A1 给出适用的两个前提条件：

1. **Human feedback demonstrably improves the response.**（人类反馈能改进 → 否则 critic 也帮不上）
2. **The LLM can produce feedback of that quality.**（模型能给出那个质量的反馈 → 否则 critic 是噪音）

这两个前提缺一不可。第 4.6 节的 self-evaluation blindness 就是 #2 被破坏的情形。

**Generator vs Critic 拆开的好处：**

- Critic prompt 可以专注 "找问题"，generator prompt 专注 "造内容"——单个 prompt 不必既会写又会改。
- 可以用更便宜的 critic（多数 review 不需要 frontier model）或更贵的 critic（让 reasoning model 当裁判）。
- 可观测性更好：两个 LLM call 分别打 trace，能精确知道是 generator 弱还是 critic 弱。

### 4.3.3 实现 C：Actor + Evaluator + Self-Reflection LM with episodic memory（Reflexion）

Shinn et al. 2023 [B/R2] 的 Reflexion 把 Reflection 从单轮 task 升级为**多 trial 的 "verbal reinforcement learning"**：不更新权重，而是更新一个文本形式的 episodic memory。

三个 LM 协作：

- **Actor**（如 ReAct 或 CoT agent）：执行任务，产出 trajectory τ_t。
- **Evaluator**：给 trajectory 打分 r_t。QA 用 exact match，决策任务用 heuristic 或 LLM judge，代码用 self-generated unit tests。
- **Self-Reflection LM**：把 `{τ_t, r_t}` 转成第一人称 critique sr_t——"我这次做错了什么、下次该怎么做"。

```python
def reflexion(task, actor, max_trials=10, Omega=3):
    memory = []   # 长期：文字反思，滑动窗口 capped at Omega
    for trial in range(max_trials):
        trajectory = actor.run(task, memory=memory)         # ReAct / CoT
        reward = evaluator.score(trajectory)                # binary / scalar
        if reward == PASS:
            return trajectory
        reflection = self_reflection_lm(task, trajectory, reward)
        memory.append(reflection)
        if len(memory) > Omega:
            memory.pop(0)                                   # 丢最旧
        environment.reset()                                 # 新 trial
    return None  # 放弃
```

具体的 reflection 例子（ALFWorld 任务，引自 [B/R2]）：

> "In this environment, my plan was to find a mug then find and use a desklamp. However, the task says to examine the mug with the desklamp. I should have looked for the desklamp first, then looked for the mug. In the next trial, I will go to desk 1, find the lamp, then look for the mug."

这条 reflection 会被 prepend 到下一次 trial 的 prompt 里，actor 据此调整策略。

**Reflexion 的硬数据：**

- **HumanEval Python pass@1: 91.0%**（Reflexion + GPT-4）vs 80.1% GPT-4 baseline vs 65.8% 此前 SOTA。+11pp。
- **ALFWorld**: 130/134 任务在 12 trials 内解决；ReAct baseline 在 trial 6 后 plateau 在 ~75%。
- **HotpotQA**: +20pp absolute over CoT/ReAct baselines（这两者多次 retry 也没改进）。
- **Leetcode Hard**: 15.0% vs 7.5% GPT-4 baseline。

**Ablation 是关键证据：**去掉 verbal self-reflection 但保留 retry → 性能跌回 baseline（60% vs 68%）。**真正起作用的是反思本身，不是单纯的重试。**

## 4.4 何时 PASS / 何时停

退出条件设计是 Reflection 工程化的关键。三种通行方案：

1. **Hard signal**：Critic 输出包含特定 sentinel（`APPROVED`、`PASS`、`<done>`）。简单但要求 critic 严格遵守格式。
2. **Score threshold**：Critic 输出 1-10 评分，>= 8 即可。容忍格式变化但需要 critic 校准。
3. **Δ-based stopping**：连续两轮改进幅度 < ε → 停。最稳健但需要可比较的 metric。

无论哪种方案都必须配 `max_iter` 兜底。Self-Refine 论文 [B/R6] 用的是 max_iter = 4，因为绝大多数 task 第 3-4 轮已经收敛，再迭代是浪费。

## 4.5 框架对比

| 框架 | 一等原语 | 怎么写 | 关键差异 |
|---|---|---|---|
| **LangGraph** | Evaluator-Optimizer（conditional edge loop） | Generator 节点 + Critic 节点，conditional edge：`approved` → END，`revise` → loop 回 Generator | 用图边构造循环，没有专门的 loop 容器，由你自己接环 |
| **AutoGen (v0.4+)** | Reflection（pub-sub Coder/Reviewer） | `CoderAgent` 发布 `CodeReviewTask` → `ReviewerAgent` 发布 `CodeReviewResult(approved=bool)` → false 则 coder 重生成 | 循环从 pub-sub 协议中"涌现"，agents 互相 republish 直到 reviewer 返回 APPROVE |
| **CrewAI** | **没有 first-class primitive** | 用 `Process.sequential` 链一个 generator task + 一个 reviewer task；reviewer 拒绝时由 app code 手动重跑 crew | 框架级别**没有 loop 原语**，必须在外层 Python 里手动 retry |
| **Google ADK** | `LoopAgent` with exit_condition | Generator 和 Critic 是 `LoopAgent` 的 sub_agents；当 critic 写 `PASS` 到 `condition_key` 时退出 | **唯一把 Reflection 抽象为一等容器的框架** |

[源：D Pattern 4]

四家对比的核心 takeaway：

- LangGraph 把 reflection 表达为**图里的一个回边**——灵活，但你要自己想清楚什么时候断环。
- AutoGen 把 reflection 表达为**pub-sub 消息回流**——分布式友好，但调试 cascading message 很痛。
- CrewAI **没有**——这是 CrewAI 的一个明显短板。如果你的核心模式是 reflection，别用 CrewAI。
- ADK 是**最直接的**——`LoopAgent(sub_agents=[gen, critic], condition_key="feedback", exit_condition="PASS", max_iterations=4)` 一行表达整个模式。

### 4.5.1 框架示例：ADK 写法

```python
from google.adk import LlmAgent, LoopAgent

generator = LlmAgent(
    name="Generator",
    instruction="Write a draft response to the user's question. Output ONLY the draft.",
    output_key="draft",
)

critic = LlmAgent(
    name="Critic",
    instruction=(
        "Review the draft in {draft}. If the draft is high quality and "
        "addresses the user's question, output exactly 'PASS' to feedback. "
        "Otherwise, output specific actionable feedback to improve the draft."
    ),
    output_key="feedback",
)

refinement_loop = LoopAgent(
    name="ReflectionLoop",
    sub_agents=[generator, critic],
    condition_key="feedback",
    exit_condition="PASS",
    max_iterations=4,
)
```

ADK 把"循环 + 退出条件 + 最大迭代"打包成一个容器。其他框架要达到同样表达力需要 20-50 行胶水代码。

## 4.6 坑（本章最重要的部分）

### 坑 1：Self-evaluation blindness（最致命）

**症状：** Self-Refine 在 GSM8K 上 +0.2pp [B/R6]。94% 的 ChatGPT self-feedback 是 "everything looks good"——但答案明明是错的。

**根因：** 当模型**没有能力识别错误**时（典型如 math reasoning、复杂代码的 edge case），critic 等于一个永远说 OK 的橡皮图章。Reflection loop 退化成"生成一稿 → 自己说没问题 → 返回错答案"。

Madaan et al. 对失败案例的归因：
- 33% 是 wrong error localization（找错了位置）
- 61% 是 inappropriate suggested fix（建议改的方向不对）
- 只有 6% 是 refiner 没正确执行 [B/R6]

**61% 的失败发生在 feedback 阶段，不是 refine 阶段。** Feedback 是瓶颈。

**对策：**

1. **用外部 verifier 替代 LLM critic.** 数学用 calculator + sympy；代码用 unit test + linter；事实用 search + 引用比对。这些是 *deterministic* 的，不会自欺。
2. **强化 critic prompt：** 显式列出 N 个常见错误模式，让 critic 逐项检查。
3. **Multi-critic ensemble：** 几个不同 prompt 的 critic 投票（参考 Pattern 3 Voting）。任何一个说有问题就 reject。
4. **判断模型是否有能力 critic 这件事：** 把 critic prompt 单独跑 N 个**已知错**和**已知对**的样本，看 precision / recall。如果 precision < 0.7 或 recall < 0.7，这个 critic 不能用。

A1 [A/A1] 把这条预条件写得很明确：

> "Two preconditions for fit: human feedback demonstrably improves the response, AND the LLM can produce feedback of that quality."

第二条不满足时，Reflection **必然失败**。这不是"调调 prompt 就能修"——这是 capability gap。

### 坑 2：Generic feedback

**症状：** Critic 输出 "this could be improved"、"good but could be better"、"the structure is okay"。Refine 后没变化。

**根因：** Critic prompt 没有强制要求 specific + actionable。

**对策：** Critic prompt 显式要求结构化输出：

```text
Review the draft. Output JSON with:
- "issues": list of {"location": "...", "problem": "...", "fix": "..."}
  Each issue must specify exactly which sentence/line/section is problematic
  AND a concrete change to make. Do not output vague comments like "could be
  improved" — that is not actionable.
- "verdict": "PASS" if no issues found, otherwise "REVISE".
```

Self-Refine 论文的 prompt 把 feedback 直接示范成 "use the formula `n*(n+1)/2`"——具体到能照做。

### 坑 3：Cost explosion

**症状：** 单 turn 1k token 的任务，3 轮 reflection 后烧到 8-10k token。

**根因：** Self-Refine 的 prefix 是 `x ∥ y₀ ∥ fb₀ ∥ ... ∥ y_t ∥ fb_t`——历史拼接，上下文线性增长。

**对策：**

1. **滑动窗口：** Reflexion 只保留最近 Omega=1-3 条反思，老的丢弃。
2. **只保留 critique，不保留所有 draft：** 只把最新 draft + 累积 critique 拼进去。
3. **Critic 输出限定长度：** `"Output at most 3 issues."`
4. **设置 max_iter = 3-4，** 不要无限循环。

Anthropic A1 [A/A1] 的判断：

> "Be sure the benefits outweigh these costs."

一般经验：第 1 轮 → 增益最大；第 2 轮 → 半数 task 还有显著增益；第 3-4 轮 → diminishing returns；> 4 轮 → 几乎不再改进，纯烧 token。

### 坑 4：Repetitive loops（生成器改不动）

**症状：** Generator 输出 v1，critic 说有问题，generator 输出 v2 跟 v1 几乎一样，critic 再说有问题，无限循环。

**根因：** Generator 没拿到足够强的修订信号——可能是 critic feedback 不具体（坑 2），也可能是 generator 真的不知道怎么改（capability gap）。

**对策：**

1. **Δ 检测：** 计算前后两稿的 edit distance / 语义相似度，太接近就直接 break + 报警。
2. **强制变化：** 在第二轮 generator prompt 里加 `"Your previous draft was rejected. You MUST make at least three substantive changes."`
3. **Critic feedback 升级：** 不只指出问题，直接给替代方案。
4. **HITL escalation：** 第 3 轮还没 PASS → 跳出 loop，交给人。

### 坑 5：被烂 critic 带偏

**症状：** Critic 自己有错（比如错把对的当错），refine 越改越糟。

**根因：** Critic 的输出没经过 sanity check，generator 全盘吸收。

**对策：**

1. **Critic 自查：** 加一道 "Are you confident in this feedback? Why?" 的 self-justification。低置信度的 feedback 不采纳。
2. **Generator 反推权：** 在 refine prompt 里加 `"If you disagree with the feedback, explain why and ignore it."` Madaan et al. 的实验显示，强模型（GPT-4）在 generator 角色下会主动 push back 错的 critique。
3. **Critic 用更强模型：** 通常 critic 比 generator 更难。把 critic 升到更高规格模型（如 generator 用 Sonnet，critic 用 Opus）。

### 坑 6：把 Reflection 用在不该用的地方

**症状：** 单轮翻译任务套 Reflection loop，4 轮后翻译质量一样。

**对策：** 见 4.7 何时不该用。

## 4.7 何时不该用

- **任务有客观正确答案且可以用代码验证。** 直接 verifier loop（test-driven generation）——确定性、无 self-blindness、可解释。Reflexion 在代码任务里之所以效果好，恰恰是因为它用了 unit test 作为 evaluator，不是 LLM critic。
- **任务延迟敏感。** Reflection 每轮 = 一次或两次额外 LLM call，3-4 轮 = 3-8× 延迟。客服首响、实时翻译等场景不行。
- **任务模型已经能稳定做好。** 模型基础能力够强时，Reflection 边际收益小。GPT-4 写"Hello World"不需要 critic。
- **Critic capability gap 存在。** 详见 4.6 坑 1。模型识别不出错的领域，Reflection 是 self-deception。
- **Generator-critic 同一个 capability 短板。** 如果你用同一个模型扮演两个角色，且这个模型在该任务上系统性地犯某类错——它当 critic 时也识别不出（这是 Self-Refine 在数学上失败的本质）。换成更强模型 / 不同模型 / 外部 verifier。

## 4.8 与其他模式的关系

- **Reflection ⊃ Reasoning Techniques（Ch.17）：** Self-evaluation 本质是 reasoning over reasoning。Tree of Thoughts [B/R4] 的 state evaluator 就是 reflection 的特例。
- **Reflection → Memory Management（Ch.8）：** Reflexion 的 episodic memory 是 reflection 的产物被存储为 long-term memory 的典型例子。
- **Reflection → Learning（Ch.9）：** Reflexion 被定位为 "verbal reinforcement learning"——反思就是 learning signal（无梯度版）。
- **Reflection → Guardrails（Ch.18）：** Critic agent 可以兼任 safety reviewer——同一个 loop，多个判据。
- **Reflection ↔ Tool Use（Ch.5）：** 当 critic 调用外部 verifier（unit test、calculator、search）时，Reflection + Tool Use 合体——这是 CRITIC 范式，也是修复 self-evaluation blindness 的标准答案。
- **Reflection vs Planning（Ch.6）：** Reflection 是事后修订；Planning 是事前分解。两者经常组合：先 plan → 执行 → reflect on results → replan。

## 4.9 小结

Reflection 是 agent 设计里最容易上手、最容易出效果、也最容易翻车的模式。三个要点压缩为一段：

1. **它低门槛、高收益。** 加 30 行代码（generator + critic + loop），在合适任务上经常能跳 20-50pp。这是 Andrew Ng 把它列为"4 patterns 之一"的原因。
2. **它的失败模式集中在 critic。** Self-evaluation blindness 是根本性的——模型识别不出自己错的领域，Reflection 退化为橡皮图章。Self-Refine 数学任务 +0.2pp 是最强的反例 [B/R6]。
3. **修复 critic 的最可靠手段是外部 verifier。** 数学用 calculator、代码用 unit test、事实用 search——deterministic verifier 不会自欺。Reflexion 在代码上拿到 91.0% pass@1 [B/R2] 就是因为它用了 unit test 作为 ground truth signal。

最后一句话送给生产环境的工程师：

> "Specific and actionable feedback is the load-bearing component." — Self-Refine [B/R6]

Critic prompt 写得糙，Reflection 等于不开。把 critic prompt 当作和 generator prompt 一样重要的对象去 eval、去迭代——这是把 Reflection 做对的全部秘密。
