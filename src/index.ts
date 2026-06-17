import 'dotenv/config';
import { type ModelMessage } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { createMockModel } from './mock/mock-model.js';
import { ToolRegistry } from './tools/registry.js';
import { allTools } from './tools/index.js';
import { agentLoop, type BudgetState } from './agent/loop.js';
import { createInterface } from 'node:readline';
import { UsageTracker, estimateMessageTokens, compactTriggerTokens } from './usage/tracker.js';
import { HandwrittenMCPClient, SDKMCPClient, MockMCPClient } from './tools/mcp-client.js';
import { SessionStore } from './session/store.js';
import { PromptContext, PromptBuilder, coreRules, toolGuide, sessionContext, deferredTools, strategies } from './context/prompt-builder.js';
import { microcompact, summarize } from './context/compressor.js';
import { createToolSearchTool } from './tools/tool-search.js';
import { applyDefense } from './context/defense.js';
import { MemoryStore } from './memory/store.js';
import { createMemoryTool } from './tools/memory-tools.js';
import { createDispatcher, type CommandContext } from './commands/index.js';
import { debugCommands } from './commands/debugger.js';
import { contextCommands } from './commands/context.js';
import { memoryCommands } from './commands/memory.js';
import { VectorStore } from './rag/store.js';
import { SqliteVectorStore } from './rag/sqlite-store.js';
import { createMockEmbedder, createDashScopeEmbedder, embed } from './rag/embedder.js';
import { createRagTools } from './tools/rag-tools.js';
import { memoryContext, ragContext } from './context/prompt-pipes.js';

// 预算由调用方持有，跨轮持续累计——agentLoop 只负责消费它
const budget: BudgetState = { used: 0, limit: 15000 };

// model 初始化选择
const deepseek = createOpenAI({
  baseURL: 'https://api.deepseek.com',
  apiKey: process.env.DEEPSEEK_API_KEY,
});

const model = process.env.DEEPSEEK_API_KEY
  ? deepseek.chat('deepseek-v4-flash')
  : createMockModel();

// token 估算口径按 model 走（见 usage/tracker.ts 的 TOKEN_WEIGHTS）
const MODEL_ID = (model as any)?.modelId ?? 'mock-model';

// ── Tools ────────────────────────────────
const registry = new ToolRegistry();
registry.register(...allTools);
registry.register(createToolSearchTool(registry));

// ── Memory ────────────────────────────────
const memoryStore = new MemoryStore('.');
memoryStore.init();
registry.register(createMemoryTool(memoryStore));

// ── RAG ────────────────────────────────
// const vectorStore = new VectorStore();
const vectorStore = new SqliteVectorStore('knowledge.db');
const embedFn = process.env.DASHSCOPE_API_KEY
  ? createDashScopeEmbedder(process.env.DASHSCOPE_API_KEY)
  : createMockEmbedder();
registry.register(...createRagTools(vectorStore, embedFn));

// MCP 实现选择：'handwritten' 或 'sdk'
const MCP_IMPLEMENTATION = (process.env.MCP_IMPLEMENTATION || 'sdk').toLowerCase() as 'handwritten' | 'sdk';
async function connectMCP() {
  const githubToken = process.env.GITHUB_PERSONAL_ACCESS_TOKEN;

  let canSpawn = true;
  try {
    const { execSync } = await import('node:child_process');
    execSync('echo test', { stdio: 'ignore' });
  } catch {
    canSpawn = false;
  }

  if (githubToken && canSpawn) {
    console.log(`\n连接 GitHub MCP Server (${MCP_IMPLEMENTATION} 实现)...`);
    try {
      const client = MCP_IMPLEMENTATION === 'handwritten'
        ? new HandwrittenMCPClient(
          'npx', ['-y', '@modelcontextprotocol/server-github'],
          { GITHUB_PERSONAL_ACCESS_TOKEN: githubToken },
        )
        : new SDKMCPClient(
          'npx', ['-y', '@modelcontextprotocol/server-github'],
          { GITHUB_PERSONAL_ACCESS_TOKEN: githubToken },
        );

      const tools = await registry.registerMCPServer('github', client);
      console.log(`  已注册 ${tools.length} 个 MCP 工具\n`);
      return;
    } catch (err) {
      console.log(`  MCP 连接失败: ${err instanceof Error ? err.message : err}`);
      console.log('  降级为 Mock MCP...');
    }
  }

  if (!githubToken) {
    console.log('\n未配置 GITHUB_PERSONAL_ACCESS_TOKEN，使用 Mock MCP');
  }

  const mockClient = new MockMCPClient();
  const tools = await registry.registerMCPServer('github', mockClient);
  console.log(`  已注册 ${tools.length} 个 Mock MCP 工具`);
}

// ── Commands ────────────────────────────────
const dispatch = createDispatcher([
  ...debugCommands,
  ...contextCommands,
  ...memoryCommands,
]);

async function autoCompact(messages: ModelMessage[], summary: string) {
  let _messages = messages.slice();
  // 触发线按 model 的 context window 比例派生，不再是魔法数
  const currentTokens = estimateMessageTokens(_messages, MODEL_ID);
  const trigger = compactTriggerTokens(MODEL_ID);
  if (currentTokens > trigger) {
    console.log(`\n  [压缩检查] ~${currentTokens} / ${trigger} tokens (${Math.round(currentTokens / trigger * 100)}%), 触发压缩...`);
    const mc = microcompact(_messages);
    _messages = mc.messages;
    if (mc.cleared > 0) console.log(`  [Microcompact] 清理了 ${mc.cleared} 个工具结果`);

    const comp = await summarize(model, _messages, summary);
    if (comp.compressedCount > 0) {
      _messages = comp.messages;
      summary = comp.summary;
      console.log(`  [Summarization] 压缩了 ${comp.compressedCount} 条消息, ~${estimateMessageTokens(_messages, MODEL_ID)} tokens`);
    }
  }
  return { messages: _messages, summary };
}

async function main() {
  await connectMCP();

  const timestamps = new Map<number, number>();
  let messages: ModelMessage[] = [];
  let summary = '';


  // Session 持久化
  const isContinue = process.argv.includes('--continue');
  const sessionId = 'default';
  const store = new SessionStore(sessionId);

  const tracker = new UsageTracker('.usage/today.jsonl');

  if (isContinue && store.exists()) {
    messages = store.load();
    const firstTokens = estimateMessageTokens(messages, MODEL_ID);
    console.log(`[Session] 恢复会话，${messages.length} 条历史消息， ~${firstTokens} tokens\n`);
  } else {
    console.log(`[Session] 新会话\n`);
  }

  // Apply three-layer defense
  const beforeTokens = estimateMessageTokens(messages, MODEL_ID);
  const defense = applyDefense(messages, timestamps, MODEL_ID);
  messages = defense.messages;
  if (defense.truncated > 0 || defense.compacted > 0 || defense.softPruned > 0 || defense.hardPruned > 0) {
    console.log(`\n=== 三层即时防线 ===`);
    console.log(`[防线前] ${messages.length} 条消息, ~${beforeTokens} tokens`);
    console.log(`[Layer 2: 截断] ${defense.truncated} 个超长结果被截断，${defense.compacted} 个消息被压缩到 ~${defense.tokenEstimate} tokens·`);
    console.log(`[Layer 3: TTL] ${defense.softPruned} 个软修剪, ${defense.hardPruned} 个硬清除`);
    console.log(`[防线后] ${messages.length} 条消息, ~${defense.tokenEstimate} tokens (节省 ${beforeTokens - defense.tokenEstimate})`);
    console.log(`====================\n`);
  }

  // Layer 1: Microcompact
  const { messages: compactedMessages, summary: compactedSummary } = await autoCompact(messages, summary);
  messages = compactedMessages;
  summary = compactedSummary;

  const builder = new PromptBuilder()
    .pipe('coreRules', coreRules())
    .pipe('toolGuide', toolGuide())
    .pipe('deferredTools', deferredTools())
    .pipe('strategies', strategies())
    .pipe('memoryContext', memoryContext(memoryStore))
    .pipe('ragContext', ragContext(vectorStore))
    .pipe('sessionContext', sessionContext());

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  function makePromptCtx(): PromptContext {
    return {
      toolRegistry: registry,
      deferredToolSummary: registry.getDeferredToolSummary(),
      sessionMessageCount: messages.length,
      sessionId,
    };
  }

  function ask() {
    rl.question('\nYou: ', async (input) => {
      const trimmed = input.trim();
      if (!trimmed || trimmed === 'exit') {
        console.log('Bye!');
        rl.close();
        return;
      }

      const ctx: CommandContext = {
        messages, timestamps, registry, builder, tracker,
        sessionStore: store, model, makePromptCtx, ask,
        memoryStore,
      };
      const handled = dispatch(trimmed, ctx);
      if (handled === 'async') return;
      if (handled) { ask(); return; }

      // Session 持久化
      const userMsg: ModelMessage = { role: 'user', content: trimmed };
      messages.push(userMsg);
      store.append(userMsg);

      const beforeLen = messages.length;

      console.log('\n--- 执行三层防线 ---');
      const before = estimateMessageTokens(messages, MODEL_ID);
      const def = applyDefense(messages, timestamps, MODEL_ID);
      messages = def.messages;
      if (def.truncated > 0 || def.softPruned > 0 || def.hardPruned > 0) {
        console.log(`\n=== 三层即时防线 ===`);
        console.log(`  [Layer 2] 截断: ${def.truncated} 条, 预算清理: ${def.compacted} 条`);
        console.log(`  [Layer 3] 软修剪: ${def.softPruned}, 硬清除: ${def.hardPruned}`);
        console.log(`  [结果] ~${before} → ~${def.tokenEstimate} tokens (节省 ${before - def.tokenEstimate})\n`);
        console.log(`====================\n`);
      }

      const currentSystem = builder.build(makePromptCtx());
      await agentLoop(model, registry, messages, currentSystem, budget, tracker);

      // 持久化本轮新增的消息（agent loop 会往 messages 里 push assistant/tool 消息）
      const newMessages = messages.slice(beforeLen);
      const now = Date.now();
      for (let i = beforeLen; i < messages.length; i++) timestamps.set(i, now);
      store.appendAll(newMessages);

      const { messages: compactedMessages, summary: compactedSummary } = await autoCompact(messages, summary);
      messages = compactedMessages;
      summary = compactedSummary;

      ask();
    });
  }

  ask();
}

main().catch(console.error);
