import 'dotenv/config';
import { type ModelMessage } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { createMockModel } from './mock/mock-model.js';
import { ToolRegistry } from './tools/registry.js';
import { allTools } from './tools/index.js';
import { agentLoop, type BudgetState } from './agent/loop.js';
import { createInterface } from 'node:readline';
import { UsageTracker } from './usage/tracker.js';
import { HandwrittenMCPClient, SDKMCPClient, MockMCPClient } from './tools/mcp-client.js';
import { ToolDefinition } from './tools/index.js';
import { SessionStore } from './session/store.js';
import { PromptContext, PromptBuilder, coreRules, toolGuide, sessionContext, deferredTools, strategies } from './context/prompt-builder.js';
import { microcompact, summarize, estimateTokens } from './context/compressor.js';
import { injectFakeHistory } from './mock/fake.js';
import { applyDefense, estimateMessageTokens } from './context/defense.js';
import { buildContextSnapshot, renderContextView, renderUsageView } from './context/view.js';

let messages: ModelMessage[] = [];

// 预算由调用方持有，跨轮持续累计——agentLoop 只负责消费它
const budget: BudgetState = { used: 0, limit: 15000 };
const tracker = new UsageTracker('.usage/today.jsonl');

// model 初始化选择
const deepseek = createOpenAI({
  baseURL: 'https://api.deepseek.com',
  apiKey: process.env.DEEPSEEK_API_KEY,
});

const model = process.env.DEEPSEEK_API_KEY
  ? deepseek.chat('deepseek-v4-flash')
  : createMockModel();

// tools 注册
const registry = new ToolRegistry();
const toolSearchTool: ToolDefinition = {
  name: 'tool_search',
  description: '获取延迟工具的完整定义。传入工具名（从系统提示的延迟工具列表中选取），返回该工具的完整参数 Schema',
  parameters: {
    type: 'object',
    properties: {
      query: { type: 'string', description: '工具名，如 "mcp__github__list_issues"。支持逗号分隔多个工具名' },
    },
    required: ['query'],
    additionalProperties: false,
  },
  isConcurrencySafe: true,
  isReadOnly: true,
  execute: async ({ query }: { query: string }) => {
    const results = registry.searchTools(query);
    if (results.length === 0) return `没有找到匹配 "${query}" 的工具`;
    console.log(`  找到匹配 "${query}" 的工具: ${results.map(t => t.name).join(', ')}`);
    return results.map(t => ({
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    }));
  },
};
registry.register(...allTools, toolSearchTool);

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

const rl = createInterface({
  input: process.stdin,
  output: process.stdout,
});


/** 
 * 这个压缩太简陋了，怎么简陋的？
 * 
 **/
async function autoCompact(messages: ModelMessage[], summary: string) {
  let _messages = messages.slice();
  // Check if compaction needed after each turn
  const currentTokens = estimateTokens(_messages);
  if (currentTokens > 4000) {
    console.log(`\n  [压缩检查] ~${currentTokens} tokens, 触发压缩...`);
    const mc = microcompact(_messages);
    _messages = mc.messages;
    if (mc.cleared > 0) console.log(`  [Microcompact] 清理了 ${mc.cleared} 个工具结果`);

    const comp = await summarize(model, _messages, summary);
    if (comp.compressedCount > 0) {
      _messages = comp.messages;
      summary = comp.summary;
      console.log(`  [Summarization] 压缩了 ${comp.compressedCount} 条消息, ~${estimateTokens(_messages)} tokens`);
    }
  }
  return { messages: _messages, summary };
}

async function main() {
  await connectMCP();

  // Session 持久化
  const isContinue = process.argv.includes('--continue');
  const sessionId = 'default';
  const store = new SessionStore(sessionId);
  const timestamps = new Map<number, number>();

  if (isContinue && store.exists()) {
    messages = store.load();
    const firstTokens = estimateTokens(messages);
    console.log(`[Session] 恢复会话，${messages.length} 条历史消息， ~${firstTokens} tokens\n`);
  } else {
    // injectFakeHistory(messages);
    console.log(`[Session] 新会话\n`);
  }

  // Apply three-layer defense
  const beforeTokens = estimateMessageTokens(messages);
  console.log(`\n=== 三层即时防线 ===`);
  console.log(`[防线前] ${messages.length} 条消息, ~${beforeTokens} tokens`);

  const defense = applyDefense(messages, timestamps);
  messages = defense.messages;
  console.log(`[Layer 2: 截断] ${defense.truncated} 个超长结果被截断，${defense.compacted} 个消息被压缩到 ~${defense.tokenEstimate} tokens·`);
  console.log(`[Layer 3: TTL] ${defense.softPruned} 个软修剪, ${defense.hardPruned} 个硬清除`);
  console.log(`[防线后] ${messages.length} 条消息, ~${defense.tokenEstimate} tokens (节省 ${beforeTokens - defense.tokenEstimate})`);
  console.log(`====================\n`);


  let summary = '';
  // Layer 1: Microcompact
  const { messages: compactedMessages, summary: compactedSummary } = await autoCompact(messages, summary);
  messages = compactedMessages;
  summary = compactedSummary;

  const deferredSummary = registry.getDeferredToolSummary();
  const promptCtx: PromptContext = {
    toolRegistry: registry,
    deferredToolSummary: deferredSummary,
    sessionMessageCount: messages.length,
    sessionId,
  };
  const builder = new PromptBuilder()
    .pipe('coreRules', coreRules())
    .pipe('toolGuide', toolGuide())
    .pipe('deferredTools', deferredTools())
    .pipe('strategies', strategies())
    .pipe('sessionContext', sessionContext());

  const SYSTEM = builder.build(promptCtx);
  console.log(`\n=== SYSTEM PROMPT ===`);
  console.log(SYSTEM);
  console.log(`=== SYSTEM PROMPT ===\n`);
  builder.debug(promptCtx);  // 显示各模块状态

  const allCount = registry.getAll().length;
  const activeTools = registry.getActiveTools();
  const estimate = registry.countTokenEstimate();

  console.log(`\n=== 工具统计 ===`);
  console.log(`  全部工具: ${allCount} 个`);
  console.log(`  活跃工具: ${activeTools.length} 个`);
  console.log(`  延迟工具: ${allCount - activeTools.length} 个`);
  console.log(`  Token 估算: ~${estimate.active} (活跃) + ~${estimate.deferred} (延迟，不占 prompt)`);
  // Quick triggers for demo
  function handleQuickTrigger(cmd: string): boolean {
    const now = Date.now();

    if (cmd === '/status' || cmd === 'status') {
      const tokens = estimateMessageTokens(messages);
      const toolMsgs = messages.filter(m => m.role === 'tool').length;
      console.log(`\n[状态] ${messages.length} 条消息 (${toolMsgs} 条工具结果), ~${tokens} tokens\n`);
      return true;
    }

    // /context: 终端可视化的 context 占用，参考 Claude Code 的 /context
    if (cmd === '/context' || cmd === 'context') {
      const snapshot = buildContextSnapshot({
        modelName: process.env.DEEPSEEK_API_KEY ? 'Deepseek V4 Flash' : 'Mock Model (开发用)',
        modelId: process.env.DEEPSEEK_API_KEY ? 'deepseek-v4-flash' : 'mock-model',
        windowTokens: 1_000_000,
        systemPromptChars: SYSTEM.length,
        toolDescriptionChars: registry.getActiveTools().reduce((a, t) => a + t.name.length + (t.description?.length || 0) + JSON.stringify(t.parameters || {}).length, 0),
        memoryChars: 0,
        skillsChars: 0,
        messages,
      });
      console.log(renderContextView(snapshot));
      return true;
    }

    if (cmd === '/usage' || cmd === 'usage') {
      console.log(renderUsageView(tracker));
      return true;
    }

    return false;
  }

  function ask() {
    rl.question('\nYou: ', async (input) => {
      const trimmed = input.trim();
      if (!trimmed || trimmed === 'exit') {
        console.log('Bye!');
        rl.close();
        return;
      }

      if (handleQuickTrigger(trimmed)) {
        ask();
        return;
      }

      // Session 持久化
      const userMsg: ModelMessage = { role: 'user', content: trimmed };
      messages.push(userMsg);
      store.append(userMsg);

      const beforeLen = messages.length;

      console.log('\n--- 执行三层防线 ---');
      const before = estimateMessageTokens(messages);
      const def = applyDefense(messages, timestamps);
      console.log(`\n=== 三层即时防线 ===`);
      messages = def.messages;
      console.log(`  [Layer 2] 截断: ${def.truncated} 条, 预算清理: ${def.compacted} 条`);
      console.log(`  [Layer 3] 软修剪: ${def.softPruned}, 硬清除: ${def.hardPruned}`);
      console.log(`  [结果] ~${before} → ~${def.tokenEstimate} tokens (节省 ${before - def.tokenEstimate})\n`);
      console.log(`====================\n`);

      await agentLoop(model, registry, messages, SYSTEM, budget, tracker);

      // 持久化本轮新增的消息（agent loop 会往 messages 里 push assistant/tool 消息）
      const newMessages = messages.slice(beforeLen);
      store.appendAll(newMessages);

      const { messages: compactedMessages, summary: compactedSummary } = await autoCompact(messages, summary);
      messages = compactedMessages;
      summary = compactedSummary;

      ask();
    });
  }

  console.log('Super Agent v0.7 — Session + Prompt Pipe (type "exit" to quit)');
  console.log('对话会自动保存。用 pnpm run continue 恢复上次对话。\n');
  ask();
}

main().catch(console.error);
