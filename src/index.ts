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

let messages: ModelMessage[] = [];

const deepseek = createOpenAI({
  baseURL: 'https://api.deepseek.com',
  apiKey: process.env.DEEPSEEK_API_KEY,
});

const model = process.env.DEEPSEEK_API_KEY
  ? deepseek.chat('deepseek-v4-flash')
  : createMockModel();


const rl = createInterface({
  input: process.stdin,
  output: process.stdout,
});

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

// 预算由调用方持有，跨轮持续累计——agentLoop 只负责消费它
const budget: BudgetState = { used: 0, limit: 15000 };
const tracker = new UsageTracker('.usage/today.jsonl');


async function main() {
  await connectMCP();

  // Session 持久化
  const isContinue = process.argv.includes('--continue');
  const sessionId = 'default';
  const store = new SessionStore(sessionId);
  if (isContinue && store.exists()) {
    messages = store.load();
    console.log(`[Session] 恢复会话，${messages.length} 条历史消息\n`);
  } else {
    console.log(`[Session] 新会话\n`);
  }

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

  console.log(`共注册 ${registry.getAll().length} 个工具：`);
  for (const tool of registry.getAll()) {
    const flags = [
      tool.isConcurrencySafe ? '可并发' : '串行',
      tool.isReadOnly ? '只读' : '读写',
    ].join(', ');
    console.log(`  - ${tool.name}（${flags}）`);
  }

  const allCount = registry.getAll().length;
  const activeTools = registry.getActiveTools();
  const estimate = registry.countTokenEstimate();

  console.log(`\n=== 工具统计 ===`);
  console.log(`  全部工具: ${allCount} 个`);
  console.log(`  活跃工具: ${activeTools.length} 个`);
  console.log(`  延迟工具: ${allCount - activeTools.length} 个`);
  console.log(`  Token 估算: ~${estimate.active} (活跃) + ~${estimate.deferred} (延迟，不占 prompt)`);

  function ask() {
    rl.question('\nYou: ', async (input) => {
      const trimmed = input.trim();
      if (!trimmed || trimmed === 'exit') {
        console.log('Bye!');
        rl.close();
        return;
      }

      // Session 持久化
      const userMsg: ModelMessage = { role: 'user', content: trimmed };
      messages.push(userMsg);
      store.append(userMsg);

      const beforeLen = messages.length;
      await agentLoop(model, registry, messages, SYSTEM, budget, tracker);

      // 持久化本轮新增的消息（agent loop 会往 messages 里 push assistant/tool 消息）
      const newMessages = messages.slice(beforeLen);
      store.appendAll(newMessages);

      ask();
    });
  }

  console.log('Super Agent v0.7 — Session + Prompt Pipe (type "exit" to quit)');
  console.log('对话会自动保存。用 pnpm run continue 恢复上次对话。\n');
  ask();
}

main().catch(console.error);
