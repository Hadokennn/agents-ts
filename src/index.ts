import 'dotenv/config';
import { type ModelMessage } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { createMockModel } from './mock/mock-model.js';
import { ToolRegistry } from './tools/registry.js';
import { allTools } from './tools/index.js';
import { agentLoop, type BudgetState } from './agent-loop.js';
import { createInterface } from 'node:readline';
import { UsageTracker } from './usage/tracker.js';
import { HandwrittenMCPClient, SDKMCPClient, MockMCPClient } from './mcp-client.js';

const deepseek = createOpenAI({
  baseURL: 'https://api.deepseek.com',
  apiKey: process.env.DEEPSEEK_API_KEY,
});

const model = process.env.DEEPSEEK_API_KEY
  ? deepseek.chat('deepseek-v4-flash')
  : createMockModel();

const registry = new ToolRegistry();
registry.register(...allTools);

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

const SYSTEM = `你是 Super Agent，一个能读代码、抓网页、生成项目的 AI 助手。
你有这些工具可用：read_file, write_file, list_directory, edit_file, glob, grep, bash, fetch_url, start_preview, get_weather, calculator。

针对常见任务的执行策略：

1. 用户让你"分析项目"或"找代码"时：
   先 list_directory 看结构 → grep 定位关键内容 → 必要时 read_file 看细节 → 最后给出归纳总结。

2. 用户给你 URL 时：
   用 fetch_url 抓取（多 URL 可以并行），再综合总结。

3. 用户让你"做一个网页应用 / 待办应用 / 任意 web demo"时（必须实际调用工具，不要只描述）：

   **重要的项目约定（不要自己重写 bootstrap）**：
   - app/index.html 已经预置在模板里，固定用 import maps 引 React + Babel Standalone 实时编译 TSX
   - app/index.html 固定加载 ./App.tsx 作为入口、固定引用 ./styles.css 作为样式
   - 你**禁止**写入或修改 app/index.html（它已经能正确工作）

   **你需要做的事**：
   - 用 write_file 至少生成这三个文件：
     1. app/styles.css — 应用样式
     2. app/App.tsx — **必须**用 \`import { createRoot } from 'react-dom/client'\` 把组件渲染到 \`document.getElementById('root')\`
     3. app/Button.tsx 或其他组件 .tsx — 可被 App.tsx import
   - .tsx 之间用相对路径 import：\`import { Button } from './Button.tsx'\`（必须带 .tsx 后缀）
   - React 用 \`import React, { useState } from 'react'\`，不要从其他源导入
   - 文件全部写完后**立即**调用 start_preview 启动预览服务器（这一步绝对不能省）
   - 最后用一段简短文本告诉用户：生成了哪些文件 + 预览地址

回答简洁直接，独立的工具调用尽量并行执行。`;

const messages: ModelMessage[] = [];
// 预算由调用方持有，跨轮持续累计——agentLoop 只负责消费它
const budget: BudgetState = { used: 0, limit: 15000 };

const tracker = new UsageTracker('.usage/today.jsonl');

function ask() {
  rl.question('\nYou: ', async (input) => {
    const trimmed = input.trim();
    if (!trimmed || trimmed === 'exit') {
      console.log('Bye!');
      rl.close();
      return;
    }

    messages.push({ role: 'user', content: trimmed });

    await agentLoop(model, registry, messages, SYSTEM, budget, tracker);

    ask();
  });
}

await connectMCP();
console.log(`共注册 ${registry.getAll().length} 个工具：`);
for (const tool of registry.getAll()) {
  const flags = [
    tool.isConcurrencySafe ? '可并发' : '串行',
    tool.isReadOnly ? '只读' : '读写',
  ].join(', ');
  console.log(`  - ${tool.name}（${flags}）`);
}
console.log('\nSuper Agent v0.5 (type "exit" to quit)\n');
ask();
