import { jsonSchema } from 'ai';
import { HandwrittenMCPClient, SDKMCPClient, MockMCPClient } from '../mcp-client.js';
export interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  isConcurrencySafe?: boolean;
  isReadOnly?: boolean;
  maxResultChars?: number;
  shouldDefer?: boolean; // 是否延迟加载
  searchHint?: string; // 搜索提示词，帮助 ToolSearch 匹配
  execute: (input: any) => Promise<unknown>;
}

const DEFAULT_MAX_RESULT_CHARS = 3000;

type MCPClientType = HandwrittenMCPClient | SDKMCPClient;

export class ToolRegistry {
  private tools = new Map<string, ToolDefinition>();
  private mcpClients: Array<MCPClientType | MockMCPClient> = [];
  private registeredMCPServers = new Set<string>();
  private discoveredTools = new Set<string>();

  // 三个状态变量构成一把读写锁
  private exclusiveLock = false;          // 当前是否有独占锁持有者
  private concurrentCount = 0;            // 当前共享锁持有数
  private waitQueue: Array<() => void> = [];  // 阻塞等待中的 resolve 函数

  register(...tools: ToolDefinition[]): void {
    for (const tool of tools) {
      this.tools.set(tool.name, tool);
    }
  }

  get(name: string): ToolDefinition | undefined {
    return this.tools.get(name);
  }

  getAll(): ToolDefinition[] {
    return Array.from(this.tools.values());
  }

  // 获取共享锁：只要没人独占就能拿，多个只读工具可以同时持有
  private async acquireConcurrent(): Promise<void> {
    while (this.exclusiveLock) {
      await new Promise<void>(r => this.waitQueue.push(r));
    }
    this.concurrentCount++;
  }

  private releaseConcurrent(): void {
    this.concurrentCount--;
    if (this.concurrentCount === 0) this.drainQueue();
  }

  // 获取独占锁：必须等所有共享锁释放、且没人持独占
  private async acquireExclusive(): Promise<void> {
    while (this.exclusiveLock || this.concurrentCount > 0) {
      await new Promise<void>(r => this.waitQueue.push(r));
    }
    this.exclusiveLock = true;
  }

  private releaseExclusive(): void {
    this.exclusiveLock = false;
    this.drainQueue();
  }

  // 锁释放时把等待队列全唤醒，让它们重新去抢锁
  private drainQueue(): void {
    const waiting = this.waitQueue.splice(0);
    for (const resolve of waiting) resolve();
  }

  toAISDKFormat(): Record<string, any> {
    const result: Record<string, any> = {};
    const activeTools = this.getActiveTools();

    for (const tool of activeTools) {
      const maxChars = tool.maxResultChars;
      const executeFn = tool.execute;
      const isSafe = tool.isConcurrencySafe === true;
      const registry = this;

      result[tool.name] = {
        description: tool.description,
        inputSchema: jsonSchema(tool.parameters as any),
        execute: async (input: any) => {
          // 在真正执行前先按 isConcurrencySafe 获取锁
          if (isSafe) {
            await registry.acquireConcurrent();
            console.log(`  [并发] ${tool.name} 获取共享锁`);
          } else {
            await registry.acquireExclusive();
            console.log(`  [串行] ${tool.name} 获取独占锁，等待其他工具完成`);
          }
          try {
            const raw = await executeFn(input);
            const text = typeof raw === 'string' ? raw : JSON.stringify(raw, null, 2);
            return truncateResult(text, maxChars);
          } finally {
            // 不管成功还是抛异常，锁都要释放
            if (isSafe) {
              registry.releaseConcurrent();
            } else {
              registry.releaseExclusive();
            }
          }
        },
      };
    }
    return result;
  }

  async registerMCPServer(
    serverName: string,
    client: MCPClientType | MockMCPClient,
  ): Promise<string[]> {
    // 检查是否已注册
    if (this.registeredMCPServers.has(serverName)) {
      console.log(`  [MCP] Server "${serverName}" 已注册，跳过`);
      return [];
    }

    await client.connect();

    // 先尝试拉取工具，成功后才登记这个 server。
    // 否则连接成功但 listTools 失败时，serverName 已被标记为已注册，
    // 后续 Mock 降级会被开头的 "已注册，跳过" 挡掉，导致 0 工具。
    let tools: Awaited<ReturnType<typeof client.listTools>>;
    try {
      tools = await client.listTools();
    } catch (err) {
      await client.close().catch(() => { });
      throw err;
    }

    this.mcpClients.push(client);
    this.registeredMCPServers.add(serverName);

    const registered: string[] = [];

    for (const tool of tools) {
      const prefixedName = `mcp__${serverName}__${tool.name}`;
      if (this.tools.has(prefixedName)) continue;

      const toolClient = client;
      const originalName = tool.name;

      this.register({
        name: prefixedName,
        description: `[MCP:${serverName}] ${tool.description}`,
        parameters: tool.inputSchema as Record<string, unknown>,
        isConcurrencySafe: true,
        isReadOnly: true,
        shouldDefer: true,
        searchHint: `${serverName} ${tool.name} ${tool.description}`,
        maxResultChars: 3000,
        execute: async (input: any) => {
          return toolClient.callTool(originalName, input);
        },
      });

      registered.push(prefixedName);
    }

    return registered;
  }

  async closeAllMCP(): Promise<void> {
    for (const client of this.mcpClients) {
      await client.close();
    }
    this.mcpClients = [];
  }

  searchTools(query: string): ToolDefinition[] {
    const q = query.trim();
    const results: ToolDefinition[] = [];

    const names = q.includes(',')
      ? q.split(',').map(n => n.trim()).filter(Boolean)
      : [q];

    for (const name of names) {
      const tool = this.tools.get(name);
      if (tool && tool.name !== 'tool_search') {
        results.push(tool);
        this.discoveredTools.add(tool.name);
      }
    }
    return results;
  }

  getActiveTools(): ToolDefinition[] {
    return this.getAll().filter(tool => {
      if (tool.shouldDefer && !this.discoveredTools.has(tool.name)) {
        return false;
      }
      return true;
    });
  }

  getDeferredToolSummary(): string {
    const deferred = this.getAll().filter(tool => {
      return tool.shouldDefer && !this.discoveredTools.has(tool.name);
    });

    if (deferred.length === 0) return '';

    const lines = deferred.map(t => {
      const hint = t.searchHint ? ` — ${t.searchHint}` : '';
      return `  - ${t.name}${hint}`;
    });

    return `\n以下工具可用，但需要先通过 tool_search 搜索获取完整定义：\n${lines.join('\n')}`;
  }

  countTokenEstimate(): { active: number; deferred: number; total: number } {
    let active = 0;
    let deferred = 0;

    for (const tool of this.tools.values()) {
      const schemaSize = JSON.stringify({
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
      }).length;
      const tokens = Math.ceil(schemaSize / 4);

      if (tool.shouldDefer && !this.discoveredTools.has(tool.name)) {
        deferred += tokens;
      } else {
        active += tokens;
      }
    }

    return { active, deferred, total: active + deferred };
  }

}

export function truncateResult(text: string, maxChars: number = DEFAULT_MAX_RESULT_CHARS): string {
  if (text.length <= maxChars) return text;

  const headSize = Math.floor(maxChars * 0.6);
  const tailSize = maxChars - headSize;
  const head = text.slice(0, headSize);
  const tail = text.slice(-tailSize);
  const dropped = text.length - headSize - tailSize;

  return `${head}\n\n... [省略 ${dropped} 字符] ...\n\n${tail}`;
}

