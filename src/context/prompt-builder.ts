import { ToolRegistry } from '../tools/index.js';

export interface PromptContext {
    toolRegistry: ToolRegistry;
    deferredToolSummary: string;
    sessionMessageCount: number;
    sessionId: string;
}

type PipeFn = (ctx: PromptContext) => string | null;

export class PromptBuilder {
    private pipes: Array<{ name: string; fn: PipeFn }> = [];

    pipe(name: string, fn: PipeFn): this {
        this.pipes.push({ name, fn });
        return this;
    }

    build(ctx: PromptContext): string {
        const sections: string[] = [];
        for (const { fn } of this.pipes) {
            const result = fn(ctx);
            if (result !== null) {
                sections.push(result);
            }
        }
        return sections.join('\n\n');
    }

    debug(ctx: PromptContext): void {
        console.log('\n=== Prompt Pipe Debug ===');
        for (const { name, fn } of this.pipes) {
            const result = fn(ctx);
            const status = result !== null
                ? `[ON] ${result.length} chars` : '[OFF]';
            console.log(`  ${name}: ${status}`);
        }
        console.log('========================\n');
    }
}

export function coreRules(): PipeFn {
    return () => `你是 Super Agent，一个能读代码、抓网页、生成项目的 AI 助手。
    你的行为准则：
    - 先读文件再修改，不要凭记忆编辑
    - 不要加没被要求的功能
    - 工具调用失败时，换一个思路而不是重复同样的操作
    - 回答要简洁直接
`;
}

export function toolGuide(): PipeFn {
    return (ctx) => {
        const activeTools = ctx.toolRegistry.getActiveTools();
        if (activeTools.length === 0) return null;
        return `你有 ${activeTools.length} 个工具可用: ${activeTools.map((tool) => tool.name).join(', ')};\n`;
    };
}

export function sessionContext(): PipeFn {
    return (ctx) => {
        if (ctx.sessionMessageCount === 0) return null;
        return `[会话信息] 当前会话 ${ctx.sessionId}，已有 ${ctx.sessionMessageCount} 条历史消息。`;
    };
}

export function deferredTools(): PipeFn {
    return (ctx) => {
        if (ctx.deferredToolSummary === '') return null;
        return `如果你需要的工具不在当前列表中，使用 tool_search 工具搜索。${ctx.deferredToolSummary}`;
    };
}

export function strategies(): PipeFn {
    return () => `针对常见任务的执行策略：

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
  `;
}