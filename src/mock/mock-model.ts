/**
 * Mock Model — 本地模拟模型
 *
 * 手动实现 LanguageModelV3 接口，不依赖 ai/test（避免 msw 等额外依赖）。
 * 模拟流式输出，不需要网络请求就能跑。
 *
 * 本地终端开发时，把 model 换成真实的 Qwen 即可（文章里有教）。
 */

const RESPONSES: Record<string, string> = {
  default: '你好！我是 Super Agent 的模拟模型。当前使用本地模拟回复，流式输出的机制和真实 API 完全一样。\n\n在 .env 里填入 DASHSCOPE_API_KEY 即可切换到真实的 Qwen 模型。',
  greeting: '你好！我是模拟模式下的 Super Agent。虽然是模拟的，但流式输出的效果和真实 API 一致 :)',
  name: '你刚才告诉我了呀！不过说实话，我是模拟模型，我能"记住"是因为代码把对话历史传给了我。这就是 messages 数组的作用——大模型本身没有记忆，是你在"提醒"它。',
  intro: '我是通义千问（模拟版），一个大语言模型。当前使用本地模拟回复，机制和真实 API 调用完全一致。',
  code: '模拟模式下我没法真的写代码，但流式输出的体验你已经感受到了。填上真实的 API Key 就能解锁完整能力。',
};

function extractUserText(prompt: any[]): string {
  const userMsgs = (prompt || []).filter((m: any) => m.role === 'user');
  const last = userMsgs[userMsgs.length - 1];
  if (!last) return '';
  return (last.content || []).map((c: any) => c.text || '').join('').toLowerCase();
}

function pickResponse(prompt: any[]): string {
  const last = extractUserText(prompt);
  if (last.includes('介绍你自己') || last.includes('你是谁')) return RESPONSES.intro;
  if (last.includes('你好') || last.includes('hello') || last.includes('hi')) return RESPONSES.greeting;
  if (last.includes('叫什么') || last.includes('名字') || last.includes('记得') || last.includes('记住')) return RESPONSES.name;
  if (last.includes('代码') || last.includes('code') || last.includes('函数')) return RESPONSES.code;
  return RESPONSES.default;
}

const USAGE = {
  inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: 20, text: 20, reasoning: undefined },
};

/** 将数组转为模拟的 ReadableStream，每个 chunk 间隔 delayMs 毫秒 */
function createDelayedStream(chunks: any[], delayMs = 30): ReadableStream {
  return new ReadableStream({
    start(controller) {
      let i = 0;
      function next() {
        if (i < chunks.length) {
          controller.enqueue(chunks[i++]);
          setTimeout(next, delayMs);
        } else {
          controller.close();
        }
      }
      next();
    },
  });
}

export function createMockModel() {
  return {
    specificationVersion: 'v2' as const,
    provider: 'mock',
    modelId: 'mock-model',

    get supportedUrls() {
      return Promise.resolve({});
    },

    async doGenerate({ prompt }: any) {
      return {
        content: [{ type: 'text' as const, text: pickResponse(prompt) }],
        finishReason: { unified: 'stop' as const, raw: undefined },
        usage: USAGE,
        warnings: [],
      };
    },

    async doStream({ prompt }: any) {
      const text = pickResponse(prompt);
      const id = 'text-1';
      const chunks: any[] = [
        { type: 'text-start', id },
        ...text.split('').map((char: string) => ({ type: 'text-delta', id, delta: char })),
        { type: 'text-end', id },
        { type: 'finish', finishReason: { unified: 'stop', raw: undefined }, usage: USAGE },
      ];

      return { stream: createDelayedStream(chunks, 30) };
    },
  };
}
