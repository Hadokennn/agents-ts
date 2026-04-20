import 'dotenv/config';
import { type ModelMessage } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { weatherTool, calculatorTool } from './tools';
import { createInterface } from 'node:readline';
import { agentLoop } from './agent-loop';

const tools = { get_weather: weatherTool, calculator: calculatorTool };

const deepseek = createOpenAI({
  baseURL: 'https://api.deepseek.com',
  apiKey: process.env.DEEPSEEK_API_KEY,
});

const kimi = createOpenAI({
  baseURL: 'https://api.moonshot.cn/v1',
  apiKey: process.env.MOONSHOT_API_KEY,
});

const rl = createInterface({
  input: process.stdin,
  output: process.stdout,
});

const SYSTEM = `你是 Super Agent，一个有工具调用能力的 AI 助手。
需要查询信息时，主动使用工具，不要编造数据。
回答要简洁直接。`;

const messages: ModelMessage[] = [];

function ask() {
  rl.question('\nYou: ', async (input) => {
    const trimmed = input.trim();
    if (!trimmed || trimmed === 'exit') {
      console.log('Bye!');
      rl.close();
      return;
    }

    messages.push({ role: 'user', content: trimmed });

    await agentLoop(deepseek.chat('deepseek-reasoner'), tools, messages, SYSTEM);

    ask();
  });
}

console.log('Super Agent v0.2 (type "exit" to quit)\n');
ask();
