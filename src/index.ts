import 'dotenv/config';
import { generateText } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { createMockModel } from './mock/mock-model';

const deepseek = createOpenAI({
  baseURL: 'https://api.deepseek.com',
  apiKey: process.env.DEEPSEEK_API_KEY,
});

const kimi = createOpenAI({
  baseURL: 'https://api.moonshot.cn/v1',
  apiKey: process.env.MOONSHOT_API_KEY,
});


function buildModelChain() {
  const chain: any[] = [];
  if (process.env.DEEPSEEK_API_KEY) chain.push(deepseek.chat('deepseek-reasoner'));
  if (process.env.MOONSHOT_API_KEY) chain.push(kimi.chat('kimi-k2-thinking'));
  chain.push(createMockModel());
  return chain;
}

async function main() {
  const models = buildModelChain();
  let lastError: unknown;
  for (const model of models) {
    try {
      const { text } = await generateText({
        model,
        prompt: '用一句话介绍你自己',
      });
      console.log(text);
      return;
    } catch (error) {
      lastError = error;
      console.error(`模型 ${model.modelId} 调用失败:`, error);
    }
  }
  throw lastError;
}

main().catch(console.error);
