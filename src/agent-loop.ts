import { streamText, type ModelMessage } from 'ai';
import { ToolRegistry } from './tools/registry.js';
import { detect, recordCall, recordResult, resetHistory } from './loop-detection.js';
import { isRetryable, calculateDelay, sleep } from './retry.js';
import { type UsageTracker, normalizeUsage } from './usage/tracker.js';

const MAX_STEPS = 15;
const MAX_RETRIES = 3;

export type BudgetState = {
  used: number;
  limit: number;
};

interface ReasoningChainStep {
  step: number;
  type: 'text' | 'tool-call' | 'tool-result';
  content: string;
  toolName?: string;
  toolInput?: unknown;
  toolOutput?: unknown;
}

export async function agentLoop(
  model: any,
  registry: ToolRegistry,
  messages: ModelMessage[],
  system: string,
  budget: BudgetState,
  tracker: UsageTracker,
) {
  let step = 0;
  let totalTokens = 0;
  const reasoningChain: ReasoningChainStep[] = [];
  resetHistory();

  while (step < MAX_STEPS) {
    step++;
    console.log(`\n--- Step ${step} ---`);

    let hasToolCall = false;
    let fullText = '';
    let shouldBreak = false;
    let lastToolCall: { name: string; input: unknown } | null = null;
    let stepResponse: any;
    let stepUsage: any;

    for (let attempt = 1; ; attempt++) {
      try {
        const result = streamText({
          model,
          system,
          tools: registry.toAISDKFormat(),
          messages,
          maxRetries: 0,
          providerOptions: { openai: { parallelToolCalls: true } },
          onError: () => {},
        });

        for await (const part of result.fullStream) {
          switch (part.type) {
            case 'text-delta':
              process.stdout.write(part.text);
              fullText += part.text;
              break;

            case 'tool-call': 
              hasToolCall = true;
              lastToolCall = { name: part.toolName, input: part.input };
              console.log(`  [调用: ${part.toolName}(${JSON.stringify(part.input)})]`);
              
              reasoningChain.push({
                step,
                type: 'tool-call',
                content: `调用工具 ${part.toolName}`,
                toolName: part.toolName,
                toolInput: part.input,
              });

              const detection = detect(part.toolName, part.input);
              if (detection.stuck) {
                console.log(`  ${detection.message}`);
                if (detection.level === 'critical') {
                  shouldBreak = true;
                } else {
                  messages.push({
                    role: 'user' as const,
                    content: `[系统提醒] ${detection.message}。请换一个思路解决问题，不要重复同样的操作。`,
                  });
                }
              }
              recordCall(part.toolName, part.input);
              break;

            case 'tool-result': 
              const output = typeof part.output === 'string' ? part.output : JSON.stringify(part.output);
              const preview = output.length > 120 ? output.slice(0, 120) + '...' : output;
              console.log(`  [结果: ${part.toolName}] ${preview}`);
              
              reasoningChain.push({
                step,
                type: 'tool-result',
                content: `工具 ${part.toolName} 返回结果`,
                toolName: part.toolName,
                toolOutput: part.output,
              });

              if (lastToolCall) {
                recordResult(lastToolCall.name, lastToolCall.input, part.output);
              }
              break;
          }
        }

        stepResponse = await result.response;
        stepUsage = await result.usage;
        break;
      } catch (error) {
        if (attempt > MAX_RETRIES || !isRetryable(error as Error)) throw error;
        const delay = calculateDelay(attempt);
        console.log(`  [重试] 第 ${attempt}/${MAX_RETRIES} 次，${delay}ms 后...`);
        await sleep(delay);
        hasToolCall = false;
        fullText = '';
        shouldBreak = false;
        lastToolCall = null;
      }
    }

    if (shouldBreak) {
      console.log('\n[循环检测触发，Agent 已停止]');
      break;
    }

    messages.push(...stepResponse!.messages);

    // 记录文本响应到推理链
    if (fullText) {
      reasoningChain.push({
        step,
        type: 'text',
        content: fullText,
      });
    }

    // 把 usage 喂给 tracker；tracker 内部按四类 token 分别累加并算 cost
    const norm = normalizeUsage(stepUsage);
    const stepRecord = tracker?.record(model?.modelId || 'mock-model', norm);
    totalTokens += norm.inputTokens + norm.outputTokens + norm.cacheReadTokens + norm.cacheWriteTokens;

    // cache 命中时才打印一行简洁状态，让 cache hit 立刻可见
    if (stepRecord && (norm.cacheReadTokens > 0 || norm.cacheWriteTokens > 0)) {
      const tag = norm.cacheReadTokens > 0 ? `\x1b[38;5;36m✓ cache hit\x1b[0m` : `\x1b[38;5;220m✎ cache write\x1b[0m`;
      const detail = norm.cacheReadTokens > 0 ? `read ${norm.cacheReadTokens}` : `write ${norm.cacheWriteTokens}`;
      console.log(`\n[${tag}] ${detail} tokens · 本步 $${stepRecord.cost.toFixed(5)}`);
    }

    if (totalTokens > budget?.limit * 0.9) {
      console.log(`  [Token] ${totalTokens}/${budget?.limit} (${Math.round(totalTokens / budget?.limit * 100)}%)`);
    }
    if (totalTokens > budget?.limit) {
      console.log('\n[Token 预算耗尽]');
      break;
    }

    if (!hasToolCall) {
      if (fullText) console.log();
      break;
    }

    console.log('  → 继续下一步...');
  }

  if (step >= MAX_STEPS) {
    console.log('\n[达到最大步数]');
  }

  // 打印推理链
  console.log('\n' + '='.repeat(60));
  console.log('🧠 推理链');
  console.log('='.repeat(60));
  
  let currentStep = 0;
  for (const chainStep of reasoningChain) {
    if (chainStep.step !== currentStep) {
      currentStep = chainStep.step;
      console.log(`\n--- 步骤 ${currentStep} ---`);
    }

    switch (chainStep.type) {
      case 'text':
        console.log(`📝 文本回复: ${chainStep.content}`);
        break;
      case 'tool-call':
        console.log(`🔧 调用工具: ${chainStep.toolName}`);
        console.log(`   参数: ${JSON.stringify(chainStep.toolInput)}`);
        break;
      case 'tool-result':
        console.log(`✅ 工具结果: ${chainStep.toolName}`);
        const resultStr = typeof chainStep.toolOutput === 'string' 
          ? chainStep.toolOutput 
          : JSON.stringify(chainStep.toolOutput, null, 2);
        console.log(`   返回: ${resultStr.length > 200 ? resultStr.slice(0, 200) + '...' : resultStr}`);
        break;
    }
  }
  console.log('\n' + '='.repeat(60) + '\n');
}
