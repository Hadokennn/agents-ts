// --- 错误分类 ---

export function isRetryable(error: unknown): boolean {
  // 检查 Symbol 类型的 AI SDK 错误（如 AI_NoOutputGeneratedError）
  if (error && typeof error === 'object') {
    const symbols = Object.getOwnPropertySymbols(error);
    for (const sym of symbols) {
      if (String(sym).includes('AI_NoOutputGeneratedError') || 
          String(sym).includes('NoOutputGeneratedError')) {
        return true;
      }
    }
  }

  if (!(error instanceof Error)) return false;

  const message = error.message || '';

  // HTTP 状态码判断
  const statusMatch = message.match(/(\d{3})/);
  if (statusMatch) {
    const status = parseInt(statusMatch[1]);
    if ([429, 529, 408].includes(status)) return true;
    if (status >= 500 && status < 600) return true;
    if (status >= 400 && status < 500) return false;
  }

  // 网络错误
  if (message.includes('ECONNRESET') || message.includes('EPIPE')) return true;
  if (message.includes('ETIMEDOUT') || message.includes('timeout')) return true;
  if (message.includes('fetch failed') || message.includes('network')) return true;
  // AI SDK 会把流式错误包装成 NoOutputGeneratedError
  if (message.includes('No output generated')) return true;

  return false;
}

// --- 指数退避 + 随机抖动 ---

export function calculateDelay(attempt: number, baseMs = 500, maxMs = 30000): number {
  const exponential = baseMs * Math.pow(2, attempt - 1);
  const capped = Math.min(exponential, maxMs);
  const jitterRange = capped * 0.25;
  const jittered = capped + (Math.random() * 2 - 1) * jitterRange;
  return Math.max(0, Math.round(jittered));
}

export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
