export interface Chunk {
    id: string;
    text: string;
    source: string;
    index: number;
    tokenEstimate: number;
}

const TARGET_TOKENS = 256;
const CHARS_PER_TOKEN = 4;
const TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN;

// --- 文档切分 ---
export function chunkDocument(source: string, text: string): Chunk[] {
    const paragraphs = text.split(/\n{2,}/);
    const chunks: Chunk[] = [];
    let current = '';
    let idx = 0;

    for (const para of paragraphs) {
        const trimmed = para.trim();
        if (!trimmed) continue;

        // 当前缓冲区 + 新段落超过目标大小，先把缓冲区存下来
        if (current.length + trimmed.length + 2 > TARGET_CHARS && current.length > 0) {
            chunks.push(makeChunk(source, current.trim(), idx++));
            current = '';
        }

        // 单个段落就超过目标大小，按句子切分
        if (trimmed.length > TARGET_CHARS) {
            // 先把当前缓冲区存下来
            if (current.length > 0) {
                chunks.push(makeChunk(source, current.trim(), idx++));
                current = '';
            }
            // ... 按句子边界（句号、问号、感叹号）继续切分
            const sentences = trimmed.split(/(?<=[。！？.!?])\s*/);
            let sentBuf = '';
            for (const sent of sentences) {
                // 句子缓冲区 + 新句子超过目标大小，先把缓冲区存下来
                if (sentBuf.length + sent.length + 1 > TARGET_CHARS && sentBuf.length > 0) {
                    chunks.push(makeChunk(source, sentBuf.trim(), idx++));
                    sentBuf = '';
                }
                sentBuf += (sentBuf ? ' ' : '') + sent;
            }
            if (sentBuf.trim()) {
                current = sentBuf.trim();
            }
        } else {
            current += (current ? '\n\n' : '') + trimmed;
        }
    }

    if (current.trim()) {
        chunks.push(makeChunk(source, current.trim(), idx++));
    }

    return chunks;
}

function makeChunk(source: string, text: string, index: number): Chunk {
    return {
        id: `${source}#${index}`,
        text,
        source,
        index,
        tokenEstimate: Math.ceil(text.length / CHARS_PER_TOKEN),
    };
}
