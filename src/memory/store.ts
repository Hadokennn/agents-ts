import fs from 'node:fs';
import path from 'node:path';

export interface MemoryEntry {
  name: string;
  description: string;
  type: 'user' | 'feedback' | 'project' | 'reference';
  content: string;
  filePath: string;
}

export interface DuplicateHit {
  filename: string;
  name: string;
  description: string;
  score: number;
}

export interface SaveResult {
  status: 'saved' | 'duplicate';
  filename: string;
  message: string;
  duplicates?: DuplicateHit[];
}

const MEMORY_DIR = '.memory';
const INDEX_FILE = 'MEMORY.md';
const INDEX_HEADER = '# Memory Index';
const MAX_INDEX_ENTRIES = 5;   // 索引最多保留多少条（仅计 entry 行），超出按新鲜度淘汰最旧
const MAX_FILE_CHARS = 4000;
const DUP_THRESHOLD = 0.34;      // 同类型记忆 名称+描述 token 重叠比，≥ 即视为疑似重复

export class MemoryStore {
  private readonly baseDir: string;

  constructor(baseDir: string = '.') {
    this.baseDir = baseDir;
  }

  private get memoryDir(): string {
    return path.join(this.baseDir, MEMORY_DIR);
  }

  private get indexPath(): string {
    return path.join(this.memoryDir, INDEX_FILE);
  }

  init(): void {
    if (!fs.existsSync(this.memoryDir)) {
      fs.mkdirSync(this.memoryDir, { recursive: true });
    }
    if (!fs.existsSync(this.indexPath)) {
      fs.writeFileSync(this.indexPath, INDEX_HEADER + '\n', 'utf-8');
    }
  }

  save(entry: Omit<MemoryEntry, 'filePath'>, opts: { force?: boolean } = {}): SaveResult {
    this.init();
    const slug = entry.name
      .toLowerCase()
      .replace(/[^a-z0-9一-鿿]+/g, '-')
      .replace(/^-|-$/g, '');
    const filename = `${entry.type}_${slug}.md`;
    const filePath = path.join(this.memoryDir, filename);
    const isUpdate = fs.existsSync(filePath);

    // 新建文件时做去重检测；显式同名更新（filename 已存在）或 force=true 时跳过。
    // 命中疑似重复则不落盘，把候选项回传给模型，由它决定是更新已有还是 force 新建。
    if (!isUpdate && !opts.force) {
      const duplicates = this.findDuplicates(entry);
      if (duplicates.length > 0) {
        const list = duplicates
          .map(d => `  - ${d.filename}（name=${d.name}）：${d.description}`)
          .join('\n');
        return {
          status: 'duplicate',
          filename,
          duplicates,
          message:
            `检测到 ${duplicates.length} 条疑似同主题记忆，未创建新文件：\n${list}\n\n` +
            `· 若属同一主题：用 read 读取上面的文件，整合后用其 name 重新 save 以更新它。\n` +
            `· 若确属不同主题：重新 save 并传 force=true 强制新建。`,
        };
      }
    }

    const fileContent = [
      '---',
      `name: ${entry.name}`,
      `description: ${entry.description}`,
      `type: ${entry.type}`,
      '---',
      '',
      entry.content,
    ].join('\n');

    fs.writeFileSync(filePath, fileContent, 'utf-8');
    const evicted = this.updateIndex(entry.name, filename, entry.description);
    for (const f of evicted) this.removeFile(f); // 淘汰即真正遗忘：连记忆文件一起删，保持索引↔磁盘一致

    return {
      status: 'saved',
      filename,
      message: isUpdate ? `已更新记忆: ${filename}` : `已保存到记忆: ${filename}`,
    };
  }

  /**
   * 把 (name, filename, description) 写入索引并返回被淘汰的 filename 列表。
   * 关键点：命中已有条目时先剔除旧行、再 append 到末尾 —— 索引顺序 == 新鲜度（LRU），
   * 这样「最近更新」一定在最后，淘汰永远从最旧（最前）开始，不会误删刚更新过的记忆。
   */
  private updateIndex(name: string, filename: string, description: string): string[] {
    const raw = fs.readFileSync(this.indexPath, 'utf-8');
    const entries = raw
      .split('\n')
      .filter(l => l.trimStart().startsWith('- '))
      .filter(l => !l.includes(`(${filename})`)); // 剔除同 filename 旧行，实现“移到末尾”
    entries.push(`- [${name}](${filename}) — ${description}`);

    const evicted: string[] = [];
    while (entries.length > MAX_INDEX_ENTRIES) {
      const oldest = entries.shift()!;
      const m = oldest.match(/\(([^)]+)\)/);
      if (m) {
        evicted.push(m[1]);
        console.log(`[memory] 索引已达 ${MAX_INDEX_ENTRIES} 条上限，淘汰最旧记忆: ${m[1]}`);
      }
    }

    fs.writeFileSync(this.indexPath, [INDEX_HEADER, '', ...entries].join('\n') + '\n', 'utf-8');
    return evicted;
  }

  list(): MemoryEntry[] {
    this.init();
    const entries: MemoryEntry[] = [];
    const files = fs.readdirSync(this.memoryDir)
      .filter(f => f.endsWith('.md') && f !== INDEX_FILE);

    for (const file of files) {
      const filePath = path.join(this.memoryDir, file);
      const raw = fs.readFileSync(filePath, 'utf-8');
      const parsed = this.parseFrontmatter(raw);
      if (parsed) {
        entries.push({ ...parsed, filePath });
      }
    }
    return entries;
  }

  search(query: string): MemoryEntry[] {
    const all = this.list();
    const keywords = query.toLowerCase().split(/\s+/);
    return all.filter(entry => {
      const text = `${entry.name} ${entry.description} ${entry.content}`.toLowerCase();
      return keywords.some(kw => text.includes(kw));
    });
  }

  loadIndex(): string {
    this.init();
    const raw = fs.readFileSync(this.indexPath, 'utf-8');
    return raw.length > MAX_FILE_CHARS ? raw.slice(0, MAX_FILE_CHARS) + '\n...(已截断)' : raw;
  }

  loadFile(filename: string): string | null {
    const filePath = path.join(this.memoryDir, filename);
    if (!fs.existsSync(filePath)) return null;
    const raw = fs.readFileSync(filePath, 'utf-8');
    return raw.length > MAX_FILE_CHARS ? raw.slice(0, MAX_FILE_CHARS) + '\n...(已截断)' : raw;
  }

  delete(filename: string): boolean {
    const filePath = path.join(this.memoryDir, filename);
    if (!fs.existsSync(filePath)) return false;
    fs.unlinkSync(filePath);

    const indexContent = fs.readFileSync(this.indexPath, 'utf-8');
    const lines = indexContent.split('\n').filter(l => !l.includes(`(${filename})`));
    fs.writeFileSync(this.indexPath, lines.join('\n'), 'utf-8');
    return true;
  }

  buildPromptSection(): string {
    this.init();
    const index = this.loadIndex();
    const entries = this.list();

    if (entries.length === 0) {
      return '[记忆系统] 当前没有存储任何记忆。你可以使用 memory 工具来保存重要信息。';
    }

    const lines = [
      `[记忆系统] 共 ${entries.length} 条记忆`,
      '',
      '记忆索引：',
      index,
      '',
      '使用 memory 工具的 read 操作来读取具体记忆内容。',
      '记忆是线索，不是事实——使用前先验证其准确性。',
      '保存前先看上面的索引：同主题已存在就用其 name 更新，不要新建重复文件。',
    ];
    return lines.join('\n');
  }

  /** 找出与待存条目同类型、且 名称+描述 token 重叠度 ≥ 阈值的疑似重复记忆（按相似度降序，最多 3 条）。 */
  private findDuplicates(entry: Omit<MemoryEntry, 'filePath'>): DuplicateHit[] {
    const candidate = this.tokenize(`${entry.name} ${entry.description}`);
    if (candidate.size === 0) return [];

    const hits = this.list()
      .filter(e => e.type === entry.type && e.name !== entry.name)
      .map(e => ({
        filename: path.basename(e.filePath),
        name: e.name,
        description: e.description,
        score: this.overlap(candidate, this.tokenize(`${e.name} ${e.description}`)),
      }))
      .filter(e => e.score >= DUP_THRESHOLD)
      .sort((a, b) => b.score - a.score);

    return hits.slice(0, 3);
  }

  /** 分词：ASCII 词（含下划线切分，长度 ≥ 2）+ 相邻中文字 bigram。用于跨命名的语义重叠判断。 */
  private tokenize(text: string): Set<string> {
    const t = text.toLowerCase();
    const tokens = new Set<string>();
    for (const w of t.split(/[^a-z0-9]+/)) {
      if (w.length >= 2) tokens.add(w);
    }
    const isCjk = (c: string) => /[一-鿿]/.test(c);
    for (let i = 0; i < t.length - 1; i++) {
      if (isCjk(t[i]) && isCjk(t[i + 1])) tokens.add(t[i] + t[i + 1]);
    }
    return tokens;
  }

  /** 重叠系数 |A∩B| / min(|A|,|B|)，对长短差异更鲁棒。 */
  private overlap(a: Set<string>, b: Set<string>): number {
    if (a.size === 0 || b.size === 0) return 0;
    let inter = 0;
    for (const x of a) if (b.has(x)) inter++;
    return inter / Math.min(a.size, b.size);
  }

  private removeFile(filename: string): void {
    const p = path.join(this.memoryDir, filename);
    if (fs.existsSync(p)) fs.unlinkSync(p);
  }

  private parseFrontmatter(raw: string): Omit<MemoryEntry, 'filePath'> | null {
    const match = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
    if (!match) return null;

    const meta: Record<string, string> = {};
    for (const line of match[1].split('\n')) {
      const idx = line.indexOf(':');
      if (idx > 0) {
        meta[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
      }
    }

    const validTypes = ['user', 'feedback', 'project', 'reference'];
    if (!meta.name || !meta.type || !validTypes.includes(meta.type)) return null;

    return {
      name: meta.name,
      description: meta.description || '',
      type: meta.type as MemoryEntry['type'],
      content: match[2].trim(),
    };
  }
}
