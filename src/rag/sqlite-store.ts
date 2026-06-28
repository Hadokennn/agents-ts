import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { DIMS } from './embedder.js';
import type { Chunk } from './chunker.js';
import type { StoredChunk } from './store.js';

export class SqliteVectorStore {
  private db: Database.Database;
  private readonly model: string;

  constructor(dbPath: string = 'knowledge.db', model: string = 'unknown') {
    this.db = new Database(dbPath);
    this.model = model;
    sqliteVec.load(this.db);       // 加载向量搜索扩展
    this.createTables();
    this.warnOnModelMismatch();
  }

  private createTables() {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        source TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        embedding TEXT NOT NULL,
        model TEXT NOT NULL DEFAULT 'unknown',
        updated_at INTEGER NOT NULL
      );
    `);

    // vec0 的维度写死在建表 DDL 里，删行(DELETE)改不了它。
    // 若已存在的向量表维度 ≠ 当前 DIMS，先丢弃重建（换维度时自动迁移）。
    this.migrateVecDimension();

    this.db.exec(`
      CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
        id TEXT PRIMARY KEY,
        embedding FLOAT[${DIMS}]
      );

      CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        text, id UNINDEXED, source UNINDEXED
      );
    `);
  }

  /** vec0 维度写在 DDL 里，CREATE IF NOT EXISTS 不会改已存在的表；维度变了需丢弃重建。 */
  private migrateVecDimension(): void {
    const row = this.db
      .prepare(`SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'chunks_vec'`)
      .get() as { sql?: string } | undefined;
    if (!row?.sql) return; // 表还不存在，交给后面的 CREATE IF NOT EXISTS
    const m = row.sql.match(/float\s*\[\s*(\d+)\s*\]/i);
    const existingDim = m ? Number(m[1]) : null;
    if (existingDim !== null && existingDim !== DIMS) {
      console.warn(
        `[rag] 向量维度变更 ${existingDim} → ${DIMS}，重建 chunks_vec（请重新 embed 语料）`,
      );
      this.db.exec('DROP TABLE IF EXISTS chunks_vec');
    }
  }

  /** 库里若存在其它模型生成的向量，与当前模型向量空间不兼容，给出醒目警告。 */
  private warnOnModelMismatch(): void {
    const rows = this.db
      .prepare(`SELECT model, COUNT(*) AS n FROM chunks GROUP BY model`)
      .all() as Array<{ model: string; n: number }>;
    const others = rows.filter(r => r.model !== this.model);
    if (others.length > 0) {
      const desc = others.map(r => `${r.model}(${r.n})`).join(', ');
      console.warn(
        `[rag] ⚠️ 库中存在其它模型的向量 [${desc}]，与当前 "${this.model}" 向量空间不兼容；` +
          `请重新 embed 这些来源，否则检索结果不可靠。`,
      );
    }
  }

  addBatch(items: Array<{ chunk: Chunk; embedding: number[] }>): void {
    const tx = this.db.transaction(() => {
      for (const { chunk, embedding } of items) {
        const now = Date.now();
        // 先删除旧记录（如果存在）
        this.db.prepare('DELETE FROM chunks WHERE id = ?').run(chunk.id);
        this.db.prepare('DELETE FROM chunks_vec WHERE id = ?').run(chunk.id);
        this.db.prepare('DELETE FROM chunks_fts WHERE id = ?').run(chunk.id);
        
        // 三表联动写入
        this.db.prepare(`INSERT INTO chunks
          (id, text, source, chunk_index, embedding, model, updated_at)
          VALUES (?, ?, ?, ?, ?, ?, ?)`)
          .run(chunk.id, chunk.text, chunk.source, chunk.index,
               JSON.stringify(embedding), this.model, now);

        this.db.prepare(`INSERT INTO chunks_vec (id, embedding)
          VALUES (?, ?)`)
          .run(chunk.id, Buffer.from(new Float32Array(embedding).buffer));

        this.db.prepare(`INSERT INTO chunks_fts (id, text, source)
          VALUES (?, ?, ?)`)
          .run(chunk.id, chunk.text, chunk.source);
      }
    });
    tx();  // 事务批量写入，比逐条快很多
  }

  vectorSearch(queryEmbedding: number[], topK: number): Array<{ chunk: StoredChunk; score: number }> {
    const buf = Buffer.from(new Float32Array(queryEmbedding).buffer);
    const rows = this.db.prepare(`
      SELECT v.id, v.distance, c.text, c.source, c.chunk_index, c.embedding
      FROM chunks_vec v
      JOIN chunks c ON c.id = v.id
      WHERE v.embedding MATCH ?
      ORDER BY v.distance
      LIMIT ?
    `).all(buf, topK) as any[];

    return rows.map(r => ({
      chunk: {
        id: r.id, text: r.text, source: r.source,
        index: r.chunk_index,
        tokenEstimate: Math.ceil(r.text.length * 0.6),
        embedding: JSON.parse(r.embedding),
        addedAt: 0,
      },
      score: 1 - r.distance,  // cosine distance → similarity
    }));
  }

  keywordSearch(query: string, topK: number): Array<{ chunk: StoredChunk; score: number }> {
    const rows = this.db.prepare(`
      SELECT f.id, bm25(chunks_fts) AS rank, c.text, c.source, c.chunk_index, c.embedding
      FROM chunks_fts f
      JOIN chunks c ON c.id = f.id
      WHERE chunks_fts MATCH ?
      ORDER BY rank
      LIMIT ?
    `).all(query, topK) as any[];

    return rows.map(r => ({
      chunk: {
        id: r.id, text: r.text, source: r.source,
        index: r.chunk_index,
        tokenEstimate: Math.ceil(r.text.length * 0.6),
        embedding: JSON.parse(r.embedding),
        addedAt: 0,
      },
      score: r.rank < 0 ? -r.rank / (1 - r.rank) : 1 / (1 + r.rank),
    }));
  }

  size(): number {
    return (this.db.prepare('SELECT COUNT(*) as n FROM chunks').get() as any).n;
  }

  sources(): string[] {
    return (this.db.prepare('SELECT DISTINCT source FROM chunks').all() as any[]).map(r => r.source);
  }

  getAll(): Array<StoredChunk> {
    const rows = this.db.prepare('SELECT * FROM chunks').all() as any[];
    return rows.map(r => ({
      id: r.id,
      text: r.text,
      source: r.source,
      index: r.chunk_index,
      tokenEstimate: Math.ceil(r.text.length * 0.6),
      embedding: JSON.parse(r.embedding),
      addedAt: r.updated_at,
    }));
  }
}
